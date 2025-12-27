# --- STANDARD ---
import gc
import json
import os
import tempfile
from datetime import date

os.environ["TF_DATA_AUTOTUNE"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY ---
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# --- CONFIG ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "15"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
LAMBDA_DIR = float(os.getenv("LAMBDA_DIR", "5.0"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
MARGIN = float(os.getenv("MARGIN", "0.01"))
PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")
SYMBOL_EMBED = int(os.getenv("SYMBOL_EMBED", "64"))
TARGET_SCALE = float(os.getenv("TARGET_SCALE", "1000.0"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))

VAL_DAYS = WINDOW_SIZE * 2

LOGGER = s.setup_logger(__file__)


def deduplicate(table):
    seen = set()
    keep_indices: list[int] = []

    dates = table[s.LOCAL_DATE]
    for i in range(table.num_rows - 1, -1, -1):
        d = dates[i].as_py()
        if d not in seen:
            seen.add(d)
            keep_indices.append(i)

    keep_indices.reverse()
    return table.take(pa.array(keep_indices))


def get_all_keys_and_deduplicated_table(
    prefix: str, schema
) -> tuple[list[str], pa.Table]:
    keys = s.list_keys_s3(prefix)
    tables = [s.read_parquet_s3(key, schema) for key in keys]
    combined = pa.concat_tables(tables, promote_options="none")
    deduplicated = deduplicate(combined)
    return keys, deduplicated


def consolidate(current_keys: list[str], table: pa.Table, prefix: str, schema):
    if len(current_keys) <= PARTIAL_FILE_LIMIT:
        return

    timestamp = s.get_now_timestamp()
    new_key = f"{prefix}{timestamp}.parquet"

    s.write_parquet_s3(new_key, table, schema, CONSOLIDATED_COMPRESSION_LEVEL)

    for key in current_keys:
        s.delete_s3(key)


def get_data_frame(prefix: str, schema):
    keys, table = get_all_keys_and_deduplicated_table(prefix, schema)

    consolidate(keys, table, prefix, schema)

    df = table.to_pandas()
    df[s.LOCAL_DATE] = pd.to_datetime(df[s.LOCAL_DATE])
    df = df.set_index(s.LOCAL_DATE).sort_index()
    return df


def get_training_data():
    stock_data = get_data_frame(s.PIVOTED_PREFIX, s.get_pivoted_schema())
    news_data = get_data_frame(s.LAGGED_PREFIX, s.get_lagged_schema())
    target_data = get_data_frame(s.TARGET_PREFIX, s.get_target_schema())

    common_index = stock_data.index.intersection(news_data.index).intersection(
        target_data.index
    )

    if common_index.empty:
        raise ValueError("No overlapping dates across inputs")

    stock_data = stock_data.loc[common_index]
    news_data = news_data.loc[common_index]
    target_data = target_data.loc[common_index]

    if not (
        stock_data.index.equals(news_data.index)
        and stock_data.index.equals(target_data.index)
    ):
        raise ValueError("Hard fail: misaligned indices")

    return stock_data, news_data, target_data


def build_windows(stock_data, news_data, target_data):
    symbols = s.get_symbols()
    symbol_count = len(symbols)
    window = WINDOW_SIZE

    # ---- STOCK FEATURES ----
    stock_cols = s.get_pivoted_schema().names[1:]  # excluding LOCAL_DATE

    per_symbol_features = len(stock_cols) // symbol_count
    assert per_symbol_features * symbol_count == len(stock_cols)

    # (time, symbols * features)
    stock_matrix = stock_data[stock_cols].to_numpy(dtype="float32")

    # (time, symbols, features)
    stock_matrix = stock_matrix.reshape(
        len(stock_data),
        symbol_count,
        per_symbol_features,
    )

    # ---- NEWS ----
    news_matrix = news_data.to_numpy(dtype="float32")

    # ---- TARGETS ----
    z_cols = [f"{sym}_zscore_1d" for sym in symbols]
    rank_cols = [f"{sym}_rank_1d" for sym in symbols]
    dir_cols = [f"{sym}_direction_1d" for sym in symbols]
    lr1_cols = [f"{sym}_log_return_1d" for sym in symbols]
    lr3_cols = [f"{sym}_log_return_3d" for sym in symbols]
    lr5_cols = [f"{sym}_log_return_5d" for sym in symbols]

    Y_z_all = target_data[z_cols].to_numpy("float32")
    Y_rank_all = target_data[rank_cols].to_numpy("float32")
    Y_dir_all = target_data[dir_cols].to_numpy("float32")
    Y_lr1_all = target_data[lr1_cols].to_numpy("float32")
    Y_lr3_all = target_data[lr3_cols].to_numpy("float32")
    Y_lr5_all = target_data[lr5_cols].to_numpy("float32")

    # ---- WINDOWING ----
    num_samples = len(stock_data) - window

    X_stock = np.empty(
        (num_samples, window, symbol_count, per_symbol_features),
        dtype="float32",
    )
    X_news = np.empty(
        (num_samples, window, news_matrix.shape[1]),
        dtype="float32",
    )

    Y_z = Y_z_all[window:]
    Y_rank = Y_rank_all[window:]
    Y_dir = Y_dir_all[window:]
    Y_lr1 = Y_lr1_all[window:]
    Y_lr3 = Y_lr3_all[window:]
    Y_lr5 = Y_lr5_all[window:]

    for i in range(num_samples):
        t = i + window
        X_stock[i] = stock_matrix[t - window : t]
        X_news[i] = news_matrix[t - window : t]

    return (
        X_stock,
        X_news,
        Y_z,
        Y_rank,
        Y_dir,
        Y_lr1,
        Y_lr3,
        Y_lr5,
    )


class SymbolAttention(layers.Layer):
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, x):
        # x: (batch, time, symbols, features)
        b, t, s, f = tf.unstack(tf.shape(x))  # type: ignore

        x = tf.reshape(x, (b * t, s, f))
        x = self.mha(x, x)
        x = tf.reshape(x, (b, t, s, f))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.mha.num_heads,
                "key_dim": self.mha.key_dim,
            }
        )
        return config


def build_model(time_steps, stock_feature_dim, news_feature_dim, symbol_count):

    stock_in = layers.Input(
        shape=(time_steps, symbol_count, stock_feature_dim), name="stock"
    )
    news_in = layers.Input(shape=(time_steps, news_feature_dim), name="news")

    # --- project per-symbol features ---
    x = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(SYMBOL_EMBED, activation="relu"))
    )(stock_in)
    # (batch, time, symbols, SYMBOL_EMBED)

    # --- symbol attention (same day) ---
    x = SymbolAttention(num_heads=4, key_dim=SYMBOL_EMBED // 4)(x)
    # x shape here: (batch, symbols, time, embed)

    # --- reduce symbols ---
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name="pool_symbols",
    )(x)
    # shape: (batch, symbols, embed)

    # --- cross-symbol mixing ---
    x = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=SYMBOL_EMBED // 4,
    )(x, x)

    # --- reduce time ---
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="pool_time",
    )(x)
    # shape: (batch, embed)

    # --- news encoder ---
    y = layers.MultiHeadAttention(num_heads=4, key_dim=16)(news_in, news_in)
    y = layers.GlobalAveragePooling1D()(y)

    # --- fuse ---
    x = layers.Concatenate()([x, y])

    # --- shared trunk ---
    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.Dense(128, activation="relu")(shared)

    # --- heads ---
    def head(name, activation=None):
        return layers.Dense(symbol_count, activation=activation, name=name)(shared)

    model = models.Model(
        inputs=[stock_in, news_in],
        outputs=[
            head("zscore_1d"),
            head("rank_1d"),
            head("direction_1d", activation="sigmoid"),
            head("log_return_1d"),
            head("log_return_3d"),
            head("log_return_5d"),
        ],
    )

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss={
            "zscore_1d": "mse",
            "rank_1d": losses.Huber(delta=0.1),
            "direction_1d": "binary_crossentropy",
            "log_return_1d": "mse",
            "log_return_3d": "mse",
            "log_return_5d": "mse",
        },
        loss_weights={
            "zscore_1d": 1.0,
            "rank_1d": 0.3,
            "direction_1d": 0.3,
            "log_return_1d": 0.05,
            "log_return_3d": 0.05,
            "log_return_5d": 0.05,
        },
    )

    return model


def write_model_s3(model, model_key):
    with tempfile.TemporaryDirectory() as tmp:
        local_model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
        model.save(local_model_path)

        with open(local_model_path, "rb") as f:
            model_data = f.read()

        s.write_bytes_s3(model_key, model_data)


def train():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.optimizer.set_jit(False)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        stock_data, news_data, target_data = get_training_data()
        LOGGER.debug("get_training_data")
        (
            X_stock,
            X_news,
            Y_z,
            Y_rank,
            Y_dir,
            Y_lr_1d,
            Y_lr_3d,
            Y_lr_5d,
        ) = build_windows(
            stock_data,
            news_data,
            target_data,
        )
        LOGGER.debug("build_windows")

        Y_lr_1d *= TARGET_SCALE
        Y_lr_3d *= TARGET_SCALE
        Y_lr_5d *= TARGET_SCALE

        Y_lr_1d = np.clip(Y_lr_1d, -10 * TARGET_SCALE, 10 * TARGET_SCALE)
        Y_lr_3d = np.clip(Y_lr_3d, -10 * TARGET_SCALE, 10 * TARGET_SCALE)
        Y_lr_5d = np.clip(Y_lr_5d, -10 * TARGET_SCALE, 10 * TARGET_SCALE)

        assert X_stock.shape[2] == len(s.get_symbols())

        symbol_count = Y_z.shape[1]
        time_steps = X_stock.shape[1]
        stock_feature_dim = X_stock.shape[3]
        news_feature_dim = X_news.shape[2]

        model = build_model(
            time_steps,
            stock_feature_dim,
            news_feature_dim,
            symbol_count,
        )

        LOGGER.info(model.summary())

        split = int(0.8 * len(X_stock))

        # keep only what fit needs
        del stock_data, news_data, target_data
        gc.collect()

        model.fit(
            x={"stock": X_stock[:split], "news": X_news[:split]},
            y={
                "zscore_1d": Y_z[:split],
                "rank_1d": Y_rank[:split],
                "direction_1d": Y_dir[:split],
                "log_return_1d": Y_lr_1d[:split],
                "log_return_3d": Y_lr_3d[:split],
                "log_return_5d": Y_lr_5d[:split],
            },
            validation_data=(
                {"stock": X_stock[split:], "news": X_news[split:]},
                {
                    "zscore_1d": Y_z[split:],
                    "rank_1d": Y_rank[split:],
                    "direction_1d": Y_dir[split:],
                    "log_return_1d": Y_lr_1d[split:],
                    "log_return_3d": Y_lr_3d[split:],
                    "log_return_5d": Y_lr_5d[split:],
                },
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.0,
            shuffle=True,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                ),
            ],
        )

        # --- persist ---
        training_date = date.today().isoformat()
        model_key = f"{s.MODEL_PREFIX}{training_date}/{s.MODEL_FILE_NAME}"
        meta_key = f"{s.MODEL_PREFIX}{training_date}/{s.META_FILE_NAME}"

        write_model_s3(model, model_key)

        meta_out = {
            "windowSize": time_steps,
            "symbolCount": symbol_count,
            "trainingCutoff": training_date,
        }
        s.write_bytes_s3(meta_key, json.dumps(meta_out).encode("utf-8"))

        s.write_json_date_s3(
            s.TRAINING_STATE_KEY,
            s.LAST_TRAINED_KEY,
            date.fromisoformat(training_date),
        )

        return s.result(LOGGER, 200, "training finished")

    except Exception:
        LOGGER.exception("Error in train")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()
