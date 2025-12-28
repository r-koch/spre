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
import pandas as pd
import pyarrow as pa
import tensorflow as tf
from tensorflow.keras import layers, losses, mixed_precision, models, optimizers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# --- CONFIG ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "15"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
LAMBDA_DIR = float(os.getenv("LAMBDA_DIR", "5.0"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
MARGIN = float(os.getenv("MARGIN", "0.01"))
NEWS_ATTENTION_HEADS = int(os.getenv("NEWS_ATTENTION_HEADS", "1"))
NEWS_EMBED = int(os.getenv("NEWS_EMBED", "8"))
PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")
SYMBOL_ATTENTION_HEADS = int(os.getenv("SYMBOL_ATTENTION_HEADS", "1"))
SYMBOL_EMBED = int(os.getenv("SYMBOL_EMBED", "8"))
TARGET_SCALE = float(os.getenv("TARGET_SCALE", "1000.0"))
VALIDATION_MAX_RATIO = float(os.getenv("VALIDATION_MAX_RATIO", "0.2"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "30"))

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


def build_training_dataset(
    stock_data: pd.DataFrame, news_data: pd.DataFrame, target_data: pd.DataFrame
):
    symbols = s.get_symbols()
    symbol_count = len(symbols)
    window = WINDOW_SIZE

    # ---------- SETUP ----------
    stock_cols = s.get_pivoted_schema().names[1:]

    stock_feature_count = len(s.PIVOTED_FEATURES)
    assert stock_feature_count * symbol_count == len(stock_cols)

    news_feature_count = news_data.shape[1]

    z_cols = [f"{sym}_zscore_1d" for sym in symbols]
    rank_cols = [f"{sym}_rank_1d" for sym in symbols]
    dir_cols = [f"{sym}_direction_1d" for sym in symbols]
    lr1_cols = [f"{sym}_log_return_1d" for sym in symbols]
    lr3_cols = [f"{sym}_log_return_3d" for sym in symbols]
    lr5_cols = [f"{sym}_log_return_5d" for sym in symbols]

    num_samples = len(stock_data) - window

    # ---- PRE-MATERIALIZE ONCE ----
    stock_values = stock_data[stock_cols].to_numpy(dtype="float32")
    news_values = news_data.to_numpy(dtype="float32")

    z_values = target_data[z_cols].to_numpy(dtype="float32")
    rank_values = target_data[rank_cols].to_numpy(dtype="float32")
    dir_values = target_data[dir_cols].to_numpy(dtype="float32")
    lr1_values = target_data[lr1_cols].to_numpy(dtype="float32")
    lr3_values = target_data[lr3_cols].to_numpy(dtype="float32")
    lr5_values = target_data[lr5_cols].to_numpy(dtype="float32")

    def generator():
        for t in range(window, len(stock_values)):
            yield (
                {
                    "stock": stock_values[t - window : t].reshape(
                        window, symbol_count, stock_feature_count
                    ),
                    "news": news_values[t - window : t],
                },
                {
                    "zscore_1d": z_values[t],
                    "rank_1d": rank_values[t],
                    "direction_1d": dir_values[t],
                    "log_return_1d": lr1_values[t],
                    "log_return_3d": lr3_values[t],
                    "log_return_5d": lr5_values[t],
                },
            )

    # ---------- TF.DATASET ----------
    output_signature = (
        {
            "stock": tf.TensorSpec(
                shape=(window, symbol_count, stock_feature_count), dtype=tf.float32  # type: ignore
            ),
            "news": tf.TensorSpec(
                shape=(window, news_feature_count), dtype=tf.float32  # type: ignore
            ),
        },
        {
            "zscore_1d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
            "rank_1d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
            "direction_1d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
            "log_return_1d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
            "log_return_3d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
            "log_return_5d": tf.TensorSpec(shape=(symbol_count,), dtype=tf.float32),  # type: ignore
        },
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)  # critical: bounded RAM

    return ds, num_samples, stock_feature_count


def build_model(time_steps, stock_feature_dim, news_feature_dim, symbol_count):

    stock_in = layers.Input(
        shape=(time_steps, symbol_count, stock_feature_dim), name="stock"
    )
    news_in = layers.Input(shape=(time_steps, news_feature_dim), name="news")

    # --- per-symbol projection ---
    x = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(SYMBOL_EMBED, activation="relu"))
    )(stock_in)
    # (batch, time, symbols, embed)

    # --- EARLY symbol pooling (critical) ---
    x = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name="pool_symbols_early",
    )(x)
    # (batch, time, embed)

    # --- temporal attention only ---
    x = layers.MultiHeadAttention(
        num_heads=SYMBOL_ATTENTION_HEADS,
        key_dim=SYMBOL_EMBED // SYMBOL_ATTENTION_HEADS,
    )(x, x)
    # (batch, time, embed)

    # --- pool time ---
    x = layers.GlobalAveragePooling1D()(x)
    # (batch, embed)

    # --- news encoder ---
    y = layers.MultiHeadAttention(
        num_heads=NEWS_ATTENTION_HEADS,
        key_dim=NEWS_EMBED,
    )(news_in, news_in)
    y = layers.GlobalAveragePooling1D()(y)

    # --- fuse ---
    x = layers.Concatenate()([x, y])

    # --- shared trunk ---
    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.Dense(128, activation="relu")(shared)

    # --- heads ---
    def head(name, activation=None):
        return layers.Dense(
            symbol_count,
            activation=activation,
            name=name,
            dtype="float32",
        )(shared)

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


def scale_targets(x, y):
    y = dict(y)  # defensive copy

    y["log_return_1d"] *= TARGET_SCALE
    y["log_return_3d"] *= TARGET_SCALE
    y["log_return_5d"] *= TARGET_SCALE

    clip = 10.0 * TARGET_SCALE
    y["log_return_1d"] = tf.clip_by_value(y["log_return_1d"], -clip, clip)
    y["log_return_3d"] = tf.clip_by_value(y["log_return_3d"], -clip, clip)
    y["log_return_5d"] = tf.clip_by_value(y["log_return_5d"], -clip, clip)

    return x, y


def split_train_validation(
    stock_data: pd.DataFrame, news_data: pd.DataFrame, target_data: pd.DataFrame
):
    n = len(stock_data)

    min_val = WINDOW_SIZE * 2
    max_val = int(n * VALIDATION_MAX_RATIO)

    if max_val < min_val:
        raise ValueError(
            f"Not enough data for validation: "
            f"len={n}, min_val={min_val}, max_val={max_val}"
        )

    val_size = max_val
    val_start = n - val_size

    # training must end early enough to form windows
    if val_start < WINDOW_SIZE:
        raise ValueError(
            f"Training set too small after split: "
            f"val_start={val_start}, window_size={WINDOW_SIZE}"
        )

    stock_train = stock_data.iloc[:val_start]
    news_train = news_data.iloc[:val_start]
    target_train = target_data.iloc[:val_start]

    # validation needs window_size lookback
    stock_val = stock_data.iloc[val_start - WINDOW_SIZE :]
    news_val = news_data.iloc[val_start - WINDOW_SIZE :]
    target_val = target_data.iloc[val_start - WINDOW_SIZE :]

    return (
        stock_train,
        news_train,
        target_train,
        stock_val,
        news_val,
        target_val,
        val_size,
    )


def train():
    try:
        # mixed_precision.set_global_policy("mixed_float16")

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_jit(False)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        stock_data, news_data, target_data = get_training_data()

        (
            stock_train,
            news_train,
            target_train,
            stock_val,
            news_val,
            target_val,
            val_size,
        ) = split_train_validation(
            stock_data,
            news_data,
            target_data,
        )

        train_ds, train_samples, stock_feature_count = build_training_dataset(
            stock_data, news_data, target_data
        )

        # train_ds, train_samples, stock_feature_count = build_training_dataset(
        #     stock_train, news_train, target_train
        # )

        # val_ds, val_samples, _ = build_training_dataset(stock_val, news_val, target_val)

        train_ds = train_ds.map(scale_targets, num_parallel_calls=1)
        # val_ds = val_ds.map(scale_targets, num_parallel_calls=1)

        train_steps = train_samples // BATCH_SIZE
        # val_steps = val_samples // BATCH_SIZE

        # if train_steps == 0 or val_steps == 0:
        #     raise ValueError(
        #         f"Invalid step count: train_steps={train_steps}, val_steps={val_steps}"
        #     )

        news_feature_dim = news_data.shape[1]

        symbol_count = len(s.get_symbols())

        model = build_model(
            time_steps=WINDOW_SIZE,
            stock_feature_dim=stock_feature_count,
            news_feature_dim=news_feature_dim,
            symbol_count=symbol_count,
        )

        LOGGER.info(model.summary())

        # keep only what fit needs
        del stock_data, news_data, target_data
        gc.collect()

        model.fit(
            train_ds,
            # validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=train_steps,
            # validation_steps=val_steps,
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
            "symbolCount": symbol_count,
            "trainingCutoff": training_date,
            "validationSize": val_size,
            "windowSize": WINDOW_SIZE,
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
