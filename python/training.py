# --- STANDARD ---
import gc
import json
import os
import tempfile
from datetime import date

# --- PROJECT ---
import shared as s
import shared_model as sm

# --- THIRD-PARTY ---
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# --- CONFIG ---
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))
PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "20"))

VALIDATION_MAX_RATIO = float(os.getenv("VALIDATION_MAX_RATIO", "0.2"))
VALIDATION_TO_WINDOW_RATIO = float(os.getenv("VALIDATION_TO_WINDOW_RATIO", "2.0"))

TARGET_SCALE_CLIP_FACTOR = float(os.getenv("TARGET_SCALE_CLIP_FACTOR", "10.0"))

STOCK_ATTENTION_HEADS = int(os.getenv("SYMBOL_ATTENTION_HEADS", "1"))
STOCK_EMBED = int(os.getenv("SYMBOL_EMBED", "8"))

NEWS_ATTENTION_HEADS = int(os.getenv("NEWS_ATTENTION_HEADS", "1"))
NEWS_EMBED = int(os.getenv("NEWS_EMBED", "8"))

SHARED_TRUNK_SIZE_1 = int(os.getenv("EARLY_STOPPING_PATIENCE", "256"))
SHARED_TRUNK_SIZE_2 = int(os.getenv("EARLY_STOPPING_PATIENCE", "128"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))

LOSSES_HUBER_DELTA = float(os.getenv("LOSSES_HUBER_DELTA", "0.1"))

LOSS_ZSCORE_1D = os.getenv("LOSS_ZSCORE_1D", "mse")
LOSS_RANK_1D = losses.Huber(delta=LOSSES_HUBER_DELTA)
LOSS_DIRECTION_1D = os.getenv("LOSS_DIRECTION_1D", "binary_crossentropy")
LOSS_LOG_RETURN_1D = os.getenv("LOSS_LOG_RETURN_1D", "mse")
LOSS_LOG_RETURN_3D = os.getenv("LOSS_LOG_RETURN_3D", "mse")
LOSS_LOG_RETURN_5D = os.getenv("LOSS_LOG_RETURN_5D", "mse")

LOSS_WEIGHT_ZSCORE_1D = float(os.getenv("LOSS_WEIGHT_ZSCORE_1D", "1.0"))
LOSS_WEIGHT_RANK_1D = float(os.getenv("LOSS_WEIGHT_RANK_1D", "0.3"))
LOSS_WEIGHT_DIRECTION_1D = float(os.getenv("LOSS_WEIGHT_DIRECTION_1D", "0.3"))
LOSS_WEIGHT_LOG_RETURN_1D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_1D", "0.05"))
LOSS_WEIGHT_LOG_RETURN_3D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_3D", "0.05"))
LOSS_WEIGHT_LOG_RETURN_5D = float(os.getenv("LOSS_WEIGHT_LOG_RETURN_5D", "0.05"))

EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "15"))
EARLY_STOPPING_RESTORE_BEST_WEIGTHS = bool(
    os.getenv("EARLY_STOPPING_RESTORE_BEST_WEIGTHS", "True")
)  # empty for False

REDUCE_LR_ON_PLATEAU_FACTOR = float(os.getenv("REDUCE_LR_ON_PLATEAU_FACTOR", "0.5"))
REDUCE_LR_ON_PLATEAU_PATIENCE = int(os.getenv("REDUCE_LR_ON_PLATEAU_PATIENCE", "5"))
REDUCE_LR_ON_PLATEAU_MIN_LR = float(os.getenv("REDUCE_LR_ON_PLATEAU_MIN_LR", "1e-5"))

MODEL_SCORE_DIRECTION_ACC_FACTOR = 0.50
MODEL_SCORE_RANK_SPEARMAN_FACTOR = 0.30
MODEL_SCORE_RMSE_LOG_RET_FACTOR = 0.20

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


def get_table(prefix: str, schema) -> pa.Table:
    keys = s.list_keys_s3(prefix)
    tables = [s.read_parquet_s3(key, schema) for key in keys]
    combined_table = pa.concat_tables(tables, promote_options="none")
    table = deduplicate(combined_table)
    consolidate(keys, table, prefix, schema)
    return table


def consolidate(current_keys: list[str], table: pa.Table, prefix: str, schema):
    if len(current_keys) <= PARTIAL_FILE_LIMIT:
        return

    timestamp = s.get_now_timestamp()
    new_key = f"{prefix}{timestamp}.parquet"

    s.write_parquet_s3(new_key, table, schema, CONSOLIDATED_COMPRESSION_LEVEL)

    for key in current_keys:
        s.delete_s3(key)


def get_training_data() -> tuple[pa.Table, pa.Table, pa.Table]:
    stock_table = get_table(s.PIVOTED_PREFIX, s.get_pivoted_schema())
    news_table = get_table(s.LAGGED_PREFIX, s.get_lagged_schema())
    target_table = get_table(s.TARGET_PREFIX, s.get_target_schema())

    # ---- build common date set ----
    stock_dates = set(stock_table[s.LOCAL_DATE].to_pylist())
    news_dates = set(news_table[s.LOCAL_DATE].to_pylist())
    target_dates = set(target_table[s.LOCAL_DATE].to_pylist())

    common_dates = stock_dates & news_dates & target_dates

    if not common_dates:
        raise ValueError("No overlapping dates across inputs")

    # ---- filter each table ----
    def filter_by_dates(table: pa.Table) -> pa.Table:
        mask = [d in common_dates for d in table[s.LOCAL_DATE].to_pylist()]
        return table.filter(pa.array(mask))

    stock_table = filter_by_dates(stock_table)
    news_table = filter_by_dates(news_table)
    target_table = filter_by_dates(target_table)

    # ---- final safety check ----
    if not (
        stock_table.num_rows == news_table.num_rows == target_table.num_rows
        and stock_table[s.LOCAL_DATE].equals(news_table[s.LOCAL_DATE])
        and stock_table[s.LOCAL_DATE].equals(target_table[s.LOCAL_DATE])
    ):
        raise ValueError("Post-alignment date mismatch (should not happen)")

    return stock_table, news_table, target_table


def table_to_numpy(table: pa.Table) -> np.ndarray:
    table = table.combine_chunks()
    arrays = [col.to_numpy(zero_copy_only=False) for col in table.itercolumns()]
    out = np.stack(arrays, axis=1).astype(dtype="float32", copy=False)
    return out


def build_training_dataset(
    stock_table: pa.Table, news_table: pa.Table, target_table: pa.Table
) -> tuple[tf.data.Dataset, int, int, int]:
    symbols = s.get_symbols()
    symbol_count = len(symbols)
    window = WINDOW_SIZE

    stock_feature_count = len(s.PIVOTED_FEATURES)
    news_feature_count = news_table.num_columns - 1

    z_cols = [f"{s}_zscore_1d" for s in symbols]
    rank_cols = [f"{s}_rank_1d" for s in symbols]
    dir_cols = [f"{s}_direction_1d" for s in symbols]
    lr1_cols = [f"{s}_log_return_1d" for s in symbols]
    lr3_cols = [f"{s}_log_return_3d" for s in symbols]
    lr5_cols = [f"{s}_log_return_5d" for s in symbols]

    stock_columns = [
        stock_table.column(col).combine_chunks().to_numpy(zero_copy_only=False)
        for col in s.get_pivoted_schema().names[1:]
    ]

    news_columns = [
        news_table.column(col).combine_chunks().to_numpy(zero_copy_only=False)
        for col in s.get_lagged_schema().names[1:]
    ]

    def target_cols(cols):
        return [
            target_table.column(c).combine_chunks().to_numpy(zero_copy_only=False)
            for c in cols
        ]

    z_arrays = target_cols(z_cols)
    rank_arrays = target_cols(rank_cols)
    dir_arrays = target_cols(dir_cols)
    lr1_arrays = target_cols(lr1_cols)
    lr3_arrays = target_cols(lr3_cols)
    lr5_arrays = target_cols(lr5_cols)

    day_count = stock_table.num_rows

    def gen():
        for t in range(window, day_count):
            stock_window = np.stack(
                [col[t - window : t] for col in stock_columns],
                axis=1,
            ).reshape(window, symbol_count, stock_feature_count)

            news_window = np.stack(
                [col[t - window : t] for col in news_columns],
                axis=1,
            )

            def take(arrs):
                return np.array([a[t] for a in arrs], dtype="float32")

            yield (
                {
                    "stock": stock_window,
                    "news": news_window,
                },
                {
                    "zscore_1d": take(z_arrays),
                    "rank_1d": take(rank_arrays),
                    "direction_1d": take(dir_arrays),
                    "log_return_1d": take(lr1_arrays),
                    "log_return_3d": take(lr3_arrays),
                    "log_return_5d": take(lr5_arrays),
                },
            )

    output_signature = (
        {
            "stock": tf.TensorSpec(
                (window, symbol_count, stock_feature_count), tf.float32  # type: ignore
            ),
            "news": tf.TensorSpec((window, news_feature_count), tf.float32),  # type: ignore
        },
        {
            "zscore_1d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
            "rank_1d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
            "direction_1d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
            "log_return_1d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
            "log_return_3d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
            "log_return_5d": tf.TensorSpec((symbol_count,), tf.float32),  # type: ignore
        },
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds, stock_feature_count, news_feature_count, symbol_count


def build_model(
    time_steps: int, stock_feature_dim: int, news_feature_dim: int, symbol_count: int
) -> models.Model:

    stock_in = layers.Input(
        shape=(time_steps, symbol_count, stock_feature_dim), name="stock"
    )
    news_in = layers.Input(shape=(time_steps, news_feature_dim), name="news")

    # --- per-symbol projection ---
    x = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(STOCK_EMBED, activation="relu"))
    )(stock_in)
    # (batch, time, symbols, embed)

    # --- pool symbols ---
    x = sm.MeanOverSymbols(name="pool_symbols")(x)

    # (batch, time, embed)

    # --- temporal attention only ---
    x = layers.MultiHeadAttention(
        num_heads=STOCK_ATTENTION_HEADS,
        key_dim=STOCK_EMBED // STOCK_ATTENTION_HEADS,
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
    shared = layers.Dense(SHARED_TRUNK_SIZE_1, activation="relu")(x)
    shared = layers.Dense(SHARED_TRUNK_SIZE_2, activation="relu")(shared)

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
            "zscore_1d": LOSS_ZSCORE_1D,
            "rank_1d": LOSS_RANK_1D,
            "direction_1d": LOSS_DIRECTION_1D,
            "log_return_1d": LOSS_LOG_RETURN_1D,
            "log_return_3d": LOSS_LOG_RETURN_3D,
            "log_return_5d": LOSS_LOG_RETURN_5D,
        },
        loss_weights={
            "zscore_1d": LOSS_WEIGHT_ZSCORE_1D,
            "rank_1d": LOSS_WEIGHT_RANK_1D,
            "direction_1d": LOSS_WEIGHT_DIRECTION_1D,
            "log_return_1d": LOSS_WEIGHT_LOG_RETURN_1D,
            "log_return_3d": LOSS_WEIGHT_LOG_RETURN_3D,
            "log_return_5d": LOSS_WEIGHT_LOG_RETURN_5D,
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

    target_scale = s.TARGET_SCALE

    y["log_return_1d"] *= target_scale
    y["log_return_3d"] *= target_scale
    y["log_return_5d"] *= target_scale

    clip = TARGET_SCALE_CLIP_FACTOR * target_scale
    y["log_return_1d"] = tf.clip_by_value(y["log_return_1d"], -clip, clip)
    y["log_return_3d"] = tf.clip_by_value(y["log_return_3d"], -clip, clip)
    y["log_return_5d"] = tf.clip_by_value(y["log_return_5d"], -clip, clip)

    return x, y


def split_train_validation(
    stock_table: pa.Table, news_table: pa.Table, target_table: pa.Table
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table, pa.Table, pa.Table, int]:
    n = stock_table.num_rows

    min_val = WINDOW_SIZE * VALIDATION_TO_WINDOW_RATIO
    max_val = int(n * VALIDATION_MAX_RATIO)

    if max_val < min_val:
        raise ValueError(
            f"Not enough data for validation: "
            f"len={n}, min_val={min_val}, max_val={max_val}"
        )

    val_start = n - max_val

    if val_start < WINDOW_SIZE:
        raise ValueError(
            f"Training set too small after split: "
            f"val_start={val_start}, window_size={WINDOW_SIZE}"
        )

    return (
        stock_table.slice(0, val_start),
        news_table.slice(0, val_start),
        target_table.slice(0, val_start),
        stock_table.slice(val_start - WINDOW_SIZE),
        news_table.slice(val_start - WINDOW_SIZE),
        target_table.slice(val_start - WINDOW_SIZE),
        max_val,
    )


def get_result_analysis(result) -> dict[str, float | dict[str, float]]:
    history = result.history

    loss_weights = {
        "zscore_1d": LOSS_WEIGHT_ZSCORE_1D,
        "rank_1d": LOSS_WEIGHT_RANK_1D,
        "direction_1d": LOSS_WEIGHT_DIRECTION_1D,
        "log_return_1d": LOSS_WEIGHT_LOG_RETURN_1D,
        "log_return_3d": LOSS_WEIGHT_LOG_RETURN_3D,
        "log_return_5d": LOSS_WEIGHT_LOG_RETURN_5D,
    }

    best_epoch = min(
        range(len(history["val_loss"])),
        key=lambda i: history["val_loss"][i],
    )

    per_head_losses = {
        name: history[f"val_{name}_loss"][best_epoch] for name in loss_weights
    }

    normalized_losses = {}
    for name, loss in per_head_losses.items():
        if name.startswith("log_return"):
            normalized_losses[name] = loss / s.TARGET_SCALE
        else:
            normalized_losses[name] = loss

    weighted_sum = sum(
        normalized_losses[name] * loss_weights[name] for name in loss_weights
    )

    weight_sum = sum(loss_weights.values())

    normalized_val_loss = weighted_sum / weight_sum
    return {
        "normalized_val_loss": normalized_val_loss,
        "val_losses": per_head_losses,
    }


def train():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        stock_table, news_table, target_table = get_training_data()

        (
            stock_train,
            news_train,
            target_train,
            stock_val,
            news_val,
            target_val,
            validation_size,
        ) = split_train_validation(
            stock_table,
            news_table,
            target_table,
        )

        (
            train_ds,
            stock_feature_count,
            news_feature_count,
            symbol_count,
        ) = build_training_dataset(stock_train, news_train, target_train)

        val_ds, _, _, _ = build_training_dataset(stock_val, news_val, target_val)

        train_ds = train_ds.map(scale_targets, num_parallel_calls=1)
        val_ds = val_ds.map(scale_targets, num_parallel_calls=1)

        model = build_model(
            time_steps=WINDOW_SIZE,
            stock_feature_dim=stock_feature_count,
            news_feature_dim=news_feature_count,
            symbol_count=symbol_count,
        )

        LOGGER.info(model.summary())

        # keep only what fit needs
        del stock_table, news_table, target_table
        gc.collect()

        result = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=REDUCE_LR_ON_PLATEAU_FACTOR,
                    patience=REDUCE_LR_ON_PLATEAU_PATIENCE,
                    min_lr=REDUCE_LR_ON_PLATEAU_MIN_LR,
                ),
            ],
        )

        # --- persist ---
        result_analysis = get_result_analysis(result)
        model_score = f"{result_analysis["normalized_val_loss"]:.4f}"

        model_key = f"{s.MODEL_PREFIX}{model_score}/{s.MODEL_FILE_NAME}"
        meta_key = f"{s.MODEL_PREFIX}{model_score}/{s.META_FILE_NAME}"

        write_model_s3(model, model_key)

        meta_data = {
            "model_score": model_score,
            "symbol_count": symbol_count,
            "training_date": date.today().isoformat(),
            "validation_size": validation_size,
            "result_analysis": result_analysis,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "window_size": WINDOW_SIZE,
                "validation_max_ratio": VALIDATION_MAX_RATIO,
                "validation_to_window_ratio": VALIDATION_TO_WINDOW_RATIO,
                "target_scale_clip_factor": TARGET_SCALE_CLIP_FACTOR,
                "stock_attention_heads": STOCK_ATTENTION_HEADS,
                "stock_embed": STOCK_EMBED,
                "news_attention_heads": NEWS_ATTENTION_HEADS,
                "news_embed": NEWS_EMBED,
                "shared_trunk_size_1": SHARED_TRUNK_SIZE_1,
                "shared_trunk_size_2": SHARED_TRUNK_SIZE_2,
                "learning_rate": LEARNING_RATE,
                "loss_zscore_1d": LOSS_ZSCORE_1D,
                "loss_rank_1d": {
                    "type": "Huber",
                    "delta": LOSSES_HUBER_DELTA,
                },
                "loss_direction_1d": LOSS_DIRECTION_1D,
                "loss_log_return_1d": LOSS_LOG_RETURN_1D,
                "loss_log_return_3d": LOSS_LOG_RETURN_3D,
                "loss_log_return_5d": LOSS_LOG_RETURN_5D,
                "loss_weight_zscore_1d": LOSS_WEIGHT_ZSCORE_1D,
                "loss_weight_rank_1d": LOSS_WEIGHT_RANK_1D,
                "loss_weight_direction_1d": LOSS_WEIGHT_DIRECTION_1D,
                "loss_weight_log_return_1d": LOSS_WEIGHT_LOG_RETURN_1D,
                "loss_weight_log_return_3d": LOSS_WEIGHT_LOG_RETURN_3D,
                "loss_weight_log_return_5d": LOSS_WEIGHT_LOG_RETURN_5D,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_restore_best_weigths": EARLY_STOPPING_RESTORE_BEST_WEIGTHS,
                "reduce_lr_on_plateau_factor": REDUCE_LR_ON_PLATEAU_FACTOR,
                "reduce_lr_on_plateau_patience": REDUCE_LR_ON_PLATEAU_PATIENCE,
                "reduce_lr_on_plateau_min_lr": REDUCE_LR_ON_PLATEAU_MIN_LR,
            },
        }
        s.write_bytes_s3(meta_key, json.dumps(meta_data).encode("utf-8"))

        LOGGER.info("training finished")

    except Exception:
        LOGGER.exception("Error in train")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()
