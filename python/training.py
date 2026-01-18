# --- STANDARD ---
import copy
import gc
import json
import os
import tempfile
from datetime import date

# --- PROJECT ---
import model1 as builder
import shared as s

# --- THIRD-PARTY ---
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tensorflow.keras import losses, optimizers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# --- CONFIG ---
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))
PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")

CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "window_size": 30,
    "validation_max_ratio": 0.2,
    "validation_to_window_ratio": 2.0,
    "target_scale_clip_factor": 10.0,
    # compile
    "learning_rate": 1e-3,
    "loss": {
        "zscore_1d": "mse",
        "rank_1d": losses.Huber(delta=0.1),
        "direction_1d": "binary_crossentropy",
        "log_return_1d": "mse",
        "log_return_3d": "mse",
        "log_return_5d": "mse",
    },
    "loss_weights": {
        "zscore_1d": 1.0,
        "rank_1d": 0.3,
        "direction_1d": 0.3,
        "log_return_1d": 0.05,
        "log_return_3d": 0.05,
        "log_return_5d": 0.05,
    },
    # fit
    "early_stopping_patience": 16,
    "early_stopping_restore_best_weigths": True,
    "reduce_lr_on_plateau_factor": 0.5,
    "reduce_lr_on_plateau_patience": 8,
    "reduce_lr_on_plateau_min_lr": 1e-5,
}


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
    window_size = CONFIG["window_size"]

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
        for t in range(window_size, day_count):
            stock_window = np.stack(
                [col[t - window_size : t] for col in stock_columns],
                axis=1,
            ).reshape(window_size, symbol_count, stock_feature_count)

            news_window = np.stack(
                [col[t - window_size : t] for col in news_columns],
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
                (window_size, symbol_count, stock_feature_count), tf.float32  # type: ignore
            ),
            "news": tf.TensorSpec((window_size, news_feature_count), tf.float32),  # type: ignore
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
    ds = ds.batch(CONFIG["batch_size"], drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return ds, stock_feature_count, news_feature_count, symbol_count


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

    clip = CONFIG["target_scale_clip_factor"] * target_scale
    y["log_return_1d"] = tf.clip_by_value(y["log_return_1d"], -clip, clip)
    y["log_return_3d"] = tf.clip_by_value(y["log_return_3d"], -clip, clip)
    y["log_return_5d"] = tf.clip_by_value(y["log_return_5d"], -clip, clip)

    return x, y


def split_train_validation(
    stock_table: pa.Table, news_table: pa.Table, target_table: pa.Table
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table, pa.Table, pa.Table, int]:
    n = stock_table.num_rows
    window_size = CONFIG["window_size"]

    min_val = window_size * CONFIG["validation_to_window_ratio"]
    max_val = int(n * CONFIG["validation_max_ratio"])

    if max_val < min_val:
        raise ValueError(
            f"Not enough data for validation: "
            f"len={n}, min_val={min_val}, max_val={max_val}"
        )

    val_start = n - max_val

    if val_start < window_size:
        raise ValueError(
            f"Training set too small after split: "
            f"val_start={val_start}, window_size={window_size}"
        )

    return (
        stock_table.slice(0, val_start),
        news_table.slice(0, val_start),
        target_table.slice(0, val_start),
        stock_table.slice(val_start - window_size),
        news_table.slice(val_start - window_size),
        target_table.slice(val_start - window_size),
        max_val,
    )


def get_result_analysis(loss_weights, result) -> dict[str, float | dict[str, float]]:
    history = result.history

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

        train_ds = train_ds.map(scale_targets, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(scale_targets, num_parallel_calls=tf.data.AUTOTUNE)

        model, model_config = builder.build(
            time_steps=CONFIG["window_size"],
            stock_feature_dim=stock_feature_count,
            news_feature_dim=news_feature_count,
            symbol_count=symbol_count,
        )

        model.compile(
            optimizer=optimizers.Adam(CONFIG["learning_rate"]),
            loss=CONFIG["loss"],
            loss_weights=CONFIG["loss_weights"],
        )

        model.summary(print_fn=LOGGER.info)

        # keep only what fit needs
        del stock_table, news_table, target_table
        gc.collect()

        result = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CONFIG["epochs"],
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=CONFIG["early_stopping_patience"],
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=CONFIG["reduce_lr_on_plateau_factor"],
                    patience=CONFIG["reduce_lr_on_plateau_patience"],
                    min_lr=CONFIG["reduce_lr_on_plateau_min_lr"],
                ),
            ],
        )

        # --- persist ---
        result_analysis = get_result_analysis(model.loss_weights, result)
        model_score = f"{result_analysis["normalized_val_loss"]:.4f}"

        model_key = f"{s.MODEL_PREFIX}{model_score}/{s.MODEL_FILE_NAME}"
        meta_key = f"{s.MODEL_PREFIX}{model_score}/{s.META_FILE_NAME}"

        write_model_s3(model, model_key)

        meta_data = {
            "model": {
                "name": builder.__name__,
                "config": model_config,
                "score": model_score,
            },
            "training": {
                "date": date.today().isoformat(),
                "symbol_count": symbol_count,
                "validation_size": validation_size,
                "config": copy.deepcopy(CONFIG),
                "result_analysis": result_analysis,
            },
        }
        s.write_bytes_s3(meta_key, json.dumps(meta_data, default=str).encode("utf-8"))

        LOGGER.info("training finished")

    except Exception:
        LOGGER.exception("Error in train")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()
