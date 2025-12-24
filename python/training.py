# --- STANDARD ---
import json
import os
import pickle
import tempfile
from datetime import date
from io import BytesIO

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY ---
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, Model, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.mixed_precision import LossScaleOptimizer  # type: ignore

# --- CONFIG ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "15"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")
VAL_DAYS = int(os.getenv("VAL_DAYS", "120"))
WINDOW = int(os.getenv("WINDOW", "60"))

LOGGER = s.setup_logger(__file__)


def deduplicate(table):
    seen = set()
    keep_indices: list[int] = []

    dates = table["localDate"]
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
    df["localDate"] = pd.to_datetime(df["localDate"])
    df = df.set_index("localDate").sort_index()
    return df


def get_training_inputs():
    stock_data = get_data_frame(s.PIVOTED_PREFIX, s.get_pivoted_schema())
    news_data = get_data_frame(s.LAGGED_PREFIX, s.get_lagged_schema())
    target_data = get_data_frame(s.TARGET_LOG_RETURNS_PREFIX, s.get_target_schema())

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


def get_training_and_validation_mask(index: pd.DatetimeIndex, cutoff_date: date):
    cutoff = pd.Timestamp(cutoff_date)

    val_end = cutoff
    val_start = cutoff - pd.Timedelta(days=VAL_DAYS - 1)

    training_mask = index < val_start
    validation_mask = (index >= val_start) & (index <= val_end)

    if not training_mask.any():
        raise ValueError("Empty training split")
    if not validation_mask.any():
        raise ValueError("Empty validation split")

    return training_mask, validation_mask


def windowed_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    shuffle: bool,
):
    X_ds = tf.data.Dataset.from_tensor_slices(X)
    Y_ds = tf.data.Dataset.from_tensor_slices(Y)

    # Sliding windows over X
    X_win = X_ds.window(WINDOW, shift=1, drop_remainder=True).flat_map(
        lambda w: w.batch(WINDOW)
    )

    # Targets aligned to window end: Y[i] for X[i-window:i]
    Y_win = Y_ds.skip(WINDOW)

    ds = tf.data.Dataset.zip((X_win, Y_win))

    if shuffle:
        ds = ds.shuffle(1024)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(D: int, out_dim: int) -> Model:
    inp = Input((WINDOW, D))

    x = layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu")(
        inp
    )

    x = layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu")(
        x
    )

    # --- long-range dependencies ---
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)
    )(x)

    # --- cross-asset interaction ---
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)

    out = layers.Dense(out_dim, activation="linear", dtype="float32")(x)

    model = Model(inp, out)

    base_opt = Adam(learning_rate=3e-4)
    opt = LossScaleOptimizer(base_opt)

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def assert_finite(name, arr):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")


def train():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        mixed_precision.set_global_policy("mixed_float16")

        stock_data, news_data, target_data = get_training_inputs()

        cutoff_date = s.get_last_processed_date()

        train_mask, val_mask = get_training_and_validation_mask(
            stock_data.index, cutoff_date
        )

        X_stock_train = stock_data.loc[train_mask].values
        X_stock_val = stock_data.loc[val_mask].values

        X_news_train = news_data.loc[train_mask].values
        X_news_val = news_data.loc[val_mask].values

        X_train = np.hstack([X_stock_train, X_news_train])
        X_val = np.hstack([X_stock_val, X_news_val])

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        Y_train = target_data.loc[train_mask].values
        Y_val = target_data.loc[val_mask].values

        assert_finite("X_train", X_train)
        assert_finite("Y_train", Y_train)
        assert_finite("X_val", X_val)
        assert_finite("Y_val", Y_val)

        stock_feature_count = X_stock_train.shape[1]
        news_feature_count = X_news_train.shape[1]
        total_feature_count = stock_feature_count + news_feature_count

        assert X_train.shape[1] == total_feature_count

        LOGGER.info(f"Training windows shape: {X_train.shape}")

        train_ds = windowed_dataset(X_train, Y_train, shuffle=True).repeat()

        val_ds = windowed_dataset(X_val, Y_val, shuffle=False)

        num_train_windows = len(X_train) - WINDOW
        steps_per_epoch = num_train_windows // BATCH_SIZE

        num_val_windows = len(X_val) - WINDOW
        validation_steps = max(1, num_val_windows // BATCH_SIZE)

        model = build_model(X_train.shape[1], Y_train.shape[1])

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[
                EarlyStopping(
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    monitor="val_loss",
                ),
                ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
            ],
        )

        model_date = cutoff_date.isoformat()
        base_prefix = f"{s.MODEL_PREFIX}{model_date}/"

        with tempfile.TemporaryDirectory() as tmp:
            local_model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
            model.save(local_model_path)

            with open(local_model_path, "rb") as f:
                model_data = f.read()

            model_key = f"{base_prefix}{s.MODEL_FILE_NAME}"
            s.write_bytes_s3(model_key, model_data)

        if mean.shape[1] != total_feature_count:
            raise ValueError(
                "Normalization vector length does not match feature layout"
            )

        meta = {
            "trainingCutoff": model_date,
            "trainingRange": [
                str(stock_data.index[train_mask].min()),
                str(stock_data.index[train_mask].max()),
            ],
            "validationRange": [
                str(stock_data.index[val_mask].min()),
                str(stock_data.index[val_mask].max()),
            ],
            "windowSize": WINDOW,
            "features": {
                "stock": stock_data.shape[1],
                "news": news_data.shape[1],
            },
            "normalization": {
                "mean": mean.astype("float32").tolist(),
                "std": std.astype("float32").tolist(),
                "stockFeatureCount": int(stock_feature_count),
                "newsFeatureCount": int(news_feature_count),
                "totalFeatureCount": int(total_feature_count),
            },
        }
        meta_data = json.dumps(meta, indent=2).encode("utf-8")

        meta_key = f"{base_prefix}{s.META_FILE_NAME}"
        s.write_bytes_s3(meta_key, meta_data)

        s.write_json_date_s3(s.TRAINING_STATE_KEY, s.LAST_TRAINED_KEY, cutoff_date)

        LOGGER.info("finished training")

    except Exception:
        LOGGER.exception("Error in train")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()
