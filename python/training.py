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
from sklearn.decomposition import PCA
from tensorflow.keras import layers, Model, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# --- CONFIG ---
VAL_DAYS = int(os.getenv("VAL_DAYS", "30"))
WINDOW = int(os.getenv("WINDOW", "20"))
PCA_DIMS = int(os.getenv("PCA_DIMS", "64"))
BATCH = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))

PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")

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


def make_windows(X: np.ndarray, Y: np.ndarray):
    xs, ys = [], []
    for i in range(WINDOW, len(X)):
        xs.append(X[i - WINDOW : i])
        ys.append(Y[i])
    return np.asarray(xs), np.asarray(ys)


def build_model(D: int, out_dim: int) -> Model:
    inp = Input((WINDOW, D))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dense(512, activation="relu")(x)
    out = layers.Dense(out_dim, activation="linear")(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train():
    try:
        stock_data, news_data, target_data = get_training_inputs()

        cutoff_date = s.get_last_processed_date()

        train_mask, val_mask = get_training_and_validation_mask(
            stock_data.index, cutoff_date
        )

        X_stock_train = stock_data.loc[train_mask].values
        X_stock_val = stock_data.loc[val_mask].values

        pca = PCA(n_components=PCA_DIMS)
        Xp_train = pca.fit_transform(X_stock_train)
        Xp_val = pca.transform(X_stock_val)

        X_news_train = news_data.loc[train_mask].values
        X_news_val = news_data.loc[val_mask].values

        X_train = np.hstack([Xp_train, X_news_train])
        X_val = np.hstack([Xp_val, X_news_val])

        Y_train = target_data.loc[train_mask].values
        Y_val = target_data.loc[val_mask].values

        Xtr_w, Ytr_w = make_windows(X_train, Y_train)
        Xv_w, Yv_w = make_windows(X_val, Y_val)

        model = build_model(Xtr_w.shape[2], Ytr_w.shape[1])

        model.fit(
            Xtr_w,
            Ytr_w,
            validation_data=(Xv_w, Yv_w),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[
                EarlyStopping(
                    patience=6, restore_best_weights=True, monitor="val_loss"
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

        buffer = BytesIO()
        pickle.dump(pca, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        pca_data = buffer.getvalue()

        pca_key = f"{base_prefix}{s.PCA_FILE_NAME}"
        s.write_bytes_s3(pca_key, pca_data)

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
            "pcaDimensions": PCA_DIMS,
            "features": {
                "stock": stock_data.shape[1],
                "news": news_data.shape[1],
            },
        }
        meta_data = json.dumps(meta, indent=2).encode("utf-8")

        meta_key = f"{base_prefix}{s.META_FILE_NAME}"
        s.write_bytes_s3(meta_key, meta_data)

        s.write_json_date_s3(s.TRAINING_STATE_KEY, s.LAST_TRAINED_KEY, cutoff_date)

    except Exception:
        LOGGER.exception("Error in train")


# --- ENTRY POINT ---
if __name__ == "__main__":
    train()
