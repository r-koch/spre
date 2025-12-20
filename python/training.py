# ---------- STANDARD LIBRARY ----------
import json
import os
import shutil
from datetime import date
from io import BytesIO

# ---------- THIRD-PARTY LIBRARIES ----------
import joblib as jl
import news_preproc as npg
import numpy as np
import pandas as pd
import pyarrow as pa
import shared as s
import stock_preproc as sp
from sklearn.decomposition import PCA
from tensorflow.keras import layers, Model, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
VAL_DAYS = int(os.getenv("VAL_DAYS", "30"))
WINDOW = int(os.getenv("WINDOW", "20"))
PCA_DIMS = int(os.getenv("PCA_DIMS", "64"))
BATCH = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
CONSOLIDATED_COMPRESSION_LEVEL = int(os.getenv("CONSOLIDATED_COMPRESSION_LEVEL", "3"))

PARTIAL_FILE_LIMIT = int(os.getenv("PARTIAL_FILE_LIMIT", "1"))
if PARTIAL_FILE_LIMIT < 1:
    raise ValueError("PARTIAL_FILE_LIMIT must be > 0")

MODEL_PREFIX = "model/localDate="
MODEL_LOCAL_PREFIX = "/tmp/model/"

PCA_FILE_NAME = "/pca.pkl"
METADATA_FILE_NAME = "/meta.json"

LOGGER = s.setup_logger(__file__)

# ---------- HELPERS ----------
def write_directory_to_s3(local_dir: str, bucket: str, prefix: str):
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{prefix}/{rel_path}"

            with open(local_path, "rb") as reader:
                s.retry_s3(
                    lambda body=reader.read(): s.s3.put_object(
                        Bucket=bucket, Key=key, Body=body
                    )
                )


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
    keys = s.list_keys_s3(BUCKET, prefix)
    tables = [s.read_parquet_s3(BUCKET, key, schema) for key in keys]
    combined = pa.concat_tables(tables, promote_options="none")
    deduplicated = deduplicate(combined)
    return keys, deduplicated


def consolidate(current_keys: list[str], table: pa.Table, prefix: str, schema):
    if len(current_keys) <= PARTIAL_FILE_LIMIT:
        return

    timestamp = s.get_now_timestamp()
    new_key = f"{prefix}{timestamp}.parquet"

    s.write_parquet_s3(table, BUCKET, new_key, schema, CONSOLIDATED_COMPRESSION_LEVEL)

    for key in current_keys:
        s.retry_s3(lambda: s.s3.delete_object(Bucket=BUCKET, Key=key))


def get_data_frame(prefix: str, schema):
    keys, table = get_all_keys_and_deduplicated_table(prefix, schema)

    consolidate(keys, table, prefix, schema)

    df = table.to_pandas()
    df["localDate"] = pd.to_datetime(df["localDate"])
    df = df.set_index("localDate").sort_index()
    return df


def get_training_inputs():
    stock_data = get_data_frame(sp.PIVOTED_PREFIX, sp.get_pivoted_schema())
    news_data = get_data_frame(npg.LAGGED_PREFIX, npg.get_lagged_schema())
    target_data = get_data_frame(sp.TARGET_LOG_RETURNS_PREFIX, sp.get_target_schema())

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


def get_cutoff_date() -> date:
    stock_last = s.read_json_date_s3(BUCKET, sp.PREPROC_STATE_KEY, s.LAST_PROCESSED_KEY)
    assert stock_last is not None
    news_last = s.read_json_date_s3(BUCKET, npg.PREPROC_STATE_KEY, s.LAST_PROCESSED_KEY)
    assert news_last is not None
    return min(stock_last, news_last)


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

        cutoff_date = get_cutoff_date()

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
                EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
                ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
            ],
        )

        date_str = cutoff_date.isoformat()
        base_prefix = f"{MODEL_PREFIX}{date_str}"

        local_model_dir = f"{MODEL_LOCAL_PREFIX}{date_str}"
        model.export(local_model_dir)

        model_prefix = f"{base_prefix}/model"
        write_directory_to_s3(local_model_dir, BUCKET, model_prefix)
        shutil.rmtree(local_model_dir)

        buffer = BytesIO()
        jl.dump(pca, buffer)
        buffer.seek(0)

        pca_key = f"{base_prefix}{PCA_FILE_NAME}"
        s.write_bytes_s3(BUCKET, pca_key, buffer.getvalue())

        meta = {
            "cutoff": date_str,
            "train_range": [
                str(stock_data.index[train_mask].min()),
                str(stock_data.index[train_mask].max()),
            ],
            "val_range": [
                str(stock_data.index[val_mask].min()),
                str(stock_data.index[val_mask].max()),
            ],
            "window": WINDOW,
            "pca_dims": PCA_DIMS,
            "features": {
                "price_features": stock_data.shape[1],
                "news_features": news_data.shape[1],
            },
        }
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")

        meta_key = f"{base_prefix}{METADATA_FILE_NAME}"
        s.write_bytes_s3(BUCKET, meta_key, meta_bytes)

    except Exception:
        LOGGER.exception("Error in train")
        return {"statusCode": 500, "body": "error"}

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    train()
