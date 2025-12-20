# ---------- STANDARD LIBRARY ----------
import json
import os
import pickle
import tempfile
from datetime import date

# ---------- THIRD-PARTY ----------
import numpy as np
import pyarrow as pa
from tensorflow.keras import models  # type: ignore

# ---------- PROJECT ----------
import shared as s
import stock_preproc as sp
import news_preproc as npg

# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")

INFERENCE_PREFIX = "inference/localDate="

DTYPE = np.float32


_inference_schema = None


def get_inference_schema():
    global _inference_schema
    if _inference_schema is None:
        # pa = pyarrow()
        _inference_schema = pa.schema(
            {
                "localDate": pa.date32(),
                "symbol": pa.string(),
                "predicted_log_return": pa.float32(),
            }
        )
    return _inference_schema


def get_model_prefix() -> str:
    model_date = s.read_json_date_s3(BUCKET, s.TRAINING_STATE_KEY, s.LAST_TRAINED_KEY)
    return f"{s.MODEL_PREFIX}{model_date}/"


def load_model_artifacts(model_prefix: str):
    model_key = f"{model_prefix}{s.MODEL_FILE_NAME}"
    pca_key = f"{model_prefix}{s.PCA_FILE_NAME}"
    meta_key = f"{model_prefix}{s.META_FILE_NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        model_data = s.read_bytes_s3(BUCKET, model_key)

        local_model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
        with open(local_model_path, "wb") as f:
            f.write(model_data)

        model = models.load_model(local_model_path, compile=False)

    pca_data = s.read_bytes_s3(BUCKET, pca_key)
    pca = pickle.loads(pca_data)

    meta_data = s.read_bytes_s3(BUCKET, meta_key)
    meta = json.loads(meta_data.decode("utf-8"))

    return model, pca, meta


def load_window(
    bucket: str, prefix: str, schema, end_date: date, window_size: int
) -> np.ndarray:
    keys = s.list_keys_s3(bucket, prefix, sort_reversed=True)
    local_date = "localDate"
    rows = []
    for key in keys:
        table = s.read_parquet_s3(bucket, key, schema)
        dates = table[local_date].to_pylist()

        for i in range(len(dates) - 1, -1, -1):
            d = dates[i]
            if d <= end_date:
                rows.append(
                    [table[col][i].as_py() for col in schema.names if col != local_date]
                )
                if len(rows) == window_size:
                    break

        if len(rows) == window_size:
            break

    if len(rows) != window_size:
        raise ValueError("Insufficient data for inference window")

    rows.reverse()
    return np.asarray(rows, dtype=DTYPE)


def build_model_input(
    stock_window: np.ndarray, news_window: np.ndarray, pca
) -> np.ndarray:
    stock_pca = pca.transform(stock_window)

    x = np.concatenate([stock_pca, news_window], axis=1)

    return x.reshape(1, *x.shape)


def infer():
    last_processed_date = s.get_last_processed_date(BUCKET)
    inference_date = last_processed_date + s.ONE_DAY

    model_prefix = get_model_prefix()
    model, pca, meta = load_model_artifacts(model_prefix)

    window_size = meta["windowSize"]

    stock_window = load_window(
        BUCKET,
        sp.PIVOTED_PREFIX,
        sp.get_pivoted_schema(),
        last_processed_date,
        window_size,
    )
    news_window = load_window(
        BUCKET,
        npg.LAGGED_PREFIX,
        npg.get_lagged_schema(),
        last_processed_date,
        window_size,
    )

    x = build_model_input(stock_window, news_window, pca)

    preds = model.predict(x, verbose=0)[0].astype(DTYPE)

    symbols = sp.get_symbols()

    inference_schema = get_inference_schema()

    inference_table = pa.Table.from_arrays(
        [
            pa.array([inference_date] * len(symbols), pa.date32()),
            pa.array(symbols, pa.string()),
            pa.array(preds, pa.float32()),
        ],
        schema=inference_schema,
    )
    inference_key = f"{INFERENCE_PREFIX}{inference_date.isoformat()}/data.parquet"
    s.write_parquet_s3(BUCKET, inference_key, inference_table, inference_schema)

    inference_meta = {
        "modelPrefix": model_prefix,
        "inferenceDate": inference_date.isoformat(),
        "trainingCutoff": meta["trainingCutoff"],
    }
    inference_meta_key = f"{INFERENCE_PREFIX}{inference_date.isoformat()}/meta.json"
    inference_meta_data = json.dumps(inference_meta, indent=2).encode("utf-8")
    s.write_bytes_s3(BUCKET, inference_meta_key, inference_meta_data)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    infer()
