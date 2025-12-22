# --- STANDARD ---
import json
import os
import pickle
import tempfile
from datetime import date

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY ---
import numpy as np
import pyarrow as pa
from tensorflow.keras import models  # type: ignore


# --- CONFIG ---
FLOAT_32 = np.float32

LOGGER = s.setup_logger(__file__)


def get_model_prefix() -> str:
    model_date = s.read_json_date_s3(s.TRAINING_STATE_KEY, s.LAST_TRAINED_KEY)
    return f"{s.MODEL_PREFIX}{model_date}/"


def load_model_artifacts(model_prefix: str):
    model_key = f"{model_prefix}{s.MODEL_FILE_NAME}"
    meta_key = f"{model_prefix}{s.META_FILE_NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        model_data = s.read_bytes_s3(model_key)

        local_model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
        with open(local_model_path, "wb") as f:
            f.write(model_data)

        model = models.load_model(local_model_path, compile=False)

    meta_data = s.read_bytes_s3(meta_key)
    meta = json.loads(meta_data.decode("utf-8"))

    return model, meta


def load_window(prefix: str, schema, end_date: date, window_size: int) -> np.ndarray:
    keys = s.list_keys_s3(prefix, sort_reversed=True)
    local_date = "localDate"
    rows = []
    for key in keys:
        table = s.read_parquet_s3(key, schema)
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
    return np.asarray(rows, dtype=FLOAT_32)


def infer(inference_date: date | None = None):
    last_processed_date = s.get_last_processed_date()
    max_inference_date = last_processed_date + s.ONE_DAY

    if inference_date is None or inference_date > max_inference_date:
        inference_date = max_inference_date

    end_date = inference_date - s.ONE_DAY

    model_prefix = get_model_prefix()
    model, meta = load_model_artifacts(model_prefix)

    norm = meta["normalization"]
    mean = np.asarray(norm["mean"], dtype=FLOAT_32)
    std = np.asarray(norm["std"], dtype=FLOAT_32)

    window_size = meta["windowSize"]

    stock_window = load_window(
        s.PIVOTED_PREFIX,
        s.get_pivoted_schema(),
        end_date,
        window_size,
    )
    news_window = load_window(
        s.LAGGED_PREFIX,
        s.get_lagged_schema(),
        end_date,
        window_size,
    )

    x = np.concatenate([stock_window, news_window], axis=1)
    x = (x - mean) / std
    x = x.reshape(1, *x.shape)

    preds = model.predict(x, verbose=0)[0].astype(FLOAT_32)

    if not np.isfinite(preds).all():
        raise ValueError("Inference produced NaN/Inf predictions")

    symbols = s.get_symbols()

    inference_schema = s.get_inference_schema()

    inference_table = pa.Table.from_arrays(
        [
            pa.array([inference_date] * len(symbols), pa.date32()),
            pa.array(symbols, pa.string()),
            pa.array(preds, pa.float32()),
        ],
        schema=inference_schema,
    )
    inference_key = f"{s.INFERENCE_PREFIX}{inference_date.isoformat()}/data.parquet"
    s.write_parquet_s3(inference_key, inference_table, inference_schema)

    inference_meta = {
        "modelPrefix": model_prefix,
        "inferenceDate": inference_date.isoformat(),
        "trainingCutoff": meta["trainingCutoff"],
    }
    inference_meta_key = f"{s.INFERENCE_PREFIX}{inference_date.isoformat()}/meta.json"
    inference_meta_data = json.dumps(inference_meta, indent=2).encode("utf-8")
    s.write_bytes_s3(inference_meta_key, inference_meta_data)
    LOGGER.info(f"finished inference for {inference_date}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    infer(date.fromisoformat("2025-12-19"))
    infer()
