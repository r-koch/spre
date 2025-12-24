# --- STANDARD ---
import json
import os
import tempfile
from datetime import date, timedelta

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY LAZY ---
_numpy = None
_pyarrow = None
_tensorflow_keras_models = None


def numpy():
    global _numpy
    if _numpy is None:
        import numpy as np

        _numpy = np
    return _numpy


def pyarrow():
    global _pyarrow
    if _pyarrow is None:
        import pyarrow as pa

        _pyarrow = pa
    return _pyarrow


def models():
    global _tensorflow_keras_models
    if _tensorflow_keras_models is None:
        from tensorflow.keras import models  # type: ignore

        _tensorflow_keras_models = models
    return _tensorflow_keras_models


# --- CONFIG ---
DEFAULT_LAST_INFERED_DATE = date.fromisoformat(
    os.getenv("INFERENCE_START_DATE", "2025-12-19")
)
LAST_INFERED_KEY = "lastInferred"

LOGGER = s.setup_logger(__file__)


def get_model_and_meta():
    model_date = s.read_json_date_s3(s.TRAINING_STATE_KEY, s.LAST_TRAINED_KEY)
    model_key = f"{s.MODEL_PREFIX}{model_date}/{s.MODEL_FILE_NAME}"
    meta_key = f"{s.MODEL_PREFIX}{model_date}/{s.META_FILE_NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        model_data = s.read_bytes_s3(model_key)

        local_model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
        with open(local_model_path, "wb") as f:
            f.write(model_data)

        model = models().load_model(local_model_path, compile=False)

    meta_data = s.read_bytes_s3(meta_key)
    meta = json.loads(meta_data.decode("utf-8"))

    return model, meta


def get_latest_pivoted_date_before(inference_date: date) -> date | None:
    keys = s.list_keys_s3(s.PIVOTED_PREFIX, sort_reversed=True)
    for key in keys:
        table = s.read_parquet_column_s3(key, [s.LOCAL_DATE])
        dates = table[s.LOCAL_DATE]

        for chunk in reversed(dates.chunks):
            for i in range(len(chunk) - 1, -1, -1):
                d = chunk[i].as_py()
                if d < inference_date:
                    return d

    return None


def load_window(prefix: str, schema, end_date: date, window_size: int):
    np = numpy()
    value_columns = [name for name in schema.names if name != s.LOCAL_DATE]
    collected_rows = []
    keys = s.list_keys_s3(prefix, sort_reversed=True)
    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table[s.LOCAL_DATE].to_numpy(zero_copy_only=False)
        arrays = [table[col].to_numpy(zero_copy_only=False) for col in value_columns]

        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= end_date:
                collected_rows.append([arr[i] for arr in arrays])
                if len(collected_rows) == window_size:
                    break

        if len(collected_rows) == window_size:
            break

    if len(collected_rows) != window_size:
        raise ValueError("Insufficient data for inference window")

    collected_rows.reverse()
    return np.stack(collected_rows, axis=0).astype(np.float32, copy=False)


def read_single_row_for_date(prefix: str, schema, target_date: date):
    value_columns = [name for name in schema.names if name != s.LOCAL_DATE]

    keys = s.list_keys_s3(prefix, sort_reversed=True)
    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table[s.LOCAL_DATE]

        for chunk in reversed(dates.chunks):
            for i in range(len(chunk) - 1, -1, -1):
                d = chunk[i].as_py()

                if d == target_date:
                    return [table[col][i].as_py() for col in value_columns]

                if d < target_date:
                    break  # older than target → stop scanning this file

    raise ValueError(f"Row not found for date {target_date}")


def get_next_inference_date(last: date) -> date:
    next = last + timedelta(days=1)
    while next.weekday() >= 5:  # 5=Sat, 6=Sun
        next += timedelta(days=1)
    return next


def infer(context=None):
    try:

        last_inferred_date = s.read_json_date_s3(
            s.INFERENCE_STATE_KEY,
            LAST_INFERED_KEY,
            DEFAULT_LAST_INFERED_DATE,
        )

        inference_date = get_next_inference_date(last_inferred_date)

        end_date = get_latest_pivoted_date_before(inference_date)
        if end_date is None:
            return s.result(LOGGER, 200, "inference is up to date")

        np = numpy()
        pa = pyarrow()

        model, meta = get_model_and_meta()

        window_size = meta["windowSize"]
        norm = meta["normalization"]
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        stock_feature_count = norm["stockFeatureCount"]

        mean_stock = mean[:, :stock_feature_count]
        mean_news = mean[:, stock_feature_count:]

        std_stock = std[:, :stock_feature_count]
        std_news = std[:, stock_feature_count:]

        symbols = s.get_symbols()

        pivoted_schema = s.get_pivoted_schema()
        lagged_schema = s.get_lagged_schema()
        inference_schema = s.get_inference_schema()

        stock_window = load_window(
            s.PIVOTED_PREFIX, pivoted_schema, end_date, window_size
        )
        news_window = load_window(s.LAGGED_PREFIX, lagged_schema, end_date, window_size)

        num_rows = stock_window.shape[0]
        num_stock_features = stock_window.shape[1]
        num_news_features = news_window.shape[1]

        assert num_stock_features == stock_feature_count, (
            f"Stock feature mismatch: training={stock_feature_count}, "
            f"inference={num_stock_features}"
        )

        expected_total = mean.shape[1]
        assert num_stock_features + num_news_features == expected_total, (
            f"Total feature mismatch: training={expected_total}, "
            f"inference={num_stock_features + num_news_features}"
        )

        x = np.empty(
            (num_rows, num_stock_features + num_news_features),
            dtype=np.float32,
        )

        while s.continue_execution(context, logger=LOGGER):

            x[:, :stock_feature_count] = (stock_window - mean_stock) / std_stock
            x[:, stock_feature_count:] = (news_window - mean_news) / std_news

            x_reshaped = x.reshape(1, *x.shape)

            preds = model(x_reshaped, training=False).numpy()[0].astype(np.float32)

            if not np.isfinite(preds).all():
                raise ValueError("Inference produced NaN/Inf predictions")

            inference_table = pa.Table.from_arrays(
                [
                    pa.array([inference_date] * len(symbols), pa.date32()),
                    pa.array(symbols, pa.string()),
                    pa.array(preds, pa.float32()),
                ],
                schema=inference_schema,
            )
            inference_key = (
                f"{s.INFERENCE_PREFIX}{inference_date.isoformat()}/data.parquet"
            )
            s.write_parquet_s3(inference_key, inference_table, inference_schema)

            training_cutoff = meta["trainingCutoff"]
            inference_meta = {
                "modelPrefix": f"{s.MODEL_PREFIX}{training_cutoff}",
                "inferenceDate": inference_date.isoformat(),
                "trainingCutoff": training_cutoff,
            }
            inference_meta_key = (
                f"{s.INFERENCE_PREFIX}{inference_date.isoformat()}/meta.json"
            )
            inference_meta_data = json.dumps(inference_meta, indent=2).encode("utf-8")
            s.write_bytes_s3(inference_meta_key, inference_meta_data)
            s.write_json_date_s3(
                s.INFERENCE_STATE_KEY, LAST_INFERED_KEY, inference_date
            )
            LOGGER.info(f"finished inference for {inference_date}")

            inference_date = get_next_inference_date(inference_date)
            end_date = get_latest_pivoted_date_before(inference_date)
            if end_date is None:
                break

            new_stock_row = read_single_row_for_date(
                s.PIVOTED_PREFIX, pivoted_schema, end_date
            )
            new_news_row = read_single_row_for_date(
                s.LAGGED_PREFIX, lagged_schema, end_date
            )

            # slide windows (drop oldest, append newest)
            stock_window[:-1] = stock_window[1:]
            stock_window[-1] = new_stock_row

            news_window[:-1] = news_window[1:]
            news_window[-1] = new_news_row

        return s.result(LOGGER, 200, f"finished infer")

    except Exception:
        LOGGER.exception("Error in infer")
        return {"statusCode": 500, "body": "error"}


# --- LAMBDA ---
def lambda_handler(event=None, context=None):
    return infer(context)


# --- ENTRY POINT ---
if __name__ == "__main__":
    infer()
