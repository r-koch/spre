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
_shared_model = None
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


def shared_model():
    global _shared_model
    if _shared_model is None:
        import shared_model as sm

        _shared_model = sm
    return _shared_model


def models():
    global _tensorflow_keras_models
    if _tensorflow_keras_models is None:
        from tensorflow.keras import models  # type: ignore

        _tensorflow_keras_models = models
    return _tensorflow_keras_models


# --- CONFIG ---
LAST_INFERRED_KEY = "lastInferred"

LOGGER = s.setup_logger(__file__)


def get_next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def parse_score_from_key(key: str) -> float | None:
    # expects: model/score={score}/model.keras
    try:
        part = key.split(s.MODEL_PREFIX, 1)[1]
        score_str = part.split("/", 1)[0]
        return float(score_str)
    except Exception:
        return None


def get_best_model_key() -> str:
    keys = s.list_keys_s3(s.MODEL_PREFIX)

    best_score = None
    best_key = None

    for key in keys:
        if not key.endswith(f"/{s.MODEL_FILE_NAME}"):
            continue

        score = parse_score_from_key(key)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_key = key
            best_score = score

    if best_score is None or best_key is None:
        raise ValueError("No scored models found in model/")

    return best_key


def get_model(model_key):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, s.MODEL_FILE_NAME)
        with open(model_path, "wb") as f:
            f.write(s.read_bytes_s3(model_key))

        shared_model()
        return models().load_model(model_path, compile=False)


def get_model_metadata(model_key):
    meta_key = model_key.replace(s.MODEL_FILE_NAME, s.META_FILE_NAME)
    return json.loads(s.read_bytes_s3(meta_key))


def get_latest_pivoted_date_before(before_date) -> date:
    return get_latest_pivoted_dates_before(before_date)[0]


def get_latest_pivoted_dates_before(before_date: date, count: int = 1) -> list[date]:
    keys = s.list_keys_s3(s.PIVOTED_PREFIX, sort_reversed=True)
    found: list[date] = []

    for key in keys:
        table = s.read_parquet_column_s3(key, [s.LOCAL_DATE])
        dates = table[s.LOCAL_DATE]

        for chunk in reversed(dates.chunks):
            for i in range(len(chunk) - 1, -1, -1):
                d = chunk[i].as_py()
                if d < before_date:
                    found.append(d)
                    if len(found) == count:
                        return found
                elif d >= before_date:
                    continue
                else:
                    break

    raise ValueError(
        f"Found {len(found)} pivoted dates before {before_date}, need {count}"
    )


def load_window(prefix, schema, end_date, window):
    rows = []
    keys = s.list_keys_s3(prefix, sort_reversed=True)

    np = numpy()

    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table[s.LOCAL_DATE]

        for chunk in reversed(dates.chunks):
            for i in range(len(chunk) - 1, -1, -1):
                if chunk[i].as_py() <= end_date:
                    rows.append(
                        [
                            table[col][i].as_py()
                            for col in schema.names
                            if col != s.LOCAL_DATE
                        ]
                    )
                    if len(rows) == window:
                        rows.reverse()
                        return np.asarray(rows, dtype="float32")

    raise ValueError("Insufficient data for inference window")


def infer(context=None):
    try:
        model_key = get_best_model_key()
        meta = get_model_metadata(model_key)

        max_date = get_next_weekday(s.get_last_processed_date())
        last_inferred = s.read_json_date_s3(s.INFERENCE_STATE_KEY, LAST_INFERRED_KEY)
        if last_inferred is None:
            inference_date = date.fromisoformat(meta["training_date"])
        else:
            inference_date = get_next_weekday(last_inferred)

        if inference_date > max_date:
            return s.result(LOGGER, 200, "inference up to date")

        model = get_model(model_key)

        window = meta["config"]["window_size"]
        symbols = s.get_symbols()

        pivoted_schema = s.get_pivoted_schema()
        lagged_schema = s.get_lagged_schema()
        inference_schema = s.get_inference_schema()

        pa = pyarrow()

        while s.continue_execution(context, logger=LOGGER):

            end_date = get_latest_pivoted_date_before(inference_date)

            stock_window = load_window(
                s.PIVOTED_PREFIX, pivoted_schema, end_date, window
            )
            news_window = load_window(s.LAGGED_PREFIX, lagged_schema, end_date, window)

            # reshape
            stock_window = stock_window.reshape(1, window, len(symbols), -1).astype(
                "float32", copy=False
            )
            news_window = news_window.reshape(1, window, -1).astype(
                "float32", copy=False
            )

            preds = model(
                {"stock": stock_window, "news": news_window},
                training=False,
            )

            (
                zscore,
                rank,
                direction,
                log_r1,
                log_r3,
                log_r5,
            ) = [p.numpy()[0].astype("float32", copy=False) for p in preds]

            inv_scale = 1.0 / s.TARGET_SCALE

            log_r1 *= inv_scale
            log_r3 *= inv_scale
            log_r5 *= inv_scale

            table = pa.Table.from_arrays(
                [
                    pa.array([inference_date] * len(symbols), pa.date32()),
                    pa.array(symbols, pa.string()),
                    pa.array(zscore, pa.float32()),
                    pa.array(rank, pa.float32()),
                    pa.array(direction, pa.float32()),
                    pa.array(log_r1, pa.float32()),
                    pa.array(log_r3, pa.float32()),
                    pa.array(log_r5, pa.float32()),
                ],
                schema=inference_schema,
            )

            key = f"{s.INFERENCE_PREFIX}{inference_date}/data.parquet"
            s.write_parquet_s3(key, table, table.schema)

            s.write_json_date_s3(
                s.INFERENCE_STATE_KEY,
                LAST_INFERRED_KEY,
                inference_date,
            )

            LOGGER.info(f"Inference completed for {inference_date}")

            inference_date = get_next_weekday(inference_date)
            if inference_date > max_date:
                break

        return s.result(LOGGER, 200, "finished inference")

    except Exception:
        LOGGER.exception("Inference failed")
        return {"statusCode": 500, "body": "error"}


# --- LAMBDA ---
def lambda_handler(event=None, context=None):
    return infer(context)


# --- ENTRY POINT ---
if __name__ == "__main__":
    infer()
