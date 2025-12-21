# ---------- STANDARD LIBRARY ----------
import json
import logging
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import cast, overload

# ---------- THIRD-PARTY LIBRARIES ----------
import boto3
from botocore.exceptions import ClientError

# ---------- THIRD-PARTY LIBRARIES LAZY ----------
_pyarrow_parquet = None


def pyarrow_parquet():
    global _pyarrow_parquet
    if _pyarrow_parquet is None:
        import pyarrow.parquet as pq

        _pyarrow_parquet = pq
    return _pyarrow_parquet


# ---------- CONFIG ----------
DEBUG = bool(os.getenv("DEBUG", "False"))
DEBUG_MAX_DATE = date.fromisoformat(os.getenv("DEBUG_MAX_DATE", "9999-11-01"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "-1"))

DEFAULT_COMPRESSION_LEVEL = 1
DEFAULT_MIN_REMAINING_MS = 60_000
ONE_DAY = timedelta(days=1)

LAST_ADDED_KEY = "lastAdded"
LAST_PROCESSED_KEY = "lastProcessed"
LAST_TRAINED_KEY = "lastTrained"

META_DATA = "metadata/"
NEWS_COLLECTOR_STATE_KEY = f"{META_DATA}news_collector_state.json"
NEWS_PREPROC_STATE_KEY = f"{META_DATA}news_preproc_state.json"
STOCK_COLLECTOR_STATE_KEY = f"{META_DATA}stock_collector_state.json"
STOCK_PREPROC_STATE_KEY = f"{META_DATA}stock_preproc_state.json"
TRAINING_STATE_KEY = f"{META_DATA}training_state.json"

MODEL_PREFIX = "model/localDate="
MODEL_FILE_NAME = "model.keras"
PCA_FILE_NAME = "pca.pkl"
META_FILE_NAME = "meta.json"

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.getenv("RETRY_MAX_DELAY_S", "2.0"))

s3 = boto3.client("s3", region_name=AWS_REGION)


def setup_logger(name: str):
    logger = logging.getLogger(os.path.basename(name))
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    if "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        logger.setLevel(logging.INFO)
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        )
        logger.setLevel(logging.DEBUG)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def continue_execution(
    context=None, min_remaining_ms=DEFAULT_MIN_REMAINING_MS, logger=None
):
    if (
        context is not None
        and int(context.get_remaining_time_in_millis()) < min_remaining_ms
    ):
        if logger:
            logger.warning(f"Stopping execution due to timeout limit.")
        return False

    return True


def retry_s3(
    operation,
    retry_count=RETRY_COUNT,
    retry_delay=RETRY_DELAY_S,
    retry_max_delay=RETRY_MAX_DELAY_S,
):
    for attempt in range(retry_count):
        try:
            return operation()
        except ClientError as e:
            if attempt < retry_count - 1:
                code = e.response.get("Error", {}).get("Code", "")
                retryable = (
                    code.startswith("5")
                    or "Throttl" in code
                    or code
                    in (
                        "SlowDown",
                        "RequestTimeout",
                        "RequestTimeTooSkewed",
                        "InternalError",
                    )
                )

                if retryable:
                    max_sleep = min(retry_delay * (2**attempt), retry_max_delay)
                    sleep = random.uniform(0, max_sleep)
                    time.sleep(sleep)
                    continue

            raise  # No retry


def read_parquet_s3(bucket: str, key: str, schema):
    data = read_bytes_s3(bucket, key)
    buffer = BytesIO(data)
    try:
        table = pyarrow_parquet().read_table(buffer, columns=schema.names)
    except Exception as e:
        raise ValueError(f"Corrupted parquet at {key}: {e}")

    if schema_mismatch(schema, table.schema):
        raise ValueError(
            f"Schema mismatch for {key}. Expected {schema}, got {table.schema}"
        )

    return table


def write_parquet_s3(
    bucket: str,
    key: str,
    table,
    schema,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
):
    assert table is not None
    assert schema is not None

    if schema_mismatch(schema, table.schema):
        raise ValueError(
            f"Schema mismatch for {key}. Expected {schema}, got {table.schema}"
        )

    buffer = BytesIO()
    pyarrow_parquet().write_table(
        table, buffer, compression="zstd", compression_level=compression_level
    )
    data = buffer.getvalue()

    write_bytes_s3(bucket, key, data)


def schema_mismatch(expected, actual) -> bool:
    if len(expected) != len(actual):
        return True

    for f_exp, f_act in zip(expected, actual):
        if f_exp.name != f_act.name:
            return True
        if not f_exp.type.equals(f_act.type):
            return True
    return False


@overload
def read_json_date_s3(bucket: str, s3_key: str, json_key: str) -> date | None: ...


@overload
def read_json_date_s3(
    bucket: str, s3_key: str, json_key: str, default_value: date
) -> date: ...


def read_json_date_s3(bucket: str, s3_key: str, json_key: str, default_value=None):
    try:
        data = read_bytes_s3(bucket, s3_key)
        json_str = json.loads(data.decode("utf-8"))
        return date.fromisoformat(json_str[json_key])
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return default_value
        raise


def write_json_date_s3(bucket: str, s3_key: str, json_key: str, value: date):
    payload = {json_key: value.isoformat()}
    data = json.dumps(payload).encode("utf-8")
    write_bytes_s3(bucket, s3_key, data)


def list_keys_s3(bucket: str, prefix: str, sort_reversed: bool = False) -> list:
    def op():
        keys = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys, reverse=sort_reversed)

    return cast(list, retry_s3(op))


def get_now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def write_bytes_s3(bucket: str, key: str, data: bytes):
    retry_s3(lambda: s3.put_object(Bucket=bucket, Key=key, Body=data))


def read_bytes_s3(bucket: str, key: str) -> bytes:
    return cast(
        bytes, retry_s3(lambda: s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    )


def get_last_processed_date(bucket: str) -> date:
    stock_last = read_json_date_s3(bucket, STOCK_PREPROC_STATE_KEY, LAST_PROCESSED_KEY)
    assert stock_last is not None
    news_last = read_json_date_s3(bucket, NEWS_PREPROC_STATE_KEY, LAST_PROCESSED_KEY)
    assert news_last is not None
    return min(stock_last, news_last)


def write_directory_s3(bucket: str, prefix: str, local_dir: str):
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            key = f"{prefix}/{rel_path}"

            with open(local_path, "rb") as reader:
                data = reader.read()
                write_bytes_s3(bucket, key, data)


def read_directory_s3(bucket: str, prefix: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    for key in list_keys_s3(bucket, prefix):
        rel = key[len(prefix) :].lstrip("/")
        path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(read_bytes_s3(bucket, key))


def debug_limit_reached(days_processed: int, date_to_process: date) -> bool:
    if not DEBUG:
        return False

    if days_processed >= DEBUG_MAX_DAYS_PER_INVOCATION:
        return True

    if date_to_process >= DEBUG_MAX_DATE:
        return True

    return False


def result(logger: logging.Logger, code: int, message: str) -> dict[str, int | str]:
    logger.info(message)
    return {"statusCode": code, "body": message}
