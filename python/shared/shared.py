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
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

# ---------- CONFIG ----------
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "60000"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.getenv("RETRY_MAX_DELAY_S", "2.0"))
TEMP_PREFIX = os.getenv("TEMP_PREFIX", "tmp/")


s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------- LOGGING CONFIG ----------
def setup_logger():
    logger = logging.getLogger(__name__)
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


# ---------- S3 UTILS ----------
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


def read_parquet_s3(bucket: str, key: str, schema: pa.Schema) -> pa.Table:
    def op():
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        buffer = BytesIO(s3_object["Body"].read())
        try:
            table = pq.read_table(buffer, columns=schema.names)
        except Exception as e:
            raise ValueError(f"Corrupted parquet at {key}: {e}")

        if schema_mismatch(schema, table.schema):
            raise ValueError(
                f"Schema mismatch for {key}. Expected {schema}, got {table.schema}"
            )

        return table

    return retry_s3(op)


def write_parquet_s3(table: pa.Table, bucket: str, key: str, schema: pa.Schema):
    if table is None:
        raise ValueError("table must not be None for write_parquet_s3")

    if schema is None:
        raise ValueError("Schema must not be None for write_parquet_s3")

    if schema_mismatch(schema, table.schema):
        raise ValueError(
            f"Schema mismatch for {key}. Expected {schema}, got {table.schema}"
        )

    buffer = BytesIO()
    pq.write_table(table, buffer, compression="zstd", compression_level=3)
    data = buffer.getvalue()

    temp_key = f"{TEMP_PREFIX}{key}"
    retry_s3(lambda: s3.put_object(Bucket=bucket, Key=temp_key, Body=data))
    retry_s3(
        lambda: s3.copy_object(
            Bucket=bucket, CopySource={"Bucket": bucket, "Key": temp_key}, Key=key
        )
    )
    retry_s3(lambda: s3.delete_object(Bucket=bucket, Key=temp_key))


def schema_mismatch(expected: pa.Schema, actual: pa.Schema) -> bool:
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
    def op():
        s3_object = s3.get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(s3_object["Body"].read().decode("utf-8"))
        return datetime.strptime(data[json_key], "%Y-%m-%d").date()

    try:
        return cast(date, retry_s3(op))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return default_value
        raise


def write_json_date_s3(bucket: str, s3_key: str, json_key: str, value: date):
    def op():
        payload = {json_key: value.isoformat()}
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(payload).encode("utf-8"),
        )

    retry_s3(op)
