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
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
REGION = os.getenv("AWS_REGION", "eu-west-1")
START_DATE = os.getenv("START_DATE", "1999-11-01")
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "60000"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.getenv("RETRY_MAX_DELAY_S", "2.0"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "2"))

ONE_DAY = timedelta(days=1)

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}stock_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}stock_preproc_state.json"
SYMBOLS_KEY = "symbols/spx.parquet"

LAST_ADDED_KEY = "lastAdded"
LAST_PROCESSED_KEY = "lastProcessed"

TEMP_PREFIX = "tmp/"
RAW_PREFIX = "raw/stock/localDate="
PIVOTED_PREFIX = "stock/pivoted/"

s3 = boto3.client("s3", region_name=REGION)


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


logger = setup_logger()


# ---------- HELPERS ----------
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
    bucket: str, s3_key: str, json_key: str, default_value: str
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
            if default_value is not None:
                return datetime.strptime(default_value, "%Y-%m-%d").date() - ONE_DAY
            else:
                return None
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


# ---------- SCHEMAS ----------

RAW_SCHEMA = pa.schema(
    {
        "localDate": pa.date32(),
        "id": pa.string(),
        "close": pa.float64(),
        "high": pa.float64(),
        "low": pa.float64(),
        "open": pa.float64(),
        "volume": pa.int64(),
    }
)


SYMBOLS_SCHEMA = pa.schema(
    {
        "id": pa.string(),
        "sector": pa.string(),
    }
)


def get_symbols():
    symbols_table = read_parquet_s3(BUCKET, SYMBOLS_KEY, SYMBOLS_SCHEMA)
    return symbols_table.column("id").to_pylist()


SYMBOLS = get_symbols()
SYMBOL_COUNT = len(SYMBOLS)
CN_CLOSE = [f"{s}_close" for s in SYMBOLS]
CN_HIGH = [f"{s}_high" for s in SYMBOLS]
CN_LOW = [f"{s}_low" for s in SYMBOLS]
CN_OPEN = [f"{s}_open" for s in SYMBOLS]
CN_VOLUME = [f"{s}_volume" for s in SYMBOLS]


def build_pivoted_schema():
    fields = [pa.field("localDate", pa.date32())]
    metrics = ["close", "high", "low", "open", "volume"]
    for sym in SYMBOLS:
        for m in metrics:
            fields.append(pa.field(f"{sym}_{m}", pa.float32()))
    return pa.schema(fields)


PIVOTED_SCHEMA = build_pivoted_schema()


def continue_execution(context=None):
    if (
        context is not None
        and int(context.get_remaining_time_in_millis()) < MIN_REMAINING_MS
    ):
        logger.warning(f"Stopping execution due to timeout limit.")
        return False

    return True


def get_raw(py_date: date) -> pa.Table:
    raw_key = f"{RAW_PREFIX}{py_date.isoformat()}/data.parquet"
    return read_parquet_s3(BUCKET, raw_key, RAW_SCHEMA)


def append_pivoted_row(pivoted_columns: dict, py_date: date, raw_table: pa.Table):
    col_close = raw_table.column("close").chunks[0]
    col_high = raw_table.column("high").chunks[0]
    col_low = raw_table.column("low").chunks[0]
    col_open = raw_table.column("open").chunks[0]
    col_volume = raw_table.column("volume").chunks[0]

    closes = col_close.to_numpy()
    highs = col_high.to_numpy()
    lows = col_low.to_numpy()
    opens_ = col_open.to_numpy()
    volumes = col_volume.to_numpy().astype("float32")

    pc = pivoted_columns
    pc["localDate"].append(py_date)
    for i in range(SYMBOL_COUNT):
        pc[CN_CLOSE[i]].append(closes[i])
        pc[CN_HIGH[i]].append(highs[i])
        pc[CN_LOW[i]].append(lows[i])
        pc[CN_OPEN[i]].append(opens_[i])
        pc[CN_VOLUME[i]].append(volumes[i])


def generate_pivoted_features(context=None):
    try:
        last_added = read_json_date_s3(BUCKET, COLLECTOR_STATE_KEY, LAST_ADDED_KEY)
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        last_processed = read_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, START_DATE
        )
        if last_processed >= last_added:
            return {"statusCode": 200, "body": "all data already processed"}

        current = last_processed + ONE_DAY

        pivoted_columns = {field.name: [] for field in PIVOTED_SCHEMA}

        days_processed = 0

        while continue_execution(context) and current <= last_added:
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            raw_table = get_raw(current)
            append_pivoted_row(pivoted_columns, current, raw_table)
            days_processed += 1
            logger.info(f"Computed pivoted features for {current}")
            current += ONE_DAY

        if days_processed == 0:
            return {
                "statusCode": 200,
                "body": "no pivoted features computed due to lambda timeout",
            }

        arrays = [
            pa.array(pivoted_columns[field.name], type=field.type)
            for field in PIVOTED_SCHEMA
        ]
        pivoted_table = pa.Table.from_arrays(arrays, schema=PIVOTED_SCHEMA)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        file_name = f"{timestamp}.parquet"
        pivoted_key = f"{PIVOTED_PREFIX}{file_name}"
        write_parquet_s3(pivoted_table, BUCKET, pivoted_key, PIVOTED_SCHEMA)

        new_last_processed = pivoted_table["localDate"].to_pylist()[-1]
        write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, new_last_processed
        )

        return {
            "statusCode": 200,
            "body": f"finished preprocessing up to and including {new_last_processed:%Y-%m-%d}",
        }

    except Exception:
        logger.exception("Error in generate_pivoted_features")
        return {"statusCode": 500, "body": "error"}


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_pivoted_features()
