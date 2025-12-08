# ---------- STANDARD LIBRARY ----------
import os
from datetime import date, datetime, timedelta, timezone

# ---------- THIRD-PARTY LIBRARIES ----------
import pyarrow as pa
import shared as s
from botocore.exceptions import ClientError

# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
START_DATE = date.fromisoformat(os.getenv("START_DATE", "1999-11-01"))
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "10000"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "2"))

ONE_DAY = timedelta(days=1)

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}stock_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}stock_preproc_state.json"
SYMBOLS_KEY = "symbols/spx.parquet"

LAST_ADDED_KEY = "lastAdded"
LAST_PROCESSED_KEY = "lastProcessed"

RAW_PREFIX = "raw/stock/localDate="
PIVOTED_PREFIX = "stock/pivoted/"


logger = s.setup_logger()

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
    symbols_table = s.read_parquet_s3(BUCKET, SYMBOLS_KEY, SYMBOLS_SCHEMA)
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


def get_raw(py_date: date) -> pa.Table:
    raw_key = f"{RAW_PREFIX}{py_date.isoformat()}/data.parquet"
    return s.read_parquet_s3(BUCKET, raw_key, RAW_SCHEMA)


def append_pivoted_row(pivoted_columns: dict, py_date: date) -> bool:
    try:
        raw_table = get_raw(py_date)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return False
        raise

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

    return True


def generate_pivoted_features(context=None):
    try:
        last_added = s.read_json_date_s3(BUCKET, COLLECTOR_STATE_KEY, LAST_ADDED_KEY)
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        default_last_processed = START_DATE - ONE_DAY
        last_processed = s.read_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, default_last_processed
        )
        if last_processed >= last_added:
            return {"statusCode": 200, "body": "all data already processed"}

        current = last_processed + ONE_DAY

        pivoted_columns = {field.name: [] for field in PIVOTED_SCHEMA}

        days_processed = 0

        while (
            s.continue_execution(context, MIN_REMAINING_MS, logger)
            and current <= last_added
        ):
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            if append_pivoted_row(pivoted_columns, current):
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
        s.write_parquet_s3(pivoted_table, BUCKET, pivoted_key, PIVOTED_SCHEMA)

        new_last_processed = pivoted_table["localDate"].to_pylist()[-1]
        s.write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, new_last_processed
        )

        return {
            "statusCode": 200,
            "body": f"finished preprocessing up to and including {new_last_processed.isoformat()}",
        }

    except Exception:
        logger.exception("Error in generate_pivoted_features")
        return {"statusCode": 500, "body": "error"}


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_pivoted_features()
