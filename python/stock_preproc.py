# ---------- STANDARD LIBRARY ----------
import os
from datetime import date, datetime, timedelta, timezone

# ---------- THIRD-PARTY LIBRARIES ----------
import shared as s
from botocore.exceptions import ClientError

# ---------- THIRD-PARTY LIBRARIES LAZY ----------
_pyarrow = None


def pyarrow():
    global _pyarrow
    if _pyarrow is None:
        import pyarrow as pa

        _pyarrow = pa
    return _pyarrow


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

LOGGER = s.setup_logger()


# ---------- SCHEMAS ----------
def get_raw_schema():
    pa = pyarrow()
    return pa.schema(
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


def get_symbols_schema():
    pa = pyarrow()
    return pa.schema(
        {
            "id": pa.string(),
            "sector": pa.string(),
        }
    )


_symbols = None


def get_symbols():
    global _symbols
    if _symbols is None:
        symbols_table = s.read_parquet_s3(BUCKET, SYMBOLS_KEY, get_symbols_schema())
        if symbols_table is None:
            raise ValueError("symbols_table is None")
        _symbols = symbols_table.column("id").to_pylist()
    return _symbols


_close_column = None


def get_close_column():
    global _close_column
    if _close_column is None:
        _close_column = [f"{symbol}_close" for symbol in get_symbols()]
    return _close_column


_high_column = None


def get_high_column():
    global _high_column
    if _high_column is None:
        _high_column = [f"{symbol}_high" for symbol in get_symbols()]
    return _high_column


_low_column = None


def get_low_column():
    global _low_column
    if _low_column is None:
        _low_column = [f"{symbol}_low" for symbol in get_symbols()]
    return _low_column


_open_column = None


def get_open_column():
    global _open_column
    if _open_column is None:
        _open_column = [f"{symbol}_open" for symbol in get_symbols()]
    return _open_column


_volume_column = None


def get_volume_column():
    global _volume_column
    if _volume_column is None:
        _volume_column = [f"{symbol}_volume" for symbol in get_symbols()]
    return _volume_column


_pivoted_schema = None


def get_pivoted_schema():
    global _pivoted_schema
    if _pivoted_schema is None:
        pa = pyarrow()
        fields = [pa.field("localDate", pa.date32())]
        for sym in get_symbols():
            fields.extend(
                [
                    pa.field(f"{sym}_close", pa.float32()),
                    pa.field(f"{sym}_high", pa.float32()),
                    pa.field(f"{sym}_low", pa.float32()),
                    pa.field(f"{sym}_open", pa.float32()),
                    pa.field(f"{sym}_volume", pa.float32()),
                ]
            )
        _pivoted_schema = pa.schema(fields)
    return _pivoted_schema


def get_raw(py_date: date):
    raw_key = f"{RAW_PREFIX}{py_date.isoformat()}/data.parquet"
    return s.read_parquet_s3(BUCKET, raw_key, get_raw_schema())


def append_pivoted_row(pivoted_columns: dict, py_date: date) -> bool:
    try:
        raw_table = get_raw(py_date)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return False
        raise

    if raw_table is None:
        raise ValueError("raw_table is None")

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

    symbols = get_symbols()
    close_cols = get_close_column()
    high_cols = get_high_column()
    low_cols = get_low_column()
    open_cols = get_open_column()
    volume_cols = get_volume_column()

    pc = pivoted_columns
    pc["localDate"].append(py_date)
    for i in range(len(symbols)):
        pc[close_cols[i]].append(closes[i])
        pc[high_cols[i]].append(highs[i])
        pc[low_cols[i]].append(lows[i])
        pc[open_cols[i]].append(opens_[i])
        pc[volume_cols[i]].append(volumes[i])

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

        pivoted_schema = get_pivoted_schema()
        pivoted_columns = {field.name: [] for field in pivoted_schema}

        days_processed = 0

        while (
            s.continue_execution(context, MIN_REMAINING_MS, LOGGER)
            and current <= last_added
        ):
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            if append_pivoted_row(pivoted_columns, current):
                days_processed += 1
                LOGGER.info(f"Computed pivoted features for {current}")

            current += ONE_DAY

        if days_processed == 0:
            return {
                "statusCode": 200,
                "body": "no pivoted features computed due to lambda timeout",
            }

        pa = pyarrow()
        arrays = [
            pa.array(pivoted_columns[field.name], type=field.type)
            for field in pivoted_schema
        ]
        pivoted_table = pa.Table.from_arrays(arrays, schema=pivoted_schema)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        file_name = f"{timestamp}.parquet"
        pivoted_key = f"{PIVOTED_PREFIX}{file_name}"
        s.write_parquet_s3(pivoted_table, BUCKET, pivoted_key, pivoted_schema)

        new_last_processed = pivoted_table["localDate"].to_pylist()[-1]
        s.write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, new_last_processed
        )

        return {
            "statusCode": 200,
            "body": f"finished preprocessing up to and including {new_last_processed.isoformat()}",
        }

    except Exception:
        LOGGER.exception("Error in generate_pivoted_features")
        return {"statusCode": 500, "body": "error"}


def lambda_handler(event=None, context=None):
    return generate_pivoted_features(context)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_pivoted_features()
