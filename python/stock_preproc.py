# ---------- STANDARD LIBRARY ----------
import os
from datetime import date, timedelta

# ---------- THIRD-PARTY LIBRARIES ----------
import shared as s
from botocore.exceptions import ClientError

# ---------- THIRD-PARTY LIBRARIES LAZY ----------
_numpy = None
_pyarrow = None


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


# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
START_DATE = date.fromisoformat(os.getenv("START_DATE", "1999-11-01"))
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "10000"))
DEBUG_MAX_DATE = date.fromisoformat(os.getenv("DEBUG_MAX_DATE", "2000-11-01"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "-1"))

MIN_PRICE = 1e-3
ONE_DAY = timedelta(days=1)

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}stock_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}stock_preproc_state.json"
SYMBOLS_KEY = "symbols/spx.parquet"

RAW_PREFIX = "raw/stock/localDate="
PIVOTED_PREFIX = "stock/pivoted/"
TARGET_LOG_RETURNS_PREFIX = "stock/target-log-returns/"

LOGGER = s.setup_logger(__file__)

# ---------- SCHEMAS ----------
_raw_schema = None


def get_raw_schema():
    global _raw_schema
    if _raw_schema is None:
        pa = pyarrow()
        _raw_schema = pa.schema(
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
    return _raw_schema


_symbols_schema = None


def get_symbols_schema():
    global _symbols_schema
    if _symbols_schema is None:
        pa = pyarrow()
        _symbols_schema = pa.schema(
            {
                "id": pa.string(),
                "sector": pa.string(),
            }
        )
    return _symbols_schema


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


_target_schema = None


def get_target_schema():
    global _target_schema
    if _target_schema is None:
        pa = pyarrow()
        fields = [pa.field("localDate", pa.date32())]
        for sym in get_symbols():
            fields.append(pa.field(f"{sym}_return", pa.float32()))
        _target_schema = pa.schema(fields)
    return _target_schema


def append_pivoted_row(pivoted_columns: dict, py_date: date):
    raw_key = f"{RAW_PREFIX}{py_date.isoformat()}/data.parquet"

    try:
        raw_table = s.read_parquet_s3(BUCKET, raw_key, get_raw_schema())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return None
        raise

    assert raw_table is not None

    closes = raw_table["close"].to_numpy().astype("float32", copy=False)
    highs = raw_table["high"].to_numpy().astype("float32", copy=False)
    lows = raw_table["low"].to_numpy().astype("float32", copy=False)
    opens_ = raw_table["open"].to_numpy().astype("float32", copy=False)
    volumes = raw_table["volume"].to_numpy().astype("float32", copy=False)

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

    return closes


def compute_log_returns(prev_closes, curr_closes):
    np = numpy()

    # compute in float64 for numerical stability, store in float32
    prev = prev_closes.astype("float64", copy=False)
    curr = curr_closes.astype("float64", copy=False)

    out = np.zeros_like(prev, dtype="float32")

    valid = (prev > MIN_PRICE) & (curr > MIN_PRICE)
    if valid.any():
        out[valid] = np.log(curr[valid] / prev[valid]).astype("float32")

    return out


def get_previous_closes(last_processed: date):
    if last_processed < START_DATE:
        return None

    keys = s.list_keys_s3(BUCKET, PIVOTED_PREFIX, sort_reversed=True)
    if not keys:
        raise ValueError(f"no pivoted data found for {last_processed}")

    schema = get_pivoted_schema()
    np = numpy()
    close_cols = get_close_column()

    pivoted_table = s.read_parquet_s3(BUCKET, keys[0], schema)
    assert pivoted_table is not None

    if pivoted_table["localDate"][-1].as_py() == last_processed:
        return np.array(
            [pivoted_table[col][-1].as_py() for col in close_cols],
            dtype="float32",
        )

    pa = pyarrow()
    for key in keys:
        pivoted_table = s.read_parquet_s3(BUCKET, key, schema)
        assert pivoted_table is not None

        indices = (pivoted_table["localDate"] == pa.scalar(last_processed)).nonzero()[0]
        if len(indices) == 1:
            row_idx = indices[0].as_py()
            return np.array(
                [pivoted_table[col][row_idx].as_py() for col in close_cols],
                dtype="float32",
            )

    raise ValueError(f"no pivoted data found for {last_processed}")


def append_target_row(
    target_columns: dict, previous_closes, curr_closes, py_date: date
):
    target_columns["localDate"].append(py_date)

    if previous_closes is None:
        for sym in get_symbols():
            target_columns[f"{sym}_return"].append(0.0)
    else:
        returns = compute_log_returns(previous_closes, curr_closes)
        for sym, value in zip(get_symbols(), returns):
            target_columns[f"{sym}_return"].append(value)


def generate_pivoted_features(context=None):
    try:
        last_added_date = s.read_json_date_s3(
            BUCKET, COLLECTOR_STATE_KEY, s.LAST_ADDED_KEY
        )
        if last_added_date is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        default_last_processed_date = START_DATE - ONE_DAY
        last_processed_date = s.read_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, s.LAST_PROCESSED_KEY, default_last_processed_date
        )
        if last_processed_date >= last_added_date:
            return {"statusCode": 200, "body": "all data already processed"}

        current_date = last_processed_date + ONE_DAY

        pivoted_schema = get_pivoted_schema()
        pivoted_columns = {field.name: [] for field in pivoted_schema}

        target_schema = get_target_schema()
        target_columns = {field.name: [] for field in target_schema}

        previous_closes = get_previous_closes(last_processed_date)

        days_processed = 0

        while s.continue_execution(context, MIN_REMAINING_MS, LOGGER):
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            if current_date >= DEBUG_MAX_DATE:
                break

            if current_date > last_added_date:
                break

            curr_closes = append_pivoted_row(pivoted_columns, current_date)
            if curr_closes is not None:
                append_target_row(
                    target_columns, previous_closes, curr_closes, current_date
                )

                previous_closes = curr_closes
                days_processed += 1

            LOGGER.info(f"Computed pivoted + target features for {current_date}")
            current_date += ONE_DAY

        if days_processed == 0:
            return {
                "statusCode": 200,
                "body": "no pivoted features computed due to lambda timeout",
            }

        pa = pyarrow()
        pivoted_arrays = [
            pa.array(pivoted_columns[field.name], type=field.type)
            for field in pivoted_schema
        ]
        pivoted_table = pa.Table.from_arrays(pivoted_arrays, schema=pivoted_schema)

        target_arrays = [
            pa.array(target_columns[field.name], type=field.type)
            for field in target_schema
        ]
        target_table = pa.Table.from_arrays(target_arrays, schema=target_schema)

        timestamp = s.get_now_timestamp()

        pivoted_key = f"{PIVOTED_PREFIX}{timestamp}.parquet"
        target_key = f"{TARGET_LOG_RETURNS_PREFIX}{timestamp}.parquet"

        s.write_parquet_s3(pivoted_table, BUCKET, pivoted_key, pivoted_schema)
        s.write_parquet_s3(target_table, BUCKET, target_key, target_schema)

        new_last_processed = pivoted_table["localDate"].to_pylist()[-1]
        s.write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, s.LAST_PROCESSED_KEY, new_last_processed
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
