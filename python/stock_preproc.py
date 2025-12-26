# --- STANDARD ---
import os
from collections import deque
from datetime import date

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY ---
from botocore.exceptions import ClientError

# --- THIRD-PARTY LAZY ---
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


# --- CONFIG ---
MIN_PRICE = 1e-3

RAW_PREFIX = "raw/stock/localDate="

MAX_LOG_RETURN_HORIZON = int(os.getenv("MAX_LOG_RETURN_HORIZON", "5"))
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "10000"))

LOGGER = s.setup_logger(__file__)

# --- SCHEMAS ---
_raw_schema = None


def get_raw_schema():
    global _raw_schema
    if _raw_schema is None:
        pa = pyarrow()
        _raw_schema = pa.schema(
            {
                s.LOCAL_DATE: pa.date32(),
                "id": pa.string(),
                "close": pa.float64(),
                "high": pa.float64(),
                "low": pa.float64(),
                "open": pa.float64(),
                "volume": pa.int64(),
            }
        )
    return _raw_schema


_close_column = None


def get_close_column():
    global _close_column
    if _close_column is None:
        _close_column = [f"{symbol}_close" for symbol in s.get_symbols()]
    return _close_column


_high_column = None


def get_high_column():
    global _high_column
    if _high_column is None:
        _high_column = [f"{symbol}_high" for symbol in s.get_symbols()]
    return _high_column


_low_column = None


def get_low_column():
    global _low_column
    if _low_column is None:
        _low_column = [f"{symbol}_low" for symbol in s.get_symbols()]
    return _low_column


_open_column = None


def get_open_column():
    global _open_column
    if _open_column is None:
        _open_column = [f"{symbol}_open" for symbol in s.get_symbols()]
    return _open_column


_volume_column = None


def get_volume_column():
    global _volume_column
    if _volume_column is None:
        _volume_column = [f"{symbol}_volume" for symbol in s.get_symbols()]
    return _volume_column


def append_pivoted_row(pivoted_columns: dict, py_date: date):
    raw_key = f"{RAW_PREFIX}{py_date.isoformat()}/data.parquet"

    try:
        raw_table = s.read_parquet_s3(raw_key, get_raw_schema())
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

    symbols = s.get_symbols()
    close_cols = get_close_column()
    high_cols = get_high_column()
    low_cols = get_low_column()
    open_cols = get_open_column()
    volume_cols = get_volume_column()

    pc = pivoted_columns
    pc[s.LOCAL_DATE].append(py_date)

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


def compute_log_return_horizon(horizon: int, close_history, curr_closes, np):
    if len(close_history) >= horizon:
        return compute_log_returns(close_history[-horizon], curr_closes)

    return np.zeros_like(curr_closes, dtype="float32")


def get_previous_closes(last_processed: date, schema) -> list:
    if last_processed < s.START_DATE:
        return []

    keys = s.list_keys_s3(s.PIVOTED_PREFIX, sort_reversed=True)
    if not keys:
        raise ValueError(f"no pivoted data found for {last_processed}")

    np = numpy()
    close_cols = get_close_column()

    collected = []

    keys = s.list_keys_s3(s.PIVOTED_PREFIX, sort_reversed=True)
    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table[s.LOCAL_DATE]

        for chunk in reversed(dates.chunks):
            for i in range(len(chunk) - 1, -1, -1):
                d = chunk[i].as_py()
                if d <= last_processed:
                    closes = np.array(
                        [table[col][i].as_py() for col in close_cols],
                        dtype="float32",
                    )
                    collected.append(closes)
                    if len(collected) == MAX_LOG_RETURN_HORIZON:
                        collected.reverse()
                        return collected
                elif d < last_processed:
                    break

    collected.reverse()
    return collected


def append_target_row(
    target_columns,
    close_history: deque,  # deque of past closes
    curr_closes,
    py_date: date,
):
    np = numpy()
    symbols = s.get_symbols()

    target_columns[s.LOCAL_DATE].append(py_date)

    # --- 1-day returns ---
    r1 = compute_log_return_horizon(1, close_history, curr_closes, np)

    # --- 3-day returns ---
    r3 = compute_log_return_horizon(3, close_history, curr_closes, np)

    # --- 5-day returns ---
    r5 = compute_log_return_horizon(5, close_history, curr_closes, np)

    # --- cross-sectional stats (1-day only) ---
    if not r1.any():
        z = np.zeros_like(r1)
    else:
        mean = float(r1.mean())
        std = float(r1.std())
        z = (r1 - mean) / std if std > 0.0 else np.zeros_like(r1)

    # rank in [0,1], stable
    order = np.argsort(r1)
    ranks = np.empty(len(r1), dtype="float32")
    n = len(r1)
    if n > 1:
        ranks[order] = np.arange(n, dtype="float32") / (n - 1)
    else:
        ranks[0] = 0.0

    direction = (r1 > 0.0).astype("int8")

    # --- write per-symbol ---
    for i, sym in enumerate(symbols):
        target_columns[f"{sym}_log_return_1d"].append(r1[i])
        target_columns[f"{sym}_log_return_3d"].append(r3[i])
        target_columns[f"{sym}_log_return_5d"].append(r5[i])
        target_columns[f"{sym}_zscore_1d"].append(z[i])
        target_columns[f"{sym}_rank_1d"].append(ranks[i])
        target_columns[f"{sym}_direction_1d"].append(direction[i])


def generate_pivoted_features(context=None):
    try:
        last_added_date = s.read_json_date_s3(
            s.STOCK_COLLECTOR_STATE_KEY, s.LAST_ADDED_KEY
        )
        if last_added_date is None:
            return s.result(LOGGER, 200, "no raw data to process")

        default_last_processed_date = s.START_DATE - s.ONE_DAY
        last_processed_date = s.read_json_date_s3(
            s.STOCK_PREPROC_STATE_KEY,
            s.LAST_PROCESSED_KEY,
            default_last_processed_date,
        )
        if last_processed_date >= last_added_date:
            return s.result(LOGGER, 200, "all data already processed")

        current_date = last_processed_date + s.ONE_DAY

        pivoted_schema = s.get_pivoted_schema()
        pivoted_columns = {field.name: [] for field in pivoted_schema}

        target_schema = s.get_target_schema()
        target_columns = {field.name: [] for field in target_schema}

        close_history = deque(
            get_previous_closes(last_processed_date, pivoted_schema),
            maxlen=MAX_LOG_RETURN_HORIZON,
        )

        days_processed = 0

        while s.continue_execution(context, MIN_REMAINING_MS, LOGGER):
            if s.debug_limit_reached(days_processed, current_date):
                break

            if current_date > last_added_date:
                break

            curr_closes = append_pivoted_row(pivoted_columns, current_date)
            if curr_closes is not None:
                append_target_row(
                    target_columns, close_history, curr_closes, current_date
                )
                close_history.append(curr_closes)

                days_processed += 1
                LOGGER.info(f"Computed pivoted + target features for {current_date}")

            current_date += s.ONE_DAY

        if days_processed == 0:
            return s.result(
                LOGGER, 200, "nothing computed. lambda timeout or debug limit too low"
            )

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

        pivoted_key = f"{s.PIVOTED_PREFIX}{timestamp}.parquet"
        target_key = f"{s.TARGET_PREFIX}{timestamp}.parquet"

        s.write_parquet_s3(pivoted_key, pivoted_table, pivoted_schema)
        s.write_parquet_s3(target_key, target_table, target_schema)

        new_last_processed = pivoted_table[s.LOCAL_DATE].to_pylist()[-1]
        s.write_json_date_s3(
            s.STOCK_PREPROC_STATE_KEY, s.LAST_PROCESSED_KEY, new_last_processed
        )

        return s.result(LOGGER, 200, "finished preprocessing")

    except Exception:
        LOGGER.exception("Error in generate_pivoted_features")
        return {"statusCode": 500, "body": "error"}


def lambda_handler(event=None, context=None):
    return generate_pivoted_features(context)


# --- ENTRY POINT ---
if __name__ == "__main__":
    generate_pivoted_features()
