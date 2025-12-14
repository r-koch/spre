# ---------- STANDARD LIBRARY ----------
import os
import unicodedata
from collections import deque
from datetime import date, datetime, timedelta, timezone

# ---------- THIRD-PARTY LIBRARIES ----------
import regex as re
import shared as s
from botocore.exceptions import ClientError

# ---------- THIRD-PARTY LIBRARIES LAZY ----------
_numpy = None
_pyarrow = None
_torch = None


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


def torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
REGION = os.getenv("AWS_REGION", "eu-west-1")
LAG_DAYS = int(os.getenv("LAG_DAYS", "5"))
START_DATE = date.fromisoformat(os.getenv("START_DATE", "1999-11-01"))
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "240000"))
MODEL_DIR_OR_NAME = os.getenv(
    "MODEL_DIR", "ProsusAI/finbert"
)  # locally use model name, in docker use env
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.getenv("RETRY_MAX_DELAY_S", "2.0"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "2"))

CONFLICT_THRESHOLD_LOW = 0.05
CONFLICT_THRESHOLD_HIGH = 0.15
EXTREME_THRESHOLD = 0.5
NEUTRAL_THRESHOLD = 0.1
ONE_DAY = timedelta(days=1)
PREVENT_DIV_BY_ZERO = 1e-9
TOKENIZER_MAX_LENGTH = 512

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}news_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}news_preproc_state.json"

LAST_ADDED_KEY = "lastAdded"
LAST_PROCESSED_KEY = "lastProcessed"

TEMP_PREFIX = "tmp/"
RAW_PREFIX = "raw/news/localDate="
SENTIMENT_PREFIX = "news/sentiment/localDate="
AGGREGATED_PREFIX = "news/aggregated/localDate="
LAGGED_PREFIX = f"news/lagged-{LAG_DAYS}/"


LOGGER = s.setup_logger()

ALLOWED_REGEX_PATTERN = re.compile(r"[A-Za-z0-9\p{P}\p{Sc}\s]+")


def get_available_memory_mb() -> int:
    memory_mb = os.getenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
    if memory_mb:
        return int(memory_mb)

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass

    return 1024


def get_batch_size() -> int:
    mem = get_available_memory_mb()
    if mem >= 3000:
        return 48
    elif mem >= 2000:
        return 32
    elif mem >= 1500:
        return 24
    elif mem >= 1000:
        return 16
    elif mem >= 500:
        return 8
    else:
        return 4


BATCH_SIZE = get_batch_size()


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
                "title": pa.string(),
                "body": pa.string(),
            }
        )

    return _raw_schema


_sentiment_schema = None


def get_sentiment_schema():
    global _sentiment_schema
    if _sentiment_schema is None:
        pa = pyarrow()
        _sentiment_schema = pa.schema(
            {
                "localDate": pa.date32(),
                "id": pa.string(),
                "sentiment_score": pa.float32(),
                "text_length": pa.int32(),
                "token_count": pa.int32(),
            }
        )

    return _sentiment_schema


_aggregated_schema = None


def get_aggregated_schema():
    global _aggregated_schema
    if _aggregated_schema is None:
        pa = pyarrow()
        _aggregated_schema = pa.schema(
            {
                "localDate": pa.date32(),
                "count_articles": pa.int32(),
                "mean_sentiment": pa.float32(),
                "median_sentiment": pa.float32(),
                "pct_positive": pa.float32(),
                "pct_negative": pa.float32(),
                "polarity_ratio": pa.float32(),
                "weighted_mean_sentiment": pa.float32(),
                "extreme_sentiment_share": pa.float32(),
                "sentiment_skewness": pa.float32(),
                "sentiment_kurtosis": pa.float32(),
                "neutral_share": pa.float32(),
                "max_sentiment": pa.float32(),
                "min_sentiment": pa.float32(),
                "article_length_variance": pa.float32(),
                "ratio_pos_neg": pa.float32(),
                "conflict_share": pa.float32(),
            }
        )
    return _aggregated_schema


_aggregated_columns = None


def get_aggregated_columns() -> list[str]:
    global _aggregated_columns
    if _aggregated_columns is None:
        _aggregated_columns = [
            name for name in get_aggregated_schema().names if name != "localDate"
        ]
    return _aggregated_columns


_lagged_schema = None


def get_lagged_schema():
    global _lagged_schema
    if _lagged_schema is None:
        pa = pyarrow()
        fields = [pa.field("localDate", pa.date32())]
        aggregated_columns = get_aggregated_columns()
        for lag in range(1, LAG_DAYS + 1):
            for col in aggregated_columns:
                fields.append(pa.field(f"{col}-{lag}", pa.float32()))

        _lagged_schema = pa.schema(fields)
    return _lagged_schema


_lagged_column_names = None


def get_lagged_column_names():
    global _lagged_column_names
    if _lagged_column_names is None:
        aggregated_columns = get_aggregated_columns()
        _lagged_column_names = [
            [f"{col}-{lag}" for col in aggregated_columns]
            for lag in range(LAG_DAYS, 0, -1)
        ]
    return _lagged_column_names


_empty_aggregated = None


def get_empty_aggregated():
    global _empty_aggregated
    if _empty_aggregated is None:
        _empty_aggregated = {
            col: 0 if col == "count_articles" else 0.0
            for col in get_aggregated_columns()
        }
    return _empty_aggregated.copy()


# ---------- MODEL (Lazy Loaded) ----------
LOCAL_FILES_ONLY = "MODEL_DIR" in os.environ

_device = None


def get_device():
    global _device
    if _device is None:
        trc = torch()
        _device = trc.device("cuda") if trc.cuda.is_available() else trc.device("cpu")
    return _device


_cpu_only = None


def is_cpu_only():
    global _cpu_only
    if _cpu_only is None:
        _cpu_only = get_device().type == "cpu"
    return _cpu_only


def init_torch():
    # lambda memory of 5308 mb equates to 4 vCPU
    vcpus = os.cpu_count() or 1
    LOGGER.debug(f"vcpu count: {vcpus}")
    trc = torch()
    trc.set_num_threads(min(vcpus, 4))
    trc.set_num_interop_threads(min(vcpus, 2))


_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR_OR_NAME, local_files_only=LOCAL_FILES_ONLY
        )
    return _tokenizer


_model = None


def get_model():
    global _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR_OR_NAME, local_files_only=LOCAL_FILES_ONLY
        )
        _model.to(get_device())
        _model.eval()

        cfg_map = _model.config.id2label
        try:
            _model.pos_idx = next(
                k for k, v in cfg_map.items() if v.lower() == "positive"
            )
            _model.neg_idx = next(
                k for k, v in cfg_map.items() if v.lower() == "negative"
            )
        except StopIteration:
            raise ValueError(f"Model has no valid positive/negative mapping: {cfg_map}")

    return _model


def clean_text(title: str, body: str) -> str:
    title_parts = ALLOWED_REGEX_PATTERN.findall(
        unicodedata.normalize("NFKC", title) if isinstance(title, str) else ""
    )
    body_parts = ALLOWED_REGEX_PATTERN.findall(
        unicodedata.normalize("NFKC", body) if isinstance(body, str) else ""
    )

    combined_parts = title_parts + body_parts
    if not combined_parts:
        return ""

    return " ".join("".join(combined_parts).split())


def fast_skew(x):
    np = numpy()
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 3:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=0)
    if s == 0:
        return 0.0
    return np.mean(((arr - m) / s) ** 3)


def fast_kurtosis(x):
    np = numpy()
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 4:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=0)
    if s == 0:
        return 0.0
    return np.mean(((arr - m) / s) ** 4) - 3


def get_sentiment(py_date: date):
    date_str = py_date.isoformat()
    raw_key = f"{RAW_PREFIX}{date_str}/data.parquet"
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"

    sentiment_schema = get_sentiment_schema()

    try:
        return s.read_parquet_s3(BUCKET, sentiment_key, sentiment_schema)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

    try:
        raw_table = s.read_parquet_s3(BUCKET, raw_key, get_raw_schema())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return None
        raise

    if raw_table is None:
        raise ValueError("raw_table is None")

    row_count = raw_table.num_rows
    if row_count == 0:
        raise ValueError(f"empty raw file for {date_str}")

    titles = raw_table["title"].to_pylist()
    bodies = raw_table["body"].to_pylist()

    np = numpy()

    cleaned_texts = [clean_text(title, body) for title, body in zip(titles, bodies)]
    text_lengths = np.fromiter(
        (len(text) for text in cleaned_texts),
        dtype=np.int32,
        count=row_count,
    )

    tokenizer = get_tokenizer()
    model = get_model()

    batch_size = BATCH_SIZE

    sentiment_scores = np.empty(row_count, dtype=np.float32)
    token_counts = np.empty(row_count, dtype=np.int32)

    trc = torch()

    with trc.inference_mode():
        for from_index in range(0, row_count, batch_size):
            to_index = min(from_index + batch_size, row_count)

            batch_texts = cleaned_texts[from_index:to_index]

            model_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=TOKENIZER_MAX_LENGTH,
                return_tensors="pt",
            )

            if is_cpu_only():
                outputs = model(**model_inputs)
                probabilites = trc.softmax(outputs.logits, dim=-1)
                batch_sentiment_scores = (
                    probabilites[:, model.pos_idx] - probabilites[:, model.neg_idx]
                ).numpy()
                batch_token_counts = (
                    (model_inputs["attention_mask"] == 1)
                    .sum(dim=1)
                    .numpy()
                    .astype(np.int32)
                )

            else:
                for k in model_inputs:
                    model_inputs[k] = model_inputs[k].to(
                        get_device(), non_blocking=True
                    )

                outputs = model(**model_inputs)
                probabilites = trc.softmax(outputs.logits, dim=-1)
                batch_sentiment_scores = (
                    (probabilites[:, model.pos_idx] - probabilites[:, model.neg_idx])
                    .cpu()
                    .numpy()
                )
                batch_token_counts = (
                    (model_inputs["attention_mask"] == 1)
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )

            sentiment_scores[from_index:to_index] = batch_sentiment_scores
            token_counts[from_index:to_index] = batch_token_counts

    pa = pyarrow()
    table = pa.Table.from_arrays(
        [
            pa.repeat(pa.scalar(py_date, type=pa.date32()), row_count),
            raw_table["id"],
            pa.array(sentiment_scores, pa.float32()),
            pa.array(text_lengths, pa.int32()),
            pa.array(token_counts, pa.int32()),
        ],
        names=sentiment_schema.names,
    )

    s.write_parquet_s3(table, BUCKET, sentiment_key, sentiment_schema)
    return table


def get_aggregated(py_date: date) -> dict[str, float]:
    aggregated_key = f"{AGGREGATED_PREFIX}{py_date.isoformat()}/data.parquet"

    aggregated_schema = get_aggregated_schema()
    aggregated_columns = get_aggregated_columns()

    try:
        aggregated_table = s.read_parquet_s3(BUCKET, aggregated_key, aggregated_schema)
        if aggregated_table is None:
            raise ValueError("aggregated_table is None")
        return {
            column_name: aggregated_table[column_name][0].as_py()
            for column_name in aggregated_columns
        }

    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

        sentiment_table = get_sentiment(py_date)
        if sentiment_table is None:
            return get_empty_aggregated()
        elif sentiment_table.num_rows == 0:
            raise ValueError(f"unexpected empty sentiment file for {py_date}")
        else:
            np = numpy()

            sentiment_scores = (
                sentiment_table["sentiment_score"]
                .to_numpy()
                .astype(np.float32, copy=False)
            )
            text_lengths = (
                sentiment_table["text_length"].to_numpy().astype(np.float32, copy=False)
            )

            absolute_scores = np.abs(sentiment_scores, dtype=np.float32)
            is_positive = sentiment_scores > 0
            is_negative = sentiment_scores < 0
            is_neutral = absolute_scores < np.float32(NEUTRAL_THRESHOLD)
            is_extreme = absolute_scores > np.float32(EXTREME_THRESHOLD)

            count_articles = sentiment_scores.size

            mean_sentiment = np.float32(sentiment_scores.mean())
            median_sentiment = np.float32(np.median(sentiment_scores))

            pct_positive = np.float32(is_positive.mean())
            pct_negative = np.float32(is_negative.mean())
            polarity_ratio = np.float32(pct_positive - pct_negative)

            weighted_mean_sentiment = np.float32(
                (sentiment_scores * text_lengths).sum(dtype=np.float32)
                / (text_lengths.sum(dtype=np.float32) + PREVENT_DIV_BY_ZERO)
            )

            extreme_sentiment_share = np.float32(is_extreme.mean())
            neutral_share = np.float32(is_neutral.mean())

            sentiment_skewness = np.float32(fast_skew(sentiment_scores))
            sentiment_kurtosis = np.float32(fast_kurtosis(sentiment_scores))

            max_sentiment = np.float32(sentiment_scores.max())
            min_sentiment = np.float32(sentiment_scores.min())

            article_length_variance = np.float32(text_lengths.var(dtype=np.float32))

            ratio_pos_neg = np.float32(
                pct_positive / (pct_negative + PREVENT_DIV_BY_ZERO)
            )

            conflict_mask = (absolute_scores >= np.float32(CONFLICT_THRESHOLD_LOW)) & (
                absolute_scores <= np.float32(CONFLICT_THRESHOLD_HIGH)
            )
            conflict_share = np.float32(conflict_mask.mean())

            pa = pyarrow()
            aggregated_table = pa.Table.from_arrays(
                [
                    pa.repeat(pa.scalar(py_date, type=pa.date32()), 1),
                    pa.array([count_articles], pa.int32()),
                    pa.array([mean_sentiment], pa.float32()),
                    pa.array([median_sentiment], pa.float32()),
                    pa.array([pct_positive], pa.float32()),
                    pa.array([pct_negative], pa.float32()),
                    pa.array([polarity_ratio], pa.float32()),
                    pa.array([weighted_mean_sentiment], pa.float32()),
                    pa.array([extreme_sentiment_share], pa.float32()),
                    pa.array([sentiment_skewness], pa.float32()),
                    pa.array([sentiment_kurtosis], pa.float32()),
                    pa.array([neutral_share], pa.float32()),
                    pa.array([max_sentiment], pa.float32()),
                    pa.array([min_sentiment], pa.float32()),
                    pa.array([article_length_variance], pa.float32()),
                    pa.array([ratio_pos_neg], pa.float32()),
                    pa.array([conflict_share], pa.float32()),
                ],
                schema=aggregated_schema,
            )

            s.write_parquet_s3(
                aggregated_table, BUCKET, aggregated_key, aggregated_schema
            )

            aggregated_values = [
                count_articles,
                mean_sentiment,
                median_sentiment,
                pct_positive,
                pct_negative,
                polarity_ratio,
                weighted_mean_sentiment,
                extreme_sentiment_share,
                sentiment_skewness,
                sentiment_kurtosis,
                neutral_share,
                max_sentiment,
                min_sentiment,
                article_length_variance,
                ratio_pos_neg,
                conflict_share,
            ]

            return dict(zip(aggregated_columns, aggregated_values))


def append_lagged_row(
    lagged_columns: dict,
    py_date: date,
    rolling_window: deque,
    aggregated_columns: list,
    lagged_column_names: list[list[str]],
):
    lagged_columns["localDate"].append(py_date)

    for aggregated_values, lagged_cols in zip(rolling_window, lagged_column_names):
        for col, lagged_col in zip(aggregated_columns, lagged_cols):
            lagged_columns[lagged_col].append(aggregated_values[col])


def generate_lagged_features(context=None):
    try:
        init_torch()

        last_added = s.read_json_date_s3(BUCKET, COLLECTOR_STATE_KEY, LAST_ADDED_KEY)
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        default_last_processed = START_DATE - ONE_DAY
        last_processed = s.read_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, default_last_processed
        )
        if last_processed > last_added:
            return {"statusCode": 200, "body": "all data already processed"}

        current_date = last_processed + ONE_DAY

        rolling_window = deque(maxlen=LAG_DAYS)
        for i in range(LAG_DAYS, 1, -1):
            lag_date = current_date - timedelta(days=i)
            aggregated_dict = get_aggregated(lag_date)
            rolling_window.append(aggregated_dict)

        lagged_schema = get_lagged_schema()
        lagged_columns = {field.name: [] for field in lagged_schema}

        aggregated_columns = get_aggregated_columns()
        lagged_column_names = get_lagged_column_names()

        days_processed = 0

        while s.continue_execution(context):
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            prev_date = current_date - ONE_DAY
            if prev_date > last_added:
                break

            aggregated_dict = get_aggregated(prev_date)
            rolling_window.append(aggregated_dict)
            append_lagged_row(
                lagged_columns,
                current_date,
                rolling_window,
                aggregated_columns,
                lagged_column_names,
            )
            days_processed += 1
            LOGGER.info(f"Computed lagged features for {current_date}")
            current_date += ONE_DAY

        if days_processed == 0:
            return {
                "statusCode": 200,
                "body": "no lagged features computed due to lambda timeout",
            }

        pa = pyarrow()
        arrays = [
            pa.array(lagged_columns[field.name], type=field.type)
            for field in lagged_schema
        ]
        lagged_table = pa.Table.from_arrays(arrays, schema=lagged_schema)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        file_name = f"{timestamp}.parquet"
        lagged_key = f"{LAGGED_PREFIX}{file_name}"
        s.write_parquet_s3(lagged_table, BUCKET, lagged_key, lagged_schema)

        new_last_processed = lagged_table["localDate"].to_pylist()[-1]
        s.write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, new_last_processed
        )

        return {
            "statusCode": 200,
            "body": f"finished preprocessing up to and including {new_last_processed:%Y-%m-%d}",
        }
    except Exception:
        LOGGER.exception("Error in generate_lagged_features")
        return {"statusCode": 500, "body": "error"}


def lambda_handler(event=None, context=None):
    return generate_lagged_features(context)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
