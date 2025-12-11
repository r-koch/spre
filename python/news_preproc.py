# ---------- STANDARD LIBRARY ----------
import multiprocessing
import os
import unicodedata
from collections import deque
from datetime import date, datetime, timedelta, timezone

# ---------- THIRD-PARTY LIBRARIES ----------
import numpy as np
import pyarrow as pa
import shared as s
import torch
from botocore.exceptions import ClientError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
REGION = os.getenv("AWS_REGION", "eu-west-1")
LAG_DAYS = int(os.getenv("LAG_DAYS", "5"))
START_DATE = date.fromisoformat(os.getenv("START_DATE", "1999-11-01"))
MIN_REMAINING_MS = int(os.getenv("MIN_REMAINING_MS", "240000"))
MODEL_NAME_OR_DIR = os.getenv(
    "MODEL_DIR", "ProsusAI/finbert"
)  # locally use model name, in docker use env
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.getenv("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.getenv("RETRY_MAX_DELAY_S", "2.0"))
DEBUG_MAX_DAYS_PER_INVOCATION = int(os.getenv("DEBUG_MAX_DAYS_PER_INVOCATION", "2"))

CONFLICT_THRESHOLD_LOW = np.float32(0.05)
CONFLICT_THRESHOLD_HIGH = np.float32(0.15)
EXTREME_THRESHOLD = np.float32(0.5)
NEUTRAL_THRESHOLD = np.float32(0.1)
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

logger = s.setup_logger()


# ---------- SCHEMAS ----------

RAW_SCHEMA = pa.schema(
    {
        "localDate": pa.date32(),
        "id": pa.string(),
        "title": pa.string(),
        "body": pa.string(),
    }
)

SENTIMENT_SCHEMA = pa.schema(
    {
        "localDate": pa.date32(),
        "id": pa.string(),
        "sentiment_score": pa.float32(),
        "text_length": pa.int32(),
        "token_count": pa.int32(),
    }
)

AGGREGATED_SCHEMA = pa.schema(
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

AGGREGATED_COLUMNS = [name for name in AGGREGATED_SCHEMA.names if name != "localDate"]


def build_lagged_schema():
    fields = [pa.field("localDate", pa.date32())]
    for lag in range(1, LAG_DAYS + 1):
        for col in AGGREGATED_COLUMNS:
            fields.append(pa.field(f"{col}-{lag}", pa.float32()))

    return pa.schema(fields)


LAGGED_SCHEMA = build_lagged_schema()

EMPTY_AGG_VALUES = {
    "count_articles": pa.array([0], pa.int32()),
    "mean_sentiment": pa.array([0.0], pa.float32()),
    "median_sentiment": pa.array([0.0], pa.float32()),
    "pct_positive": pa.array([0.0], pa.float32()),
    "pct_negative": pa.array([0.0], pa.float32()),
    "polarity_ratio": pa.array([0.0], pa.float32()),
    "weighted_mean_sentiment": pa.array([0.0], pa.float32()),
    "extreme_sentiment_share": pa.array([0.0], pa.float32()),
    "sentiment_skewness": pa.array([0.0], pa.float32()),
    "sentiment_kurtosis": pa.array([0.0], pa.float32()),
    "neutral_share": pa.array([0.0], pa.float32()),
    "max_sentiment": pa.array([0.0], pa.float32()),
    "min_sentiment": pa.array([0.0], pa.float32()),
    "article_length_variance": pa.array([0.0], pa.float32()),
    "ratio_pos_neg": pa.array([0.0], pa.float32()),
    "conflict_share": pa.array([0.0], pa.float32()),
}


# ---------- MODEL (Lazy Loaded) ----------
tokenizer = None
model = None


def is_docker():
    return "MODEL_DIR" in os.environ


if is_docker:
    vcpus = multiprocessing.cpu_count()
    torch.set_num_threads(min(vcpus, 4))
    torch.set_num_interop_threads(min(vcpus, 2))


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_OR_DIR, local_files_only=is_docker()
        )
    return tokenizer


def get_model():
    global model
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_DIR, local_files_only=is_docker()
        )
        model.to(get_device())
        model.eval()

        cfg_map = model.config.id2label
        try:
            model.pos_idx = next(
                k for k, v in cfg_map.items() if v.lower() == "positive"
            )
            model.neg_idx = next(
                k for k, v in cfg_map.items() if v.lower() == "negative"
            )
        except StopIteration:
            raise ValueError(f"Model has no valid positive/negative mapping: {cfg_map}")

    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    clean_chars = []
    for ch in text:
        cat = unicodedata.category(ch)

        # Keep: letters, digits, punctuation, currency symbols, whitespace
        if (
            cat.startswith("L")  # Letters
            or cat.startswith("N")  # Numbers
            or cat.startswith("P")  # Punctuation
            or cat == "Sc"  # Currency symbol
            or ch.isspace()
        ):
            clean_chars.append(ch)

    cleaned_text = "".join(clean_chars)
    # Collapse repeated whitespace
    return " ".join(cleaned_text.split())


@torch.no_grad()
def analyze_sentiment(texts):
    tokenizer = get_tokenizer()
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=TOKENIZER_MAX_LENGTH,
        return_tensors="pt",
    )
    token_counts = (
        (encodings["attention_mask"] == 1).sum(dim=1).cpu().numpy().astype(np.int32)
    )
    device = get_device()
    encodings = {k: v.to(device) for k, v in encodings.items()}
    model = get_model()
    outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()
    scores = (probs[:, model.pos_idx] - probs[:, model.neg_idx]).tolist()
    return scores, token_counts


def detect_available_memory_mb() -> int:
    mem_env = os.getenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
    if mem_env:
        return int(mem_env)

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb // 1024
    except Exception:
        pass

    return 1024


def choose_batch_size():
    mem = detect_available_memory_mb()
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


def fast_skew(x):
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


def get_sentiment(pa_date: pa.Date32Scalar) -> pa.Table:
    date_str = pa_date.as_py().isoformat()
    raw_key = f"{RAW_PREFIX}{date_str}/data.parquet"
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"

    try:
        return s.read_parquet_s3(BUCKET, sentiment_key, SENTIMENT_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

    try:
        raw_table = s.read_parquet_s3(BUCKET, raw_key, RAW_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            # IMPORTANT: If raw parquet is missing or empty, DO NOT create sentiment/aggregated files.
            # This preserves the signal that raw data itself is missing and prevents silent imputation.
            return None
        raise

    row_count = raw_table.num_rows
    if row_count == 0:
        return None

    titles = raw_table["title"]
    bodies = raw_table["body"]

    cleaned_texts = []
    for i in range(row_count):
        title = titles[i].as_py() or ""
        body = bodies[i].as_py() or ""
        text = f"{title} {body}"
        cleaned_text = clean_text(text)
        cleaned_texts.append(cleaned_text)

    batch_size = choose_batch_size()
    sentiment_scores = []
    token_counts = []

    for i in range(0, row_count, batch_size):
        batch_texts = cleaned_texts[i : i + batch_size]
        batch_scores, batch_token_counts = analyze_sentiment(batch_texts)

        if len(batch_scores) != len(batch_texts):
            raise ValueError(
                f"Batch sentiment mismatch for {pa_date}: expected {len(batch_texts)}, got {len(batch_scores)}"
            )

        if len(batch_token_counts) != len(batch_texts):
            raise ValueError(
                f"Batch sentiment mismatch for {pa_date}: expected {len(batch_texts)}, got {len(batch_token_counts)}"
            )

        sentiment_scores.extend(batch_scores)
        token_counts.extend(batch_token_counts)

    if len(sentiment_scores) != row_count:
        raise ValueError(
            f"Total sentiment mismatch for {pa_date}: expected {row_count}, got {len(sentiment_scores)}"
        )

    local_date_arr = pa.repeat(pa_date, row_count)
    id_arr = raw_table["id"]
    sentiment_scores_arr = pa.array(
        np.asarray(sentiment_scores, dtype=np.float32), pa.float32()
    )
    text_length_arr = pa.array(
        np.fromiter(
            (len(s) for s in cleaned_texts), dtype=np.int32, count=len(cleaned_texts)
        ),
        pa.int32(),
    )
    token_count_arr = pa.array(np.asarray(token_counts, dtype=np.int32), pa.int32())

    table = pa.Table.from_arrays(
        [
            local_date_arr,
            id_arr,
            sentiment_scores_arr,
            text_length_arr,
            token_count_arr,
        ],
        names=SENTIMENT_SCHEMA.names,
    )

    s.write_parquet_s3(table, BUCKET, sentiment_key, SENTIMENT_SCHEMA)
    return table


def get_aggregated(py_date: date) -> pa.Table:
    aggregated_key = f"{AGGREGATED_PREFIX}{py_date.isoformat()}/data.parquet"

    try:
        agg_table = s.read_parquet_s3(BUCKET, aggregated_key, AGGREGATED_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

        pa_date: pa.Date32Scalar = pa.scalar(py_date, type=pa.date32())
        sentiment_table = get_sentiment(pa_date)
        if sentiment_table is None or sentiment_table.num_rows == 0:
            agg_table = pa.Table.from_arrays(
                [pa.repeat(pa_date, 1)]
                + [EMPTY_AGG_VALUES[name] for name in AGGREGATED_COLUMNS],
                schema=AGGREGATED_SCHEMA,
            )

        else:
            scores = sentiment_table["sentiment_score"].to_numpy()
            scores_abs = np.abs(scores)
            is_positive = scores > 0
            is_negative = scores < 0
            is_neutral = scores_abs < NEUTRAL_THRESHOLD
            is_extreme = scores_abs > EXTREME_THRESHOLD
            text_lengths = sentiment_table["text_length"].to_numpy()

            count_articles = int(scores.size)

            mean_sentiment = np.float32(scores.mean())
            median_sentiment = np.float32(np.median(scores))

            pct_positive = np.float32(is_positive.mean())
            pct_negative = np.float32(is_negative.mean())
            polarity_ratio = np.float32(pct_positive - pct_negative)

            weighted_mean_sentiment = np.float32(
                (scores * text_lengths).sum()
                / (text_lengths.sum() + PREVENT_DIV_BY_ZERO)
            )

            extreme_sentiment_share = np.float32(is_extreme.mean())
            neutral_share = np.float32(is_neutral.mean())

            sentiment_skewness = np.float32(fast_skew(scores))
            sentiment_kurtosis = np.float32(fast_kurtosis(scores))

            max_sentiment = np.float32(scores.max())
            min_sentiment = np.float32(scores.min())

            article_length_variance = np.float32(text_lengths.var())

            ratio_pos_neg = np.float32(
                pct_positive / (is_negative.mean() + PREVENT_DIV_BY_ZERO)
            )

            conflict_mask = (scores_abs >= CONFLICT_THRESHOLD_LOW) & (
                scores_abs <= CONFLICT_THRESHOLD_HIGH
            )
            conflict_share = np.float32(conflict_mask.mean())

            agg_table = pa.Table.from_arrays(
                [
                    pa.repeat(pa_date, 1),
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
                schema=AGGREGATED_SCHEMA,
            )

            s.write_parquet_s3(agg_table, BUCKET, aggregated_key, AGGREGATED_SCHEMA)

    if agg_table is None:
        raise ValueError("Unexpected None result in get_aggregated()")

    if agg_table.num_rows != 1:
        raise ValueError(
            f"Aggregated result for {py_date} has {agg_table.num_rows} rows, expected 1"
        )

    return agg_table


def continue_execution(context=None):
    if (
        context is not None
        and int(context.get_remaining_time_in_millis()) < MIN_REMAINING_MS
    ):
        logger.warning(f"Stopping execution due to timeout limit.")
        return False

    return True


def append_lagged_row(lagged_columns: dict, py_date: date, window: deque):
    lagged_columns["localDate"].append(py_date)
    lag = LAG_DAYS

    for agg_table in window:
        for col in AGGREGATED_COLUMNS:
            value = agg_table[col][0].as_py()
            lagged_columns[f"{col}-{lag}"].append(value)
        lag -= 1


def generate_lagged_features(context=None):
    try:
        last_added = s.read_json_date_s3(BUCKET, COLLECTOR_STATE_KEY, LAST_ADDED_KEY)
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        default_last_processed = START_DATE - ONE_DAY
        last_processed = s.read_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, default_last_processed
        )
        if last_processed > last_added:
            return {"statusCode": 200, "body": "all data already processed"}

        current = last_processed + ONE_DAY

        rolling_window = deque(maxlen=LAG_DAYS)
        for i in range(LAG_DAYS, 1, -1):
            lag_date = current - timedelta(days=i)
            agg_table = get_aggregated(lag_date)
            rolling_window.append(agg_table)

        lagged_columns = {field.name: [] for field in LAGGED_SCHEMA}

        days_processed = 0

        while continue_execution(context):
            if days_processed == DEBUG_MAX_DAYS_PER_INVOCATION:
                break

            prev = current - ONE_DAY
            if prev > last_added:
                break

            rolling_window.append(get_aggregated(prev))
            append_lagged_row(lagged_columns, current, rolling_window)
            days_processed += 1
            logger.info(f"Computed lagged features for {current}")
            current += ONE_DAY

        if days_processed == 0:
            return {
                "statusCode": 200,
                "body": "no lagged features computed due to lambda timeout",
            }

        arrays = [
            pa.array(lagged_columns[field.name], type=field.type)
            for field in LAGGED_SCHEMA
        ]
        lagged_table = pa.Table.from_arrays(arrays, schema=LAGGED_SCHEMA)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        file_name = f"{timestamp}.parquet"
        lagged_key = f"{LAGGED_PREFIX}{file_name}"
        s.write_parquet_s3(lagged_table, BUCKET, lagged_key, LAGGED_SCHEMA)

        new_last_processed = lagged_table["localDate"].to_pylist()[-1]
        s.write_json_date_s3(
            BUCKET, PREPROC_STATE_KEY, LAST_PROCESSED_KEY, new_last_processed
        )

        return {
            "statusCode": 200,
            "body": f"finished preprocessing up to and including {new_last_processed:%Y-%m-%d}",
        }
    except Exception:
        logger.exception("Error in generate_lagged_features")
        return {"statusCode": 500, "body": "error"}


def lambda_handler(event=None, context=None):
    return generate_lagged_features(context)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
