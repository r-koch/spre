from collections import deque
import json
import uuid
import logging
import os
from typing import cast
import unicodedata
import time
import random
from botocore.exceptions import ClientError
from io import BytesIO
from datetime import date, datetime, timedelta, timezone
import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- CONFIG ----------
BUCKET = os.environ.get("BUCKET", "dev-rkoch-spre")
REGION = os.environ.get("AWS_REGION", "eu-west-1")
LAG_DAYS = int(os.environ.get("LAG_DAYS", "5"))
START_DATE = os.environ.get("START_DATE", "1999-10-27")
MIN_REMAINING_MS = int(os.environ.get("MIN_REMAINING_MS", "60000"))
MODEL_NAME_OR_DIR = os.environ.get("MODEL_DIR", "ProsusAI/finbert")  # locally use model name, in docker use env
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "3"))
RETRY_DELAY_S = float(os.environ.get("RETRY_DELAY_S", "0.25"))
RETRY_MAX_DELAY_S = float(os.environ.get("RETRY_MAX_DELAY_S", "2.0"))
TOKENIZER_MAX_LENGTH = int(os.environ.get("TOKENIZER_MAX_LENGTH", "512"))

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}news_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}news_preproc_state.json"

LAST_PROCESSED_KEY = "lastProcessed"
LAST_ADDED_KEY = "lastAdded"

RAW_PREFIX = "raw/news/localDate="
SENTIMENT_PREFIX = "news/sentiment/localDate="
AGGREGATED_PREFIX = "news/aggregated/localDate="
LAGGED_PREFIX = f"news/lagged-{LAG_DAYS}/"
TMP_PREFIX = "tmp/"

ONE_DAY = timedelta(days=1)

s3 = boto3.client("s3", region_name=REGION)

if "MODEL_DIR" in os.environ:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

# ---------- SCHEMAS ----------

RAW_SCHEMA = pa.schema({
    "localDate": pa.date32(),
    "id": pa.string(),
    "title": pa.string(),
    "body": pa.string(),
})

SENTIMENT_SCHEMA = pa.schema({
    "localDate": pa.date32(),
    "id": pa.string(),
    "title": pa.string(),
    "body": pa.string(),
    "title_clean": pa.string(),
    "body_clean": pa.string(),
    "text_clean": pa.string(),
    "sentiment_score": pa.float32(),
    "is_positive": pa.int8(),
    "is_negative": pa.int8(),
    "is_neutral": pa.int8(),
    "is_extreme": pa.int8(),
    "text_length": pa.int32(),
})

AGGREGATED_SCHEMA = pa.schema({
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
})

AGGREGATED_COLUMNS = [name for name in AGGREGATED_SCHEMA.names if name != "localDate"]

def build_lagged_schema():
    fields = [pa.field("localDate", pa.string())]

    for lag in range(1, LAG_DAYS + 1):
        for col in AGGREGATED_COLUMNS:
            fields.append(pa.field(f"{col}-{lag}", pa.float32()))

    return pa.schema(fields)

LAGGED_SCHEMA = build_lagged_schema()

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
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s")
        logger.setLevel(logging.DEBUG)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()

# ---------- MODEL (Lazy Loaded) ----------
tokenizer = None
model = None

def is_docker():
    return "MODEL_DIR" in os.environ


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_DIR, local_files_only=is_docker())
    return tokenizer


def get_model():
    global model
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_DIR, local_files_only=is_docker())
        model.to(get_device())
        model.eval()

        expected = {0: "negative", 1: "neutral", 2: "positive"}
        cfg_map = model.config.id2label
        if cfg_map != expected:
            raise ValueError(f"Unexpected FinBERT id2label mapping: {cfg_map}")
        
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- HELPERS ----------
def retry_s3(operation, retry_count=RETRY_COUNT, retry_delay=RETRY_DELAY_S, retry_max_delay=RETRY_MAX_DELAY_S):
    for attempt in range(retry_count):
        try:
            return operation()
        except ClientError as e:
            if attempt < retry_count - 1:
                code = e.response.get("Error", {}).get("Code", "")
                retryable = code.startswith("5") or "Throttl" in code or code in (
                    "SlowDown",
                    "RequestTimeout",
                    "RequestTimeTooSkewed",
                    "InternalError",
                )

                if retryable:
                    max_sleep = min(retry_delay * (2 ** attempt), retry_max_delay)
                    sleep = random.uniform(0, max_sleep)
                    time.sleep(sleep)
                    continue

            raise  # No retry


def read_parquet_s3(bucket, key, schema):
    def op():
        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = BytesIO(obj["Body"].read())
        try:
            table = pq.read_table(buf)
        except Exception as e:
            raise ValueError(f"Corrupted parquet at {key}: {e}")

        if schema != table.schema:
            raise ValueError(f"Schema mismatch for {key}. Expected {schema}, got {table.schema}")
        
        dtype_map = {}
        for field in schema:
            name = field.name
            pa_type = field.type

            if pa.types.is_int8(pa_type):
                dtype_map[name] = pd.Int8Dtype()
            elif pa.types.is_int16(pa_type):
                dtype_map[name] = pd.Int16Dtype()
            elif pa.types.is_int32(pa_type):
                dtype_map[name] = pd.Int32Dtype()
            elif pa.types.is_int64(pa_type):
                dtype_map[name] = pd.Int64Dtype()
            elif pa.types.is_floating(pa_type):
                # force float32 for your pipeline
                dtype_map[name] = np.float32
            elif pa.types.is_string(pa_type):
                dtype_map[name] = pd.StringDtype()
            elif pa.types.is_date(pa_type):
                dtype_map[name] = "datetime64[ns]"
            else:
                # Fallback: let pandas infer
                dtype_map[name] = None

        return table.to_pandas(
            types_mapper=lambda t: (
                dtype_map.get(t._name, None)
            )
        )

    return retry_s3(op)


def write_parquet_s3(df, bucket, key, schema):
    def op():
        stable_df = df.reset_index(drop=True)

        if schema is None:
            if "localDate" in stable_df.columns:
                stable_df["localDate"] = stable_df["localDate"].astype(str)

            for col in stable_df.columns:
                if col != "localDate":
                    stable_df[col] = pd.to_numeric(stable_df[col], errors="coerce").astype(float)

            stable_df = stable_df.reindex(sorted(stable_df.columns), axis=1)
        else:
            expected_cols = set(schema.names)
            incoming_cols = set(stable_df.columns)
            if incoming_cols != expected_cols:
                missing = expected_cols - incoming_cols
                extra = incoming_cols - expected_cols
                raise ValueError(f"Schema mismatch. Missing: {missing} Extra: {extra}")

            stable_df = stable_df[schema.names]

        buf = BytesIO()
        table = pa.Table.from_pandas(stable_df, schema=schema, preserve_index=False)
        pq.write_table(table, buf, compression="snappy")

        temp_key = f"{TMP_PREFIX}{uuid.uuid4().hex}"
        s3.put_object(Bucket=bucket, Key=temp_key, Body=buf.getvalue())
        s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": temp_key}, Key=key)
        s3.delete_object(Bucket=bucket, Key=temp_key)

    retry_s3(op)


def read_collector_state():
    def op():
        obj = s3.get_object(Bucket=BUCKET, Key=COLLECTOR_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_ADDED_KEY], "%Y-%m-%d").date()
    
    try:
        return retry_s3(op)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            return None
        raise


def read_preproc_state() -> date:
    def op():
        obj = s3.get_object(Bucket=BUCKET, Key=PREPROC_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_PROCESSED_KEY], "%Y-%m-%d").date()

    try:
        return cast(date, retry_s3(op))
    except ClientError:
        return datetime.strptime(START_DATE, "%Y-%m-%d").date() - ONE_DAY


def write_preproc_state(date_obj):
    def op():
        payload = {LAST_PROCESSED_KEY: date_obj.strftime("%Y-%m-%d")}
        s3.put_object(
            Bucket=BUCKET,
            Key=PREPROC_STATE_KEY,
            Body=json.dumps(payload).encode("utf-8")
        )

    retry_s3(op)


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
            cat.startswith("L") or   # Letters
            cat.startswith("N") or   # Numbers
            cat.startswith("P") or   # Punctuation
            cat == "Sc" or           # Currency symbol
            ch.isspace()
        ):
            clean_chars.append(ch)

    cleaned_text = "".join(clean_chars)
    # Collapse repeated whitespace
    return " ".join(cleaned_text.split())


@torch.no_grad()
def analyze_sentiment(batch):
    tokenizer = get_tokenizer()
    encodings  = tokenizer(batch,  padding=True, truncation=True, max_length=TOKENIZER_MAX_LENGTH, return_tensors="pt")
    device = get_device()
    encodings = {k: v.to(device) for k, v in encodings.items()}
    model = get_model()
    outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()

    labels = probs.argmax(dim=-1).tolist()
    scores = []
    for i in enumerate(labels):
        score = probs[i][2].item() - probs[i][0].item()
        scores.append(score)

    return scores


def detect_available_memory_mb() -> int:
    mem_env = os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
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


def get_sentiment(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    raw_key = f"{RAW_PREFIX}{date_str}/data.parquet"
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, sentiment_key, SENTIMENT_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

    try:
        df = read_parquet_s3(BUCKET, raw_key, RAW_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") == "NoSuchKey":
            # IMPORTANT: If raw parquet is missing or empty, DO NOT create sentiment/aggregated files.
            # This preserves the signal that raw data itself is missing and prevents silent imputation.
            return None
        raise

    if df is None or df.empty:
        return None

    df["title_clean"] = df["title"].fillna("").map(clean_text)
    df["body_clean"] = df["body"].fillna("").map(clean_text)
    df["text_clean"] = df["title_clean"] + " " + df["body_clean"]

    batch_size = choose_batch_size()

    sentiment_scores = []

    for i in range(0, len(df), batch_size):
        batch  = df["text_clean"].iloc[i : i + batch_size].tolist()
        batch_scores  = analyze_sentiment(batch)
        sentiment_scores.extend(batch_scores)

    df["sentiment_score"] = sentiment_scores
    df["is_positive"] = df["sentiment_score"] > 0
    df["is_negative"] = df["sentiment_score"] < 0
    df["is_neutral"]  = df["sentiment_score"].abs() < 0.1
    df["is_extreme"]  = df["sentiment_score"].abs() > 0.5
    df["text_length"] = df["text_clean"].str.len().fillna(0).astype(float)

    write_parquet_s3(df, BUCKET, sentiment_key, SENTIMENT_SCHEMA)
    return df


def get_aggregated(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    aggregated_key = f"{AGGREGATED_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, aggregated_key, AGGREGATED_SCHEMA)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") != "NoSuchKey":
            raise

    df = get_sentiment(local_date)
    if df is None:
        return pd.DataFrame({
            "localDate": [date_str],
            "count_articles": [0],
            "mean_sentiment": [0.0],
            "median_sentiment": [0.0],
            "pct_positive": [0.0],
            "pct_negative": [0.0],
            "polarity_ratio": [0.0],
            "weighted_mean_sentiment": [0.0],
            "extreme_sentiment_share": [0.0],
            "sentiment_skewness": [0.0],
            "sentiment_kurtosis": [0.0],
            "neutral_share": [0.0],
            "max_sentiment": [0.0],
            "min_sentiment": [0.0],
            "article_length_variance": [0.0],
            "ratio_pos_neg": [0.0],
            "conflict_share": [0.0],
        })

    result = pd.DataFrame({
        "localDate": [date_str],
        "count_articles": [len(df)],
        "mean_sentiment": [df["sentiment_score"].mean()],
        "median_sentiment": [df["sentiment_score"].median()],
        "pct_positive": [(df["is_positive"].mean())],
        "pct_negative": [(df["is_negative"].mean())],
        "polarity_ratio": [(df["is_positive"].mean() - df["is_negative"].mean())],
        "weighted_mean_sentiment": [
            (df["sentiment_score"] * df["text_length"]).sum() /
            (df["text_length"].sum() + 1e-9)
        ],
        "extreme_sentiment_share": [df["is_extreme"].mean()],
        "sentiment_skewness": [fast_skew(df["sentiment_score"].values)],
        "sentiment_kurtosis": [fast_kurtosis(df["sentiment_score"].values)],
        "neutral_share": [df["is_neutral"].mean()],
        "max_sentiment": [df["sentiment_score"].max()],
        "min_sentiment": [df["sentiment_score"].min()],
        "article_length_variance": [df["text_length"].var(ddof=0)],
        "ratio_pos_neg": [
            (df["is_positive"].mean()) /
            (df["is_negative"].mean() + 1e-9)
        ],
        "conflict_share": [
            (df["sentiment_score"].abs().between(0.05, 0.15)).mean()
        ],
    })

    write_parquet_s3(result, BUCKET, aggregated_key, AGGREGATED_SCHEMA)
    return result


def continue_execution(context=None):
    if context is not None and int(context.get_remaining_time_in_millis()) < MIN_REMAINING_MS:
        logger.warning(f"Stopping execution due to timeout limit.")
        return False
    
    return True


def build_row(current, window):
    row = {"localDate": current.strftime("%Y-%m-%d")}
    for i, agg in enumerate(window):
        lag = LAG_DAYS - i
        for col in AGGREGATED_COLUMNS:
            row[f"{col}-{lag}"] = float(agg[col].iloc[0])
    return row


def generate_lagged_features(context=None):
    try:    
        last_added = read_collector_state()
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        last_done = read_preproc_state()
        if last_done > last_added:
            return {"statusCode": 200, "body": "all data already processed"}

        current = last_done + ONE_DAY

        rolling_window = deque(maxlen=LAG_DAYS)
        for i in range(LAG_DAYS, 1, -1):
            lag_date = current - timedelta(days=i)
            agg = get_aggregated(lag_date)
            rolling_window.append(agg)

        lagged_rows = []

        while continue_execution(context):
            prev = current - ONE_DAY
            if prev > last_added:
                break

            rolling_window.append(get_aggregated(prev))

            row = build_row(current, rolling_window)
            lagged_rows.append(row)
            last_done = current
            logger.info(f"Computed lagged features for {current}")
            current += ONE_DAY

        # If nothing was computed → exit cleanly
        if not lagged_rows:
            return {"statusCode": 500, "body": "no lagged features computed. stopped early? this should not happen!"}

        # Write one new parquet chunk for this invocation
        lagged_df = pd.DataFrame(lagged_rows)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        file_name = f"{timestamp}.parquet"
        lagged_key = f"{LAGGED_PREFIX}{file_name}"
        write_parquet_s3(lagged_df, BUCKET, lagged_key, LAGGED_SCHEMA)

        # Update checkpoint atomically
        write_preproc_state(last_done)

        return {"statusCode": 200, "body": f"finished preprocessing up to and including {last_done:%Y-%m-%d}"}
    except Exception:
        logger.exception("Error in generate_lagged_features")
        return {"statusCode": 500, "body": "error"}


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
