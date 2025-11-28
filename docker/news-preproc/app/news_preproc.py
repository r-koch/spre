from collections import deque
import json
import logging
import os
import re
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
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "0.25"))

LAST_PROCESSED_KEY = "lastProcessed"
LAST_ADDED_KEY = "lastAdded"

RAW_PREFIX = "raw/news/localDate="
SENTIMENT_PREFIX = "news/sentiment/localDate="
AGGREGATED_PREFIX = "news/aggregated/localDate="

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}news_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}news_preproc_state.json"
LAGGED_OUTPUT_PREFIX = f"news/lagged-{LAG_DAYS}/"

s3 = boto3.client("s3", region_name=REGION)

if "MODEL_DIR" in os.environ:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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
        model.eval()
    return model


# ---------- HELPERS ----------
def retry_s3(operation, retry_count=RETRY_COUNT, retry_delay=RETRY_DELAY):
    for attempt in range(retry_count):
        try:
            return operation()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("Throttling", "ThrottlingException", "TooManyRequestsException",
                        "SlowDown", "RequestTimeout", "InternalError", "503", "500"):
                if attempt == retry_count - 1:
                    raise

                # Exponential backoff + jitter
                sleep = retry_delay * (2 ** attempt) * (0.5 + random.random())
                time.sleep(sleep)
            else:
                raise  # Non-retryable


def read_parquet_s3(bucket, key):
    def op():
        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = BytesIO(obj["Body"].read())
        return pq.read_table(buf).to_pandas()

    return retry_s3(op)


def write_parquet_s3(df, bucket, key):
    def op():
        stable_df = df.reset_index(drop=True)

        if "localDate" in stable_df.columns:
            stable_df["localDate"] = stable_df["localDate"].astype(str)

        for col in stable_df.columns:
            if col != "localDate":
                stable_df[col] = pd.to_numeric(stable_df[col], errors="coerce").astype(float)

        buf = BytesIO()
        table = pa.Table.from_pandas(stable_df)
        pq.write_table(table, buf, compression="snappy")
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    retry_s3(op)


def read_collector_state():
    def op():
        obj = s3.get_object(Bucket=BUCKET, Key=COLLECTOR_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_ADDED_KEY], "%Y-%m-%d").date()
    
    try:
        return retry_s3(op)
    except ClientError:
        raise ValueError(f"{COLLECTOR_STATE_KEY} is missing")


def read_preproc_state():
    def op():
        obj = s3.get_object(Bucket=BUCKET, Key=PREPROC_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_PROCESSED_KEY], "%Y-%m-%d").date()

    try:
        return retry_s3(op)
    except ClientError:
        return datetime.strptime(START_DATE, "%Y-%m-%d").date() - timedelta(days=1)


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

    cleaned = []
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
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    # Collapse repeated whitespace
    out = " ".join("".join(cleaned).split())
    return out


@torch.no_grad()
def analyze_sentiment(encodings):
    model = get_model()
    outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = probs.argmax(dim=-1).tolist()
    scores = []
    for i, label in enumerate(labels):
        if label == 0:
            score = -probs[i][0].item()
        elif label == 2:
            score = probs[i][2].item()
        else:
            score = 0.0
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

    if mem >= 2500:
        return 64
    elif mem >= 1500:
        return 48
    elif mem >= 1000:
        return 32
    elif mem >= 500:
        return 16
    else:
        return 8


def fast_skew(x):
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return 0.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 0.0
    return np.sum(((arr - m) / s) ** 3) * (n / ((n - 1) * (n - 2)))


def fast_kurtosis(x):
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n < 4:
        return 0.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 0.0
    return np.sum(((arr - m) / s) ** 4) * (n / ((n - 1) * (n - 2))) - 3


def get_sentiment(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    raw_key = f"{RAW_PREFIX}{date_str}/data.parquet"
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, sentiment_key)
    except ClientError as e:
        if e.response['Error']['Code'] != "NoSuchKey":
            raise

    try:
        df = read_parquet_s3(BUCKET, raw_key)
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            # IMPORTANT: If raw parquet is missing or empty, DO NOT create sentiment/aggregated files.
            # This preserves the signal that raw data itself is missing and prevents silent imputation.
            return None
        else:
            raise

    if df is None or df.empty:
        return None

    df["title_clean"] = df["title"].fillna("").map(clean_text)
    df["body_clean"] = df["body"].fillna("").map(clean_text)
    df["text_clean"] = df["title_clean"] + " " + df["body_clean"]

    batch_size = choose_batch_size()
    tokenizer = get_tokenizer()

    sentiment_scores = []
    headline_sentiment_scores = []

    for i in range(0, len(df), batch_size):
        text_batch  = df["text_clean"].iloc[i : i + batch_size].tolist()
        title_batch = df["title_clean"].iloc[i : i + batch_size].tolist()

        enc_text  = tokenizer(text_batch,  padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc_title = tokenizer(title_batch, padding=True, truncation=True, max_length=32, return_tensors="pt")

        text_scores  = analyze_sentiment(enc_text)
        title_scores = analyze_sentiment(enc_title)

        sentiment_scores.extend(text_scores)
        headline_sentiment_scores.extend(title_scores)

    df["sentiment_score"] = sentiment_scores
    df["headline_sentiment_score"] = headline_sentiment_scores

    df["is_positive"] = df["sentiment_score"] > 0
    df["is_negative"] = df["sentiment_score"] < 0
    df["is_neutral"]  = df["sentiment_score"].abs() < 0.1
    df["is_extreme"]  = df["sentiment_score"].abs() > 0.5
    df["text_length"] = df["text_clean"].str.len().fillna(0).astype(float)

    write_parquet_s3(df, BUCKET, sentiment_key)
    return df


def get_aggregated(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    aggregated_key = f"{AGGREGATED_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, aggregated_key)
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            pass  # features not yet aggregated, continue
        else:
            raise

    df = get_sentiment(local_date)
    if df is None:
        return None

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
        "headline_mean_sentiment": [df["headline_sentiment_score"].mean()],
        "headline_median_sentiment": [df["headline_sentiment_score"].median()],
        "article_length_variance": [df["text_length"].var()],
        "ratio_pos_neg": [
            (df["is_positive"].mean()) /
            (df["is_negative"].mean() + 1e-9)
        ],
        "conflict_share": [
            (df["sentiment_score"].abs().between(0.05, 0.15)).mean()
        ],
    })

    write_parquet_s3(result, BUCKET, aggregated_key)
    return result


def stop_early(current, context=None):
    if context is not None and int(context.get_remaining_time_in_millis()) < MIN_REMAINING_MS:
        logger.warning(f"Stopping early at {current} due to timeout limit.")
        return True
    else:
        return False


def generate_lagged_features(context=None):
    try:    
        last_done = read_preproc_state()
        if last_done is None:
            current = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        else:
            current = last_done + timedelta(days=1)

        last_added = read_collector_state()
        if last_added is None:
            return {"statusCode": 200, "body": "no raw data to process"}

        lagged_rows = []

        window = deque(maxlen=LAG_DAYS)
        for i in range(LAG_DAYS, 0, -1):
            d = current - timedelta(days=i)
            agg = get_aggregated(d)
            window.append(agg)

        while current <= last_added:

            if stop_early(current, context):
                break

            first_valid = next((a for a in window if a is not None), None)
            if first_valid is None:
                columns = []
            else:
                columns = [c for c in first_valid.columns if c not in ("localDate",)]

            row: dict[str, float | str] = {"localDate": current.strftime("%Y-%m-%d")}

            for i, agg in enumerate(window):
                lag_number = LAG_DAYS - i
                for col in columns:
                    value = np.nan
                    if agg is not None and col in agg.columns and not agg.empty:
                        value = float(agg[col].iloc[0])
                    row[f"{col}-{lag_number}"] = value

            lagged_rows.append(row)
            logger.info(f"Computed lagged features for {current}")

            if stop_early(current, context):
                break

            today_agg = get_aggregated(current)
            window.append(today_agg)

            last_done = current
            current += timedelta(days=1)

        # If nothing was computed → exit cleanly
        if not lagged_rows:
            return {"statusCode": 200, "body": "nothing to do"}

        # Write one new parquet chunk for this invocation
        out_df = pd.DataFrame(lagged_rows)
        file_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.parquet"
        out_key = f"{LAGGED_OUTPUT_PREFIX}{file_name}"
        write_parquet_s3(out_df, BUCKET, out_key)

        # Update checkpoint atomically
        write_preproc_state(last_done)

        return {"statusCode": 200, "body": "ok"}
    except Exception:
        logger.exception("Error in generate_lagged_features")
        return {"statusCode": 500, "body": "error"}


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
