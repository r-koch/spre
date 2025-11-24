from collections import deque
import json
import logging
import os
import re
import unicodedata
from io import BytesIO
from datetime import date, datetime, timedelta, timezone
import boto3
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

LAST_PROCESSED_KEY = "lastProcessed"
LAST_ADDED_KEY = "lastAdded"

RAW_PREFIX = "raw/news/localDate="
SENTIMENT_PREFIX = "news/sentiment/localDate="
AGGREGATED_PREFIX = "news/aggregated/localDate="

META_DATA = "metadata/"
COLLECTOR_STATE_KEY = f"{META_DATA}news_collector_state.json"
PREPROC_STATE_KEY = f"{META_DATA}news_preproc_state.json"
LAGGED_OUTPUT_PREFIX = f"news/lagged-{LAG_DAYS}/"

EXTRA_SYMBOLS = "•™£€¥"
ALLOWED_CHARS = rf"a-zA-Z0-9\s{re.escape(EXTRA_SYMBOLS)}.,;:!?%$\/\-@&"

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

def get_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        is_docker = "MODEL_DIR" in os.environ
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_DIR, local_files_only=is_docker)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_DIR, local_files_only=is_docker)
        model.eval()
    return tokenizer, model


# ---------- HELPERS ----------
def read_parquet_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    buf = BytesIO(obj["Body"].read())
    return pq.read_table(buf).to_pandas()


def write_parquet_s3(df, bucket, key):
    buf = BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf, compression="snappy")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def read_collector_state():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=COLLECTOR_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_ADDED_KEY], "%Y-%m-%d").date()
    except s3.exceptions.ClientError:
        raise ValueError(f"{COLLECTOR_STATE_KEY} is missing")

def read_preproc_state():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=PREPROC_STATE_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return datetime.strptime(data[LAST_PROCESSED_KEY], "%Y-%m-%d").date()
    except s3.exceptions.ClientError:
        return datetime.strptime(START_DATE, "%Y-%m-%d").date() - timedelta(days=1)


def write_preproc_state(date_obj):
    payload = {LAST_PROCESSED_KEY: date_obj.strftime("%Y-%m-%d")}
    s3.put_object(
        Bucket=BUCKET,
        Key=PREPROC_STATE_KEY,
        Body=json.dumps(payload).encode("utf-8")
    )


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(f"[^{ALLOWED_CHARS}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def analyze_sentiment(texts):
    if not texts:
        return []
    tokenizer, model = get_model()
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
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
    # 1. Prefer AWS Lambda declared memory
    mem_env = os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
    if mem_env:
        return int(mem_env)

    # 2. Fallback to /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb // 1024
    except Exception:
        pass

    # 3. Default fallback
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

# ---------- PIPELINE ----------
def get_sentiment(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    raw_key = f"{RAW_PREFIX}{date_str}/data.parquet"
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, sentiment_key)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            pass # sentiment not yet calculated, continue
        else:
            raise

    try:
        df = read_parquet_s3(BUCKET, raw_key)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            raise ValueError(f"Raw parquet {raw_key} is missing")
        else:
            raise

    if df.empty:
        raise ValueError(f"Raw parquet {raw_key} is unexpectedly empty")

    df["title_clean"] = df["title"].map(clean_text)
    df["body_clean"] = df["body"].map(clean_text)
    df["text_clean"] = df["title_clean"] + " " + df["body_clean"]

    # Sentiment
    batch_size = choose_batch_size()
    text_scores = []
    title_scores = []
    for i in range(0, len(df), batch_size):
        text_batch = df["text_clean"].iloc[i : i + batch_size].tolist()
        text_scores.extend(analyze_sentiment(text_batch))

        title_batch = df["title_clean"].iloc[i : i + batch_size].tolist()
        title_scores.extend(analyze_sentiment(title_batch))

    df["sentiment_score"] = text_scores
    df["headline_only_sentiment"] = title_scores

    # Additional fields needed for some aggregates
    df["is_positive"] = df["sentiment_score"] > 0
    df["is_negative"] = df["sentiment_score"] < 0
    df["is_neutral"]  = df["sentiment_score"].abs() < 0.1
    df["is_extreme"]  = df["sentiment_score"].abs() > 0.5
    df["text_length"] = df["text_clean"].str.len().fillna(0).astype(float)

    write_parquet_s3(df, BUCKET, sentiment_key)
    return df

def get_aggregated(local_date: date):
    date_str = local_date.strftime("%Y-%m-%d")
    sentiment_key = f"{SENTIMENT_PREFIX}{date_str}/data.parquet"
    aggregated_key = f"{AGGREGATED_PREFIX}{date_str}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, aggregated_key)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            pass  # features not yet aggregated, continue
        else:
            raise

    df = get_sentiment(local_date)

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
        "sentiment_skewness": [df["sentiment_score"].skew()],
        "sentiment_kurtosis": [df["sentiment_score"].kurtosis()],
        "neutral_share": [df["is_neutral"].mean()],
        "max_sentiment": [df["sentiment_score"].max()],
        "min_sentiment": [df["sentiment_score"].min()],
        "headline_only_mean_sentiment": [df["headline_only_sentiment"].mean()],
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
        current = last_done + timedelta(days=1)
        last_added = read_collector_state()

        columns = [
            "mean_sentiment",
            "count_articles",
            "median_sentiment",
            "pct_positive",
            "pct_negative",
            "polarity_ratio",
            "weighted_mean_sentiment",
            "extreme_sentiment_share",
            "sentiment_skewness",
            "sentiment_kurtosis",
            "neutral_share",
            "max_sentiment",
            "min_sentiment",
            "headline_only_mean_sentiment",
            "ewma_time_weighted_sentiment",
            "article_length_variance",
            "ratio_pos_neg",
            "conflict_share",
        ]

        lagged_rows = []

        window = deque(maxlen=LAG_DAYS)
        for i in range(1, LAG_DAYS + 1):
            d = current - timedelta(days=i)
            agg = get_aggregated(d)
            window.appendleft(agg)   # oldest first

        while current <= last_added:

            if stop_early(current, context):
                break

            row: dict[str, float | str] = {"localDate": current.strftime("%Y-%m-%d")}

            for i, agg in enumerate(window, start=1):
                for col in columns:
                    row[f"{col}-{i}"] = agg[col].iloc[0] if agg is not None else float("nan")

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
