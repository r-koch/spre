import logging
import os
import re
import time
from io import BytesIO
from datetime import timedelta, datetime
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
START_DATE = os.environ.get("START_DATE", "1999-11-01")
MIN_REMAINING_MS = int(os.environ.get("MIN_REMAINING_MS", "60000"))
MODEL_NAME_OR_DIR = os.environ.get("MODEL_DIR", "ProsusAI/finbert")  # locally use model name, in docker use env

RAW_PREFIX = "raw/news/localDate="
PROCESSED_PREFIX = "processed/news/localDate="
LAGGED_KEY = f"lagged_{LAG_DAYS}/news/data.parquet"

EXTRA_SYMBOLS = "•™£€¥"
ALLOWED_CHARS = rf"a-zA-Z0-9\s{re.escape(EXTRA_SYMBOLS)}.,;:!?%$\/\-@&"

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


def append_parquet_s3(df_new, bucket, key):
    try:
        existing = read_parquet_s3(bucket, key)
        combined = pd.concat([existing, df_new], ignore_index=True)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            combined = df_new
        else:
            raise
        
    combined = combined.sort_values("localDate").drop_duplicates(subset=["localDate"])
    write_parquet_s3(combined, bucket, key)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
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


# ---------- PIPELINE ----------
def process_single_day(local_date: str):
    """Processes raw news for a single day → mean & count sentiment; writes processed parquet with localDate=<d>."""
    input_key = f"{RAW_PREFIX}{local_date}/data.parquet"
    output_key = f"{PROCESSED_PREFIX}{local_date}/data.parquet"

    try:
        return read_parquet_s3(BUCKET, output_key)
    except s3.exceptions.ClientError:
        pass  # not yet processed, continue

    try:
        df = read_parquet_s3(BUCKET, input_key)
        if df.empty:
            raise ValueError(f"Raw parquet {input_key} is unexpectedly empty")

        df["title_clean"] = df["title"].map(clean_text)
        df["body_clean"] = df["body"].map(clean_text)
        df["text_clean"] = df["title_clean"] + " " + df["body_clean"]

        # Sentiment
        batch_size = 32
        scores = []
        for i in range(0, len(df), batch_size):
            batch = df["text_clean"].iloc[i : i + batch_size].tolist()
            scores.extend(analyze_sentiment(batch))
        df["sentiment_score"] = scores

        # Aggregate per day (single date)
        result = pd.DataFrame(
            {
                "localDate": [local_date],
                "count_articles": [len(df)],
                "mean_sentiment": [df["sentiment_score"].mean()],
            }
        )

        write_parquet_s3(result, BUCKET, output_key)
        return result
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            return None
        else:
            raise


def find_start_date():
    """Find earliest t not yet processed in lagged_<LAG_DAYS> file."""
    try:
        df = read_parquet_s3(BUCKET, LAGGED_KEY)
        last_date = pd.to_datetime(df["localDate"]).max().date()
        return last_date + timedelta(days=1)
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            return datetime.strptime(START_DATE, "%Y-%m-%d").date()
        else:
            raise


def generate_lagged_features(context=None):
    """Main orchestration"""
    try:    
        current = find_start_date()
        now = datetime.now().date()

        while current < now:

            if context is not None and int(context.get_remaining_time_in_millis()) < MIN_REMAINING_MS:
                logger.warning(f"Stopping early at {current} to avoid timeout (remaining_ms={context.get_remaining_time_in_millis()}).")
                break

            lag_dates = [
                (current - timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(1, LAG_DAYS + 1)
            ]

            processed_data = {}
            all_missing = True
            for d in lag_dates:
                agg = process_single_day(d)
                if agg is not None and not agg.empty:
                    all_missing = False
                    processed_data[d] = agg

            if all_missing:
                logger.warning(f"Stopping at {current}: all {LAG_DAYS} previous raw days missing.")
                break

            row = {"localDate": current.strftime("%Y-%m-%d")}
            for i, d in enumerate(lag_dates, 1):
                agg = processed_data.get(d)
                if agg is not None and not agg.empty:
                    row[f"mean_sentiment-{i}"] = agg["mean_sentiment"].iloc[0]
                    row[f"count_articles-{i}"] = agg["count_articles"].iloc[0]
                else:
                    row[f"mean_sentiment-{i}"] = float("nan")
                    row[f"count_articles-{i}"] = float("nan")

            df_row = pd.DataFrame([row])
            append_parquet_s3(df_row, BUCKET, LAGGED_KEY)

            logger.info(f"Added lagged features for {current}")
            current = current + timedelta(days=1)

        return {"statusCode": 200, "body": "done"}
    except Exception as e:
        logger.exception("Error in generate_lagged_features")
        return {"statusCode": 500, "body": "error"}


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
