import os
import re
from io import BytesIO
from datetime import timedelta, datetime
import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- CONFIG ----------
BUCKET = "dev-rkoch-spre"
REGION = "eu-west-1"
LAG_DAYS = 5  # configurable

RAW_PREFIX = "raw/news/localDate="
PROCESSED_PREFIX = "processed/news/localDate="
LAGGED_KEY = f"lagged_{LAG_DAYS}/news/data.parquet"

EXTRA_SYMBOLS = "•™£€¥"
ALLOWED_CHARS = rf"a-zA-Z0-9\s{re.escape(EXTRA_SYMBOLS)}.,;:!?%$\/\-@&"

s3 = boto3.client("s3", region_name=REGION)

# ---------- MODEL ----------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()


# ---------- HELPERS ----------
def s3_key_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


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
    if s3_key_exists(bucket, key):
        existing = read_parquet_s3(bucket, key)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
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

    if not s3_key_exists(BUCKET, input_key):
        return None  # no raw data

    if s3_key_exists(BUCKET, output_key):
        return read_parquet_s3(BUCKET, output_key)  # already processed

    df = read_parquet_s3(BUCKET, input_key)
    if df.empty:
        return None

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


def find_next_target_date():
    """Find earliest t not yet processed in lagged_<LAG_DAYS> file."""
    if not s3_key_exists(BUCKET, LAGGED_KEY):
        return "1999-11-01"

    df = read_parquet_s3(BUCKET, LAGGED_KEY)
    last_date = pd.to_datetime(df["localDate"]).max()
    return (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def generate_lagged_features():
    """Main orchestration"""
    t = find_next_target_date()

    while True:
        lag_dates = [
            (pd.to_datetime(t) - timedelta(days=i)).strftime("%Y-%m-%d")
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
            print(f"Stopping at {t}: all {LAG_DAYS} previous raw days missing.")
            break

        row = {"localDate": t}
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

        print(f"[{datetime.now().isoformat()}] Added lagged features for {t}")
        t = (pd.to_datetime(t) + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    generate_lagged_features()
