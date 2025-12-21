# --- STANDARD ---
import json
import os
from datetime import date

# --- PROJECT ---
import shared as s

# --- THIRD-PARTY ---
import boto3

# --- CONFIG ---
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

SYMBOLS = [
    sym.strip()
    for sym in os.getenv("SYMBOLS", "goog,msft,tsla").split(",")
    if sym.strip()
]

EMAIL_FROM = os.getenv("EMAIL_FROM", "spre@rkoch.dev")
EMAIL_TO = os.getenv("EMAIL_TO", "spre@rkoch.dev")


ses = boto3.client("ses", region_name=AWS_REGION)


# --- HELPERS ---
def get_latest_inference_date() -> date:
    keys = s.list_keys_s3(s.INFERENCE_PREFIX, sort_reversed=True)
    if not keys:
        raise ValueError("No inference data found")

    # inference/localDate=YYYY-MM-DD/data.parquet
    for key in keys:
        if key.endswith("/data.parquet"):
            date_str = key.split(s.INFERENCE_PREFIX)[1].split("/")[0]
            return date.fromisoformat(date_str)

    raise ValueError("No inference parquet found")


def read_inference_table(inference_date: date):
    key = f"{s.INFERENCE_PREFIX}{inference_date.isoformat()}/data.parquet"
    schema = s.get_inference_schema()
    return s.read_parquet_s3(key, schema)


def classify(predicted_log_return: float) -> str:
    return "GAIN" if predicted_log_return > 0.0 else "LOSS"


def format_email(inference_date: date, rows: list[dict]) -> str:
    lines = [
        f"Stock prediction for {inference_date.isoformat()}",
        "",
        "Symbol    Prediction    Direction",
        "-" * 36,
    ]

    for r in rows:
        lines.append(f"{r['symbol']:<8}  {r['pred']:+.5f}      {r['direction']}")

    return "\n".join(lines)


def send_email(subject: str, body: str):
    ses.send_email(
        Source=EMAIL_FROM,
        Destination={"ToAddresses": [EMAIL_TO]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": body}},
        },
    )


def notify():
    if not SYMBOLS:
        raise ValueError("SYMBOLS env var is empty")

    inference_date = get_latest_inference_date()
    table = read_inference_table(inference_date)

    df = table.to_pandas()
    df = df[df["symbol"].isin(SYMBOLS)]

    if df.empty:
        raise ValueError("No matching symbols found in inference output")

    results = []
    for _, row in df.iterrows():
        pred = float(row["predicted_log_return"])
        results.append(
            {
                "symbol": row["symbol"],
                "pred": pred,
                "direction": classify(pred),
            }
        )

    body = format_email(inference_date, results)
    subject = f"Stock prediction for {inference_date.isoformat()}"

    send_email(subject, body)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "inferenceDate": inference_date.isoformat(),
                "symbols": SYMBOLS,
            }
        ),
    }


# --- LAMBDA ---
def lambda_handler(event=None, context=None):
    notify()


# --- ENTRY POINT ---
if __name__ == "__main__":
    notify()
