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
EMAIL_FROM = os.getenv("EMAIL_FROM", "spre@rkoch.dev")
EMAIL_TO = os.getenv("EMAIL_TO", "spre@rkoch.dev")
SYMBOLS = [
    sym.strip()
    for sym in os.getenv("SYMBOLS", "goog,msft,tsla").split(",")
    if sym.strip()
]
TOP_GAINERS_COUNT = int(os.getenv("TOP_GAINERS_COUNT", "10"))

ses = boto3.client("ses", region_name=AWS_REGION)


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


def get_inference(py_date: date) -> dict[str, float]:
    key = f"{s.INFERENCE_PREFIX}{py_date.isoformat()}/data.parquet"
    schema = s.get_inference_schema()
    table = s.read_parquet_s3(key, schema)
    symbols = table["symbol"].to_pylist()
    preds = table["predicted_log_return"].to_pylist()
    return dict(zip(symbols, preds))


def get_returns(py_date: date) -> dict[str, float]:
    keys = s.list_keys_s3(s.TARGET_LOG_RETURNS_PREFIX, sort_reversed=True)
    schema = s.get_target_schema()

    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table["localDate"].to_pylist()

        for i, d in enumerate(dates):
            if d == py_date:
                out = {}
                for name in schema.names:
                    if name != "localDate":
                        sym = name.removesuffix("_return")
                        out[sym] = table[name][i].as_py()
                return out

    raise ValueError(f"No actual returns for {py_date}")


def classify(value: float) -> str:
    return "GAIN" if value > 0.0 else "LOSS"


def top_k(pred: dict[str, float], k: int):
    return sorted(pred.items(), key=lambda kv: kv[1], reverse=True)[:k]


def directional_accuracy(pred: dict[str, float], actual: dict[str, float]):
    correct = 0
    total = 0
    for sym, p in pred.items():
        if sym in actual:
            total += 1
            if p * actual[sym] > 0:
                correct += 1
    return correct, total


def top_k_hit_rate(pred: dict[str, float], actual: dict[str, float], k: int):
    hits = 0
    for sym, _ in top_k(pred, k):
        if sym in actual and actual[sym] > 0:
            hits += 1
    return hits


def sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


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

    x1 = get_latest_inference_date()
    x = x1 - s.ONE_DAY

    inf_x1 = get_inference(x1)
    inf_x = get_inference(x)

    act_x = get_returns(x)

    lines = []

    lines.append(f"{x1} prediction")

    # Predicted gain/loss of watched symbols for day X+1
    lines.append("Watched symbols")
    for sym in SYMBOLS:
        if sym in inf_x1:
            pred = inf_x1[sym]
            lines.append(f"{sym:<6}  {pred:+.5f}  {classify(pred)}")
    lines.append("")

    # Top K predicted gainers for day X+1
    lines.append(f"Top-{TOP_GAINERS_COUNT} gainers")
    for sym, val in top_k(inf_x1, TOP_GAINERS_COUNT):
        lines.append(f"{sym:<6}  {val:+.5f}")
    lines.append("")

    lines.append(f"{x} predicted vs actual")

    # Predicted vs actual for watched symbols for day X
    lines.append("Watched symbols")
    lines.append("Symbol  Predicted  Actual  Result")
    for sym in SYMBOLS:
        if sym in inf_x and sym in act_x:
            pred = inf_x[sym]
            act = act_x[sym]
            lines.append(
                f"{sym:<6}  {pred:+.5f}  {act:+.5f}  {'OK' if sign(pred) == sign(act) else 'FAIL'}"
            )
    lines.append("")

    # Directional accuracy overall for day X
    correct, total = directional_accuracy(inf_x, act_x)
    pct = (correct / total * 100.0) if total else 0.0
    lines.append(f"Directional accuracy: {correct} / {total} ({pct:.1f}%)")

    # Top K hit rate for day X
    hits = top_k_hit_rate(inf_x, act_x, TOP_GAINERS_COUNT)
    lines.append(f"Top-{TOP_GAINERS_COUNT} hit rate:      {hits} / {TOP_GAINERS_COUNT}")
    lines.append("")

    body = "\n".join(lines)
    subject = f"{x1} spre"

    send_email(subject, body)

    return {
        "statusCode": 200,
        "body": json.dumps({"date": x.isoformat()}),
    }


# --- LAMBDA ---
def lambda_handler(event=None, context=None):
    return notify()


# --- ENTRY POINT ---
if __name__ == "__main__":
    notify()
