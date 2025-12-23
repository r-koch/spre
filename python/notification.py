# --- STANDARD ---
import heapq
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
TOP_GAINERS_COUNT = int(os.getenv("TOP_GAINERS_COUNT", "10"))
WATCHED_SYMBOLS = [
    sym.strip()
    for sym in os.getenv("SYMBOLS", "goog,msft,tsla").split(",")
    if sym.strip()
]

LOGGER = s.setup_logger(__file__)

SES_CLIENT = boto3.client("ses", region_name=AWS_REGION)


def get_inferences() -> tuple[
    tuple[date, dict[str, float]],
    tuple[date, dict[str, float]],
]:
    keys = [
        k
        for k in s.list_keys_s3(s.INFERENCE_PREFIX, sort_reversed=True)
        if k.endswith("/data.parquet")
    ]

    if len(keys) < 2:
        raise ValueError(f"Expected at least 2 inference files, found {len(keys)}")

    schema = s.get_inference_schema()

    def load(key: str) -> tuple[date, dict[str, float]]:
        # inference/localDate=YYYY-MM-DD/data.parquet
        date_str = key.split(s.INFERENCE_PREFIX, 1)[1].split("/", 1)[0]

        table = s.read_parquet_s3(key, schema)
        symbols = table["symbol"].to_pylist()
        preds = table["predicted_log_return"].to_pylist()

        return date.fromisoformat(date_str), dict(zip(symbols, preds))

    latest = load(keys[0])
    previous = load(keys[1])

    return latest, previous


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


def get_top_gainers(pred: dict[str, float]) -> list[tuple[str, float]]:
    return heapq.nlargest(TOP_GAINERS_COUNT, pred.items(), key=lambda kv: kv[1])


def get_directional_accuracy(pred: dict[str, float], actual: dict[str, float]):
    correct = 0
    total = 0
    for sym, p in pred.items():
        if sym in actual:
            total += 1
            if p * actual[sym] > 0:
                correct += 1
    return correct, total


def get_top_gainers_hit_rate(
    top_gainers: list[tuple[str, float]], actual: dict[str, float]
):
    hits = 0
    for sym, _ in top_gainers:
        if sym in actual and actual[sym] > 0:
            hits += 1
    return hits


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def send_email(subject: str, body: str):
    SES_CLIENT.send_email(
        Source=EMAIL_FROM,
        Destination={"ToAddresses": [EMAIL_TO]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": body}},
        },
    )


def notify():
    try:
        if not WATCHED_SYMBOLS:
            raise ValueError("SYMBOLS env var is empty")

        (latest_date, latest_pred), (previous_date, previous_pred) = get_inferences()

        previous_actual = get_returns(previous_date)

        lines = []

        lines.append(f"{latest_date} prediction")

        # Predicted gain/loss of watched symbols for day X+1
        lines.append("Watched symbols")
        for sym in WATCHED_SYMBOLS:
            if sym in latest_pred:
                pred = latest_pred[sym]
                lines.append(
                    f"{sym:<6}  {pred:+.5f}  {"GAIN" if sign(pred) > 0 else "LOSS"}"
                )
        lines.append("")

        # Top K predicted gainers for day X+1
        top_gainers = get_top_gainers(latest_pred)
        lines.append(f"Top-{TOP_GAINERS_COUNT} gainers")
        for sym, val in top_gainers:
            lines.append(f"{sym:<6}  {val:+.5f}")
        lines.append("")

        lines.append(f"{previous_date} predicted vs actual")

        # Predicted vs actual for watched symbols for day X
        lines.append("Watched symbols")
        lines.append("Symbol  Predicted  Actual  Result")
        for sym in WATCHED_SYMBOLS:
            if sym in previous_pred and sym in previous_actual:
                pred = previous_pred[sym]
                act = previous_actual[sym]
                lines.append(
                    f"{sym:<6}  {pred:+.5f}  {act:+.5f}  {'OK' if sign(pred) == sign(act) else 'FAIL'}"
                )
        lines.append("")

        # Directional accuracy overall for day X
        correct, total = get_directional_accuracy(previous_pred, previous_actual)
        pct = (correct / total * 100.0) if total else 0.0
        lines.append(f"Directional accuracy: {correct} / {total} ({pct:.1f}%)")

        # Top K hit rate for day X
        hits = get_top_gainers_hit_rate(top_gainers, previous_actual)
        lines.append(
            f"Top-{TOP_GAINERS_COUNT} hit rate:      {hits} / {TOP_GAINERS_COUNT}"
        )
        lines.append("")

        body = "\n".join(lines)
        subject = f"{latest_date} spre"

        send_email(subject, body)

        return s.result(LOGGER, 200, f"sent notification for {latest_date}")

    except Exception:
        LOGGER.exception("Error in notify")
        return {"statusCode": 500, "body": "error"}


# --- LAMBDA ---
def lambda_handler(event=None, context=None):
    return notify()


# --- ENTRY POINT ---
if __name__ == "__main__":
    notify()
