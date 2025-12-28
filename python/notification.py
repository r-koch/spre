# --- STANDARD ---
import heapq
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

        return (
            date.fromisoformat(date_str),
            {
                "symbol": table["symbol"].to_pylist(),
                "zscore_1d": table["zscore_1d"].to_pylist(),
                "rank_1d": table["rank_1d"].to_pylist(),
                "direction_1d": table["direction_1d"].to_pylist(),
                "log_return_1d": table["log_return_1d"].to_pylist(),
                "log_return_3d": table["log_return_3d"].to_pylist(),
                "log_return_5d": table["log_return_5d"].to_pylist(),
            },
        )

    latest = load(keys[0])
    previous = load(keys[1])

    return latest, previous


def get_actual(py_date: date) -> dict[str, dict[str, float]]:
    keys = s.list_keys_s3(s.TARGET_PREFIX, sort_reversed=True)
    schema = s.get_target_schema()

    for key in keys:
        table = s.read_parquet_s3(key, schema)
        dates = table[s.LOCAL_DATE].to_pylist()

        for i, d in enumerate(dates):
            if d == py_date:
                out: dict[str, dict[str, float]] = {}
                for name in schema.names:
                    if name != s.LOCAL_DATE:
                        sym, metric = name.split("_", 1)
                        out.setdefault(sym, {})[metric] = table[name][i].as_py()
                return out

    raise ValueError(f"No actual returns for {py_date}")


def unpack(pred):
    return {
        sym: {
            "zscore_1d": pred["zscore_1d"][i],
            "rank_1d": pred["rank_1d"][i],
            "direction_1d": pred["direction_1d"][i],
            "log_return_1d": pred["log_return_1d"][i],
            "log_return_3d": pred["log_return_3d"][i],
            "log_return_5d": pred["log_return_5d"][i],
        }
        for i, sym in enumerate(pred["symbol"])
    }


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

        latest_map = unpack(latest_pred)
        prev_map = unpack(previous_pred)

        previous_actual = get_actual(previous_date)

        lines = []

        lines.append(f"{latest_date} prediction")

        # Predicted gain/loss of watched symbols for day X+1
        lines.append("Watched symbols")
        for sym in WATCHED_SYMBOLS:
            if sym in latest_map:
                m = latest_map[sym]
                lines.append(
                    f"{sym:<6} "
                    f"{m['zscore_1d']:+.3f} "
                    f"{m['rank_1d']:+.3f} "
                    f"{m['direction_1d']:.2f} "
                    f"{m['log_return_1d']:+.5f} "
                    f"{m['log_return_3d']:+.5f} "
                    f"{m['log_return_5d']:+.5f}"
                )
        lines.append("")

        # # Top K predicted gainers for day X+1
        # top_gainers = get_top_gainers(latest_pred)
        # lines.append(f"Top-{TOP_GAINERS_COUNT} gainers")
        # for sym, val in top_gainers:
        #     lines.append(f"{sym:<6}  {val:+.5f}")
        # lines.append("")

        lines.append(f"{previous_date} predicted vs actual")

        # Predicted vs actual for watched symbols for day X
        lines.append("Watched symbols")
        lines.append(
            "Symbol   "
            "z_p     z_a     "
            "r_p     r_a     "
            "d_p d_a "
            "lr1_p    lr1_a    "
            "lr3_p    lr3_a    "
            "lr5_p    lr5_a"
        )

        for sym in WATCHED_SYMBOLS:
            if sym not in prev_map or sym not in previous_actual:
                continue

            p = prev_map[sym]
            a = previous_actual[sym]

            lines.append(
                f"{sym:<6} "
                f"{p['zscore_1d']:+.3f} {a['zscore_1d']:+.3f} "
                f"{p['rank_1d']:+.3f} {a['rank_1d']:+.3f} "
                f"{int(p['direction_1d'] >= 0.5):^3} {a['direction_1d']:^3} "
                f"{p['log_return_1d']:+.5f} {a['log_return_1d']:+.5f} "
                f"{p['log_return_3d']:+.5f} {a['log_return_3d']:+.5f} "
                f"{p['log_return_5d']:+.5f} {a['log_return_5d']:+.5f}"
            )

        lines.append("")

        # # Directional accuracy overall for day X
        # correct, total = get_directional_accuracy(previous_pred, previous_actual)
        # pct = (correct / total * 100.0) if total else 0.0
        # lines.append(f"Directional accuracy: {correct} / {total} ({pct:.1f}%)")

        # # Top K hit rate for day X
        # hits = get_top_gainers_hit_rate(top_gainers, previous_actual)
        # lines.append(
        #     f"Top-{TOP_GAINERS_COUNT} hit rate:      {hits} / {TOP_GAINERS_COUNT}"
        # )
        # lines.append("")

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
