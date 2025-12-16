# ---------- STANDARD ----------
import os
from typing import Iterable
import pandas as pd
import numpy as np

# ---------- PROJECT SHARED ----------
import shared as s
import stock_preproc as sp   # schemas + prefixes
import news_preproc as npg   # schemas + prefixes

# ---------- CONFIG ----------
BUCKET = os.getenv("BUCKET", "dev-rkoch-spre")
LAG_DAYS = int(os.getenv("LAG_DAYS", "5"))

PIVOTED_PREFIX = sp.PIVOTED_PREFIX
TARGET_PREFIX  = sp.TARGET_LOG_RETURNS_PREFIX
LAGGED_PREFIX  = npg.LAGGED_PREFIX

# consolidated files written to SAME prefix
PIVOTED_CONSOLIDATED_KEY = f"{PIVOTED_PREFIX}__consolidated__.parquet"
TARGET_CONSOLIDATED_KEY  = f"{TARGET_PREFIX}__consolidated__.parquet"


# ---------- HELPERS ----------
def _read_all_tables(prefix: str, schema):
    keys = s.list_keys_s3(BUCKET, prefix)
    tables = []
    for k in keys:
        tables.append((k, s.read_parquet_s3(BUCKET, k, schema)))
    return keys, tables


def _tables_to_df(tables, drop_cols: Iterable[str] = ()):
    # concat tables to pandas, keep localDate
    dfs = []
    for _, t in tables:
        df = t.to_pandas()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def _dedup_last_wins(df: pd.DataFrame):
    # assumes 'localDate' exists
    df = df.sort_values("localDate", kind="stable")
    # keep last occurrence of each localDate
    return df.drop_duplicates(subset=["localDate"], keep="last")


def _to_indexed(df: pd.DataFrame):
    df["localDate"] = pd.to_datetime(df["localDate"])
    df = df.set_index("localDate").sort_index()
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not strictly increasing after sort")
    if df.index.has_duplicates:
        raise ValueError("Duplicates remain after dedup")
    return df


def _write_consolidated_and_cleanup(consolidated_key: str, schema, df: pd.DataFrame, keys_to_delete: list[str]):
    # write
    import pyarrow as pa
    table = pa.Table.from_pandas(df.reset_index(), schema=schema, preserve_index=False)
    s.write_parquet_s3(table, BUCKET, consolidated_key, schema)
    # delete partials
    for k in keys_to_delete:
        s.retry_s3(lambda: s.s3.delete_object(Bucket=BUCKET, Key=k))


# ---------- LOADERS ----------
def load_pivoted_df_price_flat(consolidate: bool = True) -> pd.DataFrame:
    schema = sp.get_pivoted_schema()
    keys, tables = _read_all_tables(PIVOTED_PREFIX, schema)

    # pandas
    df = _tables_to_df(tables)
    # dedup last-wins
    df = _dedup_last_wins(df)
    # index
    df = _to_indexed(df)
    # drop index col
    df_price_flat = df.drop(columns=["localDate"], errors="ignore")

    if consolidate:
        _write_consolidated_and_cleanup(
            PIVOTED_CONSOLIDATED_KEY, schema, df, keys
        )
    return df_price_flat


def load_returns_y(consolidate: bool = True) -> pd.DataFrame:
    schema = sp.get_target_schema()
    keys, tables = _read_all_tables(TARGET_PREFIX, schema)

    df = _tables_to_df(tables)
    df = _dedup_last_wins(df)
    df = _to_indexed(df)
    returns_y = df.drop(columns=["localDate"], errors="ignore")

    if consolidate:
        _write_consolidated_and_cleanup(
            TARGET_CONSOLIDATED_KEY, schema, df, keys
        )
    return returns_y


def load_news_features() -> pd.DataFrame:
    # no dedup, no consolidation
    schema = npg.get_lagged_schema()
    keys, tables = _read_all_tables(LAGGED_PREFIX, schema)

    df = _tables_to_df(tables)
    df = _to_indexed(df)
    news_features = df.drop(columns=["localDate"], errors="ignore")
    return news_features


# ---------- ALIGNMENT ----------
def load_aligned_training_inputs(consolidate_prices_and_targets: bool = True):
    df_price_flat = load_pivoted_df_price_flat(consolidate_prices_and_targets)
    news_features = load_news_features()
    returns_y     = load_returns_y(consolidate_prices_and_targets)

    common = (
        df_price_flat.index
        .intersection(news_features.index)
        .intersection(returns_y.index)
    )

    if not (
        common.equals(df_price_flat.index)
        and common.equals(news_features.index)
        and common.equals(returns_y.index)
    ):
        raise ValueError("Hard fail: index misalignment across inputs")

    return df_price_flat, news_features, returns_y


# ---------- INTEGRATION POINT ----------
# Use this at the top of your monthly retrain job:
#
# df_price_flat, news_features, returns_y = load_aligned_training_inputs(
#     consolidate_prices_and_targets=True
# )
#
# ... then proceed with PCA fit (train split only), windowing, and model.fit()
