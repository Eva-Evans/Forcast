"""Trim tab3 tables to an anchor date (backtest / train cutoff)."""
from __future__ import annotations

from datetime import date

import pandas as pd


def _cut_df(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    if "event_date" not in out.columns:
        return out
    dt = pd.to_datetime(out["event_date"], errors="coerce")
    return out.loc[dt.notna() & (dt <= cutoff)].copy()


def trim_tables_to_date(tables: dict[str, pd.DataFrame], anchor: date) -> dict[str, pd.DataFrame]:
    """Keep only rows with event_date on or before end of anchor month."""
    cutoff = pd.Timestamp(anchor.year, anchor.month, 1) + pd.offsets.MonthEnd(0)
    out: dict[str, pd.DataFrame] = {}
    for key, df in tables.items():
        if key == "bulls":
            out[key] = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        else:
            out[key] = _cut_df(df, cutoff) if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return out
