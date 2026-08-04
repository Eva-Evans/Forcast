#!/usr/bin/env python3
"""Smoke E2E: SQLite tab3 tables -> final_forecast_service -> wide forecast."""
from __future__ import annotations

import sqlite3
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.final_forecast_service import run_final_forecast_for_subdivision  # noqa: E402


def load_tables_from_sqlite(db_path: Path) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    calv = pd.read_sql(
        "SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact "
        "FROM calvings_births_raw",
        conn,
    )
    ins = pd.read_sql(
        "SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw",
        conn,
    )
    dry = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", conn)
    disp = pd.read_sql(
        "SELECT reg, event_date, disposal_reason FROM disposals_raw",
        conn,
    )
    bulls = pd.read_sql("SELECT bull_code, bull_type FROM bulls_raw", conn)
    conn.close()
    return {"calv": calv, "ins": ins, "dry": dry, "disp": disp, "bulls": bulls}


def main() -> int:
    db = ROOT / "herd_data.db"
    if not db.exists():
        print("FAIL: no herd_data.db")
        return 1
    tables = load_tables_from_sqlite(db)
    print("loaded rows:", {k: len(v) for k, v in tables.items()})
    try:
        forecast, fact, meta = run_final_forecast_for_subdivision(
            "ЖК Высокое",
            tables,
            farm_hint="ЖК Высокое",
            work_root=ROOT / ".pipeline_runtime_verify",
        )
    except Exception:
        traceback.print_exc()
        return 2
    print("meta:", meta)
    print("forecast shape:", forecast.shape)
    print("forecast columns (first 5):", list(forecast.columns[:5]))
    print("forecast index (first 8):", list(forecast.index[:8]))
    if forecast.empty or len(forecast.columns) < 2:
        print("FAIL: empty or too narrow forecast")
        return 3
    print("OK: forecast built")
    if fact is not None and not fact.empty:
        print("fact shape:", fact.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
