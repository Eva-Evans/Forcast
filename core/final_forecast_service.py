from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from config import FORECAST_HORIZON_MONTHS, PIPELINE_WORK_ROOT
from core.helpers import iter_month_ends, month_end
from core.subdivisions_registry import resolve_farm_unit, trade_rules_for
from core.lactation_stock_from_events import build_lactation_monthly_from_events
from core.tab3_to_final import (
    build_events_all,
    build_events_workbook,
    export_bulls_workbook,
    export_tab3_to_filter_folder,
)
from forecast_dynamic import latest_data_date
from prognoz_vseh_parametrov import KALUGA_TREE, SUBDIVISION_ALIASES, PipelineConfig, run_pipeline


def _month_label(y: int, m: int) -> str:
    return f"{y}-{m:02d}"


def predict_months_from_anchor(anchor: date, n_months: int) -> list[tuple[int, int]]:
    y, m = anchor.year, anchor.month
    out: list[tuple[int, int]] = []
    for _ in range(n_months):
        out.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def run_final_forecast_for_subdivision(
    subdivision_name: str,
    tables: dict[str, pd.DataFrame],
    *,
    farm_hint: str | None = None,
    horizon_months: int | None = None,
    work_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Runs prognoz_vseh_parametrov pipeline for one subdivision.
    Returns (forecast_table, fact_table, meta) — wide format: index=parameter, columns=YYYY-MM.
    """
    farm, unit = resolve_farm_unit(subdivision_name, farm_hint)
    latest = latest_data_date(tables)
    anchor_me = month_end(latest.year, latest.month)
    train_end = pd.Timestamp(anchor_me)
    n = int(horizon_months or FORECAST_HORIZON_MONTHS)
    predict_months = predict_months_from_anchor(anchor_me, n)
    month_cols = [_month_label(y, m) for y, m in predict_months]

    root = Path(work_root or PIPELINE_WORK_ROOT)
    safe = re.sub(r"[^\w\-]+", "_", unit)[:80]
    work = (root / safe).resolve()
    filter_dir = work / f"filter_{safe}"
    export_tab3_to_filter_folder(tables, filter_dir)

    events_xlsx = work / "events_cows.xlsx"
    events_csv = work / "events_cows.csv"
    events_path = build_events_workbook(
        tables,
        unit,
        farm,
        events_xlsx,
        csv_path=events_csv,
    )
    bulls_path = export_bulls_workbook(tables, work / "bulls_full.xlsx")

    lact_path = work / "lactation_stock.xlsx"
    df_events = build_events_all(tables)
    df_events["Столбец1"] = SUBDIVISION_ALIASES.get(unit, [unit])[0]
    df_events["Source.Name"] = farm
    lact_df = build_lactation_monthly_from_events(
        df_events,
        unit,
        start_floor=(2022, 1),
        end_cap=anchor_me,
    )
    lact_df.to_excel(lact_path, index=False)

    rules = trade_rules_for(farm, unit)
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])

    cfg = PipelineConfig(
        name=f"{farm} / {unit}",
        work_dir=work,
        filter_folder=str(filter_dir),
        events_path=events_path,
        events_aux_path=events_path,
        bulls_path=bulls_path,
        lactation_path=lact_path,
        output_xlsx=work / "forecast_all.xlsx",
        subdivision_names=subdiv_names,
        kuda_buy_tokens=rules["buy"],
        sold_heifer_kuda_tokens=rules["heifer_sale"],
        sold_bull_kuda_tokens=rules["bull_sale"],
        sales_require_pereezd=farm in KALUGA_TREE,
        sold_heifer_dest=rules["heifer_sale"][0] if rules.get("heifer_sale") else unit,
        sold_bull_dest="BULLS",
        train_end_ts=train_end,
        predict_months=predict_months,
        month_cols=month_cols,
    )

    forecast_table, fact_table = run_pipeline(cfg)
    meta = {
        "farm": farm,
        "unit": unit,
        "subdivision": subdivision_name,
        "latest_data": latest.isoformat(),
        "train_end": train_end.date().isoformat(),
        "horizon_months": n,
        "month_cols": month_cols,
    }
    return forecast_table, fact_table, meta
