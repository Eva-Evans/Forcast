from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from config import FORECAST_HORIZON_MONTHS, PIPELINE_WORK_ROOT
from core.forecast_params import filter_forecast_display_table
from core.helpers import month_end
from core.lactation_stock_from_events import build_or_load_lactation_monthly
from core.pipeline_artifacts import ensure_pipeline_artifacts
from core.pipeline_tables import trim_tables_to_date
from core.subdivisions_registry import resolve_farm_unit, trade_rules_for
from forecast_dynamic import latest_data_date
from prognoz_vseh_parametrov import (
    KALUGA_TREE,
    ROOT,
    SUBDIVISION_ALIASES,
    PipelineConfig,
    rem_codes_all_kaluga,
    run_pipeline,
)


def _month_label(y: int, m: int) -> str:
    return f"{y}-{m:02d}"


def predict_months_from_anchor(
    anchor: date,
    n_months: int,
    *,
    start_next_month: bool = False,
) -> list[tuple[int, int]]:
    y, m = anchor.year, anchor.month
    if start_next_month:
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
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
    anchor_date: date | None = None,
    backtest: bool = False,
    manual_baseline: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Runs prognoz_vseh_parametrov pipeline for one subdivision (Postgres / tab3 tables).

    anchor_date: if set, trim all events to this month-end and forecast the next N months
    (backtest mode). If None, anchor = month of latest event in tables (production forecast).

    Production forecast uses forecast_only=True (no fact sheet). Full fact is built only
    in backtest mode.
    """
    farm, unit = resolve_farm_unit(subdivision_name, farm_hint)
    latest = latest_data_date(tables)
    if anchor_date is not None:
        anchor_me = month_end(anchor_date.year, anchor_date.month)
        if latest < anchor_me:
            raise ValueError(
                f"В данных последняя дата {latest}, раньше выбранного якоря {anchor_me}."
            )
        tables = trim_tables_to_date(tables, anchor_me)
    else:
        anchor_me = month_end(latest.year, latest.month)

    train_end = pd.Timestamp(anchor_me)
    n = int(horizon_months or FORECAST_HORIZON_MONTHS)
    predict_months = predict_months_from_anchor(
        anchor_me,
        n,
        start_next_month=True,
    )
    month_cols = [_month_label(y, m) for y, m in predict_months]
    forecast_only = not backtest

    root = Path(work_root or PIPELINE_WORK_ROOT)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    else:
        root = root.resolve()
    safe = re.sub(r"[^\w\-]+", "_", unit)[:80]
    work = root / safe
    filter_dir = work / f"filter_{safe}"

    artifacts, df_events = ensure_pipeline_artifacts(
        work=work,
        filter_dir=filter_dir,
        tables=tables,
        unit=unit,
        farm=farm,
        train_end=train_end,
    )

    lact_path = work / "lactation_stock.xlsx"
    build_or_load_lactation_monthly(
        df_events,
        unit,
        cache_path=lact_path,
        start_floor=(2022, 1),
        end_cap=anchor_me,
    )

    rules = trade_rules_for(farm, unit)
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])

    cfg = PipelineConfig(
        name=f"{farm} / {unit}",
        work_dir=work,
        filter_folder=str(artifacts.filter_dir),
        events_path=artifacts.events_xlsx,
        events_aux_path=artifacts.events_xlsx,
        bulls_path=artifacts.bulls_xlsx,
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
        kaluga_farm=farm,
        kaluga_unit=unit,
        kaluga_internal_tokens=rem_codes_all_kaluga() if farm in KALUGA_TREE else [],
        forecast_only=forecast_only,
        sep2024_baseline=manual_baseline,
    )

    forecast_table, fact_table = run_pipeline(cfg)
    forecast_table = filter_forecast_display_table(forecast_table)
    if isinstance(fact_table, pd.DataFrame) and not fact_table.empty:
        fact_table = filter_forecast_display_table(fact_table)
    meta = {
        "farm": farm,
        "unit": unit,
        "subdivision": subdivision_name,
        "latest_data": latest.isoformat(),
        "train_end": train_end.date().isoformat(),
        "horizon_months": n,
        "month_cols": month_cols,
        "backtest": backtest,
        "forecast_only": forecast_only,
        "anchor_date": anchor_me.isoformat(),
        "manual_baseline": manual_baseline,
    }
    return forecast_table, fact_table, meta
