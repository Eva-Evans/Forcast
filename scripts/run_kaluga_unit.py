#!/usr/bin/env python3
"""Run finál pipeline for one Kaluga subdivision (3 Excel + bulls txt)."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.helpers import month_end  # noqa: E402
from core.lactation_stock_from_events import build_or_load_lactation_monthly  # noqa: E402
from core.tab3_to_final import export_bulls_workbook  # noqa: E402
from etl.bulls import read_bulls_txt  # noqa: E402
from prognoz_vseh_parametrov import (  # noqa: E402
    KALUGA_TREE,
    SUBDIVISION_ALIASES,
    PipelineConfig,
    build_kaluga_events_csv,
    build_kaluga_filter_folder,
    kaluga_trade_rules,
    run_pipeline,
)

DEFAULT_DATA = Path("/Users/sansey2/Desktop/econiva/Калуга, данные от 110726")
DEFAULT_EVENTS = Path("/Users/sansey2/Desktop/econiva/Прогноз_стада/Калуга/События-пo-korovam.csv")


def _find_events_csv() -> Path:
    if DEFAULT_EVENTS.is_file():
        return DEFAULT_EVENTS
    for p in ROOT.parent.glob("**/События-пo-korovam.csv"):
        return p
    for p in ROOT.parent.glob("**/События*.csv"):
        return p
    raise FileNotFoundError("Не найден События-пo-korovam.csv")


def _predict_months_from_anchor(anchor: date, n: int) -> list[tuple[int, int]]:
    y, m = anchor.year, anchor.month
    out: list[tuple[int, int]] = []
    for _ in range(n):
        out.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def run(farm: str, unit: str, data_dir: Path, events_csv: Path, bulls_txt: Path | None) -> Path:
    if farm not in KALUGA_TREE or unit not in KALUGA_TREE[farm]:
        raise ValueError(f"Unknown {farm!r} / {unit!r}")

    safe = unit.replace(" ", "_")
    work = ROOT / "Калуга" / "_runtime" / safe
    work.mkdir(parents=True, exist_ok=True)

    filter_dir = build_kaluga_filter_folder(farm, unit, data_dir, work / f"filter_{safe}")
    events_path = build_kaluga_events_csv(farm, unit, events_csv, work / "События-пo-korovam.xlsx")

    df_events = pd.read_excel(events_path)
    df_events["Дата"] = pd.to_datetime(df_events.get("Дата", df_events.get("Date")), errors="coerce")
    latest_ts = df_events["Дата"].max()
    if pd.isna(latest_ts):
        raise ValueError("Нет дат в событиях для подразделения.")
    latest = latest_ts.date()
    anchor_me = month_end(latest.year, latest.month)
    train_end = pd.Timestamp(anchor_me)
    n_months = 15
    predict_months = _predict_months_from_anchor(anchor_me, n_months)
    month_cols = [f"{y}-{m:02d}" for y, m in predict_months]

    lact_path = work / "lactation_stock.xlsx"
    lact_df = build_or_load_lactation_monthly(
        df_events,
        unit,
        cache_path=lact_path,
        start_floor=(2022, 1),
        end_cap=anchor_me,
    )

    bulls_path = work / "bulls_full.xlsx"
    if bulls_txt and bulls_txt.is_file():
        with bulls_txt.open("rb") as fh:
            bdf = read_bulls_txt(fh)
        if "bull_code" not in bdf.columns and len(bdf.columns):
            bdf = bdf.rename(columns={bdf.columns[0]: "bull_code"})
        if "bull_type" not in bdf.columns:
            bdf["bull_type"] = "H"
        export_bulls_workbook({"bulls": bdf}, bulls_path)
    elif not bulls_path.exists():
        export_bulls_workbook({"bulls": pd.DataFrame()}, bulls_path)

    rules = kaluga_trade_rules(farm, unit)
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])
    out_xlsx = ROOT / "Калуга" / f"прогноз_{safe}.xlsx"

    cfg = PipelineConfig(
        name=f"{farm} / {unit}",
        work_dir=work,
        filter_folder=str(filter_dir.resolve()),
        events_path=events_path,
        events_aux_path=events_path,
        bulls_path=bulls_path,
        lactation_path=lact_path,
        output_xlsx=out_xlsx,
        subdivision_names=subdiv_names,
        kuda_buy_tokens=rules["buy"],
        sold_heifer_kuda_tokens=rules["heifer_sale"],
        sold_bull_kuda_tokens=rules["bull_sale"],
        sales_require_pereezd=True,
        exit_event_types=["ВЫБЫТИЕ", "ПРОДАНА", "SOLD"],
        sold_heifer_dest=rules["heifer_sale"][0] if rules.get("heifer_sale") else unit,
        sold_bull_dest="BULLS",
        train_end_ts=train_end,
        predict_months=predict_months,
        month_cols=month_cols,
        kaluga_farm=farm,
        kaluga_unit=unit,
        kaluga_data_dir=data_dir,
    )

    forecast_table, fact_table = run_pipeline(cfg)
    print("PIPELINE_OK", unit, forecast_table.shape, "->", out_xlsx)
    if fact_table is not None and not fact_table.empty:
        print("fact", fact_table.shape)
    return out_xlsx


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaluga finál run for one unit")
    parser.add_argument("--farm", default="КН Запад")
    parser.add_argument("--unit", default="ЖК Уланово")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--events-csv", type=Path, default=None)
    parser.add_argument("--bulls-txt", type=Path, default=None)
    args = parser.parse_args()

    events = args.events_csv or _find_events_csv()
    bulls = args.bulls_txt
    if bulls is None:
        guess = args.data_dir / "Быки ЖК Уланово КН-Запад.txt"
        if args.unit == "ЖК Уланово" and guess.is_file():
            bulls = guess
        else:
            for f in args.data_dir.glob("Быки*.txt"):
                if "Улан" in f.name or args.unit.replace("ЖК ", "") in f.name:
                    bulls = f
                    break

    if not args.data_dir.is_dir():
        print("FAIL: data-dir missing:", args.data_dir, file=sys.stderr)
        return 1

    try:
        run(args.farm, args.unit, args.data_dir, events, bulls)
    except Exception as exc:
        print("FAIL:", exc, file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
