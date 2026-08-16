#!/usr/bin/env python3
"""Checkpoint audit: row counts after each filter step (Kaluga unit)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prognoz_vseh_parametrov import (  # noqa: E402
    SUBDIVISION_ALIASES,
    _filter_subdivision,
    _normalize_kaluga_event,
    build_kaluga_events_csv,
    build_kaluga_filter_folder,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def audit(farm: str, unit: str, data_dir: Path, events_csv: Path) -> None:
    aliases = SUBDIVISION_ALIASES.get(unit, [unit])
    _log(f"\n=== Аудит: {farm} / {unit} ===")
    _log(f"Алиасы Столбец1: {aliases}")
    _log(f"data_dir: {data_dir}")
    _log(f"events_csv: {events_csv}")

    calv_path = data_dir / "Отелы плюс родившиеся Калуга DZ 120726.xlsx"
    sem_path = data_dir / "Осеменения Калуга DZ 120726.xlsx"
    disp_path = data_dir / "Выбытие + Запуск Калуга DZ 120726.xlsx"
    for label, p in [("отёлы", calv_path), ("осеменения", sem_path), ("выбытие+запуск", disp_path)]:
        if not p.is_file():
            _log(f"  MISSING {label}: {p}")
            continue
        raw = pd.read_excel(p)
        _log(f"\n[{label}] файл {p.name}")
        _log(f"  строк в Excel (всего): {len(raw):,}")
        if "Source.Name" not in raw.columns or "Столбец1" not in raw.columns:
            _log(f"  колонки meta: Source.Name={'Source.Name' in raw.columns}, Столбец1={'Столбец1' in raw.columns}")
            continue
        norm = _normalize_kaluga_event(raw)
        filt = _filter_subdivision(norm, farm, unit)
        _log(f"  после normalize: {len(norm):,}")
        _log(f"  после фильтра Source.Name={farm!r} + Столбец1∈aliases: {len(filt):,}")
        if len(filt) == 0:
            top_farm = norm["Source.Name"].astype(str).str.strip().value_counts().head(5)
            top_unit = norm["Столбец1"].astype(str).str.strip().value_counts().head(8)
            _log(f"  (подсказка) топ Source.Name: {dict(top_farm)}")
            _log(f"  (подсказка) топ Столбец1: {dict(top_unit)}")

    if events_csv.is_file():
        _log(f"\n[события CSV] {events_csv.name}")
        total = 0
        matched = 0
        for chunk in pd.read_csv(events_csv, chunksize=200_000, low_memory=False):
            total += len(chunk)
            if "Столбец1" not in chunk.columns:
                continue
            m = chunk["Source.Name"].astype(str).str.strip() == farm.strip()
            m &= chunk["Столбец1"].astype(str).str.strip().isin(aliases)
            matched += int(m.sum())
        _log(f"  строк прочитано (всего в CSV): {total:,}")
        _log(f"  после фильтра {farm} / {unit}: {matched:,}")
    else:
        _log(f"\n[события CSV] NOT FOUND: {events_csv}")

    work = ROOT / "Калуга" / "_runtime" / unit.replace(" ", "_") / "audit_build"
    work.mkdir(parents=True, exist_ok=True)
    _log(f"\n[build] filter_folder -> {work / 'filter'}")
    try:
        fdir = build_kaluga_filter_folder(farm, unit, data_dir, work / "filter")
        xlsx_files = sorted(fdir.glob("*.xlsx"))
        _log(f"  годовых файлов в filter: {len(xlsx_files)}")
        for xf in xlsx_files:
            n = len(pd.read_excel(xf))
            _log(f"    {xf.name}: {n:,} строк")
    except Exception as exc:
        _log(f"  build_kaluga_filter_folder ERROR: {exc}")

    _log(f"\n[build] events xlsx -> {work / 'events.xlsx'}")
    try:
        ev_path = build_kaluga_events_csv(farm, unit, events_csv, work / "events.xlsx")
        ev = pd.read_excel(ev_path)
        _log(f"  строк в События-po-korovam.xlsx: {len(ev):,}")
    except Exception as exc:
        _log(f"  build_kaluga_events_csv ERROR: {exc}")

    _log("\n=== Аудит фильтрации завершён (finál-пайплайн не запускался) ===")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", default="КН Запад")
    parser.add_argument("--unit", default="ЖК Уланово")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/sansey2/Desktop/econiva/Калуга, данные от 110726"),
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/sansey2/Desktop/econiva/Прогноз_стада/Калуга/События-po-korovam.csv"),
    )
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        print("FAIL: data-dir missing", args.data_dir, file=sys.stderr)
        return 1
    audit(args.farm, args.unit, args.data_dir, args.events_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
