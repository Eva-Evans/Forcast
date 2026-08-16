from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.tab3_to_final import (
    build_events_all,
    build_events_workbook,
    export_bulls_workbook,
    export_tab3_to_filter_folder,
)
from prognoz_vseh_parametrov import SUBDIVISION_ALIASES

ARTIFACTS_CACHE_VERSION = 3

_TABLE_KEY_COLS: dict[str, list[str]] = {
    "calv": ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"],
    "ins": ["reg", "lact", "dim_age", "event_date", "bull", "result"],
    "dry": ["reg", "dim", "event_date", "move_reason"],
    "disp": ["reg", "event_date", "disposal_reason"],
    "bulls": ["bull_code", "bull_type"],
}


def _primary_subdiv_col(unit: str) -> str:
    return SUBDIVISION_ALIASES.get(unit, [unit])[0]


def tables_input_fingerprint(
    tables: dict[str, pd.DataFrame],
    *,
    unit: str,
    farm: str,
    train_end: date,
) -> str:
    """Stable hash of tab3 tables + train_end used to build filter/events artifacts."""
    parts: list[bytes] = []
    meta = json.dumps(
        {
            "v": ARTIFACTS_CACHE_VERSION,
            "unit": unit,
            "farm": farm,
            "train_end": train_end.isoformat(),
        },
        sort_keys=True,
    ).encode()
    parts.append(meta)
    for key in ("calv", "ins", "dry", "disp", "bulls"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            parts.append(f"{key}:0".encode())
            continue
        cols = [c for c in _TABLE_KEY_COLS[key] if c in df.columns]
        if not cols:
            parts.append(f"{key}:0".encode())
            continue
        subset = df[cols].sort_values(cols, kind="mergesort")
        parts.append(pd.util.hash_pandas_object(subset, index=False).to_numpy(dtype=np.uint64).tobytes())
    h = hashlib.sha256()
    for chunk in parts:
        h.update(chunk)
    return h.hexdigest()


def _meta_path(work: Path) -> Path:
    return work / "pipeline_artifacts.meta.json"


def _artifacts_ready(work: Path, filter_dir: Path, events_xlsx: Path, bulls_xlsx: Path) -> bool:
    if not filter_dir.is_dir():
        return False
    if not any(filter_dir.glob("Отелы_*.xlsx")):
        return False
    return events_xlsx.is_file() and bulls_xlsx.is_file()


def _load_meta(work: Path) -> dict | None:
    path = _meta_path(work)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(work: Path, fingerprint: str, *, unit: str, farm: str, train_end: date) -> None:
    work.mkdir(parents=True, exist_ok=True)
    payload = {
        "v": ARTIFACTS_CACHE_VERSION,
        "fingerprint": fingerprint,
        "unit": unit,
        "farm": farm,
        "train_end": train_end.isoformat(),
    }
    _meta_path(work).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PipelineArtifactPaths:
    filter_dir: Path
    events_xlsx: Path
    events_csv: Path
    bulls_xlsx: Path


def _events_for_unit(tables: dict[str, pd.DataFrame], unit: str, farm: str) -> pd.DataFrame:
    df_events = build_events_all(tables)
    df_events["Столбец1"] = _primary_subdiv_col(unit)
    df_events["Source.Name"] = farm
    return df_events


def ensure_pipeline_artifacts(
    *,
    work: Path,
    filter_dir: Path,
    tables: dict[str, pd.DataFrame],
    unit: str,
    farm: str,
    train_end: pd.Timestamp,
    force_rebuild: bool = False,
) -> tuple[PipelineArtifactPaths, pd.DataFrame]:
    """
    Build or reuse filter_*, events_cows.xlsx/csv, bulls_full.xlsx when fingerprint matches.
    """
    train_end_date = train_end.date() if hasattr(train_end, "date") else train_end
    fingerprint = tables_input_fingerprint(
        tables,
        unit=unit,
        farm=farm,
        train_end=train_end_date,
    )
    events_xlsx = work / "events_cows.xlsx"
    events_csv = work / "events_cows.csv"
    bulls_xlsx = work / "bulls_full.xlsx"
    paths = PipelineArtifactPaths(
        filter_dir=filter_dir,
        events_xlsx=events_xlsx,
        events_csv=events_csv,
        bulls_xlsx=bulls_xlsx,
    )

    meta = _load_meta(work)
    if (
        not force_rebuild
        and meta
        and meta.get("fingerprint") == fingerprint
        and meta.get("v") == ARTIFACTS_CACHE_VERSION
        and _artifacts_ready(work, filter_dir, events_xlsx, bulls_xlsx)
    ):
        print(f"♻️ Filter/events из кэша ({unit})")
        return paths, _events_for_unit(tables, unit, farm)

    print(f"⏳ Собираем filter/events ({unit})…")
    export_tab3_to_filter_folder(tables, filter_dir, train_end_cutoff=train_end)
    build_events_workbook(tables, unit, farm, events_xlsx, csv_path=events_csv)
    export_bulls_workbook(tables, bulls_xlsx)
    _save_meta(work, fingerprint, unit=unit, farm=farm, train_end=train_end_date)
    return paths, _events_for_unit(tables, unit, farm)
