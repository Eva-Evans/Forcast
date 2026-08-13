from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prognoz_vseh_parametrov import SUBDIVISION_ALIASES

CACHE_VERSION = 1
_LACT_GROUPS = ("L0", "L1", "L2", "L3", "L4", "L5+")


def _norm_event(s: object) -> str:
    return str(s or "").strip().upper().replace("Ё", "Е")


def _is_calving_row(row: pd.Series) -> bool:
    ev = _norm_event(row.get("Событие", row.get("Event", "")))
    tipo = _norm_event(row.get("тип_файла", ""))
    if ev == "ОТЕЛ" or "ОТEL" in ev or "CALV" in ev or "РОЖД" in ev:
        return True
    if tipo in {"ОТЕЛ", "ОТEL"}:
        return True
    return False


def _is_culling_row(row: pd.Series) -> bool:
    ev = _norm_event(row.get("Событие", row.get("Event", "")))
    tipo = _norm_event(row.get("тип_файла", ""))
    if ev in {"ЗАПУСК", "DRY", "ОСЕМЕН", "ОСЕМЕНЕНИЕ"}:
        return False
    if "ВЫБЫТ" in ev or ev in {"SOLD", "ПРОДАНА", "ПРОДАН"}:
        return True
    if tipo == "ВЫБЫТИЕ":
        return True
    if tipo == "ЗАПУСК+ВЫБЫТИЕ" and ev not in {"ЗАПУСК", "DRY", ""}:
        return True
    return False


def _lact_group(lact: float | int | None) -> str:
    if lact is None or (isinstance(lact, float) and pd.isna(lact)):
        return "L0"
    try:
        v = int(float(lact))
    except (TypeError, ValueError):
        return "L0"
    if v <= 0:
        return "L0"
    if v == 1:
        return "L1"
    if v == 2:
        return "L2"
    if v == 3:
        return "L3"
    if v == 4:
        return "L4"
    return "L5+"


def _lact_to_group_idx(lact: np.ndarray) -> np.ndarray:
    """Map lactation numbers to column indices 0..5 (L0..L5+)."""
    out = np.full(lact.shape, 5, dtype=np.int8)
    out[lact <= 0] = 0
    out[lact == 1] = 1
    out[lact == 2] = 2
    out[lact == 3] = 3
    out[lact == 4] = 4
    return out


def _cow_status_on_date(cow_events: pd.DataFrame, target_date: pd.Timestamp) -> int | None:
    """Reference implementation kept for tests / parity checks."""
    events_before = cow_events[cow_events["Дата"] <= target_date]
    if events_before.empty:
        return None

    culled = events_before[events_before.apply(_is_culling_row, axis=1)]
    if not culled.empty and pd.Timestamp(culled.iloc[-1]["Дата"]) <= target_date:
        return None

    calvings = events_before[events_before.apply(_is_calving_row, axis=1)]
    if not calvings.empty:
        last_calving = calvings.iloc[-1]
        lact = last_calving.get("LACT", last_calving.get("Lact"))
        try:
            return int(float(lact))
        except (TypeError, ValueError):
            return 0
    return 0


def _filter_events_for_unit(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    out = df.copy()
    if "Столбец1" not in out.columns:
        return out
    aliases = {str(a).strip() for a in SUBDIVISION_ALIASES.get(unit, [unit])}
    col = out["Столбец1"].astype(str).str.strip()
    return out.loc[col.isin(aliases)].copy()


def _month_range_from_events(
    df: pd.DataFrame,
    *,
    start_floor: tuple[int, int] = (2022, 1),
    end_cap: date | None = None,
) -> list[tuple[int, int]]:
    if df.empty or "Дата" not in df.columns:
        y1, m1 = start_floor
        if end_cap is not None:
            y2, m2 = end_cap.year, end_cap.month
        else:
            y2, m2 = y1, m1
    else:
        dmin = pd.Timestamp(df["Дата"].min()).normalize()
        dmax = pd.Timestamp(df["Дата"].max()).normalize()
        y1, m1 = start_floor
        if (dmin.year, dmin.month) > start_floor:
            y1, m1 = dmin.year, dmin.month
        if end_cap is not None:
            y2, m2 = end_cap.year, end_cap.month
        else:
            y2, m2 = dmax.year, dmax.month

    months: list[tuple[int, int]] = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def _vectorized_event_flags(work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ev = (
        work.get("Событие", work.get("Event", pd.Series("", index=work.index)))
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("Ё", "Е", regex=False)
    )
    tipo = (
        work.get("тип_файла", pd.Series("", index=work.index))
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("Ё", "Е", regex=False)
    )
    is_calving = (
        (ev == "ОТЕЛ")
        | ev.str.contains("ОТEL", na=False)
        | ev.str.contains("CALV", na=False)
        | ev.str.contains("РОЖД", na=False)
        | tipo.isin(["ОТЕЛ", "ОТEL"])
    ).to_numpy(dtype=bool)
    not_cull_ev = ~ev.isin(["ЗАПУСК", "DRY", "ОСЕМЕН", "ОСЕМЕНЕНИЕ"])
    is_culling = (
        not_cull_ev
        & (
            ev.str.contains("ВЫБЫТ", na=False)
            | ev.isin(["SOLD", "ПРОДАНА", "ПРОДАН"])
            | (tipo == "ВЫБЫТИЕ")
            | ((tipo == "ЗАПУСК+ВЫБЫТИЕ") & ~ev.isin(["ЗАПУСК", "DRY", ""]))
        )
    ).to_numpy(dtype=bool)
    return is_calving, is_culling


def _calving_lact_values(work: pd.DataFrame) -> np.ndarray:
    lact = pd.Series(np.nan, index=work.index, dtype="float64")
    for col in ("LACT", "Lact"):
        if col in work.columns:
            lact = lact.fillna(pd.to_numeric(work[col], errors="coerce"))
    return lact.fillna(0).astype(np.int32).to_numpy()


def _cow_slice_bounds(ids: np.ndarray) -> np.ndarray:
    if len(ids) == 0:
        return np.array([0], dtype=np.int64)
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    return np.concatenate(([0], changes, [len(ids)])).astype(np.int64)


def _accumulate_cow_counts(
    counts: np.ndarray,
    month_ends: np.ndarray,
    dates: np.ndarray,
    is_calving: np.ndarray,
    is_culling: np.ndarray,
    row_lact: np.ndarray,
) -> None:
    calv_dates = dates[is_calving]
    cull_dates = dates[is_culling]
    calv_lact = row_lact[is_calving]

    n_months = len(month_ends)
    culled = np.zeros(n_months, dtype=bool)
    if len(cull_dates):
        ci = np.searchsorted(cull_dates, month_ends, side="right") - 1
        valid = ci >= 0
        culled[valid] = cull_dates[ci[valid]] <= month_ends[valid]

    lact_at_month = np.zeros(n_months, dtype=np.int32)
    if len(calv_dates):
        ki = np.searchsorted(calv_dates, month_ends, side="right") - 1
        has_calv = ki >= 0
        lact_at_month[has_calv] = calv_lact[ki[has_calv]]

    active = ~culled
    if not active.any():
        return

    group_idx = _lact_to_group_idx(lact_at_month)
    active_idx = np.flatnonzero(active)
    np.add.at(counts, (active_idx, group_idx[active_idx]), 1)


def _prepare_events(df_events: pd.DataFrame, unit: str) -> pd.DataFrame:
    df = _filter_events_for_unit(df_events, unit)
    if df.empty:
        raise ValueError(f"Нет событий для подразделения «{unit}».")

    work = df.copy()
    work["Дата"] = pd.to_datetime(work.get("Дата", work.get("Date")), errors="coerce")
    work = work[work["Дата"].notna()].copy()
    if "ID" not in work.columns:
        work["ID"] = ""
    if "REG" not in work.columns:
        work["REG"] = ""
    work["ID"] = work["ID"].astype(str).fillna("")
    mask_id = (work["ID"] == "") | (work["ID"].str.lower() == "nan")
    work.loc[mask_id, "ID"] = work.loc[mask_id, "REG"].astype(str)
    work = work[work["ID"].astype(str).str.strip() != ""].copy()
    return work.sort_values(["ID", "Дата"], kind="mergesort")


def _month_end_timestamps(all_months: list[tuple[int, int]]) -> np.ndarray:
    return np.array(
        [
            (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).to_datetime64()
            for y, m in all_months
        ],
        dtype="datetime64[ns]",
    )


def build_lactation_monthly_from_events(
    df_events: pd.DataFrame,
    unit: str,
    *,
    start_floor: tuple[int, int] = (2022, 1),
    end_cap: date | None = None,
) -> pd.DataFrame:
    """
    Помесячное поголовье L0…L5+ на последний день месяца (логика ноутбука, фильтр подразделения Калуга/Высокое).
    """
    work = _prepare_events(df_events, unit)
    all_months = _month_range_from_events(work, start_floor=start_floor, end_cap=end_cap)
    month_ends = _month_end_timestamps(all_months)

    is_calving_all, is_culling_all = _vectorized_event_flags(work)
    dates = work["Дата"].to_numpy(dtype="datetime64[ns]")
    ids = work["ID"].to_numpy()
    row_lact = _calving_lact_values(work)
    bounds = _cow_slice_bounds(ids)
    counts = np.zeros((len(all_months), len(_LACT_GROUPS)), dtype=np.int64)

    for s, e in zip(bounds[:-1], bounds[1:]):
        _accumulate_cow_counts(
            counts,
            month_ends,
            dates[s:e],
            is_calving_all[s:e],
            is_culling_all[s:e],
            row_lact[s:e],
        )

    rows: list[dict] = []
    for i, (year, month) in enumerate(all_months):
        c = counts[i]
        row = {
            "год": year,
            "месяц": month,
            "L0": int(c[0]),
            "L1": int(c[1]),
            "L2": int(c[2]),
            "L3": int(c[3]),
            "L4": int(c[4]),
            "L5+": int(c[5]),
        }
        row["всего"] = sum(row[k] for k in _LACT_GROUPS)
        rows.append(row)

    return pd.DataFrame(rows)


def lactation_cache_fingerprint(
    df_events: pd.DataFrame,
    unit: str,
    *,
    start_floor: tuple[int, int] = (2022, 1),
    end_cap: date | None = None,
) -> str:
    """Stable hash of inputs that affect lactation monthly stock."""
    work = _prepare_events(df_events, unit)
    key_cols = [c for c in ("ID", "Дата", "Событие", "Event", "тип_файла", "LACT", "Lact", "REG") if c in work.columns]
    if not key_cols:
        payload = b""
    else:
        subset = work[key_cols].sort_values(key_cols, kind="mergesort")
        payload = pd.util.hash_pandas_object(subset, index=False).to_numpy(dtype=np.uint64).tobytes()
    meta = json.dumps(
        {
            "v": CACHE_VERSION,
            "unit": unit,
            "start_floor": list(start_floor),
            "end_cap": end_cap.isoformat() if end_cap else None,
            "rows": len(work),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(meta + payload).hexdigest()


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + ".meta.json")


def load_cached_lactation(cache_path: Path, fingerprint: str) -> pd.DataFrame | None:
    meta_path = _cache_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("fingerprint") != fingerprint or meta.get("v") != CACHE_VERSION:
        return None
    try:
        return pd.read_excel(cache_path)
    except (OSError, ValueError):
        return None


def save_lactation_cache(df: pd.DataFrame, cache_path: Path, fingerprint: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(cache_path, index=False)
    meta = {
        "v": CACHE_VERSION,
        "fingerprint": fingerprint,
        "rows": len(df),
    }
    _cache_meta_path(cache_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_or_load_lactation_monthly(
    df_events: pd.DataFrame,
    unit: str,
    *,
    cache_path: Path | None = None,
    start_floor: tuple[int, int] = (2022, 1),
    end_cap: date | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build lactation monthly stock or load from cache when events fingerprint matches.
    """
    fingerprint = lactation_cache_fingerprint(
        df_events,
        unit,
        start_floor=start_floor,
        end_cap=end_cap,
    )
    if cache_path is not None and not force_rebuild:
        cached = load_cached_lactation(cache_path, fingerprint)
        if cached is not None:
            print(f"♻️ Лактации из кэша: {cache_path} ({len(cached)} мес.)")
            return cached

    print(f"⏳ Считаем лактации по событиям ({unit})…")
    df = build_lactation_monthly_from_events(
        df_events,
        unit,
        start_floor=start_floor,
        end_cap=end_cap,
    )
    if cache_path is not None:
        save_lactation_cache(df, cache_path, fingerprint)
        print(f"💾 Лактации сохранены: {cache_path}")
    return df
