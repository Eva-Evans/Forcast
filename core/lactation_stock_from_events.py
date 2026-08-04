from __future__ import annotations

from datetime import date

import pandas as pd

from prognoz_vseh_parametrov import SUBDIVISION_ALIASES


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


def _cow_status_on_date(cow_events: pd.DataFrame, target_date: pd.Timestamp) -> int | None:
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
    work = work.sort_values(["ID", "Дата"], kind="mergesort")

    all_months = _month_range_from_events(work, start_floor=start_floor, end_cap=end_cap)
    groups = ["L0", "L1", "L2", "L3", "L4", "L5+"]
    monthly_status: dict[tuple[int, int], dict[str, int]] = {}

    for cow_id, cow_events in work.groupby("ID", sort=False):
        cow_events = cow_events.sort_values("Дата", kind="mergesort")
        for year, month in all_months:
            last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
            lact = _cow_status_on_date(cow_events, last_day)
            if lact is None:
                continue
            g = _lact_group(lact)
            key = (year, month)
            if key not in monthly_status:
                monthly_status[key] = {x: 0 for x in groups}
            monthly_status[key][g] += 1

    rows: list[dict] = []
    for year, month in all_months:
        counts = monthly_status.get((year, month), {g: 0 for g in groups})
        row = {
            "год": year,
            "месяц": month,
            "L0": counts.get("L0", 0),
            "L1": counts.get("L1", 0),
            "L2": counts.get("L2", 0),
            "L3": counts.get("L3", 0),
            "L4": counts.get("L4", 0),
            "L5+": counts.get("L5+", 0),
        }
        row["всего"] = sum(row[k] for k in groups)
        rows.append(row)

    return pd.DataFrame(rows)
