# Снимок стада ЖК Высокое: _actual_nonbirth_snapshot_from_tables
# Запуск: python snapshot_zhk_vysokoe.py  или %run snapshot_zhk_vysokoe.py в ноутбуке

import pandas as pd
from datetime import date

# =============================================================================
# Загрузка сырых Excel → таблицы с колонками reg / event_date / … (как в backtest)
# =============================================================================

FOLDER = "фильтр_ЖК_Высокое"
YEARS = [2022, 2023, 2024, 2025]

BASELINE_2025_12_31 = {
    "Т 0-2": 210, "Т 2-6": 584, "Т 6-12": 3, "Т 12-18": 0, "Т 18+": 0,
    "Б 0-2": 26, "Б 2-6": 0, "Б 6-12": 0, "Б 18+": 0,
}


def _norm_id(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "nat"):
        return ""
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    return s


def _norm_event_type(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip().upper().replace("Ё", "Е")


def _norm_sex_marker_backtest(val) -> str:
    s = str(val).strip().upper() if val is not None and not (isinstance(val, float) and pd.isna(val)) else ""
    if s in ("F", "Ж", "FEMALE", "ЖЕНСКИЙ", "2"):
        return "F"
    if s in ("M", "М", "MALE", "МУЖСКОЙ", "1"):
        return "M"
    return ""


def calendar_months(birth_date, ref_date) -> int | None:
    """Календарные полные месяцы (как в ручном пересчёте 29.11 → 1 мес. на 29.12)."""
    if pd.isna(birth_date) or pd.isna(ref_date):
        return None
    b = pd.Timestamp(birth_date)
    r = pd.Timestamp(ref_date)
    m = (r.year - b.year) * 12 + (r.month - b.month)
    if r.day < b.day:
        m -= 1
    return m if m >= 0 else None


def _heifer_group_label(months: int | None, prefix: str) -> str | None:
    """prefix: 'Т' (телки) или 'Н' (нетели)."""
    if months is None:
        return None
    if months < 2:
        return f"{prefix} 0-2"
    if months < 6:
        return f"{prefix} 2-6"
    if months < 12:
        return f"{prefix} 6-12"
    if months < 18:
        return f"{prefix} 12-18"
    return f"{prefix} 18+"


def _young_group_label(sex: str, months: int | None) -> str | None:
    if months is None:
        return None
    if sex == "F":
        return _heifer_group_label(months, "Т")
    if sex == "M":
        if months < 2:
            return "Б 0-2"
        if months < 6:
            return "Б 2-6"
        if months < 12:
            return "Б 6-12"
        return "Б 18+"
    return None


def _read_yearly(folder: str, prefix: str, years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        try:
            parts.append(pd.read_excel(f"{folder}/{prefix}_{y}.xlsx"))
        except FileNotFoundError:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_vysokoe_raw_tables(folder: str = FOLDER, years: list[int] | None = None) -> dict[str, pd.DataFrame]:
    years = years or YEARS
    return {
        "calv": _read_yearly(folder, "Отелы", years),
        "ins": _read_yearly(folder, "Осеменения", years),
        "dry": _read_yearly(folder, "Запуск", years),
        "disp": _read_yearly(folder, "Выбытие", years),
    }


def excel_to_backtest_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Приводит Excel ЖК Высокое к колонкам reg, event_date, …"""

    calv_raw = raw.get("calv", pd.DataFrame())
    if not calv_raw.empty:
        calv_df = pd.DataFrame({
            "reg": calv_raw.get("REG", pd.Series(dtype=object)).map(_norm_id),
            "event_date": pd.to_datetime(calv_raw.get("Дата"), errors="coerce"),
            "birth_date": pd.to_datetime(calv_raw.get("BDAT"), errors="coerce"),
            "event_type": calv_raw.get("Событие", calv_raw.get("Event", pd.Series(dtype=object))),
            "sex": calv_raw.get("GNDR", calv_raw.get("Пол", pd.Series(dtype=object))),
            "mother_reg": calv_raw.get("DREG1", calv_raw.get("DREG", pd.Series(dtype=object))),
        })
    else:
        calv_df = pd.DataFrame(columns=["reg", "event_date", "birth_date", "event_type", "sex", "mother_reg"])

    def _simple_event(df_raw: pd.DataFrame, with_lact: bool = False, with_result: bool = False) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            cols = ["reg", "event_date"]
            if with_lact:
                cols.append("lact")
            if with_result:
                cols.append("result")
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame({
            "reg": df_raw.get("REG", pd.Series(dtype=object)).map(_norm_id),
            "event_date": pd.to_datetime(df_raw.get("Дата"), errors="coerce"),
        })
        if with_lact:
            out["lact"] = pd.to_numeric(df_raw.get("LACT"), errors="coerce")
        if with_result:
            res = df_raw.get("R", df_raw.get("Result", pd.Series(dtype=object)))
            out["result"] = res.astype(str).str.strip().str.upper()
        return out

    return {
        "calv_df": calv_df,
        "ins_df": _simple_event(raw.get("ins", pd.DataFrame()), with_lact=True, with_result=True),
        "dry_df": _simple_event(raw.get("dry", pd.DataFrame())),
        "disp_df": _simple_event(raw.get("disp", pd.DataFrame())),
    }


# =============================================================================
# Ваши функции (логика backtest; возраст молодняка — календарные месяцы → Т/Б группы)
# =============================================================================

def actual_birth_stats_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    """Рождения в месяце month_end_date (упрощённо: строки РОЖДЕН в отёлах)."""
    as_of = pd.Timestamp(as_of_date or month_end_date).normalize()
    month_end = pd.Timestamp(month_end_date).normalize()
    month_start = month_end.replace(day=1)

    calv = calv_df.copy() if isinstance(calv_df, pd.DataFrame) else pd.DataFrame()
    if calv.empty:
        return {"ожидаемые_тёлочки": 0.0, "ожидаемые_бычки": 0.0, "всего_рождений": 0.0}

    calv["event_date_n"] = pd.to_datetime(calv.get("event_date"), errors="coerce").dt.normalize()
    calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).map(_norm_event_type)
    calv["sex_norm"] = calv.get("sex", pd.Series(dtype=object)).map(_norm_sex_marker_backtest)
    calv = calv[(calv["event_date_n"].notna()) & (calv["event_date_n"] <= as_of)]

    born = calv[calv["event_type_n"].isin(["РОЖД", "РОЖД.", "РОЖДЕН"])]
    born = born[(born["event_date_n"] >= month_start) & (born["event_date_n"] <= month_end)]

    f = float((born["sex_norm"] == "F").sum())
    m = float((born["sex_norm"] == "M").sum())
    return {"ожидаемые_тёлочки": f, "ожидаемые_бычки": m, "всего_рождений": f + m}


def _actual_birth_stats_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=as_of_date)


def _actual_nonbirth_snapshot_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    dry_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    as_of_date: date,
) -> dict[str, float]:
    out = {
        "Дойные коровы": 0.0,
        "Сухостойные коровы": 0.0,
        "Т 0-2": 0.0,
        "Т 2-6": 0.0,
        "Т 6-12": 0.0,
        "Т 12-18": 0.0,
        "Т 18+": 0.0,
        "Б 0-2": 0.0,
        "Б 2-6": 0.0,
        "Б 6-12": 0.0,
        "Б 18+": 0.0,
        "Нетели": 0.0,
    }
    as_of_ts = pd.Timestamp(as_of_date).normalize()

    disp = disp_df.copy() if isinstance(disp_df, pd.DataFrame) else pd.DataFrame()
    if not disp.empty:
        disp["event_date_n"] = pd.to_datetime(disp.get("event_date"), errors="coerce").dt.normalize()
        disp["reg_s"] = disp.get("reg", pd.Series(dtype=object)).map(_norm_id)
        disp = disp[(disp["event_date_n"].notna()) & (disp["event_date_n"] <= as_of_ts) & (disp["reg_s"] != "")]
    disposed: set[str] = set(disp["reg_s"].astype(str).tolist()) if not disp.empty else set()

    ins = ins_df.copy() if isinstance(ins_df, pd.DataFrame) else pd.DataFrame()
    if not ins.empty:
        ins["event_date_n"] = pd.to_datetime(ins.get("event_date"), errors="coerce").dt.normalize()
        ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(_norm_id)
        ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
        ins = ins[(ins["event_date_n"].notna()) & (ins["event_date_n"] <= as_of_ts) & (ins["reg_s"] != "")]
    cows_from_ins = set(ins.loc[ins["lact_n"] > 0, "reg_s"].astype(str).tolist()) if not ins.empty else set()
    neteli_from_ins = set(ins.loc[ins["lact_n"] <= 0, "reg_s"].astype(str).tolist()) if not ins.empty else set()

    dry = dry_df.copy() if isinstance(dry_df, pd.DataFrame) else pd.DataFrame()
    if not dry.empty:
        dry["event_date_n"] = pd.to_datetime(dry.get("event_date"), errors="coerce").dt.normalize()
        dry["reg_s"] = dry.get("reg", pd.Series(dtype=object)).map(_norm_id)
        dry = dry[(dry["event_date_n"].notna()) & (dry["event_date_n"] <= as_of_ts) & (dry["reg_s"] != "")]
    if not dry.empty:
        last_dry = (
            dry.sort_values(["reg_s", "event_date_n"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
            .set_index("reg_s")["event_date_n"]
            .to_dict()
        )
    else:
        last_dry = {}

    calv = calv_df.copy() if isinstance(calv_df, pd.DataFrame) else pd.DataFrame()
    if not calv.empty:
        calv["event_date_n"] = pd.to_datetime(calv.get("event_date"), errors="coerce").dt.normalize()
        calv["birth_date_n"] = pd.to_datetime(calv.get("birth_date"), errors="coerce").dt.normalize()
        calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).map(_norm_event_type)
        calv["reg_s"] = calv.get("reg", pd.Series(dtype=object)).map(_norm_id)
        calv["mother_reg_s"] = calv.get("mother_reg", pd.Series(dtype=object)).map(_norm_id)
        calv["sex_norm"] = calv.get("sex", pd.Series(dtype=object)).map(_norm_sex_marker_backtest)
        calv = calv[(calv["event_date_n"].notna()) & (calv["event_date_n"] <= as_of_ts)]
        born = calv.loc[calv["event_type_n"].isin(["РОЖД", "РОЖД.", "РОЖДЕН"])].copy()
    else:
        born = pd.DataFrame()

    if not born.empty:
        born["birth_dt_n"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
    else:
        born["birth_dt_n"] = pd.NaT

    mother_with_calv = (
        set(born.loc[born["mother_reg_s"] != "", "mother_reg_s"].astype(str).tolist()) if not born.empty else set()
    )
    if not born.empty:
        last_calv_by_mother = (
            born.loc[born["mother_reg_s"] != "", ["mother_reg_s", "event_date_n"]]
            .sort_values(["mother_reg_s", "event_date_n"], kind="mergesort")
            .drop_duplicates(subset=["mother_reg_s"], keep="last")
            .set_index("mother_reg_s")["event_date_n"]
            .to_dict()
        )
    else:
        last_calv_by_mother = {}

    cow_candidates: set[str] = set()
    cow_candidates |= cows_from_ins
    cow_candidates |= set(last_dry.keys())
    cow_candidates |= mother_with_calv
    cows_alive = {reg for reg in cow_candidates if reg and reg not in disposed}

    dry_count = 0
    for reg in cows_alive:
        dry_dt = last_dry.get(reg)
        if dry_dt is None or pd.isna(dry_dt):
            continue
        calv_dt = last_calv_by_mother.get(reg)
        if calv_dt is None or pd.isna(calv_dt):
            dry_count += 1
        elif pd.Timestamp(dry_dt) > pd.Timestamp(calv_dt):
            dry_count += 1
    doy_count = max(0, len(cows_alive) - dry_count)

    neteli_alive = {
        reg
        for reg in neteli_from_ins
        if reg and reg not in disposed and reg not in cows_alive and reg not in mother_with_calv
    }

    calf_excluded = set(cows_alive) | set(neteli_alive)
    if not born.empty:
        calves = born.loc[(born["reg_s"] != "") & (born["sex_norm"].isin(["F", "M"])), ["reg_s", "sex_norm", "birth_dt_n"]].copy()
    else:
        calves = pd.DataFrame(columns=["reg_s", "sex_norm", "birth_dt_n"])

    if not calves.empty:
        calves = calves[calves["birth_dt_n"].notna()].copy()
        calves = calves[~calves["reg_s"].astype(str).isin(disposed)]
        calves = calves[~calves["reg_s"].astype(str).isin(calf_excluded)]
        calves = calves[pd.to_datetime(calves["birth_dt_n"]) <= as_of_ts]

        for _, row in calves.iterrows():
            months = calendar_months(row["birth_dt_n"], as_of_ts)
            label = _young_group_label(row["sex_norm"], months)
            if label and label in out:
                out[label] += 1.0

    out["Дойные коровы"] = float(doy_count)
    out["Сухостойные коровы"] = float(dry_count)
    out["Нетели"] = float(len(neteli_alive))
    return out


def print_snapshot(title: str, snap: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    for k in ["Дойные коровы", "Сухостойные коровы", "Нетели"]:
        print(f"  {k:22s}: {int(snap.get(k, 0))}")
    print()
    young_keys = ["Т 0-2", "Т 2-6", "Т 6-12", "Т 12-18", "Т 18+", "Б 0-2", "Б 2-6", "Б 6-12", "Б 18+"]
    for k in young_keys:
        line = f"  {k:10s}: {int(snap.get(k, 0))}"
        if baseline and k in baseline:
            d = int(snap.get(k, 0)) - int(baseline[k])
            if d:
                line += f"   (эталон {baseline[k]}, Δ {d:+d})"
        print(line)
    t = sum(snap.get(k, 0) for k in young_keys[:5])
    b = sum(snap.get(k, 0) for k in young_keys[5:])
    n = snap.get("Нетели", 0)
    print(f"\n  Итого нетели: {int(n)}   телки (Т): {int(t)}   бычки (Б): {int(b)}")
    print(f"  Молодняк (Т+Б): {int(t + b)}   Т+Б+нетели: {int(t + b + n)}")


YOUNG_AND_NETELI_KEYS = [
    "Нетели",
    "Т 0-2", "Т 2-6", "Т 6-12", "Т 12-18", "Т 18+",
    "Б 0-2", "Б 2-6", "Б 6-12", "Б 18+",
]


def month_end_date(year: int, month: int) -> date:
    from calendar import monthrange
    return date(year, month, monthrange(year, month)[1])


def iter_months_reverse(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) >= (end_year, end_month):
        yield y, m
        m -= 1
        if m == 0:
            m = 12
            y -= 1


def monthly_young_neteli_history(
    tables: dict[str, pd.DataFrame],
    start_year: int = 2025,
    start_month: int = 12,
    end_year: int = 2022,
    end_month: int = 1,
) -> pd.DataFrame:
    """Помесячный снимок: нетели + Т/Б (без дойных и сухостойных)."""
    rows = []
    calv, ins, dry, disp = tables["calv_df"], tables["ins_df"], tables["dry_df"], tables["disp_df"]

    for year, month in iter_months_reverse(start_year, start_month, end_year, end_month):
        as_of = month_end_date(year, month)
        snap = _actual_nonbirth_snapshot_from_tables(calv, ins, dry, disp, as_of)
        row = {"год": year, "месяц": month, "дата_снимка": as_of.isoformat()}
        total = 0.0
        for k in YOUNG_AND_NETELI_KEYS:
            v = float(snap.get(k, 0.0))
            row[k] = int(v)
            total += v
        row["Всего без дойных и сухостойных"] = int(total)
        rows.append(row)

    return pd.DataFrame(rows)


def monthly_dry_milking_fact_history(
    tables: dict[str, pd.DataFrame],
    *,
    start_year: int = 2022,
    start_month: int = 1,
    end_year: int = 2025,
    end_month: int = 12,
) -> pd.DataFrame:
    """Факт сухостойных / дойных / фуражных на конец каждого месяца (снимок из событий)."""
    calv, ins, dry, disp = tables["calv_df"], tables["ins_df"], tables["dry_df"], tables["disp_df"]
    rows: list[dict[str, float | int]] = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        as_of = month_end_date(y, m)
        snap = _actual_nonbirth_snapshot_from_tables(calv, ins, dry, disp, as_of)
        rows.append(
            {
                "год": y,
                "месяц": m,
                "сухостойные": float(snap.get("Сухостойные коровы", 0.0)),
                "дойные": float(snap.get("Дойные коровы", 0.0)),
                "фуражные": float(snap.get("Дойные коровы", 0.0))
                + float(snap.get("Сухостойные коровы", 0.0)),
            }
        )
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pd.DataFrame(rows)


# =============================================================================
# Запуск (после ячейки «# фильтр ЖК Высокое»)
# =============================================================================

def main_snapshot() -> pd.DataFrame:
    raw = load_vysokoe_raw_tables()
    for name, df in raw.items():
        print(f"{name}: {len(df):,} строк")

    tables = excel_to_backtest_tables(raw)
    snap_1130 = _actual_nonbirth_snapshot_from_tables(
        tables["calv_df"], tables["ins_df"], tables["dry_df"], tables["disp_df"], date(2025, 11, 30),
    )
    snap_1231 = _actual_nonbirth_snapshot_from_tables(
        tables["calv_df"], tables["ins_df"], tables["dry_df"], tables["disp_df"], date(2025, 12, 31),
    )

    print_snapshot("Снимок 30.11.2025  (_actual_nonbirth_snapshot_from_tables)", snap_1130)
    print()
    print_snapshot("Снимок 31.12.2025  (_actual_nonbirth_snapshot_from_tables)", snap_1231, baseline=BASELINE_2025_12_31)

    birth_dec = _actual_birth_stats_month_from_tables(
        tables["calv_df"], tables["ins_df"], date(2025, 12, 31),
    )
    print("\nРождения в декабре 2025:", birth_dec)

    print("\n" + "=" * 72)
    print("ПОМЕСЯЧНО: нетели + молодняк (Т/Б), без дойных и сухостойных")
    print("  Период: декабрь 2025 → январь 2022 (снимок на конец месяца)")
    print("=" * 72)

    df_monthly = monthly_young_neteli_history(tables)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(df_monthly.to_string(index=False))

    out_xlsx = "снимок_молодняк_нетели_2022_01_2025_12.xlsx"
    df_monthly.to_excel(out_xlsx, index=False)
    print(f"\n✅ Сохранено: {out_xlsx}")
    return df_monthly


if __name__ == "__main__":
    main_snapshot()

