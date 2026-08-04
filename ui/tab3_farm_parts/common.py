from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import streamlit as st

from etl.bulls import read_bulls_txt
from etl.calvings_births import read_calvings_excel
from etl.disposals import _as_excel_source, _read_excel_best_header, read_disposals_excel
from etl.dryoff import read_dryoff_excel
from etl.inseminations import clean_inseminations, read_inseminations_excel

TAB3_TABLES = {
    "calv": "tab3_calvings_farm_raw",
    "ins": "tab3_inseminations_farm_raw",
    "dry": "tab3_dryoff_farm_raw",
    "disp": "tab3_disposals_farm_raw",
    "bulls": "tab3_bulls_farm_raw",
}

TAB3_CACHE_TABLE = "tab3_forecast_cache"

TAB3_MAP_TABLE = "tab3_subdivision_farm_map"

TAB3_CAPACITY_TABLE = "tab3_capacity_places"

TAB3_CACHE_SCHEMA_VERSION = "2026-03-03.v8"

TAB3_UI_STATE_VERSION = "2026-02-26.v3"

TAB3_SHOW_TRANSFER_SNAPSHOT = False

TAB3_SHOW_TRANSFER_FLOWS = False

TAB3_UNASSIGNED_FARM = "ВНЕ ХОЗЯЙСТВА"

FARM_BACKTEST_TARGETS: list[str] = [
    "Дойные коровы",
    "Сухостойные коровы",
    "Тёлки 0–3 мес",
    "Бычки 0–2 мес",
    "Тёлки 3–8 мес",
    "Тёлки ≥9 мес",
    "Нетели",
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]

FARM_BACKTEST_BIRTH_TARGETS = {
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
}

FARM_PERCENT_TARGETS = {
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
}

_STOPWORDS = {
    "ОСЕМЕН", "ОСЕМЕНЕНИЯ", "INSEM", "INSEMINATION",
    "ОТЕЛ", "ОТЕЛЫ", "ОТЕЛА", "РОДИВ", "РОДИВШ", "CALV", "BIRTH", "BORN",
    "ЗАПУСК", "DRY", "DRYOFF",
    "ВЫБЫТИЕ", "DISPOSAL", "DISPOSALS",
    "БЫК", "БЫКИ", "BULL", "BULLS",
    "ПЛЮС", "DZ", "XLS", "XLSX", "TXT", "ДАННЫЕ", "ЖК", "МТФ", "РЖК",
}

_STOP_PREFIXES = (
    "ОСЕМЕН", "ОТЕЛ", "РОДИВ", "ЗАПУСК", "ВЫБЫТ", "БЫК", "DISPOS", "INSEM", "CALV", "BIRTH", "BORN", "DRY",
)

@dataclass
class FarmUploadBundle:
    farm_name: str
    calv: Any | None = None
    ins: Any | None = None
    dry: Any | None = None
    disp: Any | None = None
    bulls: list[Any] = field(default_factory=list)

def _rewind(file_obj: Any) -> None:
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass

def _find_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {str(c).strip().upper(): c for c in df.columns}
    for x in cands:
        k = str(x).strip().upper()
        if k in cols:
            return cols[k]
    return None

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.normalize()

def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        return s.replace(".0", "")
    return s

def _norm_sex(x: Any) -> Optional[str]:
    if x is None:
        return None
    v = str(x).strip().upper().replace("Ё", "Е")
    if v in {"", "NAN", "NONE", "NULL", "0", "0.0"}:
        return None
    if v in {"F", "Ж"} or "ТЕЛ" in v or "ТЁЛ" in v or "HEIF" in v or "FEMALE" in v:
        return "F"
    if v in {"M", "М"} or "БЫЧ" in v or "BULL" in v or "MALE" in v:
        return "M"
    return None

def _norm_event_type(x: Any) -> str:
    if x is None:
        return ""
    v = str(x).strip().upper().replace("Ё", "Е")
    if "ОТЕЛ" in v or "CALV" in v:
        return "ОТЕЛ"
    if "РОЖ" in v or "BORN" in v or "BIRTH" in v:
        return "РОЖДЕН"
    return v

def _fallback_calvings(df_raw: pd.DataFrame) -> pd.DataFrame:
    mother_col = _find_col(df_raw, "DREG", "DREG1", "REG", "MOTHER_REG", "MOTHER")
    date_col = _find_col(df_raw, "DATE", "EVENT_DATE", "ARDAT", "CARX")
    ev_col = _find_col(df_raw, "EVENT", "EVENT_TYPE", "EVENTTYPE")
    sex_col = _find_col(df_raw, "GNDR", "GENDER", "SEX")
    lact_col = _find_col(df_raw, "LACT", "LACTATION")

    calf_cols = []
    for k in ("CALF1", "CALF2", "CALF3", "CALF4", "CALF5"):
        c = _find_col(df_raw, k)
        if c:
            calf_cols.append(c)

    if mother_col is None or date_col is None:
        raise ValueError("Не нашёл колонки матери/даты в файле отёлов (нужны DREG1/DATE или аналоги).")

    dts = _to_dt(df_raw[date_col])
    ev = df_raw[ev_col].map(_norm_event_type) if ev_col else "ОТЕЛ"
    mother = df_raw[mother_col].map(_norm_id)
    lact = pd.to_numeric(df_raw[lact_col], errors="coerce") if lact_col else pd.Series([pd.NA] * len(df_raw))

    out_rows: list[dict[str, Any]] = []
    for i in range(len(df_raw)):
        if pd.isna(dts.iloc[i]):
            continue
        mr = mother.iloc[i]
        if not mr:
            continue
        out_rows.append(
            {
                "reg": mr,
                "mother_reg": "",
                "birth_date": pd.NaT,
                "sex": None,
                "event_type": ev.iloc[i] if isinstance(ev, pd.Series) else "ОТЕЛ",
                "event_date": dts.iloc[i],
                "lact": lact.iloc[i],
            }
        )

    if calf_cols:
        sx = df_raw[sex_col].map(_norm_sex) if sex_col else None
        for i in range(len(df_raw)):
            dt = dts.iloc[i]
            if pd.isna(dt):
                continue
            mr = mother.iloc[i]
            if not mr:
                continue
            for cc in calf_cols:
                calf = _norm_id(df_raw[cc].iloc[i])
                if not calf or calf in {"0", "-"}:
                    continue
                out_rows.append(
                    {
                        "reg": calf,
                        "mother_reg": mr,
                        "birth_date": dt,
                        "sex": (sx.iloc[i] if sx is not None else None),
                        "event_type": "РОЖДЕН",
                        "event_date": dt,
                        "lact": pd.NA,
                    }
                )

    return pd.DataFrame(out_rows)

def _fallback_inseminations(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    lact_c = _find_col(df_raw, "LACT", "LACTATION")
    dim_c = _find_col(df_raw, "DIM", "DIM_AGE", "DAYS", "ВОЗРАСТ")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    bull_c = _find_col(df_raw, "REMARK", "BULL", "B", "BULL_CODE", "БЫК")
    res_c = _find_col(df_raw, "R", "RESULT", "RES", "RESULT ")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле осеменений.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "lact": pd.to_numeric(df_raw[lact_c], errors="coerce") if lact_c else 0,
            "dim_age": pd.to_numeric(df_raw[dim_c], errors="coerce") if dim_c else pd.NA,
            "event_date": _to_dt(df_raw[date_c]),
            "bull": df_raw[bull_c].map(_norm_id) if bull_c else "",
            "result": df_raw[res_c].astype(str).str.strip() if res_c else "",
        }
    )

def _fallback_disposals(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    reason_c = _find_col(df_raw, "REMARK", "DISPOSAL_REASON", "REM", "ПРИЧИНА ВЫБЫТИЯ")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле выбытия.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "event_date": _to_dt(df_raw[date_c]),
            "disposal_reason": df_raw[reason_c].astype(str).str.strip() if reason_c else "",
        }
    )

def _fallback_dryoff(df_raw: pd.DataFrame) -> pd.DataFrame:
    reg_c = _find_col(df_raw, "REG", "DREG", "IDREG")
    date_c = _find_col(df_raw, "DATE", "EVENT_DATE", "ДАТА")
    dim_c = _find_col(df_raw, "DIM", "ВОЗРАСТ", "DIM_AGE", "DAYS")
    reason_c = _find_col(df_raw, "CARX", "ПРИЧИНА ВЫБЫТИЯ", "REASON", "REM", "REMARK")

    if reg_c is None or date_c is None:
        raise ValueError("Не нашёл REG/DATE в файле запусков.")

    return pd.DataFrame(
        {
            "reg": df_raw[reg_c].map(_norm_id),
            "dim": pd.to_numeric(df_raw[dim_c], errors="coerce") if dim_c else pd.NA,
            "event_date": _to_dt(df_raw[date_c]),
            "move_reason": df_raw[reason_c].astype(str).str.strip() if reason_c else "",
        }
    )

def _filename_event_flags(filename: str) -> dict[str, bool]:
    n = filename.upper().replace("Ё", "Е")
    return {
        "calv": any(x in n for x in ("ОТЕЛ", "ОТЁЛ", "РОДИВ", "CALV", "BIRTH", "BORN")),
        "ins": any(x in n for x in ("ОСЕМЕН", "INSEM")),
        "dry": any(x in n for x in ("ЗАПУСК", "DRY", "DRYOFF")),
        "disp": any(x in n for x in ("ВЫБЫТИ", "DISPOS")),
    }


def _detect_kind(filename: str) -> Optional[str]:
    n = filename.upper().replace("Ё", "Е")
    if any(x in n for x in ("БЫК", "BULL")) and not any(
        x in n for x in ("ОСЕМЕН", "ОТЕЛ", "ОТЁЛ", "РОДИВ", "ЗАПУСК", "ВЫБЫТИ", "CALV", "INSEM", "DISPOS")
    ):
        return "bulls"
    flags = _filename_event_flags(filename)
    n_types = sum(flags.values())
    if n_types >= 2:
        return "multi_events"
    if flags["calv"]:
        return "calv"
    if flags["ins"]:
        return "ins"
    if flags["dry"]:
        return "dry"
    if flags["disp"]:
        return "disp"
    low = filename.lower()
    if low.endswith((".xlsx", ".xls", ".xlsm")):
        return "multi_events"
    return None


def _same_upload_file(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    return getattr(a, "name", None) == getattr(b, "name", None)


def _event_series(df: pd.DataFrame) -> pd.Series:
    col = _find_col(
        df,
        "event_type",
        "EVENT_TYPE",
        "EVENT",
        "EVENTS",
        "Событие",
        "СОБЫТИЕ",
        "ТИП СОБЫТИЯ",
    )
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].astype(str).str.upper().replace("Ё", "Е").str.strip()


def _event_column_name(df: pd.DataFrame) -> Optional[str]:
    return _find_col(
        df,
        "event_type",
        "EVENT_TYPE",
        "EVENT",
        "EVENTS",
        "Событие",
        "СОБЫТИЕ",
        "ТИП СОБЫТИЯ",
    )


def _column_inventory(df: pd.DataFrame, *, limit: int = 35) -> str:
    cols = [str(c) for c in df.columns]
    if len(cols) <= limit:
        return ", ".join(cols) if cols else "(нет колонок)"
    head = ", ".join(cols[:limit])
    return f"{head}, … (+{len(cols) - limit})"


def _event_value_sample(ev: pd.Series, *, top_n: int = 10) -> str:
    vals = ev.astype(str).str.strip()
    vals = vals[~vals.isin(["", "nan", "None", "NaT", "NAT"])]
    if vals.empty:
        return "(нет непустых значений — маски отёлов/осеменений дадут 0 строк)"
    vc = vals.value_counts().head(top_n)
    parts = [f"{repr(k)}×{int(v)}" for k, v in vc.items()]
    return ", ".join(parts)


def _bundle_slot_summary(bundle: FarmUploadBundle) -> str:
    def _name(f: Any) -> str:
        return str(getattr(f, "name", "—") or "—")

    return (
        f"слоты: отёлы={_name(bundle.calv)}, осеменения={_name(bundle.ins)}, "
        f"запуск={_name(bundle.dry)}, выбытие={_name(bundle.disp)}, "
        f"быки={len(bundle.bulls)} файл(ов)"
    )


def _diagnose_events_workbook(filename: str, file_obj: Any) -> str:
    """Текст для пользователя: почему из файла могло получиться 0 отёлов/осеменений."""
    lines: list[str] = [f"── {filename}"]
    kind = _detect_kind(filename)
    dedicated = _dedicated_file_kind(filename)
    lines.append(
        f"   распознавание по имени: kind={kind!r}, dedicated={dedicated!r} "
        f"(флаги {_filename_event_flags(filename)})"
    )
    _rewind(file_obj)
    try:
        raw = _raw_excel_from_file(file_obj)
    except Exception as exc:
        lines.append(f"   ❌ Excel не прочитан: {exc}")
        return "\n".join(lines)

    lines.append(f"   строк на листе: {len(raw)}")
    lines.append(f"   колонки: {_column_inventory(raw)}")

    ev_col = _event_column_name(raw)
    if ev_col:
        lines.append(f"   колонка события: «{ev_col}»")
    else:
        lines.append(
            "   колонка события: НЕ НАЙДЕНА "
            "(ожидаем Event / EVENT / Событие / event_type — иначе нужны отдельные файлы "
            "«Отелы…», «Осеменения…» в имени)"
        )

    ev = _event_series(raw)
    lines.append(f"   примеры Event/Событие: {_event_value_sample(ev)}")

    m_calv, m_ins, m_dry, m_disp = _apply_filename_hint_masks(filename, len(raw), ev)
    lines.append(
        f"   строк по маскам в сыром файле: отёлы={int(m_calv.sum())}, "
        f"осеменения={int(m_ins.sum())}, запуск={int(m_dry.sum())}, выбытие={int(m_disp.sum())}"
    )

    _rewind(file_obj)
    parsed = _parse_events_workbook(filename, file_obj)
    for key, label in (
        ("calv", "отёлы"),
        ("ins", "осеменения"),
        ("dry", "запуск"),
        ("disp", "выбытие"),
    ):
        part = parsed.get(key)
        n = len(part) if isinstance(part, pd.DataFrame) else 0
        lines.append(f"   после парсера «{label}»: {n} строк")

    if dedicated and len(raw) and all(
        len(parsed.get(k, pd.DataFrame())) == 0 for k in ("calv", "ins", "dry", "disp")
    ):
        lines.append(
            "   подсказка: файл с одним типом в имени, но ETL не извлёк строк — "
            "проверьте REG/DATE (или DREG1/DATE в отёлах) и строку заголовка в Excel."
        )
    elif kind == "multi_events" and ev_col and int(m_calv.sum()) == 0 and int(m_ins.sum()) == 0:
        lines.append(
            "   подсказка: в Event нет подстрок ОТЕЛ/CALV/BRED/ОСЕМЕН — переименуйте значения "
            "или загрузите отдельные файлы «Отелы…» и «Осеменения…»."
        )

    return "\n".join(lines)


def _mask_calv_events(ev: pd.Series) -> pd.Series:
    return ev.str.contains(r"ОТЕЛ|CALV|\bBIRTH\b|BORN|РОЖД", regex=True, na=False)


def _mask_ins_events(ev: pd.Series) -> pd.Series:
    return ev.str.contains(r"ОСЕМЕН|INSEM|\bBRED\b", regex=True, na=False)


def _mask_dry_events(ev: pd.Series) -> pd.Series:
    return ev.str.contains(r"ЗАПУСК|\bDRY\b|DRYOFF", regex=True, na=False)


def _mask_disp_events(ev: pd.Series) -> pd.Series:
    return ev.str.contains(r"ВЫБЫТ|DISPOS|\bSOLD\b|ПРОДАН", regex=True, na=False)


def _read_combined_dry_disp_excel(file_obj: Any) -> pd.DataFrame:
    _rewind(file_obj)
    try:
        return read_disposals_excel(file_obj, include_meta=True)
    except Exception:
        _rewind(file_obj)
        return read_dryoff_excel(file_obj, include_meta=True)


def _split_dry_disp_frames(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Один Excel «Запуск + Выбытие»: делим по колонке события."""
    if full is None or full.empty:
        empty_dry = pd.DataFrame(columns=["reg", "dim", "event_date", "move_reason", "__farm", "__subdivision"])
        empty_disp = pd.DataFrame(columns=["reg", "event_date", "disposal_reason", "__farm", "__subdivision"])
        return empty_dry, empty_disp

    work = full.copy()
    ev = _event_series(work)
    m_dry = _mask_dry_events(ev)
    m_disp = _mask_disp_events(ev)
    if not m_dry.any() and not m_disp.any():
        m_disp = pd.Series([True] * len(work), index=work.index)

    dry_src = work[m_dry].copy() if m_dry.any() else work.iloc[0:0].copy()
    disp_src = work[m_disp].copy() if m_disp.any() else work.iloc[0:0].copy()

    for c in ("reg", "event_date", "__farm", "__subdivision"):
        if c not in dry_src.columns:
            dry_src[c] = pd.NA
        if c not in disp_src.columns:
            disp_src[c] = pd.NA

    dim_col = "dim" if "dim" in dry_src.columns else None
    if dim_col is None and "age_dim" in dry_src.columns:
        dry_src["dim"] = pd.to_numeric(dry_src["age_dim"], errors="coerce")
    elif "dim" not in dry_src.columns:
        dry_src["dim"] = pd.NA

    reason_d = None
    for c in ("disposal_reason", "remark", "note"):
        if c in dry_src.columns:
            reason_d = dry_src[c].astype(str)
            break
    dry_out = pd.DataFrame(
        {
            "reg": dry_src["reg"].map(_norm_id),
            "dim": pd.to_numeric(dry_src.get("dim"), errors="coerce"),
            "event_date": pd.to_datetime(dry_src["event_date"], errors="coerce", dayfirst=True),
            "move_reason": reason_d if reason_d is not None else "",
            "__farm": dry_src["__farm"],
            "__subdivision": dry_src["__subdivision"],
        }
    )

    reason_x = None
    for c in ("disposal_reason", "remark", "note"):
        if c in disp_src.columns:
            reason_x = disp_src[c].astype(str).str.strip()
            break
    disp_out = pd.DataFrame(
        {
            "reg": disp_src["reg"].map(_norm_id),
            "event_date": pd.to_datetime(disp_src["event_date"], errors="coerce", dayfirst=True),
            "disposal_reason": reason_x if reason_x is not None else "",
            "__farm": disp_src["__farm"],
            "__subdivision": disp_src["__subdivision"],
        }
    )
    return dry_out, disp_out


def _raw_excel_from_file(file_obj: Any) -> pd.DataFrame:
    _rewind(file_obj)
    return _read_excel_best_header(_as_excel_source(file_obj), max_header=25)


def _read_normalized_events_table(file_obj: Any) -> pd.DataFrame:
    for reader in (
        read_disposals_excel,
        read_dryoff_excel,
        read_inseminations_excel,
        read_calvings_excel,
    ):
        _rewind(file_obj)
        try:
            return reader(file_obj, include_meta=True)
        except Exception:
            continue
    raw = _raw_excel_from_file(file_obj)
    return raw.rename(columns={c: str(c).strip() for c in raw.columns})


def _empty_typed_frame(kind: str) -> pd.DataFrame:
    cols = {
        "calv": ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date", "__farm", "__subdivision"],
        "ins": ["reg", "lact", "dim_age", "event_date", "bull", "result", "__farm", "__subdivision"],
        "dry": ["reg", "dim", "event_date", "move_reason", "__farm", "__subdivision"],
        "disp": ["reg", "event_date", "disposal_reason", "__farm", "__subdivision"],
    }
    return pd.DataFrame(columns=cols[kind])


def _apply_filename_hint_masks(
    filename: str, n_rows: int, ev: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = ev.index
    z = pd.Series([False] * n_rows, index=idx)
    m_calv, m_ins, m_dry, m_disp = z.copy(), z.copy(), z.copy(), z.copy()
    if ev.astype(str).str.strip().ne("").any():
        m_calv = _mask_calv_events(ev)
        m_ins = _mask_ins_events(ev)
        m_dry = _mask_dry_events(ev)
        m_disp = _mask_disp_events(ev)
        return m_calv, m_ins, m_dry, m_disp

    flags = _filename_event_flags(filename)
    kind = _detect_kind(filename)
    if kind == "multi_events" or sum(flags.values()) >= 2:
        all_rows = pd.Series([True] * n_rows, index=idx)
        return all_rows, all_rows, all_rows, all_rows
    if kind == "calv" or flags["calv"]:
        m_calv = pd.Series([True] * n_rows, index=idx)
    elif kind == "ins" or flags["ins"]:
        m_ins = pd.Series([True] * n_rows, index=idx)
    elif kind == "dry" or flags["dry"]:
        m_dry = pd.Series([True] * n_rows, index=idx)
    elif kind == "disp" or flags["disp"]:
        m_disp = pd.Series([True] * n_rows, index=idx)
    return m_calv, m_ins, m_dry, m_disp


def _dedicated_file_kind(filename: str) -> Optional[str]:
    """Один тип события по имени файла — читаем целиком через ETL, без фильтра «Событие»."""
    kind = _detect_kind(filename)
    if kind == "multi_events":
        flags = _filename_event_flags(filename)
        if flags["dry"] and flags["disp"] and not flags["calv"] and not flags["ins"]:
            return None
        return None
    flags = _filename_event_flags(filename)
    if sum(flags.values()) == 1:
        return next(k for k, v in flags.items() if v)
    if kind in ("calv", "ins", "dry", "disp"):
        return kind
    return None


def _parse_dedicated_file(filename: str, file_obj: Any) -> Optional[dict[str, pd.DataFrame]]:
    kind = _dedicated_file_kind(filename)
    if kind is None:
        return None
    empty = {
        "calv": _empty_typed_frame("calv"),
        "ins": _empty_typed_frame("ins"),
        "dry": _empty_typed_frame("dry"),
        "disp": _empty_typed_frame("disp"),
    }
    _rewind(file_obj)
    try:
        if kind == "calv":
            try:
                df = read_calvings_excel(file_obj, include_meta=True)
            except Exception:
                _rewind(file_obj)
                df = _fallback_calvings(_raw_excel_from_file(file_obj))
            empty["calv"] = df
            return empty
        if kind == "ins":
            try:
                df = clean_inseminations(read_inseminations_excel(file_obj, include_meta=True))
            except Exception:
                _rewind(file_obj)
                df = _fallback_inseminations(_raw_excel_from_file(file_obj))
            empty["ins"] = df
            return empty
        if kind == "dry":
            try:
                df = read_dryoff_excel(file_obj, include_meta=True)
            except Exception:
                _rewind(file_obj)
                df = _fallback_dryoff(_raw_excel_from_file(file_obj))
            df = df.copy()
            if "move_reason" not in df.columns and "disposal_reason" in df.columns:
                df["move_reason"] = df["disposal_reason"]
            empty["dry"] = df
            return empty
        if kind == "disp":
            try:
                df = read_disposals_excel(file_obj, include_meta=True)
            except Exception:
                _rewind(file_obj)
                df = _fallback_disposals(_raw_excel_from_file(file_obj))
            empty["disp"] = df
            return empty
    except Exception:
        return None
    return None


def _parse_events_workbook(filename: str, file_obj: Any) -> dict[str, pd.DataFrame]:
    """Разнести строки одного Excel по calv / ins / dry / disp."""
    dedicated = _parse_dedicated_file(filename, file_obj)
    kind = _dedicated_file_kind(filename)
    if dedicated is not None and kind is not None:
        part = dedicated.get(kind)
        if isinstance(part, pd.DataFrame) and not part.empty:
            return dedicated
    _rewind(file_obj)
    out: dict[str, pd.DataFrame] = {
        "calv": _empty_typed_frame("calv"),
        "ins": _empty_typed_frame("ins"),
        "dry": _empty_typed_frame("dry"),
        "disp": _empty_typed_frame("disp"),
    }
    try:
        raw = _raw_excel_from_file(file_obj)
    except Exception:
        return out
    if raw.empty:
        return out

    ev = _event_series(raw)
    m_calv, m_ins, m_dry, m_disp = _apply_filename_hint_masks(filename, len(raw), ev)

    if not (m_calv.any() or m_ins.any() or m_dry.any() or m_disp.any()):
        blank = pd.Series([""] * len(raw), index=raw.index)
        m_calv, m_ins, m_dry, m_disp = _apply_filename_hint_masks(filename, len(raw), blank)

    hint = _dedicated_file_kind(filename)
    if hint == "calv" and len(raw) and not m_calv.any():
        m_calv = pd.Series(True, index=raw.index)
    elif hint == "ins" and len(raw) and not m_ins.any():
        m_ins = pd.Series(True, index=raw.index)

    overlap = (m_calv.astype(int) + m_ins.astype(int) + m_dry.astype(int) + m_disp.astype(int)) > 1
    if overlap.any():
        m_calv &= ~overlap
        m_ins &= ~overlap
        m_dry &= ~overlap
        m_disp &= ~overlap

    if m_calv.any():
        try:
            out["calv"] = _fallback_calvings(raw.loc[m_calv].copy())
        except Exception:
            pass
    if m_ins.any():
        try:
            out["ins"] = _fallback_inseminations(raw.loc[m_ins].copy())
        except Exception:
            pass

    if m_dry.any() or m_disp.any():
        try:
            norm = _read_normalized_events_table(file_obj)
            nev = _event_series(norm)
            if nev.astype(str).str.strip().ne("").any():
                md = _mask_dry_events(nev)
                mp = _mask_disp_events(nev)
            else:
                _, _, md, mp = _apply_filename_hint_masks(filename, len(norm), nev)
            subset = norm.loc[md | mp].copy()
            if not subset.empty:
                dry_df, disp_df = _split_dry_disp_frames(subset)
                if md.any():
                    out["dry"] = dry_df
                if mp.any():
                    out["disp"] = disp_df
        except Exception:
            pass

    return out


def _unique_bundle_event_files(bundle: FarmUploadBundle) -> list[Any]:
    seen: set[int] = set()
    uniq: list[Any] = []
    for f in (bundle.calv, bundle.ins, bundle.dry, bundle.disp):
        if f is None:
            continue
        fid = id(f)
        if fid in seen:
            continue
        seen.add(fid)
        uniq.append(f)
    return uniq


def _merge_parsed_parts(parts: dict[str, list[pd.DataFrame]], key: str) -> pd.DataFrame:
    frames = [df for df in parts.get(key, []) if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return _empty_typed_frame(key)
    return pd.concat(frames, ignore_index=True, sort=False)


def _assign_multi_event_file(bundle: FarmUploadBundle, f: Any) -> bool:
    """True если заменили уже загруженные слоты."""
    replaced = any(x is not None for x in (bundle.calv, bundle.ins, bundle.dry, bundle.disp))
    bundle.calv = f
    bundle.ins = f
    bundle.dry = f
    bundle.disp = f
    return replaced


def _extract_farm_name(filename: str, kind: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", filename, flags=re.IGNORECASE)
    tokens = re.findall(r"[0-9A-ZА-ЯЁ]+", stem.upper().replace("Ё", "Е"))

    out: list[str] = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if any(t.startswith(pref) for pref in _STOP_PREFIXES):
            continue
        if t.isdigit() and len(t) >= 4:
            continue
        if len(t) <= 1:
            continue
        out.append(t)

    name = " ".join(out).strip()
    return name or "ХОЗЯЙСТВО_1"


def _group_files(files: list[Any]) -> tuple[dict[str, FarmUploadBundle], pd.DataFrame]:
    bundles: dict[str, FarmUploadBundle] = {}
    rows: list[dict[str, str]] = []

    for f in files:
        kind = _detect_kind(f.name)
        if kind is None:
            rows.append({"Файл": f.name, "Тип": "не распознан", "Подразделение": "—", "Статус": "пропущен"})
            continue

        farm = _extract_farm_name(f.name, kind)
        b = bundles.setdefault(farm, FarmUploadBundle(farm_name=farm))

        status = "ok"
        if kind == "calv":
            if b.calv is not None:
                status = "заменён (последний файл)"
            b.calv = f
        elif kind == "ins":
            if b.ins is not None:
                status = "заменён (последний файл)"
            b.ins = f
        elif kind == "dry":
            if b.dry is not None:
                status = "заменён (последний файл)"
            b.dry = f
        elif kind == "disp":
            if b.disp is not None:
                status = "заменён (последний файл)"
            b.disp = f
        elif kind == "multi_events":
            if _assign_multi_event_file(b, f):
                status = "заменён (последний файл)"
            else:
                status = "ok (несколько типов событий в одном файле)"
            kind = "all-events"
        else:
            b.bulls.append(f)

        rows.append({"Файл": f.name, "Тип": kind, "Подразделение": farm, "Статус": status})

    return bundles, pd.DataFrame(rows, columns=["Файл", "Тип", "Подразделение", "Статус"])


def _bundle_has_core_files(bundle: FarmUploadBundle) -> bool:
    return any(x is not None for x in (bundle.calv, bundle.ins, bundle.dry, bundle.disp))


def _bundle_has_only_bulls(bundle: FarmUploadBundle) -> bool:
    return bool(bundle.bulls) and not _bundle_has_core_files(bundle)


def _merge_aux_bull_bundles(
    bundles: dict[str, FarmUploadBundle],
) -> tuple[dict[str, FarmUploadBundle], dict[str, str]]:
    """
    Если пользователь загрузил один комплект из 4 Excel по всему хозяйству
    и отдельно несколько txt/xlsx по быкам для подразделений, не считаем
    эти bull-only файлы отдельными "подразделениями с неполным комплектом".

    Возвращает:
    - bundles после слияния
    - mapping source_bundle_name -> target_bundle_name для bull-only комплектов
    """
    if not isinstance(bundles, dict) or not bundles:
        return bundles, {}

    core_bundle_names = [name for name, bundle in bundles.items() if _bundle_has_core_files(bundle)]
    if len(core_bundle_names) != 1:
        return bundles, {}

    target_name = core_bundle_names[0]
    merged = dict(bundles)
    attached: dict[str, str] = {}

    for name, bundle in list(merged.items()):
        if name == target_name:
            continue
        if not _bundle_has_only_bulls(bundle):
            continue
        merged[target_name].bulls.extend(bundle.bulls)
        attached[name] = target_name
        del merged[name]

    return merged, attached

def _prepare_tables(bundle: FarmUploadBundle) -> dict[str, pd.DataFrame]:
    if not _unique_bundle_event_files(bundle):
        raise ValueError("Нужны Excel с событиями (отёлы, осеменения, запуск, выбытие — отдельно или в одном файле).")

    collected: dict[str, list[pd.DataFrame]] = {k: [] for k in ("calv", "ins", "dry", "disp")}
    for f in _unique_bundle_event_files(bundle):
        parsed = _parse_events_workbook(getattr(f, "name", "upload.xlsx"), f)
        for key in collected:
            df = parsed.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                collected[key].append(df)

    calv_df = _merge_parsed_parts(collected, "calv")
    ins_df = _merge_parsed_parts(collected, "ins")
    dry_df = _merge_parsed_parts(collected, "dry")
    disp_df = _merge_parsed_parts(collected, "disp")

    missing = [label for label, df in (
        ("отёлы", calv_df),
        ("осеменения", ins_df),
        ("запуски", dry_df),
        ("выбытие", disp_df),
    ) if df.empty]
    if missing:
        diag_parts = [_bundle_slot_summary(bundle), ""]
        for f in _unique_bundle_event_files(bundle):
            _rewind(f)
            diag_parts.append(_diagnose_events_workbook(getattr(f, "name", "upload.xlsx"), f))
        diag_text = "\n".join(diag_parts)
        raise ValueError(
            "Не хватает данных после разбора файлов: "
            + ", ".join(missing)
            + ".\n\nДиагностика (что увидел разборщик):\n"
            + diag_text
            + "\n\nНужна колонка Event/Событие с узнаваемыми значениями "
            "(CALVING, BRED, DRY, …) или отдельные файлы с «Отелы» / «Осеменения» в имени."
        )

    calv_df = calv_df.copy()
    for c in ("reg", "mother_reg", "birth_date", "sex", "event_type", "event_date", "__farm", "__subdivision"):
        if c not in calv_df.columns:
            calv_df[c] = pd.NA
    calv_df["reg"] = calv_df["reg"].map(_norm_id)
    calv_df["mother_reg"] = calv_df["mother_reg"].map(_norm_id)
    calv_df["birth_date"] = pd.to_datetime(calv_df["birth_date"], errors="coerce", dayfirst=True)
    calv_df["event_date"] = pd.to_datetime(calv_df["event_date"], errors="coerce", dayfirst=True)
    calv_df["sex"] = calv_df["sex"].map(_norm_sex)
    calv_df["event_type"] = calv_df["event_type"].map(_norm_event_type)
    if "lact" not in calv_df.columns:
        calv_df["lact"] = pd.NA

    ins_df = ins_df.copy()
    for c in ("reg", "lact", "dim_age", "event_date", "bull", "result", "__farm", "__subdivision"):
        if c not in ins_df.columns:
            ins_df[c] = pd.NA
    ins_df["reg"] = ins_df["reg"].map(_norm_id)
    ins_df["lact"] = pd.to_numeric(ins_df["lact"], errors="coerce")
    ins_df["dim_age"] = pd.to_numeric(ins_df["dim_age"], errors="coerce")
    ins_df["event_date"] = pd.to_datetime(ins_df["event_date"], errors="coerce", dayfirst=True)
    ins_df["bull"] = ins_df["bull"].map(_norm_id)
    ins_df["result"] = ins_df["result"].astype(str).str.strip()

    dry_df = dry_df.copy()
    for c in ("reg", "dim", "event_date", "move_reason", "__farm", "__subdivision"):
        if c not in dry_df.columns:
            dry_df[c] = pd.NA
    dry_df["reg"] = dry_df["reg"].map(_norm_id)
    dry_df["dim"] = pd.to_numeric(dry_df["dim"], errors="coerce")
    dry_df["event_date"] = pd.to_datetime(dry_df["event_date"], errors="coerce", dayfirst=True)
    if "move_reason" not in dry_df.columns:
        dry_df["move_reason"] = ""
    if dry_df["move_reason"].astype(str).str.strip().eq("").all() and "disposal_reason" in dry_df.columns:
        dry_df["move_reason"] = dry_df["disposal_reason"].astype(str)

    disp_df = disp_df.copy()
    for c in ("reg", "event_date", "disposal_reason", "__farm", "__subdivision"):
        if c not in disp_df.columns:
            disp_df[c] = pd.NA
    disp_df["reg"] = disp_df["reg"].map(_norm_id)
    disp_df["event_date"] = pd.to_datetime(disp_df["event_date"], errors="coerce", dayfirst=True)

    try:
        ins_df = clean_inseminations(ins_df)
    except Exception:
        pass

    bulls_frames: list[pd.DataFrame] = []
    for bf in bundle.bulls:
        try:
            _rewind(bf)
            bdf = read_bulls_txt(bf)
            if not bdf.empty:
                for c in ("bull_code", "bull_type"):
                    if c not in bdf.columns:
                        bdf[c] = pd.NA
                bdf = bdf[["bull_code", "bull_type"]].copy()
                bdf["bull_code"] = bdf["bull_code"].map(_norm_id)
                bdf["bull_type"] = bdf["bull_type"].astype(str).str.strip()
                bulls_frames.append(bdf)
        except Exception:
            continue

    bulls_df = (
        pd.concat(bulls_frames, ignore_index=True).drop_duplicates(subset=["bull_code"], keep="first")
        if bulls_frames
        else pd.DataFrame(columns=["bull_code", "bull_type"])
    )

    return {
        "calv": calv_df[["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date", "__farm", "__subdivision"]].copy(),
        "ins": ins_df[["reg", "lact", "dim_age", "event_date", "bull", "result", "__farm", "__subdivision"]].copy(),
        "dry": dry_df[["reg", "dim", "event_date", "move_reason", "__farm", "__subdivision"]].copy(),
        "disp": disp_df[["reg", "event_date", "disposal_reason", "__farm", "__subdivision"]].copy(),
        "bulls": bulls_df[["bull_code", "bull_type"]].copy(),
    }

def _json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _params_hash(params: dict) -> str:
    payload = {
        "__cache_schema_version__": TAB3_CACHE_SCHEMA_VERSION,
        "params": params or {},
    }
    return _json_hash(payload)

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _farm_param_overrides_state() -> dict[str, dict]:
    raw = st.session_state.get("tab3_farm_param_overrides")
    if not isinstance(raw, dict):
        raw = {}
    st.session_state["tab3_farm_param_overrides"] = raw
    return raw

def _subdivision_param_overrides_state() -> dict[str, dict]:
    raw = st.session_state.get("tab3_subdivision_param_overrides")
    if not isinstance(raw, dict):
        raw = {}
    st.session_state["tab3_subdivision_param_overrides"] = raw
    return raw

def _is_admin_mode() -> bool:
    return bool(st.session_state.get("is_admin", False))

def _build_farm_params(base_params: dict, farm_override: dict | None) -> dict:
    params = deepcopy(base_params or {})
    if isinstance(farm_override, dict) and farm_override:
        _deep_merge(params, farm_override)
    params.pop("HERD_CAPACITY", None)
    params.pop("herd_capacity", None)
    params["DISABLE_CAPACITY"] = True
    params["APPLY_CAPACITY"] = False
    return params

def _build_subdivision_params(
    base_params: dict,
    farm_override: dict | None = None,
    subdivision_override: dict | None = None,
) -> dict:
    params = _build_farm_params(base_params, farm_override)
    if isinstance(subdivision_override, dict) and subdivision_override:
        _deep_merge(params, subdivision_override)
    params.pop("HERD_CAPACITY", None)
    params.pop("herd_capacity", None)
    params["DISABLE_CAPACITY"] = True
    params["APPLY_CAPACITY"] = False
    return params


__all__ = [name for name in globals() if not name.startswith("__")]
