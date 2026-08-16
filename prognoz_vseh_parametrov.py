#!/usr/bin/env python3
"""
Единый прогноз всех параметров (логика ЖК_Высокое_финал.ipynb).

Обучение: январь 2022 — сентябрь 2024.
Горизонт таблиц: октябрь 2024 — декабрь 2025 (15 месяцев).
Выход: Excel с листами «прогноз» и «факт» (строки = параметры, столбцы = месяцы).

Для Калуги начальные остатки на 30.09.2024 (фураж / сухостой / L1–L5+)
берутся из Калуга/база_30.09.2024_поголовье.xlsx (если есть строка подразделения).

ЖК Высокое (по умолчанию):
  python prognoz_vseh_parametrov.py

Калуга (хозяйство + подразделение):
  python Калуга/prognoz_vseh_parametrov.py --farm "КН Восток" --unit "ЖК Аристово"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd


def _setting(name: str, default: str = "") -> str:
    """Env или Streamlit Cloud Secrets (st.secrets)."""
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


def _pipeline_fast_enabled() -> bool:
    return _setting("PIPELINE_FAST", "1").lower() not in ("0", "false", "no")


ROOT = Path(__file__).resolve().parent
EXTRACTED = ROOT / "finale_pipeline_extracted.py"
TRAIN_END = pd.Timestamp("2024-09-30")
KALUGA_TRIM_DATA_DIR = ROOT / "Калуга" / "данные-калуга" / "до_2024_09"
KALUGA_FULL_DATA_DIR = ROOT / "Калуга" / "данные-калуга"
KALUGA_SEP2024_BASELINE_XLSX = ROOT / "Калуга" / "база_30.09.2024_поголовье.xlsx"

PREDICT_MONTHS: list[tuple[int, int]] = [(2024, m) for m in (10, 11, 12)] + [
    (2025, m) for m in range(1, 13)
]

MONTH_COLS = [f"{y}-{m:02d}" for y, m in PREDICT_MONTHS]

KALUGA_TREE: dict[str, list[str]] = {
    "КН Восток": [
        "ЖК Аристово",
        "ЖК Болдасовка",
        "ЖК Сугуново",
        "РМ Детчино",
        "РМ Кольцово",
    ],
    "КН Юг": ["ЖК Богданино", "ЖК Бушовка", "РМ КН-Юг"],
    "КН Запад": ["ЖК Гусево", "ЖК Уланово", "МТФ КН-Запад"],
    "КН Хвастовичи": ["ЖК Пеневичи"],
}

# Aliases in CSV (Столбец1)
SUBDIVISION_ALIASES: dict[str, list[str]] = {
    "ЖК Сугуново": ["ЖК Сугуново", "ЖК Сугоново"],
    "МТФ КН-Запад": ["МТФ КН-Запад", "МТФ Запад Медынь"],
}

# Внутренние продажи/покупки (Калуга): «внутри» = перевод в любое подразделение того же хозяйства.
# _match_kuda ищет подстроку в поле «Куда» (REM). heifer_sale — все коды хозяйства; buy — коды текущего ЖК/РМ.
KALUGA_FARM_INTERNAL_REM: dict[str, list[str]] = {
    "КН Восток": [
        "ЖК БОЛД",
        "ЖК БОЛДА",
        "ЖК БОЛДАСОВКА",
        "ЖК БОЛТ",
        "ЖК ДЕТЧ",
        "ЖК ДЕТ",
        "ЖК ДЕТЧИНО",
        "ЖК_АРИСТОВО",
        "ЖК_АРИСТ",
        "ЖК_БЛД",
        "ЖК_БОЛД",
        "ЖК_БОЛДАСОВКА",
        "ДЕТЧ",
        "ДЕТЧИНО",
        "ЖКАРИСТОВО",
        "ЖКБОЛДАСОВКА",
        "РМ_ДЕТ",
        "РМ_ДЕТЧИНО",
        "РМ_КОЛЬЦОВО",
        "КОЛЬЦОВО",
        "КОЛЬЦ",
        "ЖК_БОЛДА",
        "ЖКАРС",
        "ЖК АРС",
        "РМ_КЛЦ",
        "КОЛЬЦОВ",
        "ЖКСГН",
        "ЖК_СУГОНОВО",
        "СУГОНОВО",
        "СУГУНОВО",
        "АРИСТОВО",
        "БОЛДАСОВКА",
        "БОЛД",
        "РМ_ДТЧ",
    ],
    "КН Юг": [
        "ЖК_БОГД",
        "ЖК_БОГДАНИНО",
        "БОГДАНИНО",
        "ЖК_БГД",
        "ЖК_БУШОВКА",
        "БУШОВКА",
        "ЖК_БШ",
        "РМ_КН-ЮГ",
        "РМ_КН_ЮГ",
        "КН-ЮГ",
        "КН_ЮГ",
    ],
    "КН Запад": [
        "МТФ_КН-ЗАПАД",
        "МТФ_ЗАПАД",
        "МТФ КН-ЗАПАД",
        "ЗАПАД",
        "ЖК_ГУСЕВО",
        "ГУСЕВО",
        "ЖК_ГСВ",
        "ЖК_УЛАНОВО",
        "УЛАНОВО",
    ],
    "КН Хвастовичи": ["ЖК_ПЕНЕВИЧИ", "ПЕНЕВИЧИ", "ПЕНЕВ"],
}

# Доп. REM для покупки в конкретное подразделение (если не покрывает _unit_rem_codes).
KALUGA_UNIT_BUY_REM_EXTRAS: dict[str, list[str]] = {
    "ЖК Аристово": ["ЖКАРС", "АРИСТОВО"],
    "ЖК Болдасовка": ["ЖК_БОЛД", "БОЛДАСОВКА"],
    "ЖК Сугуново": ["ЖКСГН", "ЖК_СУГОНОВО", "СУГОНОВО"],
    "РМ Детчино": ["РМ_ДЕТ", "РМ_ДЕТЧИНО", "ДЕТЧ", "РМ_ДТЧ"],
    "РМ Кольцово": ["РМ_КЛЦ", "КОЛЬЦ", "КОЛЬЦОВ"],
    "ЖК Богданино": ["ЖК_БОГД", "БОГДАНИНО", "ЖК_БГД"],
    "ЖК Бушовка": ["БУШОВКА", "ЖК_БШ"],
    "РМ КН-Юг": ["КН-ЮГ", "КН_ЮГ"],
    "ЖК Гусево": ["ЖК_ГСВ", "ГУСЕВО"],
    "МТФ КН-Запад": ["МТФ_ЗАПАД", "ЗАПАД"],
    "ЖК Пеневичи": ["ПЕНЕВ"],
}

VYSOKOE_HEIFER_SALE = [
    "МТФ_ВЫСОКОЕ",
    "МТФ ВЫСОКОЕ",
    "МТФВЫСОКОЕ",
    "МТФ_ВЫСОК",
    "МТФ ВЫСОК",
    "МТФВЫСОК",
]
VYSOKOE_BULL_SALE = ["БЫЧКИ", "БЫЧ"]
VYSOKOE_BUY = ["ЖК_ВЫСОК", "ЖК_ВЫСОКОЕ", "ЖКВЫСОК", "ЖКВЫСОКО", "ЖКВЫСОКОЕ"]


def _unit_rem_codes(unit: str) -> list[str]:
    u = unit.strip()
    codes = [u.upper().replace(" ", "_"), u.upper().replace(" ", ""), u.upper()]
    if u.startswith("ЖК "):
        n = u[3:].upper()
        codes += [f"ЖК_{n}", f"ЖК{n[:4]}", n, n.replace(" ", "_")]
    if u.startswith("РМ "):
        n = u[3:].upper().replace(" ", "_")
        codes += [f"РМ_{n}", f"РМ{n[:3]}", n]
    if u.startswith("МТФ "):
        n = u[4:].upper().replace(" ", "_")
        codes += [f"МТФ_{n}", f"МТФ{n[:3]}", n]
    return list(dict.fromkeys(c for c in codes if c))


def rem_codes_for_subdivision(unit: str) -> list[str]:
    """REM «Куда» для покупки в данное подразделение."""
    codes: list[str] = []
    for name in SUBDIVISION_ALIASES.get(unit, [unit]):
        codes.extend(_unit_rem_codes(name))
    codes.extend(_unit_rem_codes(unit))
    codes.extend(KALUGA_UNIT_BUY_REM_EXTRAS.get(unit, []))
    return list(dict.fromkeys(c for c in codes if c))


def rem_codes_for_farm_internal(farm: str) -> list[str]:
    """REM «Куда» для внутренней продажи: любое подразделение того же хозяйства."""
    codes: list[str] = []
    for u in KALUGA_TREE.get(farm, []):
        codes.extend(rem_codes_for_subdivision(u))
    codes.extend(KALUGA_FARM_INTERNAL_REM.get(farm, []))
    return list(dict.fromkeys(c for c in codes if c))


def rem_codes_all_kaluga() -> list[str]:
    """REM/Кuda всех подразделений Калуги (для ПЕРЕЕЗД → «внутри»)."""
    codes: list[str] = []
    for farm, units in KALUGA_TREE.items():
        codes.extend(rem_codes_for_farm_internal(farm))
        for u in units:
            codes.extend(rem_codes_for_subdivision(u))
    out: list[str] = []
    for c in dict.fromkeys(codes):
        cu = c.strip().upper()
        if len(cu) < 3 or cu in ("-", "NAN") or cu.isdigit():
            continue
        out.append(cu)
    return out


def kaluga_trade_rules(farm: str, unit: str) -> dict[str, list[str]]:
    return {
        "buy": rem_codes_for_subdivision(unit),
        "heifer_sale": rem_codes_for_farm_internal(farm),
        "bull_sale": ["БЫЧКИ", "БЫЧ"],
    }


def normalize_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Калуга: REM→Куда, Event→Событие (как в finál)."""
    df = df.copy()
    if "Date" in df.columns:
        df["Дата"] = pd.to_datetime(df.get("Дата", df["Date"]), errors="coerce")
    else:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    if "Event" in df.columns:
        raw = df["Event"].astype(str).str.strip().str.upper()
        if "Событие" not in df.columns:
            df["Событие"] = df["Event"]
        so = df["Событие"].astype(str).str.strip().str.upper()
        df.loc[raw.str.contains("SOLD", na=False), "Событие"] = "ПРОДАНА"
        df.loc[raw.str.contains("BRED", na=False), "Событие"] = "ОСЕМЕН"
        df.loc[so.str.contains("SOLD", na=False), "Событие"] = "ПРОДАНА"
        df.loc[so.str.contains("BRED", na=False), "Событие"] = "ОСЕМЕН"
    if "Куда" not in df.columns and "REM" in df.columns:
        df["Куда"] = df["REM"].astype(str).str.strip()
    elif "Куда" in df.columns:
        empty = df["Куда"].isna() | (df["Куда"].astype(str).str.strip() == "")
        if "REM" in df.columns:
            df.loc[empty, "Куда"] = df.loc[empty, "REM"].astype(str).str.strip()
    if "LACT" not in df.columns and "Lact" in df.columns:
        df["LACT"] = df["Lact"]
    if "Lact" not in df.columns and "LACT" in df.columns:
        df["Lact"] = df["LACT"]
    if "ключ_коровы" not in df.columns:
        if "REG" in df.columns:
            df["ключ_коровы"] = df["REG"].astype(str)
    return df


@dataclass
class PipelineConfig:
    name: str = "ЖК Высокое"
    work_dir: Path = field(default_factory=lambda: ROOT)
    filter_folder: str = "фильтр_ЖК_Высокое"
    events_path: Path = field(default_factory=lambda: ROOT / "d1" / "События-пo-korovam.xlsx")
    events_aux_path: Path = field(
        default_factory=lambda: ROOT / "d1" / "События-пo-korovam (1).xlsx"
    )
    bulls_path: Path = field(default_factory=lambda: ROOT / "быки_полная_база.xlsx")
    lactation_path: Path = field(
        default_factory=lambda: ROOT / "d1" / "поголовье_по_лактациям_январь2022_декабрь2025.xlsx"
    )
    output_xlsx: Path = field(default_factory=lambda: ROOT / "прогноз_всех_параметров_ЖК_Высокое.xlsx")
    subdivision_names: list[str] = field(default_factory=lambda: ["ЖК Высокое"])
    kuda_buy_tokens: list[str] = field(default_factory=lambda: list(VYSOKOE_BUY))
    sold_heifer_kuda_tokens: list[str] = field(default_factory=lambda: list(VYSOKOE_HEIFER_SALE))
    sold_bull_kuda_tokens: list[str] = field(default_factory=lambda: list(VYSOKOE_BULL_SALE))
    sales_require_pereezd: bool = False
    exit_event_types: list[str] = field(default_factory=lambda: ["ВЫБЫТИЕ", "ПРОДАНА", "SOLD"])
    sold_heifer_dest: str = "МТФ_ВЫСОКОЕ"
    sold_bull_dest: str = "БЫЧКИ"
    kaluga_farm: str | None = None
    kaluga_unit: str | None = None
    kaluga_data_dir: Path | None = None
    kaluga_internal_tokens: list[str] = field(default_factory=list)
    # Остатки на 30.09.2024: dry, furazh, milk, L1..L5+ (из база_30.09.2024_поголовье.xlsx)
    sep2024_baseline: dict[str, float] | None = None
    baseline_xlsx: Path | None = field(default_factory=lambda: KALUGA_SEP2024_BASELINE_XLSX)
    detail_log_path: Path | None = None
    forecast_only: bool = field(
        default_factory=lambda: os.environ.get("PIPELINE_FORECAST_ONLY", "0").strip()
        in ("1", "true", "True", "yes")
    )
    fast_train: bool = field(default_factory=_pipeline_fast_enabled)
    train_end_ts: pd.Timestamp | None = None
    predict_months: list[tuple[int, int]] | None = None
    month_cols: list[str] | None = None

    def resolved_train_end(self) -> pd.Timestamp:
        if self.train_end_ts is not None and not pd.isna(self.train_end_ts):
            return pd.Timestamp(self.train_end_ts).normalize()
        return TRAIN_END

    def resolved_predict_months(self) -> list[tuple[int, int]]:
        return self.predict_months if self.predict_months else PREDICT_MONTHS

    def resolved_month_cols(self) -> list[str]:
        if self.month_cols:
            return list(self.month_cols)
        return [_col_label(y, m) for y, m in self.resolved_predict_months()]


def _month_key(y: int, m: int) -> tuple[int, int]:
    return (y, m)


def _col_label(y: int, m: int) -> str:
    return f"{y}-{m:02d}"


def load_cell_sources() -> dict[int, str]:
    text = EXTRACTED.read_text(encoding="utf-8")
    parts = re.split(r"# ===== NOTEBOOK CELL (\d+) =====\n", text)
    cells: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        cells[int(parts[i])] = parts[i + 1]
    return cells


def remove_dict_assignment(src: str, var_name: str) -> str:
    pattern = rf"(\n{re.escape(var_name)}\s*=\s*)\{{"
    m = re.search(pattern, src)
    if not m:
        return src
    start = m.start() + 1
    depth = 0
    i = m.end() - 1
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                inject = f"{var_name} = _INJECT.get('{var_name}', {{}})\n"
                return src[:start] + inject + src[end:]
        i += 1
    return src


def _trade_preamble(cfg: PipelineConfig) -> str:
    heifer = [t.upper() for t in cfg.sold_heifer_kuda_tokens]
    bull = [t.upper() for t in cfg.sold_bull_kuda_tokens]
    buy = [t.upper() for t in cfg.kuda_buy_tokens]
    exit_ev = [e.upper() for e in cfg.exit_event_types]
    kaluga_all = [t.upper() for t in cfg.kaluga_internal_tokens]
    return f"""
_SUBDIV_NAMES = {list(cfg.subdivision_names)!r}
_HEIFER_SALE_KUDA = {heifer!r}
_BULL_SALE_KUDA = {bull!r}
_KUDA_BUY = {buy!r}
_KALUGA_INTERNAL_KUDA = {kaluga_all!r}
_EXIT_EVENTS = {exit_ev!r}
_REQUIRE_PEREEZD = {cfg.sales_require_pereezd!r}

def _subdiv_mask(series):
    return series.astype(str).str.strip().isin(_SUBDIV_NAMES)

def _match_kuda(series, tokens):
    def ok(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        u = str(x).upper().strip()
        return any(t in u for t in tokens)
    return series.apply(ok)

def _dest_rem_kuda(row):
    parts = []
    for c in ('Куда', 'REM'):
        if c not in row.index:
            continue
        val = row[c]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if s and s.lower() not in ('nan', '-', ''):
            parts.append(s.upper())
    return ' '.join(parts)

def _bull_sale_blob(row):
    parts = []
    for c in ('Куда', 'CARX'):
        if c not in row.index:
            continue
        val = row[c]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if s and s.lower() not in ('nan', '-', ''):
            parts.append(s.upper())
    return ' '.join(parts)

def _is_pereezd(row):
    if 'CARX' not in row.index:
        return False
    val = row['CARX']
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return 'ПЕРЕЕЗД' in str(val).upper()

def _heifer_internal_dest(dest, pereezd):
    if any(t in dest for t in _HEIFER_SALE_KUDA):
        return True
    if pereezd and _KALUGA_INTERNAL_KUDA and any(t in dest for t in _KALUGA_INTERNAL_KUDA):
        return True
    return False
"""


def _apply_trade_patches(src: str, cfg: PipelineConfig) -> str:
    needs_trade = (
        "ЖК Высокое" in src
        or "Столбец1" in src
        or "kuda_heifers" in src
        or "calculate_parameters" in src
    )
    if not needs_trade:
        return src
    if "_HEIFER_SALE_KUDA" not in src:
        src = _trade_preamble(cfg) + src
    src = src.replace("df['Столбец1'] == 'ЖК Высокое'", "_subdiv_mask(df['Столбец1'])")
    src = src.replace("(df['Столбец1'] == 'ЖК Высокое')", "_subdiv_mask(df['Столбец1'])")
    src = src.replace(
        "df_events = df_events[df_events['Столбец1'] == 'ЖК Высокое'].copy()",
        "df_events = df_events[_subdiv_mask(df_events['Столбец1'])].copy()",
    )
    old_mask = (
        "mask_kuda = (\n"
        "    df_filtered['Куда'].str.upper().str.strip().isin(['МТФ_ВЫСОКОЕ', 'МТФ ВЫСОКОЕ']) |\n"
        "    df_filtered['Куда'].str.upper().str.strip().str.contains('МТФ_ВЫСОКОЕ', na=False) |\n"
        "    df_filtered['Куда'].str.upper().str.strip().str.contains('МТФ ВЫСОКОЕ', na=False)\n"
        ")"
    )
    src = src.replace(old_mask, "mask_kuda = _match_kuda(df_filtered['Куда'], _HEIFER_SALE_KUDA)")
    if cfg.kaluga_internal_tokens:
        src = src.replace(
            "mask_kuda = _match_kuda(df_filtered['Куда'], _HEIFER_SALE_KUDA)",
            "_dest_trade = df_filtered['Куда'].astype(str).str.upper()\n"
            "if 'REM' in df_filtered.columns:\n"
            "    _dest_trade = (_dest_trade + ' ' + df_filtered['REM'].astype(str).str.upper())\n"
            "_pereezd_m = (\n"
            "    df_filtered['CARX'].astype(str).str.upper().str.contains('ПЕРЕЕЗД', na=False)\n"
            "    if 'CARX' in df_filtered.columns\n"
            "    else pd.Series(False, index=df_filtered.index)\n"
            ")\n"
            "_bull_blob = df_filtered.apply(_bull_sale_blob, axis=1)\n"
            "mask_kuda = (\n"
            "    (_match_kuda(_dest_trade, _HEIFER_SALE_KUDA) | (_pereezd_m & _match_kuda(_dest_trade, _KALUGA_INTERNAL_KUDA)))\n"
            "    & ~_match_kuda(_bull_blob, _BULL_SALE_KUDA)\n"
            ")",
        )
    src = src.replace(
        "(df['Куда'].str.upper().str.contains('БЫЧКИ', na=False))",
        "_match_kuda(df['Куда'], _BULL_SALE_KUDA)",
    )
    src = _patch_df_bulls_sold_block(src)
    src = _patch_cow_info_bdat_in_calculate_parameters(src)
    src = re.sub(r"kuda_list = \[.*?\]", "kuda_list = list(_KUDA_BUY)", src, count=0)
    src = re.sub(r"kuda_heifers = \[.*?\]", "kuda_heifers = list(_HEIFER_SALE_KUDA)", src, count=0)
    src = src.replace(
        "df['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА'])",
        "df['Событие'].astype(str).str.strip().str.upper().isin(_EXIT_EVENTS)",
    )
    src = src.replace(
        "mask_exit = df['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА'])",
        "mask_exit = df['Событие'].astype(str).str.strip().str.upper().isin(_EXIT_EVENTS)",
    )
    src = src.replace(
        "if event == 'ОСЕМЕН' and r_val == 'P':",
        "if event in ('ОСЕМЕН', 'BRED') and r_val == 'P':",
    )
    needle = "df_heifers_sold = df_filtered[mask_kuda & mask_lact].copy()"
    if needle in src and cfg.sales_require_pereezd and not cfg.kaluga_internal_tokens:
        src = src.replace(
            needle,
            needle
            + "\nif _REQUIRE_PEREEZD and 'CARX' in df_heifers_sold.columns:\n"
            + "    df_heifers_sold = df_heifers_sold[df_heifers_sold['CARX'].astype(str).str.upper().str.contains('ПЕРЕЕЗД', na=False)].copy()",
        )
    src = src.replace(
        "if lact == 0 and 'БЫЧКИ' in kuda:",
        "if lact == 0 and any(v in kuda for v in _BULL_SALE_KUDA):",
    )
    return src


def _patch_df_bulls_sold_block(src: str) -> str:
    """Ячейка 23: продажа бычков — выбытие, LACT=0, БЫЧКИ в Кuda или CARX."""
    new_block = (
        "_kuda_u = df['Куда'].astype(str).str.upper() if 'Куда' in df.columns else pd.Series('', index=df.index)\n"
        "_rem_u = df['REM'].astype(str).str.upper() if 'REM' in df.columns else pd.Series('', index=df.index)\n"
        "_carx_u = df['CARX'].astype(str).str.upper() if 'CARX' in df.columns else pd.Series('', index=df.index)\n"
        "_bull_blob_all = (_kuda_u.fillna('') + ' ' + _rem_u.fillna('') + ' ' + _carx_u.fillna('')).str.replace('NAN', '', regex=False)\n"
        "_mask_bull_sale = False\n"
        "for _t in _BULL_SALE_KUDA:\n"
        "    _mask_bull_sale = _mask_bull_sale | _bull_blob_all.str.contains(str(_t), na=False, regex=False)\n"
        "_lact0_b = pd.to_numeric(df['LACT'], errors='coerce').fillna(-1) == 0\n"
        "_mask_exit_bull = df['Событие'].astype(str).str.strip().str.upper().isin(_EXIT_EVENTS)\n"
        "df_bulls_sold = df[_subdiv_mask(df['Столбец1']) & _mask_bull_sale & _lact0_b & _mask_exit_bull].copy()"
    )
    for old in (
        "df_bulls_sold = df[\n"
        "    (df['Столбец1'] == 'ЖК Высокое') &\n"
        "    (df['Куда'].str.upper().str.contains('БЫЧКИ', na=False)) &\n"
        "    (df['LACT'] == 0)\n"
        "].copy()",
        "df_bulls_sold = df[\n"
        "    _subdiv_mask(df['Столбец1']) &\n"
        "    _match_kuda(df['Куда'], _BULL_SALE_KUDA) &\n"
        "    (df['LACT'] == 0)\n"
        "].copy()",
    ):
        if old in src:
            src = src.replace(old, new_block)
            break
    return src


def _patch_cow_info_bdat_in_calculate_parameters(src: str) -> str:
    old = (
        "        bdat = group.iloc[0]['BDAT']\n"
        "        if pd.isna(bdat):\n"
        "            continue\n"
    )
    new = (
        "        _bd = group['BDAT'].dropna()\n"
        "        if _bd.empty:\n"
        "            continue\n"
        "        bdat = _bd.iloc[0]\n"
    )
    if old in src:
        src = src.replace(old, new)
    return src


def _patch_bulls_remark_and_type(src: str) -> str:
    """Ячейка 5: Remark/Примечание + справочник быков Type/Плем."""
    old_bulls = (
        "df_bulls['Плем'] = df_bulls['Плем'].astype(str).str.strip().str.upper()\n"
        "df_bulls['тип_семени'] = df_bulls['Плем'].apply(lambda x: 'секс' if x == 'S' else 'обычное')\n"
        "bull_type_dict = dict(zip(df_bulls['Бык'], df_bulls['тип_семени']))"
    )
    new_bulls = (
        "_type_col = next((c for c in ('Type', 'Плем', 'TYPE') if c in df_bulls.columns), None)\n"
        "_name_col = next((c for c in ('Бык', 'Bull', 'Name', 'Имя') if c in df_bulls.columns), None)\n"
        "if _type_col is None:\n"
        "    df_bulls['Плем'] = ''\n"
        "    _type_col = 'Плем'\n"
        "if _name_col is None:\n"
        "    df_bulls['Бык'] = df_bulls.get('Бык', pd.Series(dtype=str))\n"
        "    _name_col = 'Бык'\n"
        "df_bulls['_ptype'] = df_bulls[_type_col].astype(str).str.strip().str.upper()\n"
        "df_bulls['тип_семени'] = df_bulls['_ptype'].apply(lambda x: 'секс' if x == 'S' else 'обычное')\n"
        "bull_type_dict = dict(\n"
        "    zip(df_bulls[_name_col].astype(str).str.strip(), df_bulls['тип_семени'])\n"
        ")"
    )
    if old_bulls in src:
        src = src.replace(old_bulls, new_bulls)
    old_semen = (
        "all_semen['тип_семени_быка'] = all_semen['Примечание'].apply(\n"
        "    lambda x: bull_type_dict.get(str(x).strip(), 'неизвестно') if pd.notna(x) else 'неизвестно'\n"
        ")"
    )
    new_semen = (
        "if 'Примечание' not in all_semen.columns and 'Remark' in all_semen.columns:\n"
        "    all_semen['Примечание'] = all_semen['Remark']\n"
        "elif 'Remark' in all_semen.columns:\n"
        "    all_semen['Примечание'] = all_semen['Примечание'].fillna(all_semen['Remark'])\n"
        "_semen_bull_name = (\n"
        "    all_semen['Примечание']\n"
        "    if 'Примечание' in all_semen.columns\n"
        "    else all_semen.get('Remark', pd.Series(index=all_semen.index, dtype=object))\n"
        ")\n"
        "all_semen['тип_семени_быка'] = _semen_bull_name.apply(\n"
        "    lambda x: bull_type_dict.get(str(x).strip(), 'неизвестно')\n"
        "    if pd.notna(x) and str(x).strip() not in ('', 'nan')\n"
        "    else 'неизвестно'\n"
        ")"
    )
    if old_semen in src:
        src = src.replace(old_semen, new_semen)
    return src


def _patch_birth_event_rozhd(src: str) -> str:
    """Калуга/DZ: рождения = РОЖД/ОТЕЛ (не только РОЖДЕН)."""
    birth_isin = (
        "df['Событие'].astype(str).str.upper().str.strip().str.replace('NAN', '', regex=False)"
        ".isin(('РОЖД', 'РОЖДЕН', 'CALF', 'CALVED', 'РОЖДЕНИЕ', "
        "'ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'))"
    )
    calv_isin = (
        "df['Событие'].astype(str).str.upper().str.strip().str.replace('NAN', '', regex=False)"
        ".isin(('ОТЕЛ', 'ОТЁЛ', 'CALVING', 'CALVED', 'ОТЕЛЕНИЕ', "
        "'РОЖД', 'РОЖДЕН', 'CALF', 'РОЖДЕНИЕ'))"
    )
    birth_block_old = (
        "    # Только события \"Рожден\"\n"
        "    birth_mask = df['Событие'].str.upper().str.strip() == 'РОЖДЕН'\n"
        "    df = df[birth_mask].copy()"
    )
    birth_block_new = (
        "    # Рождения: РОЖД/РОЖДЕН или отёлы (Калуга — Отелы_YYYY)\n"
        f"    birth_mask = {birth_isin}\n"
        "    if not birth_mask.any() and len(df) > 0:\n"
        f"        birth_mask = {calv_isin}\n"
        "    df = df[birth_mask].copy()"
    )
    if birth_block_old in src:
        src = src.replace(birth_block_old, birth_block_new)
    else:
        src = src.replace(
            "birth_mask = df['Событие'].str.upper().str.strip() == 'РОЖДЕН'",
            f"birth_mask = {birth_isin}",
        )
    src = src.replace(
        "all_calvings = all_calvings[all_calvings['Событие'].str.upper().str.strip() == 'РОЖДЕН']",
        "all_calvings = all_calvings[all_calvings['Событие'].astype(str).str.upper().str.strip()"
        ".isin(('РОЖД', 'РОЖДЕН', 'CALF', 'CALVED', 'РОЖДЕНИЕ', "
        "'ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'))]",
    )
    src = src.replace(
        "print(f\"  Телочки: {train_df['телочки'].sum()} ({train_df['телочки'].sum()/train_df['отелы'].sum()*100:.1f}%)\")\n"
        "print(f\"  Бычки: {train_df['бычки'].sum()} ({train_df['бычки'].sum()/train_df['отелы'].sum()*100:.1f}%)\")",
        "_birth_total = train_df['отелы'].sum()\n"
        "if _birth_total:\n"
        "    print(f\"  Телочки: {train_df['телочки'].sum()} ({train_df['телочки'].sum()/_birth_total*100:.1f}%)\")\n"
        "    print(f\"  Бычки: {train_df['бычки'].sum()} ({train_df['бычки'].sum()/_birth_total*100:.1f}%)\")\n"
        "else:\n"
        "    print(\"  Телочки: 0 (нет данных)\")\n"
        "    print(\"  Бычки: 0 (нет данных)\")",
    )
    return src


# Типы событий «отёл» (Калуга/DZ: CALVING, РОЖД, … — не только ОТЕЛ).
_CALVING_EVENT_ISIN = (
    "df['Событие'].astype(str).str.upper().str.strip().str.replace('NAN', '', regex=False)"
    ".isin(('ОТЕЛ', 'ОТЁЛ', 'CALVING', 'CALVED', 'ОТЕЛЕНИЕ', "
    "'РОЖД', 'РОЖДЕН', 'CALF', 'РОЖДЕНИЕ'))"
)
_CALVING_EVENT_ISIN_EVENTS = (
    "df_events_train['Событие'].astype(str).str.upper().str.strip().str.replace('NAN', '', regex=False)"
    ".isin(('ОТЕЛ', 'ОТЁЛ', 'CALVING', 'CALVED', 'ОТЕЛЕНИЕ', "
    "'РОЖД', 'РОЖДЕН', 'CALF', 'РОЖДЕНИЕ'))"
)


def _patch_calving_event_types(src: str) -> str:
    """Finál cells 1/3/7/10: распознавание отёлов из Калуги (CALVING, РОЖД, …)."""
    old_otel = (
        "otel_mask = df['Событие'].str.upper().str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])"
    )
    new_otel = (
        f"otel_mask = {_CALVING_EVENT_ISIN}\n"
        "    if not otel_mask.any() and len(df) > 0 and 'Дата' in df.columns:\n"
        "        otel_mask = pd.to_datetime(df['Дата'], errors='coerce').notna()"
    )
    src = src.replace(old_otel, new_otel)
    src = src.replace(
        "df_calving_cows_train = df_events_train[(df_events_train['Событие'] == 'ОТЕЛ') &\n"
        "                                        (df_events_train['LACT'] >= 2)].copy()",
        "df_calving_cows_train = df_events_train["
        f"({_CALVING_EVENT_ISIN_EVENTS}) & "
        "(pd.to_numeric(df_events_train['LACT'], errors='coerce') >= 2)].copy()",
    )
    src = src.replace(
        "df_dry_train = df_events_train[df_events_train['тип_файла'] == 'ЗАПУСК'].copy()",
        "df_dry_train = df_events_train["
        "df_events_train['Событие'].astype(str).str.upper().str.strip() == 'ЗАПУСК'].copy()",
    )
    return src


def _patch_gridsearch_min_samples(src: str) -> str:
    """Не падать на GridSearchCV при <3 обучающих месяцах."""
    cell1_block = (
        "tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))\n"
        "grid_search = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)\n"
        "grid_search.fit(X_train, y_train)\n\n"
        "print(f\"Лучшие параметры: {grid_search.best_params_}\")\n"
        "print(f\"Лучшая MAE: {abs(grid_search.best_score_):.2f}\")\n\n"
        "final_model = XGBRegressor(**grid_search.best_params_, random_state=42)\n"
        "final_model.fit(X_train, y_train)"
    )
    cell1_new = (
        "_default_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.08, "
        "'subsample': 0.8, 'colsample_bytree': 0.8}\n"
        "_n_cv = min(3, len(train_clean) - 1)\n"
        "if _n_cv >= 2 and len(train_clean) >= 3:\n"
        "    tscv = TimeSeriesSplit(n_splits=_n_cv)\n"
        "    grid_search = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=tscv, "
        "scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)\n"
        "    grid_search.fit(X_train, y_train)\n"
        "    best_params = grid_search.best_params_\n"
        "    print(f\"Лучшие параметры: {best_params}\")\n"
        "    print(f\"Лучшая MAE: {abs(grid_search.best_score_):.2f}\")\n"
        "else:\n"
        "    best_params = _default_params\n"
        "    print(f\"  Мало обучающих месяцев ({len(train_clean)}), параметры по умолчанию\")\n\n"
        "final_model = XGBRegressor(**best_params, random_state=42)\n"
        "final_model.fit(X_train, y_train)"
    )
    if cell1_block in src:
        src = src.replace(cell1_block, cell1_new, 1)
    return src


def _patch_split_calvings_lact(src: str) -> str:
    """Cell 3/7: LACT из Lact; если лактация не заполнена — эвристика 85/15."""
    old = (
        "    df['LACT'] = df['LACT'].fillna(0)\n\n"
        "    cows = df[df['LACT'] >= 2].copy()\n"
        "    heifers = df[df['LACT'] == 1].copy()\n\n"
        "    return cows, heifers"
    )
    new = (
        "    if 'LACT' not in df.columns and 'Lact' in df.columns:\n"
        "        df['LACT'] = df['Lact']\n"
        "    df['LACT'] = pd.to_numeric(df.get('LACT', 0), errors='coerce').fillna(0)\n\n"
        "    cows = df[df['LACT'] >= 2].copy()\n"
        "    heifers = df[df['LACT'] == 1].copy()\n"
        "    if len(cows) == 0 and len(heifers) == 0 and len(df) > 0:\n"
        "        n_h = max(1, int(round(len(df) * 0.15)))\n"
        "        heifers = df.iloc[:n_h].copy()\n"
        "        cows = df.iloc[n_h:].copy()\n\n"
        "    return cows, heifers"
    )
    if old in src:
        src = src.replace(old, new)
    return src


_PREDICT_MONTHS_BLOCK = (
    "predict_months = []\n"
    "for month in [10, 11, 12]:\n"
    "    predict_months.append((2024, month))\n"
    "for month in range(1, 13):\n"
    "    predict_months.append((2025, month))"
)


def _patch_predict_months_loops(src: str) -> str:
    """Finál cells: горизонт прогноза из _PREDICT_MONTHS (UI), не Oct2024–Dec2025."""
    if _PREDICT_MONTHS_BLOCK in src:
        src = src.replace(_PREDICT_MONTHS_BLOCK, "predict_months = list(_PREDICT_MONTHS)")
    return src


def _patch_cell10_furazh_baseline(src: str, baseline: dict[str, float] | None) -> str:
    """Если furazh_forecast пуст (cell 12 ещё не считался) — база с якоря."""
    if not baseline or baseline.get("furazh") is None:
        return src
    fur = int(round(float(baseline["furazh"])))
    old = "    furazh = furazh_forecast.get((year, month), 0)"
    new = (
        "    furazh = furazh_forecast.get((year, month))\n"
        f"    if furazh is None or (isinstance(furazh, (int, float)) and furazh <= 0):\n"
        f"        furazh = {fur}"
    )
    if old in src:
        src = src.replace(old, new, 1)
    return src


def _patch_cell25_trade_and_features(src: str) -> str:
    """Ячейка 25: BDAT с строки события, LACT numeric, str-колонки в add_features."""
    old_loop = (
        "                bdat = info.get('bdat')\n"
        "                has_success = info.get('has_success', False)\n"
        "                kuda = str(event['Куда']).upper().strip()\n"
        "                lact = event.get('LACT', 0)\n"
        "                event_date = event['Дата']\n"
        "                \n"
        "                # Бычки\n"
        "                if lact == 0 and any(v in kuda for v in _BULL_SALE_KUDA):"
    )
    new_loop = (
        "                bdat = info.get('bdat')\n"
        "                if bdat is None or (isinstance(bdat, float) and pd.isna(bdat)):\n"
        "                    bdat = event.get('BDAT', bdat)\n"
        "                has_success = info.get('has_success', False)\n"
        "                kuda = _dest_rem_kuda(event)\n"
        "                lact = pd.to_numeric(event.get('LACT', event.get('Lact', -1)), errors='coerce')\n"
        "                if pd.isna(lact):\n"
        "                    lact = -1\n"
        "                event_date = event['Дата']\n"
        "                _bull_dest = _bull_sale_blob(event)\n"
        "                \n"
        "                # Бычки\n"
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):"
    )
    if old_loop in src:
        src = src.replace(old_loop, new_loop)
    src = src.replace(
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age == '0-6':\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age in ('0-2', '2-6', '0-6'):\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
    )
    src = src.replace(
        "                if lact == 0 and any(v in kuda for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age == '0-6':\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age in ('0-2', '2-6', '0-6'):\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
    )
    src = src.replace(
        "                # Продажа телок\n"
        "                if lact == 0 and any(v in kuda for v in [x.upper() for x in kuda_heifers]):\n"
        "                    age = get_age_group_exact(bdat, event_date)\n"
        "                    if age:\n"
        "                        row[f'продажа_телки_{age}_внутри'] += 1\n"
        "                    continue",
        "                # Продажа телок (хозяйство / ПЕРЕЕЗД в подразделение Калуги)\n"
        "                _pz = _is_pereezd(event)\n"
        "                if lact == 0 and _heifer_internal_dest(kuda, _pz) and not any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_exact(bdat, event_date)\n"
        "                    if age:\n"
        "                        row[f'продажа_телки_{age}_внутри'] += 1\n"
        "                    continue",
    )
    src = src.replace(
        "                if lact == 0 and any(v in kuda for v in [x.upper() for x in kuda_list]):\n",
        "                if lact == 0 and any(v in kuda for v in _KUDA_BUY):\n",
    )
    src = src.replace(
        "    semen = df_raw[df_raw['Событие'].str.strip() == 'ОСЕМЕН']\n",
        "    _ev = df_raw['Событие'].astype(str).str.strip().str.upper()\n"
        "    semen = df_raw[_ev.isin(('ОСЕМЕН', 'BRED', 'ОСЕМЕНЕНИЕ'))]\n",
    )
    src = src.replace(
        "    semen_p = semen[semen['R'].str.strip() == 'P'].groupby(['год', 'месяц']).size().reset_index(name='осеменения_успешные')\n",
        "    semen_p = semen[semen['R'].astype(str).str.strip() == 'P']"
        ".groupby(['год', 'месяц']).size().reset_index(name='осеменения_успешные')\n",
    )
    src = src.replace(
        "    semen_failed = semen[~semen['R'].str.strip().isin(['P'])].groupby(['год', 'месяц']).size().reset_index(name='осеменения_неуспешные')\n",
        "    semen_failed = semen[~semen['R'].astype(str).str.strip().isin(['P'])]"
        ".groupby(['год', 'месяц']).size().reset_index(name='осеменения_неуспешные')\n",
    )
    src = src.replace(
        "    calvings = df_raw[df_raw['Событие'].str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])]\n",
        "    calvings = df_raw[_ev.isin(('ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ', 'РОЖД', 'РОЖДЕН'))]\n",
    )
    src = src.replace(
        "    dry = df_raw[df_raw['Событие'].str.strip() == 'ЗАПУСК']\n",
        "    dry = df_raw[_ev.isin(('ЗАПУСК', 'DRY'))]\n",
    )
    src = src.replace(
        "    exits = df_raw[df_raw['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА'])]\n",
        "    exits = df_raw[_ev.isin(_EXIT_EVENTS)]\n" if "_EXIT_EVENTS" in src else
        "    exits = df_raw[_ev.isin(('ВЫБЫТИЕ', 'ПРОДАНА', 'SOLD'))]\n",
    )
    return src


def _patch_cell25_bull_sales_full(src: str) -> str:
    """Ячейка 25: продажа бычков по всем возрастным группам (не только 0-6)."""
    bull_cols = [
        "продажа_бычки_0-2_внутри",
        "продажа_бычки_2-6_внутри",
        "продажа_бычки_6-12_внутри",
        "продажа_бычки_12-18_внутри",
        "продажа_бычки_18+_внутри",
    ]
    bull_target_lines = "\n".join(f"        '{c}'," for c in bull_cols)
    src = src.replace("        'продажа_бычки_0-6_внутри',", bull_target_lines)
    src = src.replace(
        "            row['продажа_бычки_0-6_внутри'] = 0\n",
        "            for age in ['0-2', '2-6', '6-12', '12-18', '18+']:\n"
        "                row[f'продажа_бычки_{age}_внутри'] = 0\n",
    )
    for old_bull in (
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age in ('0-2', '2-6', '0-6'):\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
        "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
        "                    age = get_age_group_bulls(bdat, event_date)\n"
        "                    if age == '0-6':\n"
        "                        row['продажа_бычки_0-6_внутри'] += 1\n"
        "                    continue",
    ):
        src = src.replace(
            old_bull,
            "                if lact == 0 and any(v in _bull_dest for v in _BULL_SALE_KUDA):\n"
            "                    age = get_age_group_exact(bdat, event_date)\n"
            "                    if age:\n"
            "                        row[f'продажа_бычки_{age}_внутри'] += 1\n"
            "                    continue",
        )
    return src


def _patch_cell25_vectorize(src: str) -> str:
    """Ячейка 25: vectorized calculate_parameters (без iterrows по событиям)."""
    if "def calculate_parameters(df):" not in src or "for _, event in month_events.iterrows():" not in src:
        return src

    new_fn = '''def calculate_parameters(df):
    print("\\n📊 Расчет параметров (vectorized)...")
    age_groups = ['0-2', '2-6', '6-12', '12-18', '18+']
    all_months = [
        (y, m) for y in range(2022, 2028) for m in range(1, 13)
    ]
    idx = pd.MultiIndex.from_tuples(all_months, names=['год', 'месяц'])

    def _empty_params():
        cols = {f'продажа_телки_{a}_внутри': 0 for a in age_groups}
        cols.update({f'покупка_телки_{a}_внутри': 0 for a in age_groups})
        cols.update({f'продажа_бычки_{a}_внутри': 0 for a in age_groups})
        cols['покупка_нетели_внутри'] = 0
        return pd.DataFrame(cols, index=idx).reset_index()

    _bdat_by_cow = df.groupby('ключ_коровы', sort=False)['BDAT'].first()
    _ev_ok = (
        df['Событие'].astype(str).str.strip().isin(('ОСЕМЕН', 'BRED'))
        & (df['R'].astype(str).str.strip() == 'P')
    )
    _has_success = (
        _ev_ok.groupby(df['ключ_коровы'], sort=False).any()
        .reindex(_bdat_by_cow.index, fill_value=False)
    )

    _ev = df['Событие'].astype(str).str.strip().str.upper()
    ex = df.loc[_ev.isin(_EXIT_EVENTS if '_EXIT_EVENTS' in globals() else ('ВЫБЫТИЕ', 'ПРОДАНА', 'SOLD'))].copy()
    if ex.empty:
        out = _empty_params()
        print(f"  Рассчитано {len(out)} месяцев")
        return out

    ex['год'] = ex['Дата'].dt.year
    ex['месяц'] = ex['Дата'].dt.month
    ex['lact_n'] = pd.to_numeric(ex.get('LACT', ex.get('Lact', -1)), errors='coerce').fillna(-1)
    ex = ex.loc[ex['lact_n'] == 0].copy()
    if ex.empty:
        out = _empty_params()
        print(f"  Рассчитано {len(out)} месяцев")
        return out

    _kuda_u = ex['Куда'].astype(str).str.upper() if 'Куда' in ex.columns else pd.Series('', index=ex.index)
    _rem_u = ex['REM'].astype(str).str.upper() if 'REM' in ex.columns else pd.Series('', index=ex.index)
    kuda = (_kuda_u.fillna('') + ' ' + _rem_u.fillna('')).str.replace('NAN', '', regex=False).str.strip()
    _carx_u = ex['CARX'].astype(str).str.upper() if 'CARX' in ex.columns else pd.Series('', index=ex.index)
    _bull_blob = (_kuda_u.fillna('') + ' ' + _carx_u.fillna('')).str.replace('NAN', '', regex=False)
    _pz = _carx_u.str.contains('ПЕРЕЕЗД', na=False)

    bdat = ex['ключ_коровы'].map(_bdat_by_cow)
    if 'BDAT' in ex.columns:
        bdat = bdat.fillna(pd.to_datetime(ex['BDAT'], errors='coerce'))
    hs = ex['ключ_коровы'].map(_has_success).fillna(False).astype(bool)

    def _age_months_vec(bdat_s, edate_s):
        bdat_s = pd.to_datetime(bdat_s, errors='coerce')
        edate_s = pd.to_datetime(edate_s, errors='coerce')
        months = (edate_s.dt.year - bdat_s.dt.year) * 12 + (edate_s.dt.month - bdat_s.dt.month)
        months = months - (edate_s.dt.day < bdat_s.dt.day).astype('int64')
        return months

    def _age_group_exact_vec(bdat_s, edate_s):
        m = _age_months_vec(bdat_s, edate_s)
        out = pd.Series(pd.NA, index=m.index, dtype=object)
        ok = m.notna() & (m >= 0)
        out.loc[ok & (m < 2)] = '0-2'
        out.loc[ok & (m >= 2) & (m < 6)] = '2-6'
        out.loc[ok & (m >= 6) & (m < 12)] = '6-12'
        out.loc[ok & (m >= 12) & (m < 18)] = '12-18'
        out.loc[ok & (m >= 18)] = '18+'
        return out

    def _age_group_bulls_vec(bdat_s, edate_s):
        return _age_group_exact_vec(bdat_s, edate_s)

    def _blob_match(blob, tokens):
        m = pd.Series(False, index=blob.index)
        for t in tokens:
            m = m | blob.str.contains(str(t), na=False, regex=False)
        return m

    is_bull = _blob_match(_bull_blob, _BULL_SALE_KUDA if '_BULL_SALE_KUDA' in globals() else ['БЫЧКИ'])
    is_buy = _blob_match(kuda, _KUDA_BUY if '_KUDA_BUY' in globals() else [])
    is_sell = _blob_match(kuda, _HEIFER_SALE_KUDA if '_HEIFER_SALE_KUDA' in globals() else [])
    if '_KALUGA_INTERNAL_KUDA' in globals() and _KALUGA_INTERNAL_KUDA:
        is_sell = is_sell | (_pz & _blob_match(kuda, _KALUGA_INTERNAL_KUDA))

    cat_bull = is_bull
    cat_buy = is_buy & ~cat_bull
    cat_sell = is_sell & ~cat_bull & ~is_buy

    def _counts_by_month_age(frame, ages, col_prefix):
        base = pd.DataFrame(0, index=idx, columns=ages, dtype=int)
        if frame is None or len(frame) == 0 or 'возраст_группа' not in frame.columns:
            out = base.copy()
            out.columns = [f'{col_prefix}_{a}_внутри' for a in ages]
            return out.reset_index()
        g = (
            frame.groupby(['год', 'месяц', 'возраст_группа'], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        for a in ages:
            if a not in g.columns:
                g[a] = 0
        g = g.reindex(columns=ages, fill_value=0).reindex(idx, fill_value=0).astype(int)
        g.columns = [f'{col_prefix}_{a}_внутри' for a in ages]
        return g.reset_index()

    df_params = _empty_params()

    if cat_bull.any():
        bulls = ex.loc[cat_bull].copy()
        bulls['возраст_группа'] = _age_group_exact_vec(bdat.loc[cat_bull], bulls['Дата'])
        bulls = bulls[bulls['возраст_группа'].notna()]
        part = _counts_by_month_age(bulls, age_groups, 'продажа_бычки')
        for c in part.columns:
            if c not in ('год', 'месяц'):
                df_params[c] = part[c].to_numpy()

    if (cat_buy & hs).any():
        preg = ex.loc[cat_buy & hs]
        g = preg.groupby(['год', 'месяц']).size().reindex(idx, fill_value=0).astype(int)
        df_params['покупка_нетели_внутри'] = g.to_numpy()

    if (cat_buy & ~hs).any():
        buy_h = ex.loc[cat_buy & ~hs].copy()
        buy_h['возраст_группа'] = _age_group_exact_vec(bdat.loc[cat_buy & ~hs], buy_h['Дата'])
        buy_h = buy_h[buy_h['возраст_группа'].notna()]
        part = _counts_by_month_age(buy_h, age_groups, 'покупка_телки')
        for c in part.columns:
            if c not in ('год', 'месяц'):
                df_params[c] = part[c].to_numpy()

    if cat_sell.any():
        sell = ex.loc[cat_sell].copy()
        sell['возраст_группа'] = _age_group_exact_vec(bdat.loc[cat_sell], sell['Дата'])
        sell = sell[sell['возраст_группа'].notna()]
        part = _counts_by_month_age(sell, age_groups, 'продажа_телки')
        for c in part.columns:
            if c not in ('год', 'месяц'):
                df_params[c] = part[c].to_numpy()

    print(f"  Рассчитано {len(df_params)} месяцев")
    return df_params
'''
    start = src.index("def calculate_parameters(df):")
    assign = "df_params = calculate_parameters(df)"
    end = src.index(assign)
    return src[:start] + new_fn + "\n\n" + assign + "\n" + src[end + len(assign) :]


def _patch_cell25_train_test_split(src: str, cfg: PipelineConfig) -> str:
    """Ячейка 25: обучение до train_end, прогноз — cfg.predict_months (не Oct2024–Dec2025)."""
    if "train_data = df_features" not in src:
        return src
    te = cfg.resolved_train_end()
    ty, tm = int(te.year), int(te.month)
    pm = cfg.resolved_predict_months()
    old_train = (
        "train_data = df_features[\n"
        "    (df_features['год'] < 2024) | \n"
        "    ((df_features['год'] == 2024) & (df_features['месяц'] <= 9))\n"
        "].copy()"
    )
    new_train = (
        f"train_data = df_features[\n"
        f"    (df_features['год'] < {ty}) | \n"
        f"    ((df_features['год'] == {ty}) & (df_features['месяц'] <= {tm}))\n"
        f"].copy()"
    )
    src = src.replace(old_train, new_train)
    old_test = (
        "test_data = df_features[\n"
        "    ((df_features['год'] == 2024) & (df_features['месяц'] >= 10)) |\n"
        "    (df_features['год'] == 2025)\n"
        "].copy()"
    )
    new_test = (
        f"_predict_keys = {list(pm)!r}\n"
        "test_data = df_features[\n"
        "    df_features.apply(lambda r: (int(r['год']), int(r['месяц'])) in set(_predict_keys), axis=1)\n"
        "].copy()"
    )
    src = src.replace(old_test, new_test)
    return src


def _patch_skip_gridsearch_when_fast(src: str) -> str:
    """PIPELINE_FAST=1: без GridSearchCV — один проход XGB с фиксированными параметрами."""
    old = """    try:
        tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
        grid_search = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"    Лучшие параметры: {best_params}")
    except:
        best_params = {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1, 
                       'subsample': 0.8, 'colsample_bytree': 0.8}
        print(f"    Использую стандартные параметры")"""
    new = """    best_params = {
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.08,
        'subsample': 0.8, 'colsample_bytree': 0.8,
    }"""
    if old in src:
        src = src.replace(old, new)
    return src


def _patch_semen_remark_in_folder_files(src: str) -> str:
    """Колонки осеменений из DZ: Remark вместо Примечание."""
    for yr in (2022, 2023, 2024):
        old = f'df_semen_{yr} = pd.read_excel(f"{{folder}}/Осеменения_{yr}.xlsx")'
        new = (
            f"_df_s{yr} = pd.read_excel(f\"{{folder}}/Осеменения_{yr}.xlsx\")\n"
            f"if 'Remark' in _df_s{yr}.columns and 'Примечание' not in _df_s{yr}.columns:\n"
            f"    _df_s{yr}['Примечание'] = _df_s{yr}['Remark']\n"
            f"df_semen_{yr} = _df_s{yr}"
        )
        src = src.replace(old, new)
    return src


def patch_cell_source(src: str, cfg: PipelineConfig) -> str:
    folder_path = Path(cfg.filter_folder)
    if not folder_path.is_absolute():
        folder_path = (cfg.work_dir / folder_path).resolve()
    src = src.replace('folder = "фильтр_ЖК_Высокое"', f'folder = r"{folder_path}"')
    src = src.replace("pd.read_excel('События-по-коровам.xlsx')", f'read_filter_excel(r"{cfg.events_path}")')
    src = src.replace(
        "df = pd.read_excel('События-по-коровам.xlsx')",
        f'df = read_filter_excel(r"{cfg.events_path}")',
    )
    aux = cfg.events_aux_path if cfg.events_aux_path.exists() else cfg.events_path
    src = src.replace('pd.read_excel("События-по-коровам (1).xlsx")', f'read_filter_excel(r"{aux}")')
    bulls_stub = (
        f"_bp = Path(r'{cfg.bulls_path}')\n"
        "if _bp.exists():\n"
        "    df_bulls = pd.read_excel(_bp)\n"
        "else:\n"
        '    df_bulls = pd.DataFrame({"Плем": [], "Бык": []})\n'
    )
    src = src.replace("df_bulls = pd.read_excel('быки_полная_база.xlsx')", bulls_stub)
    src = src.replace(
        'pd.read_excel("поголовье_по_лактациям_январь2022_декабрь2025.xlsx")',
        f'pd.read_excel(r"{cfg.lactation_path}")',
    )
    tokens = repr(cfg.kuda_buy_tokens)
    src = src.replace(
        "kuda_list = ['ЖК_ВЫСОК', 'ЖК_ВЫСОКОЕ', 'ЖКВЫСОК', 'ЖКВЫСОКО', 'ЖКВЫСОКОЕ']",
        "kuda_list = list(_KUDA_BUY)" if "_KUDA_BUY" in src else f"kuda_list = {tokens}",
    )
    src = src.replace('sold_heifer_dest = "МТФ_ВЫСОКОЕ"', f'sold_heifer_dest = "{cfg.sold_heifer_dest}"')
    src = src.replace("sold_heifer_dest = 'МТФ_ВЫСОКОЕ'", f"sold_heifer_dest = '{cfg.sold_heifer_dest}'")
    for var in (
        "forecast_model1",
        "calving_forecast",
        "culling_forecast",
        "furazh_forecast",
        "status_forecast",
    ):
        src = remove_dict_assignment(src, var)
    if cfg.fast_train:
        src = re.sub(
            r"'n_estimators':\s*\[[^\]]+\]",
            "'n_estimators': [100]",
            src,
        )
        src = re.sub(
            r"'max_depth':\s*\[[^\]]+\]",
            "'max_depth': [4]",
            src,
        )
        src = re.sub(
            r"'learning_rate':\s*\[[^\]]+\]",
            "'learning_rate': [0.08]",
            src,
        )
        src = src.replace("verbose=1", "verbose=0")
        src = _patch_skip_gridsearch_when_fast(src)
    src = src.replace("n_jobs=-1", "n_jobs=1")
    src = _patch_bulls_remark_and_type(src)
    src = _patch_semen_remark_in_folder_files(src)
    src = _patch_birth_event_rozhd(src)
    src = _patch_calving_event_types(src)
    src = _patch_split_calvings_lact(src)
    src = _patch_predict_months_loops(src)
    src = _patch_gridsearch_min_samples(src)
    src = _apply_trade_patches(src, cfg)
    src = _patch_cell23_vectorize(src)
    src = _patch_cell25_vectorize(src)
    src = _patch_cell25_bull_sales_full(src)
    src = _patch_cell25_train_test_split(src, cfg)
    src = _patch_cell25_trade_and_features(src)
    src = src.replace('pd.read_excel(f"{folder}/', 'read_filter_excel(f"{folder}/')
    train_end = cfg.resolved_train_end()
    train_end_s = train_end.strftime('%Y-%m-%d')
    src = re.sub(r"MAX_DATE_PROB = pd\.Timestamp\('[^']+'\)", f"MAX_DATE_PROB = pd.Timestamp('{train_end_s}')", src)
    src = re.sub(r"MAX_DATE = pd\.Timestamp\('[^']+'\)", f"MAX_DATE = pd.Timestamp('{train_end_s}')", src)
    src = re.sub(r"MAX_DATE_TRAIN = pd\.Timestamp\('[^']+'\)", f"MAX_DATE_TRAIN = pd.Timestamp('{train_end_s}')", src)
    predict_end = cfg.resolved_predict_months()[-1]
    predict_end_ts = pd.Timestamp(f'{predict_end[0]}-{predict_end[1]:02d}-01') + pd.offsets.MonthEnd(0)
    pe_s = predict_end_ts.strftime('%Y-%m-%d')
    src = re.sub(
        r"df_events = df_events\[df_events\['Дата'\] <= pd\.Timestamp\('[^']+'\)\]",
        f"df_events = df_events[df_events['Дата'] <= pd.Timestamp('{pe_s}')]",
        src,
    )
    src = _patch_lact_and_death_events(src)
    if cfg.kaluga_farm:
        ev_read = f'pd.read_excel(r"{cfg.events_path}", engine="openpyxl")'
        src = src.replace(
            f"df = {ev_read}",
            f"df = normalize_events_df({ev_read})",
        )
    return src


def _patch_cell23_vectorize(src: str) -> str:
    """Ячейка 23: cow_info / возраст / месячная агрегация без iterrows/apply по строкам."""
    if "cow_info = {}" not in src or "df_heifers_sold.apply(get_animal_age" not in src:
        return src

    old_cow = (
        "cow_info = {}\n"
        "for cow_key, group in df.groupby('ключ_коровы'):\n"
        "    group = group.sort_values('Дата')\n"
        "    if len(group) == 0:\n"
        "        continue\n"
        "    \n"
        "    bdat = group.iloc[0]['BDAT']\n"
        "    if pd.isna(bdat):\n"
        "        continue\n"
        "    \n"
        "    has_success = False\n"
        "    for _, row in group.iterrows():\n"
        "        event = str(row.get('Событие', '')).strip()\n"
        "        r_val = str(row.get('R', '')).strip()\n"
        "        if event in ('ОСЕМЕН', 'BRED') and r_val == 'P':\n"
        "            has_success = True\n"
        "            break\n"
        "    \n"
        "    cow_info[cow_key] = {\n"
        "        'bdat': bdat,\n"
        "        'has_success': has_success\n"
        "    }\n"
        "\n"
        "print(f\"  Информация о {len(cow_info):,} животных\")"
    )
    # Also match unpatched ОСЕМЕН-only variant
    old_cow_alt = old_cow.replace(
        "if event in ('ОСЕМЕН', 'BRED') and r_val == 'P':",
        "if event == 'ОСЕМЕН' and r_val == 'P':",
    )
    new_cow = (
        "# Векторизовано: BDAT = first after sort; has_success = any ОСЕМЕН/BRED + R=P\n"
        "_bdat_by_cow = df.groupby('ключ_коровы', sort=False)['BDAT'].first()\n"
        "_ev_ok = (\n"
        "    df['Событие'].astype(str).str.strip().isin(('ОСЕМЕН', 'BRED'))\n"
        "    & (df['R'].astype(str).str.strip() == 'P')\n"
        ")\n"
        "_has_success = (\n"
        "    _ev_ok.groupby(df['ключ_коровы'], sort=False).any()\n"
        "    .reindex(_bdat_by_cow.index, fill_value=False)\n"
        ")\n"
        "_valid_cows = _bdat_by_cow.dropna()\n"
        "# cow_info оставляем лёгким (для совместимости); возраст/P берутся из Series\n"
        "cow_info = {'_n': int(len(_valid_cows))}\n"
        "_bdat_map = _bdat_by_cow\n"
        "print(f\"  Информация о {len(_valid_cows):,} животных\")\n"
        "\n"
        "def _age_months_vec(bdat, edate):\n"
        "    bdat = pd.to_datetime(bdat, errors='coerce')\n"
        "    edate = pd.to_datetime(edate, errors='coerce')\n"
        "    months = (edate.dt.year - bdat.dt.year) * 12 + (edate.dt.month - bdat.dt.month)\n"
        "    months = months - (edate.dt.day < bdat.dt.day).astype('int64')\n"
        "    return months\n"
        "\n"
        "def _age_group_exact_vec(bdat, edate):\n"
        "    m = _age_months_vec(bdat, edate)\n"
        "    out = pd.Series(pd.NA, index=m.index, dtype=object)\n"
        "    ok = m.notna() & (m >= 0)\n"
        "    out.loc[ok & (m < 2)] = '0-2'\n"
        "    out.loc[ok & (m >= 2) & (m < 6)] = '2-6'\n"
        "    out.loc[ok & (m >= 6) & (m < 12)] = '6-12'\n"
        "    out.loc[ok & (m >= 12) & (m < 18)] = '12-18'\n"
        "    out.loc[ok & (m >= 18)] = '18+'\n"
        "    return out\n"
        "\n"
        "def _age_group_bulls_vec(bdat, edate):\n"
        "    return _age_group_exact_vec(bdat, edate)\n"
        "\n"
        "def _attach_age_group(frame, bull=False):\n"
        "    if frame is None or len(frame) == 0:\n"
        "        out = frame.copy() if frame is not None else pd.DataFrame()\n"
        "        if len(out):\n"
        "            out['возраст_группа'] = pd.Series(dtype=object)\n"
        "        return out\n"
        "    out = frame.copy()\n"
        "    bdat = out['ключ_коровы'].map(_bdat_map)\n"
        "    if bull:\n"
        "        out['возраст_группа'] = _age_group_bulls_vec(bdat, out['Дата'])\n"
        "    else:\n"
        "        out['возраст_группа'] = _age_group_exact_vec(bdat, out['Дата'])\n"
        "    out = out[out['возраст_группа'].notna()].copy()\n"
        "    out['год'] = out['Дата'].dt.year\n"
        "    out['месяц'] = out['Дата'].dt.month\n"
        "    return out"
    )
    if old_cow in src:
        src = src.replace(old_cow, new_cow)
    elif old_cow_alt in src:
        src = src.replace(old_cow_alt, new_cow)
    else:
        return src

    # Age applies → vectorized attach
    src = src.replace(
        "def get_animal_age(row):\n"
        "    cow_key = row['ключ_коровы']\n"
        "    event_date = row['Дата']\n"
        "    \n"
        "    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')\n"
        "    if len(cow_events) == 0:\n"
        "        return None\n"
        "    \n"
        "    bdat = cow_events.iloc[0]['BDAT']\n"
        "    if pd.isna(bdat):\n"
        "        return None\n"
        "    \n"
        "    return get_age_group_exact(bdat, event_date)\n"
        "\n"
        "df_heifers_sold['возраст_группа'] = df_heifers_sold.apply(get_animal_age, axis=1)\n"
        "df_heifers_sold = df_heifers_sold[df_heifers_sold['возраст_группа'].notna()].copy()\n"
        "df_heifers_sold['год'] = df_heifers_sold['Дата'].dt.year\n"
        "df_heifers_sold['месяц'] = df_heifers_sold['Дата'].dt.month",
        "df_heifers_sold = _attach_age_group(df_heifers_sold, bull=False)",
    )
    src = src.replace(
        "def get_animal_age_bulls(row):\n"
        "    cow_key = row['ключ_коровы']\n"
        "    event_date = row['Дата']\n"
        "    \n"
        "    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')\n"
        "    if len(cow_events) == 0:\n"
        "        return None\n"
        "    \n"
        "    bdat = cow_events.iloc[0]['BDAT']\n"
        "    if pd.isna(bdat):\n"
        "        return None\n"
        "    \n"
        "    return get_age_group_bulls(bdat, event_date)\n"
        "\n"
        "df_bulls_sold['возраст_группа'] = df_bulls_sold.apply(get_animal_age_bulls, axis=1)\n"
        "df_bulls_sold = df_bulls_sold[df_bulls_sold['возраст_группа'].notna()].copy()\n"
        "df_bulls_sold['год'] = df_bulls_sold['Дата'].dt.year\n"
        "df_bulls_sold['месяц'] = df_bulls_sold['Дата'].dt.month",
        "df_bulls_sold = _attach_age_group(df_bulls_sold, bull=True)",
    )
    src = src.replace(
        "df_buy['is_pregnant'] = df_buy['ключ_коровы'].apply(\n"
        "    lambda cow_key: cow_info.get(cow_key, {}).get('has_success', False)\n"
        ")",
        "df_buy['is_pregnant'] = df_buy['ключ_коровы'].map(_has_success).fillna(False).astype(bool)",
    )
    src = src.replace(
        "def get_animal_age_buy(row):\n"
        "    cow_key = row['ключ_коровы']\n"
        "    event_date = row['Дата']\n"
        "    \n"
        "    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')\n"
        "    if len(cow_events) == 0:\n"
        "        return None\n"
        "    \n"
        "    bdat = cow_events.iloc[0]['BDAT']\n"
        "    if pd.isna(bdat):\n"
        "        return None\n"
        "    \n"
        "    return get_age_group_exact(bdat, event_date)\n"
        "\n"
        "df_heifers_buy['возраст_группа'] = df_heifers_buy.apply(get_animal_age_buy, axis=1)\n"
        "df_heifers_buy = df_heifers_buy[df_heifers_buy['возраст_группа'].notna()].copy()\n"
        "df_heifers_buy['год'] = df_heifers_buy['Дата'].dt.year\n"
        "df_heifers_buy['месяц'] = df_heifers_buy['Дата'].dt.month",
        "df_heifers_buy = _attach_age_group(df_heifers_buy, bull=False)",
    )

    old_months = (
        "data = []\n"
        "for year, month in all_months:\n"
        "    row = {'год': year, 'месяц': month}\n"
        "    \n"
        "    # 1. Продажа телок (МТФ_ВЫСОКОЕ, LACT=0)\n"
        "    for age in age_groups:\n"
        "        val = len(df_heifers_sold[\n"
        "            (df_heifers_sold['год'] == year) & \n"
        "            (df_heifers_sold['месяц'] == month) & \n"
        "            (df_heifers_sold['возраст_группа'] == age)\n"
        "        ])\n"
        "        row[f'продажа_телки_{age}_внутри'] = val\n"
        "    \n"
        "    # 2. Продажа бычков (БЫЧКИ, LACT=0)\n"
        "    for age in bull_groups:\n"
        "        val = len(df_bulls_sold[\n"
        "            (df_bulls_sold['год'] == year) & \n"
        "            (df_bulls_sold['месяц'] == month) & \n"
        "            (df_bulls_sold['возраст_группа'] == age)\n"
        "        ])\n"
        "        row[f'продажа_бычки_{age}_внутри'] = val\n"
        "    \n"
        "    # 3. Покупка телок (без P)\n"
        "    for age in age_groups:\n"
        "        val = len(df_heifers_buy[\n"
        "            (df_heifers_buy['год'] == year) & \n"
        "            (df_heifers_buy['месяц'] == month) & \n"
        "            (df_heifers_buy['возраст_группа'] == age)\n"
        "        ])\n"
        "        row[f'покупка_телки_{age}_внутри'] = val\n"
        "    \n"
        "    # 4. Покупка нетелей (с P)\n"
        "    val = len(df_pregnant[\n"
        "        (df_pregnant['год'] == year) & \n"
        "        (df_pregnant['месяц'] == month)\n"
        "    ])\n"
        "    row['покупка_нетели_внутри'] = val\n"
        "    \n"
        "    data.append(row)\n"
        "\n"
        "df_results = pd.DataFrame(data)"
    )
    new_months = (
        "def _counts_by_month_age(frame, ages, col_prefix):\n"
        "    idx = pd.MultiIndex.from_tuples(all_months, names=['год', 'месяц'])\n"
        "    base = pd.DataFrame(0, index=idx, columns=ages, dtype=int)\n"
        "    if frame is None or len(frame) == 0 or 'возраст_группа' not in getattr(frame, 'columns', []):\n"
        "        out = base.copy()\n"
        "        out.columns = [f'{col_prefix}_{a}_внутри' for a in ages]\n"
        "        return out.reset_index()\n"
        "    g = (\n"
        "        frame.groupby(['год', 'месяц', 'возраст_группа'], observed=False)\n"
        "        .size()\n"
        "        .unstack(fill_value=0)\n"
        "    )\n"
        "    for a in ages:\n"
        "        if a not in g.columns:\n"
        "            g[a] = 0\n"
        "    g = g.reindex(columns=ages, fill_value=0).reindex(idx, fill_value=0).astype(int)\n"
        "    g.columns = [f'{col_prefix}_{a}_внутри' for a in ages]\n"
        "    return g.reset_index()\n"
        "\n"
        "df_results = _counts_by_month_age(df_heifers_sold, age_groups, 'продажа_телки')\n"
        "_bull_part = _counts_by_month_age(df_bulls_sold, age_groups, 'продажа_бычки')\n"
        "df_results = df_results.merge(_bull_part, on=['год', 'месяц'], how='left')\n"
        "_buy_part = _counts_by_month_age(df_heifers_buy, age_groups, 'покупка_телки')\n"
        "df_results = df_results.merge(_buy_part, on=['год', 'месяц'], how='left')\n"
        "_preg_idx = pd.MultiIndex.from_tuples(all_months, names=['год', 'месяц'])\n"
        "if len(df_pregnant):\n"
        "    _preg = df_pregnant.groupby(['год', 'месяц']).size().reindex(_preg_idx, fill_value=0).astype(int)\n"
        "else:\n"
        "    _preg = pd.Series(0, index=_preg_idx, dtype=int)\n"
        "df_results['покупка_нетели_внутри'] = _preg.to_numpy()\n"
        "df_results = df_results.fillna(0)\n"
        "for _c in df_results.columns:\n"
        "    if _c not in ('год', 'месяц'):\n"
        "        df_results[_c] = df_results[_c].astype(int)"
    )
    if old_months in src:
        src = src.replace(old_months, new_months)
    src = src.replace("bull_groups = ['0-2', '2-6', '0-6']", "bull_groups = age_groups")
    return src


def _patch_cell10_dry_fact_from_snapshot(src: str, cfg: PipelineConfig) -> str:
    """Заменить рекурсию с SUHOSTOYNYE_BASE_FEB_2022=261 на снимок из таблиц."""
    if "SUHOSTOYNYE_BASE_FEB_2022 = 261" not in src:
        return src
    te = cfg.resolved_train_end()
    ey, em = int(te.year), int(te.month)
    src = src.replace(
        "# Начальное значение: февраль 2022 = 261\nSUHOSTOYNYE_BASE_FEB_2022 = 261\n\n",
        "# Факт сухостойных для обучения — снимок на конец месяца (headAnalyzer / tab3 compute)\n",
    )
    recursive = (
        "# Рассчитываем сухостойных рекурсивно\n"
        "df_train_months['сухостойные'] = 0\n"
        "suhostoynye_prev = SUHOSTOYNYE_BASE_FEB_2022\n\n"
        "for idx, row in df_train_months.iterrows():\n"
        "    suhostoynye_current = (suhostoynye_prev +\n"
        "                           row['запуски'] -\n"
        "                           row['отелы_коров'] -\n"
        "                           row['выбытия_сухостойных'])\n\n"
        "    df_train_months.at[idx, 'сухостойные'] = suhostoynye_current\n"
        "    suhostoynye_prev = suhostoynye_current\n\n"
    )
    snapshot = (
        "from snapshot_zhk_vysokoe import load_vysokoe_raw_tables, excel_to_backtest_tables, monthly_dry_milking_fact_history\n"
        "_stock_raw = load_vysokoe_raw_tables(folder)\n"
        "_stock_bt = excel_to_backtest_tables(_stock_raw)\n"
        "_stock_hist = monthly_dry_milking_fact_history(\n"
        f"    _stock_bt, start_year=2022, start_month=1, end_year={ey}, end_month={em},\n"
        ")\n"
        'df_train_months = df_train_months.drop(columns=["сухостойные"], errors="ignore")\n'
        "df_train_months = df_train_months.merge(\n"
        '    _stock_hist[["год", "месяц", "сухостойные"]], on=["год", "месяц"], how="left",\n'
        ")\n"
        'df_train_months["сухостойные"] = df_train_months["сухостойные"].fillna(0)\n'
        'print("  (факт сухостойных: снимок из таблиц, без SUHOSTOYNYE_BASE)")\n\n'
    )
    if recursive not in src:
        return src
    return src.replace(recursive, snapshot, 1)


def load_dry_milk_monthly_fact(cfg: "PipelineConfig") -> pd.DataFrame:
    """Помесячный факт сухостойных / дойных / фуражных (снимок из таблиц)."""
    from snapshot_zhk_vysokoe import (  # noqa: WPS433
        excel_to_backtest_tables,
        load_vysokoe_raw_tables,
        monthly_dry_milking_fact_history,
    )

    folder_path = Path(cfg.filter_folder)
    if not folder_path.is_absolute():
        folder_path = (cfg.work_dir / folder_path).resolve()
    raw = load_vysokoe_raw_tables(str(folder_path))
    bt = excel_to_backtest_tables(raw)
    te = cfg.resolved_train_end()
    return monthly_dry_milking_fact_history(
        bt,
        start_year=2022,
        start_month=1,
        end_year=int(te.year),
        end_month=int(te.month),
    )


def _patch_lact_and_death_events(src: str) -> str:
    """Finál cell 18: LACT (Калуга) и DIED вместо ПАЛА."""
    src = src.replace(
        "    df = df_culling.copy()\n    df['Дата'] = pd.to_datetime(df['Дата'])",
        "    df = df_culling.copy()\n    if 'Lact' not in df.columns and 'LACT' in df.columns:\n"
        "        df['Lact'] = df['LACT']\n    df['Дата'] = pd.to_datetime(df['Дата'])",
    )
    src = src.replace(
        "death_mask = df['Событие'].str.upper().str.strip() == 'ПАЛА'",
        "death_mask = df['Событие'].astype(str).str.upper().str.strip().isin("
        "['ПАЛА', 'DIED', 'ПАЛ', 'DEAD', 'MORT', 'ПАДЕЖ'])",
    )
    src = src.replace(
        "        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'падеж_телочки', 'падеж_бычки'])",
        "        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'падеж_телочки', 'падеж_бычки', 'падеж_всего'])",
    )
    src = src.replace(
        "print(f\"\\nВсего месяцев: {len(monthly_deaths)}\")\n"
        "print(f\"Всего падежей: {monthly_deaths['падеж_всего'].sum():.0f}\")",
        "print(f\"\\nВсего месяцев: {len(monthly_deaths)}\")\n"
        "_death_total = monthly_deaths['падеж_всего'].sum() if len(monthly_deaths) else 0\n"
        "print(f\"Всего падежей: {_death_total:.0f}\")",
    )
    if ("df_culling_2022 = pd.read_excel" in src or "df_culling_2022 = read_filter_excel" in src) and "_ensure_lact_col" not in src:
        helper = (
            "\ndef _ensure_lact_col(df):\n"
            "    if df is None or len(df) == 0:\n"
            "        return df\n"
            "    out = df.copy()\n"
            '    if "Lact" not in out.columns and "LACT" in out.columns:\n'
            '        out["Lact"] = out["LACT"]\n'
            "    return out\n"
        )
        src = helper + src
        for yr in (2022, 2023, 2024, 2025):
            for reader in ("pd.read_excel", "read_filter_excel"):
                old = f"df_culling_{yr} = {reader}(f\"{{folder}}/Выбытие_{yr}.xlsx\")"
                if old in src:
                    src = src.replace(
                        old,
                        f"df_culling_{yr} = _ensure_lact_col({reader}(f\"{{folder}}/Выбытие_{yr}.xlsx\"))",
                    )
    src = src.replace(
        "print(f\"  Из них сухостойные: {total_suh} ({total_suh/total_furazh*100:.1f}%)\")\n"
        "print(f\"  Из них дойные: {total_doy} ({total_doy/total_furazh*100:.1f}%)\")",
        "if total_furazh:\n"
        "    print(f\"  Из них сухостойные: {total_suh} ({total_suh/total_furazh*100:.1f}%)\")\n"
        "    print(f\"  Из них дойные: {total_doy} ({total_doy/total_furazh*100:.1f}%)\")\n"
        "else:\n"
        "    print(f\"  Из них сухостойные: {total_suh}\")\n"
        "    print(f\"  Из них дойные: {total_doy}\")",
    )
    return src




def baseline_from_lactation_at_train_end(
    lact_df: pd.DataFrame,
    train_end: pd.Timestamp,
    dry_milk_fact: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """Остатки на конец месяца train_end из lactation_stock + dry/milk fact."""
    if not isinstance(lact_df, pd.DataFrame) or lact_df.empty:
        return None
    y, m = int(train_end.year), int(train_end.month)
    row = lact_df[(lact_df["год"] == y) & (lact_df["месяц"] == m)]
    if row.empty:
        return None
    r = row.iloc[-1]
    lcols = ["L1", "L2", "L3", "L4", "L5+"]
    furazh = sum(float(r.get(c, 0) or 0) for c in lcols if c in r.index)
    if furazh <= 0:
        return None
    out: dict[str, float] = {
        "furazh": furazh,
        "L1": float(r.get("L1", 0) or 0),
        "L2": float(r.get("L2", 0) or 0),
        "L3": float(r.get("L3", 0) or 0),
        "L4": float(r.get("L4", 0) or 0),
        "L5+": float(r.get("L5+", 0) or 0),
    }
    if isinstance(dry_milk_fact, pd.DataFrame) and not dry_milk_fact.empty:
        dm = dry_milk_fact[(dry_milk_fact["год"] == y) & (dry_milk_fact["месяц"] == m)]
        if not dm.empty:
            drow = dm.iloc[-1]
            if "сухостойные" in drow.index and pd.notna(drow["сухостойные"]):
                out["dry"] = float(drow["сухостойные"])
            if "дойные" in drow.index and pd.notna(drow["дойные"]):
                out["milk"] = float(drow["дойные"])
            if "фуражные" in drow.index and pd.notna(drow["фуражные"]):
                out["furazh"] = float(drow["фуражные"])
    if "dry" in out and "milk" not in out:
        out["milk"] = max(0.0, out["furazh"] - out["dry"])
    elif "milk" in out and "dry" not in out:
        out["dry"] = max(0.0, out["furazh"] - out["milk"])
    return out

def load_sep2024_baseline(
    farm: str | None,
    unit: str | None,
    path: Path | None = None,
) -> dict[str, float] | None:
    """Читает остатки на 30.09.2024 из Калуга/база_30.09.2024_поголовье.xlsx."""
    xlsx = path or KALUGA_SEP2024_BASELINE_XLSX
    if not farm or not unit or not xlsx or not Path(xlsx).is_file():
        return None
    try:
        df = pd.read_excel(xlsx, sheet_name="пайплайн_подразделения")
    except Exception:
        try:
            df = pd.read_excel(xlsx)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Не удалось прочитать базу 30.09.2024: {exc}")
            return None
    if df.empty or "Подразделение" not in df.columns:
        return None
    farm_n = str(farm).strip()
    unit_n = str(unit).strip()
    sub = df.copy()
    if "Хозяйство" in sub.columns:
        sub = sub[sub["Хозяйство"].astype(str).str.strip() == farm_n]
    hit = sub[sub["Подразделение"].astype(str).str.strip() == unit_n]
    if hit.empty and "имя_в_файле" in sub.columns:
        hit = sub[sub["имя_в_файле"].astype(str).str.strip() == unit_n]
    if hit.empty:
        aliases = SUBDIVISION_ALIASES.get(unit_n, [])
        if aliases:
            hit = sub[sub["Подразделение"].astype(str).str.strip().isin(aliases)]
    if hit.empty:
        print(f"⚠️ База 30.09.2024: нет строки для {farm_n} / {unit_n}")
        return None
    row = hit.iloc[0]

    def _num(col: str) -> float | None:
        if col not in row.index or pd.isna(row[col]):
            return None
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            return None
        if v < 0:
            return None
        return v

    out = {
        "dry": _num("Сухостой коровы"),
        "furazh": _num("Фуражные коровы"),
        "milk": _num("Дойные коровы"),
        "L1": _num("1 лактации"),
        "L2": _num("2 лактации"),
        "L3": _num("3 лактации"),
        "L4": _num("4 лактации"),
        "L5+": _num("5 и более лактации"),
    }
    # Пустая/нулевая фураж — не используем (нет данных в Forecast)
    if out.get("furazh") is None or out["furazh"] <= 0:
        print(f"⚠️ База 30.09.2024 для {unit_n}: фуражные пустые/0 — пропуск")
        return None
    if out.get("milk") is None and out.get("dry") is not None:
        out["milk"] = max(0.0, float(out["furazh"]) - float(out["dry"]))
    if out.get("dry") is None and out.get("milk") is not None:
        out["dry"] = max(0.0, float(out["furazh"]) - float(out["milk"]))
    print(
        f"📦 База 30.09.2024 ({farm_n} / {unit_n}): "
        f"фур={out.get('furazh')}, сух={out.get('dry')}, "
        f"L1–L5+={out.get('L1')},{out.get('L2')},{out.get('L3')},{out.get('L4')},{out.get('L5+')}"
    )
    return {k: float(v) for k, v in out.items() if v is not None}


def apply_sep2024_baseline_to_stock_fact(
    stock_fact: pd.DataFrame,
    baseline: dict[str, float] | None,
    *,
    anchor: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Подставляет факт сухостой/дойные/фураж на якорный месяц в помесячную таблицу."""
    if not baseline or not isinstance(stock_fact, pd.DataFrame) or stock_fact.empty:
        return stock_fact
    df = stock_fact.copy()
    te = pd.Timestamp(anchor).normalize() if anchor is not None else TRAIN_END
    ay, am = int(te.year), int(te.month)
    mask = (df["год"] == ay) & (df["месяц"] == am)
    if not mask.any():
        row = {"год": ay, "месяц": am}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        mask = (df["год"] == ay) & (df["месяц"] == am)
    if "сухостойные" in df.columns and "dry" in baseline:
        df.loc[mask, "сухостойные"] = baseline["dry"]
    if "дойные" in df.columns and "milk" in baseline:
        df.loc[mask, "дойные"] = baseline["milk"]
    if "фуражные" in df.columns and "furazh" in baseline:
        df.loc[mask, "фуражные"] = baseline["furazh"]
    elif "фуражные" not in df.columns and "furazh" in baseline:
        df["фуражные"] = np.nan
        df.loc[mask, "фуражные"] = baseline["furazh"]
    return df


def furazh_base_sep_2024(cfg: PipelineConfig, stock_fact: pd.DataFrame | None = None) -> int:
    """База фуражных на сен 2024: сначала факт 30.09.2024, иначе снимок/лактации."""
    if cfg.sep2024_baseline and cfg.sep2024_baseline.get("furazh"):
        return int(round(float(cfg.sep2024_baseline["furazh"])))
    df = stock_fact
    if not isinstance(df, pd.DataFrame) or df.empty:
        try:
            df = load_dry_milk_monthly_fact(cfg)
        except Exception:
            df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        row = df[(df["год"] == 2024) & (df["месяц"] == 9)]
        if len(row):
            if "фуражные" in row.columns and pd.notna(row["фуражные"].iloc[0]):
                return int(round(float(row["фуражные"].iloc[0])))
            return int(round(float(row["сухостойные"].iloc[0]) + float(row["дойные"].iloc[0])))
    if not cfg.lactation_path.exists():
        return 2909
    lact = pd.read_excel(cfg.lactation_path)
    cols = [c for c in ("L1", "L2", "L3", "L4", "L5+") if c in lact.columns]
    if not cols:
        return 2909
    if "год" in lact.columns and "месяц" in lact.columns:
        row = lact[(lact["год"] == 2024) & (lact["месяц"] == 9)]
        if len(row):
            return int(row[cols].sum(axis=1).iloc[0])
    return 2909


def _patch_cell10_dry_sep2024_baseline(src: str, baseline: dict[str, float] | None) -> str:
    """Стартовый сухостой для прогноза = факт 30.09.2024 (если есть в базе)."""
    if not baseline or baseline.get("dry") is None:
        return src
    dry = int(round(float(baseline["dry"])))
    old = "suhostoynye_prev = df_train_months.iloc[-1]['сухостойные']"
    new = (
        f"suhostoynye_prev = {dry}  # факт 30.09.2024 из базы\n"
        f'print(f"  Сухостойных на сентябрь 2024 (база 30.09.2024): {dry}")'
    )
    if old in src:
        src = src.replace(old, new, 1)
    return src


def _patch_cell20_lact_sep2024_baseline(src: str, baseline: dict[str, float] | None) -> str:
    """L1–L5+ на сен 2024 из базы 30.09.2024 (не из файла Высокого / дефолтов)."""
    need = ("L1", "L2", "L3", "L4", "L5+")
    if not baseline or any(baseline.get(k) is None for k in need):
        return src
    vals = {k: int(round(float(baseline[k]))) for k in need}
    old = (
        "last_actual = train_df[train_df['год'] == 2024][train_df['месяц'] == 9]\n"
        "if len(last_actual) > 0:\n"
        "    prev_values = {\n"
        "        'L1': last_actual['L1'].values[0],\n"
        "        'L2': last_actual['L2'].values[0],\n"
        "        'L3': last_actual['L3'].values[0],\n"
        "        'L4': last_actual['L4'].values[0],\n"
        "        'L5+': last_actual['L5+'].values[0]\n"
        "    }\n"
        "else:\n"
        "    prev_values = {'L1': 633, 'L2': 834, 'L3': 527, 'L4': 334, 'L5+': 266}\n"
    )
    new = (
        "# База L1–L5+ на 30.09.2024 (Forecast / база_30.09.2024_поголовье.xlsx)\n"
        "prev_values = {\n"
        f"    'L1': {vals['L1']},\n"
        f"    'L2': {vals['L2']},\n"
        f"    'L3': {vals['L3']},\n"
        f"    'L4': {vals['L4']},\n"
        f"    'L5+': {vals['L5+']},\n"
        "}\n"
        'print("  (L1–L5+ из базы факт 30.09.2024)")\n'
    )
    if old in src:
        return src.replace(old, new, 1)
    # fallback: replace only defaults dict if structure drifted
    return src.replace(
        "prev_values = {'L1': 633, 'L2': 834, 'L3': 527, 'L4': 334, 'L5+': 266}",
        "prev_values = {"
        f"'L1': {vals['L1']}, 'L2': {vals['L2']}, 'L3': {vals['L3']}, "
        f"'L4': {vals['L4']}, 'L5+': {vals['L5+']}"
        "}",
        1,
    )


def build_injections(state: dict[str, Any]) -> dict[str, Any]:
    inj: dict[str, Any] = {}
    if "forecast_model1" in state:
        inj["forecast_model1"] = state["forecast_model1"]
    if "calving_forecast" in state:
        inj["calving_forecast"] = state["calving_forecast"]
    if "culling_forecast" in state:
        inj["culling_forecast"] = state["culling_forecast"]
    if "furazh_forecast" in state:
        inj["furazh_forecast"] = state["furazh_forecast"]
    if "status_forecast" in state:
        inj["status_forecast"] = state["status_forecast"]
    return inj


def run_cell(cell_id: int, src: str, ns: dict[str, Any], inj: dict[str, Any]) -> None:
    ns["_INJECT"] = inj
    ns["__file__"] = str(EXTRACTED)
    ns["Path"] = Path
    ns["normalize_events_df"] = normalize_events_df
    ns["read_filter_excel"] = read_filter_excel
    exec(compile(src, f"finale_cell_{cell_id}", "exec"), ns)  # noqa: S102


CELL_TITLES: dict[int, str] = {
    1: "отёлы (модель 1)",
    3: "отёлы коровы/нетели",
    5: "приплод",
    7: "отёлы по лактациям",
    10: "сухостойные / дойные",
    12: "фуражные (баланс)",
    14: "выбытие",
    16: "выбытие по лактациям",
    18: "падёж",
    20: "поголовье по лактациям",
    23: "продажи/покупки (факт)",
    25: "продажи/покупки (прогноз)",
}


class PipelineDetailLogger:
    """Подробный лог пайплайна (файл + дублирование в stdout)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._fh: TextIO | None = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def _write(self, text: str) -> None:
        print(text, end="", flush=True)
        if self._fh:
            self._fh.write(text)
            self._fh.flush()

    def banner(self, title: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n{'=' * 72}\n[{ts}] {title}\n{'=' * 72}\n")

    def line(self, msg: str = "") -> None:
        self._write(msg + "\n")

    def kv(self, indent: int = 0, **fields: Any) -> None:
        pad = "  " * indent
        for k, v in fields.items():
            self._write(f"{pad}{k}: {v}\n")


def _log_dataframe_summary(log: PipelineDetailLogger | None, ns: dict[str, Any]) -> None:
    if not log:
        return
    for key in sorted(ns.keys()):
        if key.startswith("_") or key in ("pd", "np", "Path"):
            continue
        val = ns[key]
        if isinstance(val, pd.DataFrame):
            dmax = dmin = None
            for col in ("дата_месяц", "Дата", "Date"):
                if col in val.columns:
                    s = pd.to_datetime(val[col], errors="coerce")
                    if s.notna().any():
                        dmin, dmax = s.min(), s.max()
                    break
            log.kv(
                1,
                object=key,
                type="DataFrame",
                rows=len(val),
                cols=len(val.columns),
                date_min=dmin,
                date_max=dmax,
            )
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            log.kv(1, object=key, type="list[dict]", rows=len(val))


def _log_cell_artifacts(log: PipelineDetailLogger | None, cell_id: int, ns: dict[str, Any]) -> None:
    if not log:
        return
    title = CELL_TITLES.get(cell_id, f"ячейка {cell_id}")
    log.line(f"  Параметр / блок: {title}")

    if cell_id == 1:
        train = ns.get("monthly_train")
        if isinstance(train, pd.DataFrame):
            n = len(train[train["дата_месяц"] <= TRAIN_END]) if "дата_месяц" in train.columns else len(train)
            log.kv(2, monthly_train_rows=len(train), train_rows_le_sep2024=n)
        log.kv(2, forecast_months=len(ns.get("state_forecast_model1", {})))
    elif cell_id == 10:
        rows = ns.get("state_dry_milk", [])
        train_r = [r for r in rows if (r["год"], r["месяц"]) <= (2024, 9)]
        log.kv(2, status_months=len(rows), train_months=len(train_r))
    elif cell_id == 23 and ns.get("state_sales_fact") is not None:
        df = ns["state_sales_fact"]
        if hasattr(df, "shape"):
            log.kv(2, sales_fact_rows=df.shape[0], sales_fact_cols=df.shape[1])
            num_cols = [c for c in df.columns if "внутри" in str(c)]
            if num_cols:
                log.kv(2, sales_fact_sum=int(df[num_cols].sum().sum()))
    elif cell_id == 25:
        res = ns.get("state_sales_forecast", [])
        log.kv(2, sales_forecast_models=len(res))

    _log_dataframe_summary(log, ns)


def log_kaluga_input_sources(cfg: PipelineConfig, log: PipelineDetailLogger | None) -> None:
    if not log:
        return
    log.banner(f"Источники данных — {cfg.name}")
    log.kv(
        data_dir=str(cfg.kaluga_data_dir),
        filter_folder=cfg.filter_folder,
        events_path=str(cfg.events_path),
        lactation_path=str(cfg.lactation_path),
        output_xlsx=str(cfg.output_xlsx),
        train_end=str(cfg.resolved_train_end().date()),
        predict_months=cfg.resolved_predict_months(),
        month_cols=cfg.resolved_month_cols(),
    )
    fdir = Path(cfg.filter_folder)
    if fdir.is_dir():
        for p in sorted(fdir.glob("*.xlsx")):
            try:
                n = len(pd.read_excel(p))
            except Exception as exc:  # noqa: BLE001
                n = f"err:{exc}"
            log.kv(1, filter_file=p.name, rows=n)
    if cfg.events_path.is_file():
        st = cfg.events_path.stat()
        log.kv(1, events_file=str(cfg.events_path.name), events_bytes=st.st_size)
        try:
            peek = pd.read_excel(cfg.events_path, nrows=1000)
            dts = pd.to_datetime(peek.get("Дата", peek.get("Date")), errors="coerce")
            log.kv(
                1,
                events_peek_rows=len(peek),
                events_peek_date_min=dts.min(),
                events_peek_date_max=dts.max(),
            )
        except Exception as exc:  # noqa: BLE001
            log.kv(1, events_peek_error=str(exc))


def kaluga_farm_dir(data_dir: Path, farm: str) -> Path:
    return data_dir / "по_хозяйствам" / re.sub(r"[^\w\-]+", "_", farm.strip()).strip("_")


def kaluga_farm_workbook_paths(data_dir: Path, farm: str) -> tuple[Path, Path, Path]:
    d = kaluga_farm_dir(data_dir, farm)
    return (
        d / f"Отелы плюс родившиеся {farm} DZ 120726.xlsx",
        d / f"Осеменения {farm} DZ 120726.xlsx",
        d / f"Выбытие + Запуск {farm} DZ 120726.xlsx",
    )


def after_cell_1(ns: dict[str, Any]) -> None:
    fm: dict[tuple[int, int], float] = {}
    for k, v in ns.get("forecasts", {}).items():
        fm[k] = v
    for r in ns.get("results", []):
        fm[(r["год"], r["месяц"])] = r["прогноз"]
    ns["state_forecast_model1"] = fm


def after_cell_3(ns: dict[str, Any]) -> None:
    fm = ns["state_forecast_model1"]
    calving: dict[tuple[int, int], dict[str, int]] = {}
    for r in ns.get("results_cows", []):
        key = (r["год"], r["месяц"])
        total = fm.get(key, r["прогноз"])
        cows = int(r["прогноз"])
        heifers = max(0, int(round(total - cows)))
        calving[key] = {"коровы": cows, "нетели": heifers}
    ns["state_calving_forecast"] = calving
    ns["state_results_cows"] = list(ns.get("results_cows", []))


def after_cell_5(ns: dict[str, Any]) -> None:
    fm = ns["state_forecast_model1"]
    scalar = {k: int(v) for k, v in fm.items()}
    ns["state_calving_scalar"] = scalar
    ns["state_births"] = list(ns.get("results", []))


def after_cell_7(ns: dict[str, Any]) -> None:
    ns["state_lact_calvings"] = list(ns.get("results", []))


def after_cell_10(ns: dict[str, Any]) -> None:
    status = {}
    for r in ns.get("results", []):
        status[(r["год"], r["месяц"])] = {
            "сухостойные": r["сухостойные"],
            "дойные": r["дойные"],
        }
    ns["state_status"] = status
    ns["state_dry_milk"] = list(ns.get("results", []))


def after_cell_14(ns: dict[str, Any]) -> None:
    cull = {}
    for r in ns.get("results", []):
        cull[(r["год"], r["месяц"])] = dict(r["прогноз"])
    ns["state_culling"] = cull


def after_cell_12(ns: dict[str, Any]) -> None:
    furazh = {}
    for r in ns.get("results", []):
        furazh[(r["год"], r["месяц"])] = r["фуражные_текущий"]
    ns["state_furazh_balance"] = furazh
    ns["state_furazh_formula_rows"] = list(ns.get("results", []))


def after_cell_16(ns: dict[str, Any]) -> None:
    ns["state_culling_split"] = list(ns.get("adjusted_results", []))


def after_cell_18(ns: dict[str, Any]) -> None:
    ns["state_mortality"] = list(ns.get("all_predictions", []))


def after_cell_20(ns: dict[str, Any]) -> None:
    ns["state_lact_stock"] = list(ns.get("results", []))


def after_cell_23(ns: dict[str, Any]) -> None:
    ns["state_sales_fact"] = ns.get("df_results")


def after_cell_25(ns: dict[str, Any]) -> None:
    r = ns.get("results")
    if isinstance(r, pd.DataFrame):
        ns["state_sales_forecast"] = r
    else:
        ns["state_sales_forecast"] = pd.DataFrame()


def run_young_stock(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Факт и прогноз молодняка + нетели (snapshot + XGB как forecast_young_groups)."""
    from forecast_young_groups import (  # noqa: WPS433
        build_predict_row,
        create_features,
        load_event_features,
        load_history,
        month_key,
        train_models,
    )
    from snapshot_zhk_vysokoe import (  # noqa: WPS433
        YOUNG_AND_NETELI_KEYS,
        excel_to_backtest_tables,
        load_vysokoe_raw_tables,
        monthly_young_neteli_history,
    )

    import forecast_young_groups as fyg  # noqa: WPS433
    import snapshot_zhk_vysokoe as snap  # noqa: WPS433

    folder_path = Path(cfg.filter_folder)
    if not folder_path.is_absolute():
        folder_path = (cfg.work_dir / folder_path).resolve()
    fyg.FOLDER = str(folder_path)
    snap.FOLDER = str(folder_path)
    # Кэш снимка — рядом с фильтром подразделения (не общий файл Высокого)
    hist_cache = folder_path / "снимок_молодняк_нетели_2022_01_2025_12.xlsx"
    fyg.HIST_XLSX = str(hist_cache)

    hist = load_history()
    ev = load_event_features()
    df = hist.merge(ev, on=["год", "месяц", "дата_месяц"], how="left").fillna(0)
    targets = snap.YOUNG_AND_NETELI_KEYS + ["Всего без дойных и сухостойных"]
    df = create_features(df, snap.YOUNG_AND_NETELI_KEYS)
    meta = {"год", "месяц", "дата_месяц", "дата_снимка"}
    feature_cols = [c for c in df.columns if c not in meta and c not in targets]

    train_end = cfg.resolved_train_end()
    predict_months = cfg.resolved_predict_months()
    if predict_months:
        py, pm = predict_months[0]
        ey, em = predict_months[-1]
    else:
        py, pm, ey, em = 2024, 10, 2025, 12

    train_df = df[df["дата_месяц"] <= train_end].copy()
    models = train_models(train_df, feature_cols, targets)
    print(
        f"  [молодняк] обучение: {len(train_df)} мес. (≤ {train_end.date()}), "
        f"признаков {len(feature_cols)}, целей {len(targets)}"
    )
    monthly_avg: dict[str, dict[int, float]] = {}
    for col in targets + ["запуски", "выбытия", "успешные", "всего_осеменений"]:
        if col in train_df.columns:
            monthly_avg[col] = train_df.groupby("месяц")[col].mean().to_dict()

    forecasts: dict[str, dict[tuple[int, int], float]] = {t: {} for t in targets}
    for _, r in train_df.iterrows():
        k = month_key(int(r["год"]), int(r["месяц"]))
        for t in targets:
            forecasts[t][k] = float(r[t])

    pred_rows = []
    trend = len(train_df)
    for year, month in predict_months:
        trend += 1
        X_pred = build_predict_row(year, month, trend, forecasts, monthly_avg, feature_cols)
        row = {"год": year, "месяц": month}
        for t in snap.YOUNG_AND_NETELI_KEYS:
            val = max(0, int(round(models[t].predict(X_pred)[0])))
            forecasts[t][month_key(year, month)] = float(val)
            row[t] = val
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    if cfg.forecast_only:
        return pd.DataFrame(columns=["год", "месяц"] + snap.YOUNG_AND_NETELI_KEYS), pred_df

    fact_rows = df[
        (df["дата_месяц"] >= pd.Timestamp(f"{py}-{pm:02d}-01"))
        & (df["дата_месяц"] <= pd.Timestamp(f"{ey}-{em:02d}-01"))
    ][["год", "месяц"] + snap.YOUNG_AND_NETELI_KEYS].copy()

    if fact_rows.empty:
        raw = load_vysokoe_raw_tables(str(folder_path))
        tables = excel_to_backtest_tables(raw)
        fact_rows = monthly_young_neteli_history(tables, ey, em, 2022, 1)
        fact_rows = fact_rows[
            (fact_rows["год"] > py) | ((fact_rows["год"] == py) & (fact_rows["месяц"] >= pm))
        ][["год", "месяц"] + snap.YOUNG_AND_NETELI_KEYS]

    return fact_rows, pred_df


def align_dry_with_furazh(ns: dict[str, Any]) -> None:
    """Дойные = фуражные (баланс) − сухостойные (модель 4), как в finál."""
    furazh = ns.get("state_furazh_balance", {})
    dry = ns.get("state_dry_milk", [])
    if not furazh or not dry:
        return
    aligned = []
    for r in dry:
        key = (r["год"], r["месяц"])
        f = int(furazh.get(key, r.get("фуражные", 0)))
        suh = int(r["сухостойные"])
        doy = max(0, f - suh)
        aligned.append({"год": r["год"], "месяц": r["месяц"], "сухостойные": suh, "дойные": doy, "фуражные": f})
    ns["state_dry_milk"] = aligned
    ns["state_furazh_forecast"] = furazh


def run_pipeline(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EXTRACTED.exists():
        raise FileNotFoundError(f"Нет {EXTRACTED} — перегенерируйте из ЖК_Высокое_финал.ipynb")

    detail = PipelineDetailLogger(cfg.detail_log_path)
    detail.banner(f"СТАРТ ПАЙПЛАЙНА — {cfg.name}")
    if cfg.forecast_only:
        detail.line("  режим: PIPELINE_FORECAST_ONLY (без листа «факт», без merge факта Oct2024+)")
    log_kaluga_input_sources(cfg, detail)

    sys.path.insert(0, str(ROOT))
    _prev_cwd = os.getcwd()
    os.chdir(cfg.work_dir)
    cells = load_cell_sources()
    predict_months = cfg.resolved_predict_months()
    ns: dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "_PREDICT_MONTHS": predict_months,
        "_TRAIN_END_TS": cfg.resolved_train_end(),
    }
    print(
        f"Горизонт прогноза: обучение до {cfg.resolved_train_end().date()}, "
        f"месяцы {[f'{y}-{m:02d}' for y, m in predict_months]}"
    )
    try:
        try:
            train_end = cfg.resolved_train_end()
            if cfg.sep2024_baseline is None and cfg.kaluga_farm and cfg.kaluga_unit:
                cfg.sep2024_baseline = load_sep2024_baseline(
                    cfg.kaluga_farm, cfg.kaluga_unit, cfg.baseline_xlsx
                )
            if cfg.sep2024_baseline is None and cfg.lactation_path.exists():
                lact_probe = pd.read_excel(cfg.lactation_path)
                cfg.sep2024_baseline = baseline_from_lactation_at_train_end(lact_probe, train_end)
            ns["state_dry_milk_fact"] = load_dry_milk_monthly_fact(cfg)
            ns["state_dry_milk_fact"] = apply_sep2024_baseline_to_stock_fact(
                ns["state_dry_milk_fact"],
                cfg.sep2024_baseline,
                anchor=train_end,
            )
            sep = furazh_base_sep_2024(cfg, ns["state_dry_milk_fact"])
            src_label = "база 30.09.2024" if cfg.sep2024_baseline else "из таблиц"
            print(
                f"Факт снимка (сух/дой/фур): {len(ns['state_dry_milk_fact'])} мес.; "
                f"FURAZH_BASE сен 2024 = {sep} ({src_label})"
            )
            if cfg.sep2024_baseline:
                detail.kv(2, **{f"sep2024_{k}": v for k, v in cfg.sep2024_baseline.items()})
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Факт сухостойных из снимка: {exc}")
            ns["state_dry_milk_fact"] = pd.DataFrame()

        order = [
            (1, after_cell_1),
            (3, after_cell_3),
            (5, after_cell_5),
            (7, after_cell_7),
            (10, after_cell_10),
            (14, after_cell_14),
            (12, after_cell_12),
            (16, after_cell_16),
            (18, after_cell_18),
            (20, after_cell_20),
        ]
        if not cfg.forecast_only:
            order.append((23, after_cell_23))
        order.append((25, after_cell_25))

        inj: dict[str, Any] = {}
        for cell_id, hook in order:
            print("\n" + "=" * 72)
            print(f"Шаг: ячейка finál {cell_id} ({cfg.name})")
            print("=" * 72)
            detail.banner(f"ЯЧЕЙКА {cell_id} — {CELL_TITLES.get(cell_id, '?')} — {cfg.name}")
            src = patch_cell_source(cells[cell_id], cfg)
            if cell_id == 12:
                inj["calving_forecast"] = ns.get("state_calving_forecast", {})
                inj["culling_forecast"] = ns.get("state_culling", {})
                src = re.sub(
                    r"FURAZH_BASE_SEP_2024\s*=\s*\d+",
                    f"FURAZH_BASE_SEP_2024 = {furazh_base_sep_2024(cfg, ns.get('state_dry_milk_fact'))}",
                    src,
                )
            if cell_id == 3:
                inj["forecast_model1"] = ns.get("state_forecast_model1", {})
            if cell_id == 5:
                inj["calving_forecast"] = ns.get("state_calving_scalar", {})
            if cell_id == 10:
                inj["furazh_forecast"] = ns.get("state_furazh_forecast", ns.get("state_furazh_balance", {}))
                src = _patch_cell10_dry_fact_from_snapshot(src, cfg)
                src = _patch_cell10_dry_sep2024_baseline(src, cfg.sep2024_baseline)
                src = _patch_cell10_furazh_baseline(src, cfg.sep2024_baseline)
            if cell_id == 16:
                inj["culling_forecast"] = ns.get("state_culling", {})
                inj["status_forecast"] = ns.get("state_status", {})
            if cell_id == 18:
                inj["calving_forecast"] = ns.get("state_calving_scalar", {})
            if cell_id == 20:
                inj["furazh_forecast"] = ns.get("state_furazh_balance", {})
                src = _patch_cell20_lact_sep2024_baseline(src, cfg.sep2024_baseline)
                if not cfg.lactation_path.exists():
                    print(f"⚠️ Нет {cfg.lactation_path} — пропуск ячейки 20")
                    continue

            try:
                run_cell(cell_id, src, ns, inj)
                hook(ns)
                _log_cell_artifacts(detail, cell_id, ns)
                detail.line("  Статус: OK")
            except Exception as exc:  # noqa: BLE001
                detail.line(f"  Статус: ОШИБКА — {exc}")
                detail.line(traceback.format_exc())
                print(f"⚠️ Ячейка {cell_id} ошибка: {exc}")
                if cell_id in (23, 25) and not cfg.forecast_only:
                    detail.close()
                    raise
            if cell_id == 12:
                align_dry_with_furazh(ns)
                inj["furazh_forecast"] = ns.get("state_furazh_balance", {})

        try:
            detail.banner(f"Молодняк и нетели (XGB) — {cfg.name}")
            fact_young, pred_young = run_young_stock(cfg)
            ns["state_young_fact"] = fact_young
            ns["state_young_pred"] = pred_young
            detail.kv(young_fact_rows=len(fact_young), young_pred_rows=len(pred_young))
            detail.line("  Статус: OK")
        except Exception as exc:  # noqa: BLE001
            detail.line(f"  Статус: ОШИБКА — {exc}")
            detail.line(traceback.format_exc())
            print(f"⚠️ Молодняк: {exc}")
            ns["state_young_fact"] = pd.DataFrame()
            ns["state_young_pred"] = pd.DataFrame()

        forecast_table, fact_table = assemble_tables(ns, cfg)
        detail.banner(f"СБОРКА ТАБЛИЦ — {cfg.name}")
        detail.kv(
            forecast_params=len(forecast_table),
            forecast_cols=len(forecast_table.columns),
            fact_params=len(fact_table),
            fact_cols=len(fact_table.columns),
        )
        cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(cfg.output_xlsx, engine="openpyxl") as writer:
            forecast_table.to_excel(writer, sheet_name="прогноз")
            if not cfg.forecast_only:
                fact_table.to_excel(writer, sheet_name="факт")
            else:
                pd.DataFrame(
                    [{"параметр": "факт отложен (PIPELINE_FORECAST_ONLY=1)", **{c: "" for c in cfg.resolved_month_cols()}}]
                ).to_excel(writer, sheet_name="факт", index=False)
        detail.kv(output=str(cfg.output_xlsx), size_bytes=cfg.output_xlsx.stat().st_size)
        detail.banner(f"ЗАВЕРШЕНО — {cfg.name}")
        detail.close()
        print(f"\n✅ Сохранено: {cfg.output_xlsx}")
        return forecast_table, fact_table
    finally:
        os.chdir(_prev_cwd)


def _series_from_rows(
    rows: list[dict],
    value_key: str | Callable[[dict], Any],
    nested: str | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        y, m = int(r["год"]), int(r["месяц"])
        if nested:
            val = r[value_key][nested] if isinstance(value_key, str) else value_key(r)
        elif callable(value_key):
            val = value_key(r)
        else:
            val = r[value_key]
        out[_col_label(y, m)] = float(val)
    return out


def assemble_tables(ns: dict[str, Any], cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    predict_months = cfg.resolved_predict_months()
    month_cols = cfg.resolved_month_cols()
    predict_keys = set(predict_months)

    def _predict_mask(df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda r: (int(r["год"]), int(r["месяц"])) in predict_keys, axis=1)

    params: list[tuple[str, dict[str, float]]] = []
    facts: list[tuple[str, dict[str, float]]] = []
    skip_fact = cfg.forecast_only

    fm = ns.get("state_forecast_model1", {})
    for label, key in [("отёлы_всего", None)]:
        d = {_col_label(y, m): float(fm.get((y, m), 0)) for y, m in predict_months}
        params.append((label, d))

    cows = ns.get("state_results_cows", [])
    params.append(("отёлы_коров", _series_from_rows(cows, "прогноз")))
    facts.append(("отёлы_коров", _series_from_rows(cows, "факт")))
    params.append(
        (
            "отёлы_нетели",
            {
                _col_label(r["год"], r["месяц"]): max(
                    0,
                    ns["state_forecast_model1"].get((r["год"], r["месяц"]), 0) - r["прогноз"],
                )
                for r in cows
            },
        )
    )

    births = ns.get("state_births", [])
    params.append(("приплод_телочки", _series_from_rows(births, "прогноз_телочки")))
    params.append(("приплод_бычки", _series_from_rows(births, "прогноз_бычки")))
    facts.append(("приплод_телочки", _series_from_rows(births, "факт_телочки")))
    facts.append(("приплод_бычки", _series_from_rows(births, "факт_бычки")))

    for lact in ("L1", "L2", "L3+"):
        rows = ns.get("state_lact_calvings", [])
        params.append(
            (
                f"отёлы_{lact}",
                {
                    _col_label(r["год"], r["месяц"]): float(r["прогноз"].get(lact, 0))
                    for r in rows
                },
            )
        )
        facts.append(
            (
                f"отёлы_{lact}",
                {
                    _col_label(r["год"], r["месяц"]): float(r["факт"].get(lact, 0))
                    for r in rows
                },
            )
        )

    dry = ns.get("state_dry_milk", [])
    for name in ("сухостойные", "дойные", "фуражные"):
        params.append((name, _series_from_rows(dry, name)))

    dmf = ns.get("state_dry_milk_fact")
    if isinstance(dmf, pd.DataFrame) and not dmf.empty:
        def _fact_series(col: str) -> dict[str, float]:
            out: dict[str, float] = {}
            for _, row in dmf.iterrows():
                key = (int(row["год"]), int(row["месяц"]))
                if key not in predict_keys:
                    continue
                out[_col_label(key[0], key[1])] = float(row[col])
            return out

        facts.append(("сухостойные", _fact_series("сухостойные")))
        facts.append(("дойные", _fact_series("дойные")))
        facts.append(("фуражные", _fact_series("фуражные")))

    furazh_rows = ns.get("state_furazh_formula_rows", [])
    params.append(
        (
            "фуражные_баланс",
            {_col_label(r["год"], r["месяц"]): float(r["фуражные_текущий"]) for r in furazh_rows},
        )
    )

    cull = ns.get("state_culling", {})
    for lact in ("L0", "L1", "L2", "L3", "L4", "L5+"):
        params.append(
            (
                f"выбытие_{lact}",
                {_col_label(y, m): float(cull.get((y, m), {}).get(lact, 0)) for y, m in predict_months},
            )
        )

    split = ns.get("state_culling_split", [])
    params.append(("выбытие_сухостойных", _series_from_rows(split, "сухостойные")))
    params.append(("выбытие_дойных", _series_from_rows(split, "дойные")))

    mort = ns.get("state_mortality", [])
    if mort:
        for col in ("падеж_телочки", "падеж_бычки", "падеж_всего"):
            params.append((col, _series_from_rows(mort, col)))

    stock = ns.get("state_lact_stock", [])
    for lact in ("L1", "L2", "L3", "L4", "L5+"):
        params.append(
            (
                f"остаток_{lact}",
                {_col_label(r["год"], r["месяц"]): float(r["прогноз"].get(lact, 0)) for r in stock},
            )
        )
    params.append(("остаток_фуражные", _series_from_rows(stock, "фуражные")))

    sales_f = ns.get("state_sales_forecast")
    if isinstance(sales_f, pd.DataFrame) and not sales_f.empty:
        for col in sales_f.columns:
            if col in ("год", "месяц") or not col.endswith("_прогноз"):
                continue
            base = col[: -len("_прогноз")]
            sub = sales_f.loc[_predict_mask(sales_f)]
            params.append(
                (
                    base,
                    {_col_label(int(r["год"]), int(r["месяц"])): float(r[col]) for _, r in sub.iterrows()},
                )
            )
            fact_col = f"{base}_факт"
            if fact_col in sales_f.columns:
                facts.append(
                    (
                        base,
                        {
                            _col_label(int(r["год"]), int(r["месяц"])): float(r[fact_col])
                            for _, r in sub.iterrows()
                        },
                    )
                )

    sales_fact = ns.get("state_sales_fact")
    if isinstance(sales_fact, pd.DataFrame) and not sales_fact.empty:
        for col in sales_fact.columns:
            if col in ("год", "месяц"):
                continue
            sub = sales_fact.loc[_predict_mask(sales_fact)]
            facts.append(
                (
                    col,
                    {_col_label(int(r["год"]), int(r["месяц"])): float(r[col]) for _, r in sub.iterrows()},
                )
            )

    yfact = ns.get("state_young_fact")
    ypred = ns.get("state_young_pred")
    if isinstance(yfact, pd.DataFrame) and not yfact.empty:
        from snapshot_zhk_vysokoe import YOUNG_AND_NETELI_KEYS  # noqa: WPS433

        for col in YOUNG_AND_NETELI_KEYS:
            if col in yfact.columns:
                sub = yfact.loc[_predict_mask(yfact)]
                facts.append(
                    (
                        f"остаток_{col}",
                        {_col_label(int(r["год"]), int(r["месяц"])): float(r[col]) for _, r in sub.iterrows()},
                    )
                )
    if isinstance(ypred, pd.DataFrame) and not ypred.empty:
        from snapshot_zhk_vysokoe import YOUNG_AND_NETELI_KEYS  # noqa: WPS433

        for col in YOUNG_AND_NETELI_KEYS:
            if col in ypred.columns:
                sub = ypred.loc[_predict_mask(ypred)]
                params.append(
                    (
                        f"остаток_{col}",
                        {_col_label(int(r["год"]), int(r["месяц"])): float(r[col]) for _, r in sub.iterrows()},
                    )
                )

    if skip_fact:
        facts.clear()

    def to_df(items: list[tuple[str, dict[str, float]]]) -> pd.DataFrame:
        rows = []
        for name, data in items:
            row = {"параметр": name}
            for c in month_cols:
                row[c] = data.get(c, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    return to_df(params), to_df(facts)




def read_filter_excel(path: str | Path) -> pd.DataFrame:
    """Чтение filter_* / events: CSV если рядом есть (быстрее), иначе Excel."""
    p = Path(path)
    csv = p.with_suffix(".csv")
    if csv.is_file() and csv.stat().st_size > 0:
        return normalize_events_df(pd.read_csv(csv, low_memory=False))
    return normalize_events_df(pd.read_excel(p))

# --- Kaluga: фильтр данных и временная папка Excel ---


def _normalize_kaluga_event(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Event" in out.columns and "Событие" not in out.columns:
        out["Событие"] = out["Event"]
    if "Event" in out.columns:
        ev = out["Event"].astype(str).str.upper().str.strip()
        out.loc[ev.str.contains("DIED", na=False), "Событие"] = "ПАЛА"
        out.loc[ev.str.contains("SOLD", na=False), "Событие"] = "ПРОДАНА"
        out.loc[ev.str.contains("BRED", na=False), "Событие"] = "ОСЕМЕН"
    if "Date" in out.columns and "Дата" not in out.columns:
        out["Дата"] = out["Date"]
    if "LACT" in out.columns and "Lact" not in out.columns:
        out["Lact"] = out["LACT"]
    elif "Lact" in out.columns and "LACT" not in out.columns:
        out["LACT"] = out["Lact"]
    return out


def _filter_subdivision(df: pd.DataFrame, farm: str, unit: str) -> pd.DataFrame:
    aliases = SUBDIVISION_ALIASES.get(unit, [unit])
    m = df["Source.Name"].astype(str).str.strip() == farm.strip()
    m &= df["Столбец1"].astype(str).str.strip().isin(aliases)
    return df.loc[m].copy()


def _read_farm_event_tables(data_dir: Path, farm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calv_path, sem_path, disp_path = kaluga_farm_workbook_paths(data_dir, farm)
    for p in (calv_path, sem_path, disp_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Ожидается файл по хозяйству в {data_dir / 'по_хозяйствам'}: {p}"
            )
    calv = _normalize_kaluga_event(pd.read_excel(calv_path))
    sem = _normalize_kaluga_event(pd.read_excel(sem_path))
    disp = _normalize_kaluga_event(pd.read_excel(disp_path))
    return calv, sem, disp


def _merge_trim_and_fact_period(
    trim: pd.DataFrame,
    full: pd.DataFrame,
    cutoff: pd.Timestamp = TRAIN_END,
) -> pd.DataFrame:
    """Обучение: ≤ cutoff из trim; факт прогнозного окна: > cutoff из полного split."""
    trim = trim.copy()
    full = full.copy()
    trim["Дата"] = pd.to_datetime(trim["Дата"], errors="coerce")
    full["Дата"] = pd.to_datetime(full["Дата"], errors="coerce")
    left = trim.loc[trim["Дата"].notna() & (trim["Дата"] <= cutoff)]
    right = full.loc[full["Дата"].notna() & (full["Дата"] > cutoff)]
    if left.empty and right.empty:
        return trim.iloc[0:0]
    out = pd.concat([left, right], ignore_index=True)
    return out.drop_duplicates()


def _unit_event_tables(
    farm: str,
    unit: str,
    trim_dir: Path,
    full_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calv_t, sem_t, disp_t = _read_farm_event_tables(trim_dir, farm)
    calv = _filter_subdivision(calv_t, farm, unit)
    sem = _filter_subdivision(sem_t, farm, unit)
    disp = _filter_subdivision(disp_t, farm, unit)
    if full_dir and full_dir.resolve() != trim_dir.resolve() and kaluga_farm_dir(full_dir, farm).is_dir():
        calv_f, sem_f, disp_f = _read_farm_event_tables(full_dir, farm)
        calv = _merge_trim_and_fact_period(
            calv,
            _filter_subdivision(calv_f, farm, unit),
        )
        sem = _merge_trim_and_fact_period(
            sem,
            _filter_subdivision(sem_f, farm, unit),
        )
        disp = _merge_trim_and_fact_period(
            disp,
            _filter_subdivision(disp_f, farm, unit),
        )
    return calv, sem, disp


def build_kaluga_filter_folder(
    farm: str,
    unit: str,
    data_dir: Path,
    out_dir: Path,
) -> Path:
    """Фильтр: train ≤2024-09 из до_2024_09; факт Oct2024+ из полного по_хозяйствам."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.xlsx"):
        old.unlink()

    full_dir = KALUGA_FULL_DATA_DIR if KALUGA_FULL_DATA_DIR.is_dir() else None
    if os.environ.get("PIPELINE_FORECAST_ONLY", "0").strip() in ("1", "true", "True"):
        full_dir = None
    calv, sem, disp_all = _unit_event_tables(farm, unit, data_dir, full_dir)

    print(
        f"  фильтр {unit}: отёлы {len(calv)}, осеменения {len(sem)}, "
        f"выбытие+запуск {len(disp_all)} (trim={data_dir.name}"
        f"{', +полный split для факта' if full_dir else ''})"
    )

    dry_ev = disp_all[disp_all["Event"].astype(str).str.upper().str.strip().isin(["DRY", "ЗАПУСК", "ЗАПУСКА"])]
    cull = disp_all[
        ~disp_all["Event"].astype(str).str.upper().str.strip().isin(["DRY", "ЗАПУСК", "ЗАПУСКА"])
    ]

    datasets = (
        ("Отелы", calv),
        ("Осеменения", sem),
        ("Запуск", dry_ev),
        ("Выбытие", cull),
    )
    ref_cols: list[str] = []
    for _, data in datasets:
        if len(data.columns):
            ref_cols = list(data.columns)
            break

    for prefix, data in datasets:
        cols = list(data.columns) if len(data.columns) else ref_cols
        work = data.copy()
        if not work.empty:
            work["Дата"] = pd.to_datetime(work["Дата"], errors="coerce")
        for year in (2022, 2023, 2024, 2025):
            if work.empty:
                part = pd.DataFrame(columns=cols)
            else:
                part = work[work["Дата"].dt.year == year]
            (out_dir / f"{prefix}_{year}.xlsx").parent.mkdir(parents=True, exist_ok=True)
            part.to_excel(out_dir / f"{prefix}_{year}.xlsx", index=False)

    return out_dir


def _resolve_kaluga_events_csv(data_dir: Path, root: Path) -> Path:
    matches = sorted(data_dir.glob("Событ*.csv"))
    if matches:
        return matches[0]
    fallback_dir = root / "Калуга"
    if fallback_dir.is_dir():
        matches = sorted(fallback_dir.glob("Событ*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Не найден CSV событий в {data_dir}")


def _resolve_kaluga_events_xlsx(work: Path) -> Path | None:
    cached = work / "events_cows.xlsx"
    if cached.is_file():
        return cached
    matches = sorted(work.glob("Событ*.xlsx"))
    return matches[0] if matches else None


def build_kaluga_events_csv(
    farm: str,
    unit: str,
    csv_path: Path,
    out_xlsx: Path,
    *,
    fact_csv_path: Path | None = None,
) -> Path:
    aliases = SUBDIVISION_ALIASES.get(unit, [unit])

    def _load_filtered(path: Path, after_cutoff: bool | None) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
            if "Столбец1" not in chunk.columns:
                continue
            m = chunk["Source.Name"].astype(str).str.strip() == farm.strip()
            m &= chunk["Столбец1"].astype(str).str.strip().isin(aliases)
            if not m.any():
                continue
            sub = chunk.loc[m].copy()
            dcol = "Дата" if "Дата" in sub.columns else "Date"
            if dcol in sub.columns and after_cutoff is not None:
                dts = pd.to_datetime(sub[dcol], errors="coerce")
                if after_cutoff:
                    sub = sub.loc[dts > TRAIN_END]
                else:
                    sub = sub.loc[dts.notna() & (dts <= TRAIN_END)]
            if len(sub):
                chunks.append(sub)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    df = _load_filtered(csv_path, after_cutoff=None if fact_csv_path is None else False)
    if fact_csv_path and fact_csv_path.is_file() and fact_csv_path.resolve() != csv_path.resolve():
        extra = _load_filtered(fact_csv_path, after_cutoff=True)
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True).drop_duplicates()
    if df.empty:
        raise ValueError(f"Нет строк для {farm} / {unit}")
    df = normalize_events_df(df)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    return out_xlsx


def kaluga_config(farm: str, unit: str) -> PipelineConfig:
    if farm not in KALUGA_TREE:
        raise ValueError(f"Неизвестное хозяйство: {farm}. Доступно: {list(KALUGA_TREE)}")
    if unit not in KALUGA_TREE[farm]:
        raise ValueError(f"Подразделение {unit} не входит в {farm}")

    safe = re.sub(r"[^\w\-]+", "_", unit)
    work = ROOT / "Калуга" / "_runtime" / safe
    data_dir = KALUGA_TRIM_DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Нет обрезанных данных: {data_dir}")
    filter_dir = build_kaluga_filter_folder(farm, unit, data_dir, work / f"фильтр_{safe}")
    forecast_only = os.environ.get("PIPELINE_FORECAST_ONLY", "0").strip() in ("1", "true", "True")
    events_xlsx = _resolve_kaluga_events_xlsx(work)
    if events_xlsx is not None and forecast_only:
        events = events_xlsx
    else:
        csv_path = _resolve_kaluga_events_csv(data_dir, ROOT)
        try:
            full_csv = _resolve_kaluga_events_csv(ROOT / "Калуга", ROOT)
        except FileNotFoundError:
            full_csv = csv_path
        fact_csv = (
            None
            if forecast_only
            else (full_csv if full_csv.is_file() and csv_path.resolve() != full_csv.resolve() else None)
        )
        out_xlsx = work / "События-пo-korovam.xlsx"
        events = build_kaluga_events_csv(
            farm,
            unit,
            csv_path,
            out_xlsx,
            fact_csv_path=fact_csv,
        )
        if not events.is_file() and events_xlsx is not None:
            events = events_xlsx
    rules = kaluga_trade_rules(farm, unit)
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])
    lact = work / "поголовье_по_лактациям.xlsx"
    src_lact = ROOT / "d1" / "поголовье_по_лактациям_январь2022_декабрь2025.xlsx"
    if not lact.exists() and src_lact.exists():
        pd.read_excel(src_lact).to_excel(lact, index=False)
    baseline = load_sep2024_baseline(farm, unit, KALUGA_SEP2024_BASELINE_XLSX)
    return PipelineConfig(
        name=f"{farm} / {unit}",
        work_dir=work,
        filter_folder=str(filter_dir.resolve()),
        events_path=events,
        lactation_path=lact,
        output_xlsx=ROOT / "Калуга" / f"прогноз_всех_параметров_{safe}.xlsx",
        subdivision_names=subdiv_names,
        kuda_buy_tokens=rules["buy"],
        sold_heifer_kuda_tokens=rules["heifer_sale"],
        sold_bull_kuda_tokens=rules["bull_sale"],
        sales_require_pereezd=False,
        exit_event_types=["ВЫБЫТИЕ", "ПРОДАНА", "SOLD"],
        sold_heifer_dest=rules["heifer_sale"][0] if rules["heifer_sale"] else unit,
        sold_bull_dest="БЫЧКИ",
        kaluga_farm=farm,
        kaluga_unit=unit,
        kaluga_data_dir=data_dir,
        kaluga_internal_tokens=rem_codes_all_kaluga(),
        sep2024_baseline=baseline,
        baseline_xlsx=KALUGA_SEP2024_BASELINE_XLSX,
        forecast_only=forecast_only,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Прогноз всех параметров (finál pipeline)")
    parser.add_argument("--farm", help="Калуга: хозяйство (КН Восток, …)")
    parser.add_argument("--unit", "--subdivision", dest="unit", help="Калуга: подразделение")
    parser.add_argument(
        "--list-units",
        action="store_true",
        help="Показать дерево хозяйств и подразделений",
    )
    parser.add_argument(
        "--detail-log",
        type=Path,
        help="Файл подробного лога пайплайна (дописывается)",
    )
    parser.add_argument(
        "--baseline-xlsx",
        type=Path,
        default=None,
        help="Excel с остатками на 30.09.2024 (по умолчанию Калуга/база_30.09.2024_поголовье.xlsx)",
    )
    args = parser.parse_args(argv)

    if args.list_units:
        for farm, units in KALUGA_TREE.items():
            print(f"\n{farm}:")
            for u in units:
                print(f"  - {u}")
        return

    if args.farm and args.unit:
        cfg = kaluga_config(args.farm, args.unit)
    elif args.farm or args.unit:
        parser.error("Для Калуги укажите оба: --farm и --unit")
    else:
        cfg = PipelineConfig()

    if args.baseline_xlsx is not None:
        cfg.baseline_xlsx = args.baseline_xlsx
        if cfg.kaluga_farm and cfg.kaluga_unit:
            cfg.sep2024_baseline = load_sep2024_baseline(
                cfg.kaluga_farm, cfg.kaluga_unit, cfg.baseline_xlsx
            )

    if args.detail_log:
        cfg.detail_log_path = args.detail_log

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
