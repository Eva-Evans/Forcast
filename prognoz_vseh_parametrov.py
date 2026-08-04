#!/usr/bin/env python3
"""
Единый прогноз всех параметров (логика ЖК_Высокое_финал.ipynb).

Обучение: январь 2022 — сентябрь 2024.
Горизонт таблиц: октябрь 2024 — декабрь 2025 (15 месяцев).
Выход: Excel с листами «прогноз» и «факт» (строки = параметры, столбцы = месяцы).

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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

# Справочник REM («Кuda») для внутренних продаж/покупок (Калуга).
# heifer_sale_rem — REM при CARX=ПЕРЕЕЗД, LACT=0 (куда ушли телки).
# buy_rem — REM, по которому животное считается пришедшим в это подразделение.
KALUGA_TRADE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "ЖК Аристово": {
        "heifer_sale_rem": ["РМ_КОЛЬЦОВО", "ЖК_БОЛДАСОВКА", "РМ_КЛЦ", "ЖК_БЛД", "КОЛЬЦОВ"],
        "buy_rem": ["ЖК_АРИСТОВО", "ЖКАРС", "АРИСТОВО"],
    },
    "ЖК Болдасовка": {
        "heifer_sale_rem": ["РМ_КОЛЬЦОВО", "ЖК_АРИСТОВО", "ЖКАРС", "РМ_КЛЦ"],
        "buy_rem": ["ЖК_БОЛДАСОВКА", "ЖК_БОЛД", "БОЛДАСОВКА", "ЖК_БЛД"],
    },
    "ЖК Сугуново": {
        "heifer_sale_rem": ["ЖКСГН", "ЖК_СУГОНОВО", "РМ_КОЛЬЦОВО", "ЖК_БОЛДАСОВКА"],
        "buy_rem": ["ЖКСГН", "ЖК_СУГОНОВО", "СУГОНОВО", "СУГУНОВО"],
    },
    "РМ Детчино": {
        "heifer_sale_rem": ["РМ_КОЛЬЦОВО", "ЖК_БОЛДАСОВКА", "ЖК_АРИСТОВО"],
        "buy_rem": ["РМ_ДЕТЧИНО", "ДЕТЧИНО", "РМ_ДТЧ"],
    },
    "РМ Кольцово": {
        "heifer_sale_rem": ["ЖК_БОЛДАСОВКА", "ЖК_АРИСТОВО", "РМ_ДЕТЧИНО"],
        "buy_rem": ["РМ_КОЛЬЦОВО", "РМ_КЛЦ", "КОЛЬЦОВО", "КОЛЬЦОВ"],
    },
    "ЖК Богданино": {
        "heifer_sale_rem": ["ЖК_БОГДАНИНО", "РМ_КН-ЮГ", "РМ_КН_ЮГ", "ЖК_БУШОВКА"],
        "buy_rem": ["ЖК_БОГДАНИНО", "БОГДАНИНО", "ЖК_БГД"],
    },
    "ЖК Бушовка": {
        "heifer_sale_rem": ["ЖК_БОГДАНИНО", "РМ_КН-ЮГ", "РМ_КН_ЮГ"],
        "buy_rem": ["ЖК_БУШОВКА", "БУШОВКА", "ЖК_БШ"],
    },
    "РМ КН-Юг": {
        "heifer_sale_rem": ["ЖК_БОГДАНИНО", "ЖК_БУШОВКА"],
        "buy_rem": ["РМ_КН-ЮГ", "РМ_КН_ЮГ", "КН-ЮГ", "КН_ЮГ"],
    },
    "ЖК Гусево": {
        "heifer_sale_rem": ["МТФ_КН-ЗАПАД", "МТФ_ЗАПАД", "ЖК_УЛАНОВО"],
        "buy_rem": ["ЖК_ГУСЕВО", "ГУСЕВО", "ЖК_ГСВ"],
    },
    "ЖК Уланово": {
        "heifer_sale_rem": ["МТФ_КН-ЗАПАД", "ЖК_ГУСЕВО"],
        "buy_rem": ["ЖК_УЛАНОВО", "УЛАНОВО"],
    },
    "МТФ КН-Запад": {
        "heifer_sale_rem": ["ЖК_ГУСЕВО", "ЖК_УЛАНОВО"],
        "buy_rem": ["МТФ_КН-ЗАПАД", "МТФ_ЗАПАД", "МТФ КН-ЗАПАД", "ЗАПАД"],
    },
    "ЖК Пеневичи": {
        "heifer_sale_rem": ["ЖК_ПЕНЕВИЧИ", "ПЕНЕВИЧИ"],
        "buy_rem": ["ЖК_ПЕНЕВИЧИ", "ПЕНЕВИЧИ", "ПЕНЕВ"],
    },
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


def kaluga_trade_rules(farm: str, unit: str) -> dict[str, list[str]]:
    if unit in KALUGA_TRADE_OVERRIDES:
        o = KALUGA_TRADE_OVERRIDES[unit]
        return {
            "buy": o["buy_rem"],
            "heifer_sale": o["heifer_sale_rem"],
            "bull_sale": ["БЫЧКИ", "БЫЧ"],
        }
    siblings = [u for u in KALUGA_TREE[farm] if u != unit]
    buy = _unit_rem_codes(unit)
    heifer: list[str] = []
    for s in siblings:
        heifer.extend(_unit_rem_codes(s))
    return {
        "buy": buy,
        "heifer_sale": list(dict.fromkeys(heifer)),
        "bull_sale": ["БЫЧКИ", "БЫЧ"],
    }


def _canonical_sobytie_series(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip().str.upper().str.replace("Ё", "Е", regex=False)
    out = s.copy()
    rules: list[tuple[pd.Series, str]] = [
        (s.str.contains("CALV|ОТЕЛ", na=False, regex=True), "ОТЕЛ"),
        (s.str.contains("BRED|ОСЕМ", na=False, regex=True), "ОСЕМЕН"),
        (s.str.contains("DRY|ЗАПУСК", na=False, regex=True), "ЗАПУСК"),
        (s.str.contains("SOLD|ПРОД", na=False, regex=True), "ПРОДАНА"),
        (s.str.contains("DIED|DEAD|ПАЛ|МЕР", na=False, regex=True), "ПАЛА"),
        (s.str.contains("BORN|РОЖ", na=False, regex=True), "РОЖДЕН"),
        (s.isin(["ВЫБЫТИЕ", "EXIT", "CULL", "SOLD"]), "ВЫБЫТИЕ"),
    ]
    for mask, label in rules:
        out.loc[mask] = label
    return out


def normalize_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Калуга: REM→Куда, Event/event_type→Событие (как в finál)."""
    df = df.copy()
    if "Date" in df.columns:
        df["Дата"] = pd.to_datetime(df.get("Дата", df["Date"]), errors="coerce")
    elif "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    else:
        df["Дата"] = pd.NaT

    if "Event" not in df.columns and "event_type" in df.columns:
        df["Event"] = df["event_type"]
    if "Event" in df.columns:
        ev_raw = df["Event"]
        if "Событие" not in df.columns:
            df["Событие"] = ev_raw
        else:
            empty = df["Событие"].isna() | (
                df["Событие"].astype(str).str.strip().isin(["", "nan", "None", "NaT"])
            )
            df.loc[empty, "Событие"] = df.loc[empty, "Event"]
        canon = _canonical_sobytie_series(df["Событие"].fillna(df["Event"]))
        df["Событие"] = canon
        raw = df["Event"].astype(str).str.strip().str.upper()
        so = df["Событие"].astype(str).str.strip().str.upper()
        df.loc[raw.str.contains("SOLD", na=False), "Событие"] = "ПРОДАНА"
        df.loc[raw.str.contains("BRED", na=False), "Событие"] = "ОСЕМЕН"
        df.loc[so.str.contains("SOLD", na=False), "Событие"] = "ПРОДАНА"
        df.loc[so.str.contains("BRED", na=False), "Событие"] = "ОСЕМЕН"
    elif "Событие" not in df.columns:
        df["Событие"] = ""
    if "Кuda" in df.columns and "Куда" not in df.columns:
        df["Куда"] = df["Кuda"].astype(str).str.strip()
    if "Куда" not in df.columns and "REM" in df.columns:
        df["Куда"] = df["REM"].astype(str).str.strip()
    elif "Куда" in df.columns:
        empty = df["Куда"].isna() | (df["Куда"].astype(str).str.strip() == "")
        if "REM" in df.columns:
            df.loc[empty, "Куда"] = df.loc[empty, "REM"].astype(str).str.strip()
    if "LACT" not in df.columns and "Lact" in df.columns:
        df["LACT"] = df["Lact"]
    if "ключ_коровы" not in df.columns:
        id_s = df.get("ID", pd.Series("", index=df.index)).astype(str).fillna("")
        if "REG" in df.columns:
            mask_id = (id_s == "") | (id_s.str.lower() == "nan")
            id_s = id_s.where(~mask_id, df["REG"].astype(str).fillna(""))
        if "BDAT" in df.columns:
            bdat = pd.to_datetime(df["BDAT"], errors="coerce")
        else:
            bdat = pd.Series(pd.NaT, index=df.index)
        bdat_key = bdat.dt.strftime("%Y%m%d").fillna("")
        df["ключ_коровы"] = id_s.astype(str) + "_" + bdat_key.astype(str)
        bad_key = df["ключ_коровы"] == "_"
        if bad_key.any():
            df.loc[bad_key, "ключ_коровы"] = "без_ключа_" + df.index[bad_key].astype(str)
    return df

@dataclass
class PipelineConfig:
    name: str = "ЖК Высокое"
    work_dir: Path = field(default_factory=lambda: ROOT)
    filter_folder: str = "фильтр_ЖК_Высокое"
    events_path: Path = field(default_factory=lambda: ROOT / "d1" / "События-по-коровам.xlsx")
    events_aux_path: Path = field(
        default_factory=lambda: ROOT / "d1" / "События-по-коровам (1).xlsx"
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
    return f"""
_SUBDIV_NAMES = {list(cfg.subdivision_names)!r}
_HEIFER_SALE_KUDA = {heifer!r}
_BULL_SALE_KUDA = {bull!r}
_KUDA_BUY = {buy!r}
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
"""


def _apply_trade_patches(src: str, cfg: PipelineConfig) -> str:
    if "ЖК Высокое" not in src and "Столбец1" not in src:
        return src
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
    src = src.replace(
        "(df['Куда'].str.upper().str.contains('БЫЧКИ', na=False))",
        "_match_kuda(df['Куда'], _BULL_SALE_KUDA)",
    )
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
    if needle in src and cfg.sales_require_pereezd:
        src = src.replace(
            needle,
            needle
            + "\nif _REQUIRE_PEREEZD and 'CARX' in df_heifers_sold.columns:\n"
            + "    df_heifers_sold = df_heifers_sold[df_heifers_sold['CARX'].astype(str).str.upper().str.contains('ПЕРЕЕЗД', na=False)].copy()",
        )
    return src


def patch_cell_source(src: str, cfg: PipelineConfig) -> str:
    folder_path = Path(cfg.filter_folder)
    if not folder_path.is_absolute():
        folder_path = (cfg.work_dir / folder_path).resolve()
    src = src.replace('folder = "фильтр_ЖК_Высокое"', f'folder = r"{folder_path}"')
    src = src.replace(
        "pd.read_excel('События-по-коровам.xlsx')",
        f'read_filter_excel(r"{cfg.events_path}")',
    )
    src = src.replace('pd.read_excel(f"{folder}/', 'read_filter_excel(f"{folder}/')
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
    src = src.replace("n_jobs=-1", "n_jobs=1")
    src = _apply_trade_patches(src, cfg)
    train_end = cfg.resolved_train_end()
    train_end_s = train_end.strftime("%Y-%m-%d")
    src = re.sub(
        r"MAX_DATE_PROB = pd\.Timestamp\('[^']+'\)",
        f"MAX_DATE_PROB = pd.Timestamp('{train_end_s}')",
        src,
    )
    src = re.sub(
        r"MAX_DATE = pd\.Timestamp\('[^']+'\)",
        f"MAX_DATE = pd.Timestamp('{train_end_s}')",
        src,
    )
    src = re.sub(
        r"MAX_DATE_TRAIN = pd\.Timestamp\('[^']+'\)",
        f"MAX_DATE_TRAIN = pd.Timestamp('{train_end_s}')",
        src,
    )
    predict_end = cfg.resolved_predict_months()[-1]
    predict_end_ts = pd.Timestamp(f"{predict_end[0]}-{predict_end[1]:02d}-01") + pd.offsets.MonthEnd(0)
    pe_s = predict_end_ts.strftime("%Y-%m-%d")
    src = re.sub(
        r"df_events = df_events\[df_events\['Дата'\] <= pd\.Timestamp\('[^']+'\)\]",
        f"df_events = df_events[df_events['Дата'] <= pd.Timestamp('{pe_s}')]",
        src,
    )
    return src


def furazh_base_sep_2024(cfg: PipelineConfig) -> int:
    if not cfg.lactation_path.exists():
        return 2909
    df = pd.read_excel(cfg.lactation_path)
    cols = [c for c in ("L1", "L2", "L3", "L4", "L5+") if c in df.columns]
    if not cols:
        return 2909
    if "год" in df.columns and "месяц" in df.columns:
        te = cfg.resolved_train_end()
        row = df[(df["год"] == te.year) & (df["месяц"] == te.month)]
        if len(row):
            return int(row[cols].sum(axis=1).iloc[0])
        row = df[(df["год"] == 2024) & (df["месяц"] == 9)]
        if len(row):
            return int(row[cols].sum(axis=1).iloc[0])
    return 2909


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
    ns["state_sales_forecast"] = ns.get("results")


def run_young_stock(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Факт и прогноз молодняка + нетели (snapshot + XGB как forecast_young_groups)."""
    from forecast_young_groups import (  # noqa: WPS433
        build_predict_row,
        create_features,
        iter_predict_months,
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

    train_end = cfg.resolved_train_end()
    pred_months = cfg.resolved_predict_months()
    predict_start = pred_months[0]
    predict_end = pred_months[-1]
    fyg.TRAIN_END = pd.Timestamp(train_end)
    fyg.PREDICT_START = predict_start
    fyg.PREDICT_END = predict_end

    hist = load_history()
    ev = load_event_features()
    df = hist.merge(ev, on=["год", "месяц", "дата_месяц"], how="left").fillna(0)
    targets = snap.YOUNG_AND_NETELI_KEYS + ["Всего без дойных и сухостойных"]
    df = create_features(df, snap.YOUNG_AND_NETELI_KEYS)
    meta = {"год", "месяц", "дата_месяц", "дата_снимка"}
    feature_cols = [c for c in df.columns if c not in meta and c not in targets]

    train_df = df[df["дата_месяц"] <= train_end].copy()
    models = train_models(train_df, feature_cols, targets)
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
    for year, month in iter_predict_months(predict_start, predict_end):
        trend += 1
        X_pred = build_predict_row(year, month, trend, forecasts, monthly_avg, feature_cols)
        row = {"год": year, "месяц": month}
        for t in snap.YOUNG_AND_NETELI_KEYS:
            val = max(0, int(round(models[t].predict(X_pred)[0])))
            forecasts[t][month_key(year, month)] = float(val)
            row[t] = val
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    fact_start = pd.Timestamp(year=predict_start[0], month=predict_start[1], day=1)
    fact_end = pd.Timestamp(year=predict_end[0], month=predict_end[1], day=1)
    fact_rows = df[
        (df["дата_месяц"] >= fact_start)
        & (df["дата_месяц"] <= fact_end)
    ][["год", "месяц"] + snap.YOUNG_AND_NETELI_KEYS].copy()

    if fact_rows.empty:
        raw = load_vysokoe_raw_tables(str(folder_path))
        tables = excel_to_backtest_tables(raw)
        fact_rows = monthly_young_neteli_history(
            tables, predict_end[0], predict_end[1], 2022, 1
        )
        pm = (fact_rows["год"] > predict_start[0]) | (
            (fact_rows["год"] == predict_start[0]) & (fact_rows["месяц"] >= predict_start[1])
        )
        fact_rows = fact_rows.loc[pm][["год", "месяц"] + snap.YOUNG_AND_NETELI_KEYS]

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

    sys.path.insert(0, str(ROOT))
    os.chdir(cfg.work_dir)
    cells = load_cell_sources()
    ns: dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
    }

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
        (23, after_cell_23),
        (25, after_cell_25),
    ]

    inj: dict[str, Any] = {}
    for cell_id, hook in order:
        print("\n" + "=" * 72)
        print(f"Шаг: ячейка finál {cell_id} ({cfg.name})")
        print("=" * 72)
        src = patch_cell_source(cells[cell_id], cfg)
        if cell_id == 12:
            inj["calving_forecast"] = ns.get("state_calving_forecast", {})
            inj["culling_forecast"] = ns.get("state_culling", {})
            src = re.sub(
                r"FURAZH_BASE_SEP_2024\s*=\s*\d+",
                f"FURAZH_BASE_SEP_2024 = {furazh_base_sep_2024(cfg)}",
                src,
            )
        if cell_id == 3:
            inj["forecast_model1"] = ns.get("state_forecast_model1", {})
        if cell_id == 5:
            inj["calving_forecast"] = ns.get("state_calving_scalar", {})
        if cell_id == 10:
            inj["furazh_forecast"] = ns.get("state_furazh_forecast", ns.get("state_furazh_balance", {}))
        if cell_id == 16:
            inj["culling_forecast"] = ns.get("state_culling", {})
            inj["status_forecast"] = ns.get("state_status", {})
        if cell_id == 18:
            inj["calving_forecast"] = ns.get("state_calving_scalar", {})
        if cell_id == 20:
            inj["furazh_forecast"] = ns.get("state_furazh_balance", {})
            if not cfg.lactation_path.exists():
                print(f"⚠️ Нет {cfg.lactation_path} — пропуск ячейки 20")
                continue

        try:
            run_cell(cell_id, src, ns, inj)
            hook(ns)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Ячейка {cell_id} ошибка: {exc}")
            if cell_id in (23, 25):
                raise
        if cell_id == 12:
            align_dry_with_furazh(ns)
            inj["furazh_forecast"] = ns.get("state_furazh_balance", {})

    try:
        fact_young, pred_young = run_young_stock(cfg)
        ns["state_young_fact"] = fact_young
        ns["state_young_pred"] = pred_young
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Молодняк: {exc}")
        ns["state_young_fact"] = pd.DataFrame()
        ns["state_young_pred"] = pd.DataFrame()

    forecast_table, fact_table = assemble_tables(ns, cfg)
    cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cfg.output_xlsx, engine="openpyxl") as writer:
        forecast_table.to_excel(writer, sheet_name="прогноз")
        fact_table.to_excel(writer, sheet_name="факт")
    print(f"\n✅ Сохранено: {cfg.output_xlsx}")
    return forecast_table, fact_table


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
    month_cols = cfg.resolved_month_cols()
    predict_months = cfg.resolved_predict_months()
    py0, pm0 = predict_months[0]

    def _predict_period_mask(df: pd.DataFrame) -> pd.Series:
        return (df["год"] > py0) | ((df["год"] == py0) & (df["месяц"] >= pm0))

    params: list[tuple[str, dict[str, float]]] = []
    facts: list[tuple[str, dict[str, float]]] = []

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
            sub = sales_f.loc[_predict_period_mask(sales_f)]
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
            sub = sales_fact.loc[_predict_period_mask(sales_fact)]
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
                sub = yfact.loc[_predict_period_mask(yfact)]
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
                sub = ypred.loc[_predict_period_mask(ypred)]
                params.append(
                    (
                        f"остаток_{col}",
                        {_col_label(int(r["год"]), int(r["месяц"])): float(r[col]) for _, r in sub.iterrows()},
                    )
                )

    def to_df(items: list[tuple[str, dict[str, float]]]) -> pd.DataFrame:
        rows = []
        for name, data in items:
            row = {"параметр": name}
            for c in month_cols:
                row[c] = data.get(c, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    return to_df(params), to_df(facts)


# --- Kaluga: фильтр данных и временная папка Excel ---


def _normalize_kaluga_event(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_events_df(df)


def read_filter_excel(path: str | Path) -> pd.DataFrame:
    """Чтение годового Excel из filter_* с колонкой «Событие» (из Event при необходимости)."""
    return normalize_events_df(pd.read_excel(path))


def _filter_subdivision(df: pd.DataFrame, farm: str, unit: str) -> pd.DataFrame:
    aliases = SUBDIVISION_ALIASES.get(unit, [unit])
    m = df["Source.Name"].astype(str).str.strip() == farm.strip()
    m &= df["Столбец1"].astype(str).str.strip().isin(aliases)
    return df.loc[m].copy()


def build_kaluga_filter_folder(
    farm: str,
    unit: str,
    data_dir: Path,
    out_dir: Path,
) -> Path:
    """Готовит фильтр_<unit> с годовыми Excel как у ЖК Высокое."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calv_path = data_dir / "Отелы плюс родившиеся Калуга DZ 120726.xlsx"
    sem_path = data_dir / "Осеменения Калуга DZ 120726.xlsx"
    disp_path = data_dir / "Выбытие + Запуск Калуга DZ 120726.xlsx"

    calv = _normalize_kaluga_event(pd.read_excel(calv_path))
    sem = _normalize_kaluga_event(pd.read_excel(sem_path))
    disp = _normalize_kaluga_event(pd.read_excel(disp_path))

    calv = _filter_subdivision(calv, farm, unit)
    sem = _filter_subdivision(sem, farm, unit)
    disp_all = _filter_subdivision(disp, farm, unit)

    dry_ev = disp_all[disp_all["Event"].astype(str).str.upper().str.strip().isin(["DRY", "ЗАПУСК", "ЗАПУСКА"])]
    cull = disp_all[
        ~disp_all["Event"].astype(str).str.upper().str.strip().isin(["DRY", "ЗАПУСК", "ЗАПУСКА"])
    ]

    for prefix, data in (
        ("Отелы", calv),
        ("Осеменения", sem),
        ("Запуск", dry_ev),
        ("Выбытие", cull),
    ):
        if data.empty:
            continue
        data = data.copy()
        data["Дата"] = pd.to_datetime(data["Дата"], errors="coerce")
        for year in (2022, 2023, 2024, 2025):
            part = data[data["Дата"].dt.year == year]
            if len(part):
                part.to_excel(out_dir / f"{prefix}_{year}.xlsx", index=False)

    return out_dir


def build_kaluga_events_csv(farm: str, unit: str, csv_path: Path, out_xlsx: Path) -> Path:
    aliases = SUBDIVISION_ALIASES.get(unit, [unit])
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
        if "Столбец1" not in chunk.columns:
            continue
        m = chunk["Source.Name"].astype(str).str.strip() == farm.strip()
        m &= chunk["Столбец1"].astype(str).str.strip().isin(aliases)
        if m.any():
            chunks.append(chunk.loc[m])
    if not chunks:
        raise ValueError(f"Нет строк для {farm} / {unit}")
    df = pd.concat(chunks, ignore_index=True)
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
    data_dir = ROOT / "Калуга" / "данные-калуга"
    filter_dir = build_kaluga_filter_folder(farm, unit, data_dir, work / f"фильтр_{safe}")
    events = build_kaluga_events_csv(
        farm,
        unit,
        ROOT / "Калуга" / "События-по-коровам.csv",
        work / "События-по-коровам.xlsx",
    )
    rules = kaluga_trade_rules(farm, unit)
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])
    lact = work / "поголовье_по_лактациям.xlsx"
    src_lact = ROOT / "d1" / "поголовье_по_лактациям_январь2022_декабрь2025.xlsx"
    if not lact.exists() and src_lact.exists():
        pd.read_excel(src_lact).to_excel(lact, index=False)
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
        sales_require_pereezd=True,
        exit_event_types=["ВЫБЫТИЕ", "ПРОДАНА", "SOLD"],
        sold_heifer_dest=rules["heifer_sale"][0] if rules["heifer_sale"] else unit,
        sold_bull_dest="БЫЧКИ",
        kaluga_farm=farm,
        kaluga_unit=unit,
        kaluga_data_dir=data_dir,
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

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
