"""Параметры finál-прогноза для UI/Excel: порядок строк и русские подписи."""
from __future__ import annotations

from typing import Any

import pandas as pd

AGE_GROUPS: tuple[str, ...] = ("0-2", "2-6", "6-12", "12-18", "18+")

# Внутренние ключи (как в assemble_tables) — порядок строк в выводе.
FORECAST_DISPLAY_PARAMS: tuple[str, ...] = (
    "отёлы_всего",
    "отёлы_коров",
    "отёлы_нетели",
    "приплод_телочки",
    "приплод_бычки",
    "падеж_телочки",
    "падеж_бычки",
    "сухостойные",
    "дойные",
    "фуражные",
    "остаток_L1",
    "остаток_L2",
    "остаток_L3",
    "остаток_L4",
    "остаток_L5+",
    *(f"продажа_телки_{a}_внутри" for a in AGE_GROUPS),
    *(f"продажа_бычки_{a}_внутри" for a in AGE_GROUPS),
    *(f"покупка_телки_{a}_внутри" for a in AGE_GROUPS),
    "покупка_нетели_внутри",
    "остаток_Нетели",
    *(f"остаток_Т {a}" for a in AGE_GROUPS),
    "остаток_Б 0-2",
    "остаток_Б 2-6",
    "остаток_Б 6-12",
    "остаток_Б 18+",
)

FORECAST_PARAM_LABELS_RU: dict[str, str] = {
    "отёлы_всего": "Отёлы всего",
    "отёлы_коров": "Отёлы коров",
    "отёлы_нетели": "Отёлы нетелей",
    "приплод_телочки": "Приплод телочки",
    "приплод_бычки": "Приплод бычки",
    "падеж_телочки": "Падёж телочки",
    "падеж_бычки": "Падёж бычки",
    "сухостойные": "Сухостойные",
    "дойные": "Дойные",
    "фуражные": "Фуражные",
    "остаток_L1": "Коровы в 1 лактации",
    "остаток_L2": "Коровы во 2 лактации",
    "остаток_L3": "Коровы в 3 лактации",
    "остаток_L4": "Коровы в 4 лактации",
    "остаток_L5+": "Коровы в 5 лактации",
    **{f"продажа_телки_{a}_внутри": f"Продажа телочек внутри хоз {a}" for a in AGE_GROUPS},
    **{f"продажа_бычки_{a}_внутри": f"Продажа бычков {a}" for a in AGE_GROUPS},
    **{f"покупка_телки_{a}_внутри": f"Покупка телочек внутри хоз {a}" for a in AGE_GROUPS},
    "покупка_нетели_внутри": "Покупка нетелей внутри хоз",
    "остаток_Нетели": "Нетели",
    **{f"остаток_Т {a}": f"Кол-во телочек {a}" for a in AGE_GROUPS},
    "остаток_Б 0-2": "Кол-во бычков 0-2",
    "остаток_Б 2-6": "Кол-во бычков 2-6",
    "остаток_Б 6-12": "Кол-во бычков 6-12",
    "остаток_Б 18+": "Кол-во бычков 18+",
}

# Поля ручной базы на дату обучения (ключ → подпись UI).
MANUAL_BASELINE_FIELDS: tuple[tuple[str, str], ...] = (
    ("dry", "Сухостойные"),
    ("milk", "Дойные"),
    ("furazh", "Фуражные"),
    ("L1", "1 лактация"),
    ("L2", "2 лактация"),
    ("L3", "3 лактация"),
    ("L4", "4 лактация"),
    ("L5+", "5 и более лактаций"),
)


def manual_baseline_from_inputs(raw: dict[str, Any]) -> dict[str, float] | None:
    """Собрать sep2024_baseline из полей UI; None если ничего не введено."""
    out: dict[str, float] = {}
    for key, _label in MANUAL_BASELINE_FIELDS:
        val = raw.get(key)
        if val is None or val == "":
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f < 0:
            continue
        if f == 0:
            continue
        out[key] = f
    if not out:
        return None
    if "furazh" not in out and "dry" in out and "milk" in out:
        out["furazh"] = out["dry"] + out["milk"]
    elif "milk" not in out and "furazh" in out and "dry" in out:
        out["milk"] = max(0.0, out["furazh"] - out["dry"])
    elif "dry" not in out and "furazh" in out and "milk" in out:
        out["dry"] = max(0.0, out["furazh"] - out["milk"])
    return out


def filter_forecast_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить нужные строки и заменить «параметр» на русские подписи."""
    if df is None or df.empty or "параметр" not in df.columns:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    order = {name: i for i, name in enumerate(FORECAST_DISPLAY_PARAMS)}
    out = df.loc[df["параметр"].astype(str).isin(order)].copy()
    out["_ord"] = out["параметр"].map(order)
    out = out.sort_values("_ord").drop(columns=["_ord"])
    out["параметр"] = out["параметр"].astype(str).map(
        lambda k: FORECAST_PARAM_LABELS_RU.get(k, k)
    )
    month_cols = [c for c in out.columns if c != "параметр"]
    return out[["параметр", *month_cols]]
