# Прогноз групп молодняка + нетели (без дойных/сухостойных)
# Обучение: январь 2022 — сентябрь 2024
# Прогноз: октябрь 2024 — декабрь 2025
# Запуск: python forecast_young_groups.py  или  %run forecast_young_groups.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

from snapshot_zhk_vysokoe import (  # noqa: E402
    YOUNG_AND_NETELI_KEYS,
    excel_to_backtest_tables,
    load_vysokoe_raw_tables,
    monthly_young_neteli_history,
)

FOLDER = "фильтр_ЖК_Высокое"
HIST_XLSX = "снимок_молодняк_нетели_2022_01_2025_12.xlsx"
OUT_XLSX = "прогноз_молодняк_нетели_окт2024_дек2025.xlsx"
TRAIN_END = pd.Timestamp("2024-09-01")
PREDICT_START = (2024, 10)
PREDICT_END = (2025, 12)

months_ru = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def _read_yearly(prefix: str, years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = Path(FOLDER) / f"{prefix}_{y}.xlsx"
        if p.exists():
            parts.append(pd.read_excel(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def process_semen(df_semen: pd.DataFrame) -> pd.DataFrame:
    if df_semen.empty:
        return df_semen
    df = df_semen.copy()
    df["Дата"] = pd.to_datetime(df["Дата"])
    df["BDAT"] = pd.to_datetime(df["BDAT"], errors="coerce")
    df["животное_ключ"] = df["REG"].fillna("").astype(str)
    mask = (df["животное_ключ"] == "") | (df["животное_ключ"] == "nan")
    df.loc[mask, "животное_ключ"] = (
        df.loc[mask, "ID"].astype(str) + "_" + df.loc[mask, "BDAT"].astype(str)
    )
    df = df.sort_values(["животное_ключ", "Дата"])
    df["R"] = df["R"].astype(str).str.strip()
    c_mask = df["R"] == "C"
    df["Дата_исправленная"] = df["Дата"]
    for idx in df[c_mask].index:
        key = df.loc[idx, "животное_ключ"]
        cur = df.loc[idx, "Дата"]
        prev = df[(df["животное_ключ"] == key) & (df["Дата"] < cur) & (df.index != idx)].sort_values(
            "Дата", ascending=False
        )
        if len(prev):
            df.loc[idx, "Дата_исправленная"] = prev.iloc[0]["Дата"]
    df["тип_осеменения"] = df["R"]
    return df


def aggregate_count_monthly(df: pd.DataFrame, date_col: str = "Дата", name: str = "cnt") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["год", "месяц", "дата_месяц", name])
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["месяц"] = d[date_col].dt.month
    d["год"] = d[date_col].dt.year
    d["дата_месяц"] = pd.to_datetime(d["год"].astype(str) + "-" + d["месяц"].astype(str) + "-01")
    return d.groupby(["год", "месяц", "дата_месяц"]).size().reset_index(name=name)


def aggregate_semen_monthly(df_semen: pd.DataFrame) -> pd.DataFrame:
    if df_semen.empty:
        return pd.DataFrame(columns=["год", "месяц", "дата_месяц", "всего_осеменений", "успешные"])
    df = df_semen.copy()
    df["месяц"] = df["Дата_исправленная"].dt.month
    df["год"] = df["Дата_исправленная"].dt.year
    df["дата_месяц"] = pd.to_datetime(df["год"].astype(str) + "-" + df["месяц"].astype(str) + "-01")
    total = df.groupby(["год", "месяц", "дата_месяц"]).size().reset_index(name="всего_осеменений")
    ok = df[df["тип_осеменения"] == "P"].groupby(["год", "месяц", "дата_месяц"]).size().reset_index(name="успешные")
    return total.merge(ok, on=["год", "месяц", "дата_месяц"], how="left").fillna(0)


def load_history() -> pd.DataFrame:
    path = Path(HIST_XLSX)
    if path.exists():
        df = pd.read_excel(path)
    else:
        print(f"⚠️ {HIST_XLSX} не найден — считаем снимки (может занять время)...")
        raw = load_vysokoe_raw_tables()
        tables = excel_to_backtest_tables(raw)
        df = monthly_young_neteli_history(tables, 2025, 12, 2022, 1)
        df.to_excel(path, index=False)

    df["дата_месяц"] = pd.to_datetime(df["год"].astype(str) + "-" + df["месяц"].astype(str) + "-01")
    return df.sort_values("дата_месяц").reset_index(drop=True)


def load_event_features() -> pd.DataFrame:
    years = [2022, 2023, 2024, 2025]
    dry = aggregate_count_monthly(_read_yearly("Запуск", years), name="запуски")
    cull = aggregate_count_monthly(_read_yearly("Выбытие", years), name="выбытия")
    sem = aggregate_semen_monthly(process_semen(_read_yearly("Осеменения", years)))

    m = dry.merge(cull, on=["год", "месяц", "дата_месяц"], how="outer")
    m = m.merge(sem, on=["год", "месяц", "дата_месяц"], how="outer").fillna(0)
    return m


def create_features(df: pd.DataFrame, target_keys: list[str]) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["месяц_синус"] = np.sin(2 * np.pi * df["месяц"] / 12)
    df["месяц_косинус"] = np.cos(2 * np.pi * df["месяц"] / 12)
    df["квартал"] = ((df["месяц"] - 1) // 3 + 1).astype(int)
    df["тренд"] = np.arange(1, len(df) + 1)

    total_col = "Всего без дойных и сухостойных"
    for lag in [1, 2, 3, 6]:
        df[f"{total_col}_lag{lag}"] = df[total_col].shift(lag).fillna(0)
    df[f"{total_col}_ma3"] = df[total_col].rolling(3, min_periods=1).mean().fillna(0)

    for col in target_keys:
        for lag in [1, 2, 3, 6]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(0)
        df[f"{col}_ma3"] = df[col].rolling(3, min_periods=1).mean().fillna(0)

    for col in ["запуски", "выбытия", "успешные", "всего_осеменений"]:
        if col not in df.columns:
            df[col] = 0.0
        for lag in [1, 2, 3, 6]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(0)
        df[f"{col}_ma3"] = df[col].rolling(3, min_periods=1).mean().fillna(0)

    return df


def train_models(train_df: pd.DataFrame, feature_cols: list[str], targets: list[str]) -> dict[str, XGBRegressor]:
    models: dict[str, XGBRegressor] = {}
    X = train_df[feature_cols].fillna(0)

    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.1, 0.5],
        "reg_lambda": [0.5, 1.0],
    }
    default_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
    }

    for target in targets:
        y = train_df[target].values
        try:
            tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(train_df) - 1)))
            grid = GridSearchCV(
                XGBRegressor(random_state=42, verbosity=0),
                param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X, y)
            params = grid.best_params_
        except Exception:
            params = default_params

        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X, y)
        models[target] = model
        mae = mean_absolute_error(y, model.predict(X))
        print(f"  {target}: MAE train = {mae:.2f}")

    return models


def month_key(year: int, month: int) -> tuple[int, int]:
    return year, month


def prev_month(year: int, month: int, delta: int) -> tuple[int, int]:
    y, m = year, month - delta
    while m <= 0:
        m += 12
        y -= 1
    return y, m


def get_value(series: dict[tuple[int, int], float], year: int, month: int, fallback: float = 0.0) -> float:
    return float(series.get(month_key(year, month), fallback))


def build_predict_row(
    year: int,
    month: int,
    trend: int,
    forecasts: dict[str, dict[tuple[int, int], float]],
    monthly_avg: dict[str, dict[int, float]],
    feature_cols: list[str],
) -> pd.DataFrame:
    row: dict[str, float] = {
        "год": year,
        "месяц": month,
        "месяц_синус": np.sin(2 * np.pi * month / 12),
        "месяц_косинус": np.cos(2 * np.pi * month / 12),
        "квартал": (month - 1) // 3 + 1,
        "тренд": trend,
    }

    for col in ["запуски", "выбытия", "успешные", "всего_осеменений"]:
        row[col] = monthly_avg.get(col, {}).get(month, 0.0)
        for lag in [1, 2, 3, 6]:
            py, pm = prev_month(year, month, lag)
            row[f"{col}_lag{lag}"] = monthly_avg.get(col, {}).get(pm, row[col])
        row[f"{col}_ma3"] = (row[f"{col}_lag1"] + row[f"{col}_lag2"] + row[f"{col}_lag3"]) / 3

    total_col = "Всего без дойных и сухостойных"
    f_total = forecasts.get(total_col, {})
    for lag in [1, 2, 3, 6]:
        py, pm = prev_month(year, month, lag)
        row[f"{total_col}_lag{lag}"] = get_value(f_total, py, pm, monthly_avg.get(total_col, {}).get(pm, 0))
    row[f"{total_col}_ma3"] = (
        row[f"{total_col}_lag1"] + row[f"{total_col}_lag2"] + row[f"{total_col}_lag3"]
    ) / 3

    for target in YOUNG_AND_NETELI_KEYS:
        f = forecasts.get(target, {})
        for lag in [1, 2, 3, 6]:
            py, pm = prev_month(year, month, lag)
            row[f"{target}_lag{lag}"] = get_value(f, py, pm, monthly_avg.get(target, {}).get(pm, 0))
        row[f"{target}_ma3"] = (
            row[f"{target}_lag1"] + row[f"{target}_lag2"] + row[f"{target}_lag3"]
        ) / 3

    pred_df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    return pred_df[feature_cols].fillna(0)


def iter_predict_months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def main() -> None:
    print("=" * 80)
    print("ПРОГНОЗ: нетели + молодняк (Т/Б), без дойных и сухостойных")
    print("  Обучение: январь 2022 — сентябрь 2024")
    print("  Прогноз: октябрь 2024 — декабрь 2025")
    print("=" * 80)

    hist = load_history()
    events = load_event_features()
    df = hist.merge(events, on=["год", "месяц", "дата_месяц"], how="left").fillna(0)

    targets = YOUNG_AND_NETELI_KEYS + ["Всего без дойных и сухостойных"]
    df = create_features(df, YOUNG_AND_NETELI_KEYS)

    meta = {"год", "месяц", "дата_месяц", "дата_снимка"}
    feature_cols = [c for c in df.columns if c not in meta and c not in targets]

    train_mask = df["дата_месяц"] <= TRAIN_END
    train_df = df.loc[train_mask].copy()
    print(f"\nОбучающих месяцев: {len(train_df)}, признаков: {len(feature_cols)}")

    print("\nОбучение XGBoost (отдельная модель на каждую группу)...")
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

    actual_full = df.set_index(["год", "месяц"])
    rows_out = []
    trend = len(train_df)

    print("\n" + "=" * 80)
    print("РЕКУРСИВНЫЙ ПРОГНОЗ")
    print("=" * 80)

    for year, month in iter_predict_months(PREDICT_START, PREDICT_END):
        trend += 1
        X_pred = build_predict_row(year, month, trend, forecasts, monthly_avg, feature_cols)
        pred: dict[str, int] = {}
        for t in targets:
            val = max(0, int(round(models[t].predict(X_pred)[0])))
            pred[t] = val
            forecasts[t][month_key(year, month)] = float(val)

        fact = {}
        for t in targets:
            try:
                fact[t] = int(actual_full.loc[(year, month), t])
            except KeyError:
                fact[t] = np.nan

        label = f"{months_ru[month]}{str(year)[-2:]}"
        print(f"\n{label}: всего прогноз {pred['Всего без дойных и сухостойных']}, факт {fact.get('Всего без дойных и сухостойных', '—')}")

        row = {"год": year, "месяц": month, "месяц_название": label}
        for t in targets:
            row[f"{t}_прогноз"] = pred[t]
            row[f"{t}_факт"] = fact.get(t)
            if pd.notna(fact.get(t)):
                row[f"{t}_ошибка"] = pred[t] - int(fact[t])
        rows_out.append(row)

    result = pd.DataFrame(rows_out)
    result.to_excel(OUT_XLSX, index=False)
    print(f"\n✅ Сохранено: {OUT_XLSX}")

    print("\n" + "=" * 80)
    print("MAE на периоде прогноза (где есть факт)")
    print("=" * 80)
    for t in targets:
        sub = result.dropna(subset=[f"{t}_факт"])
        if sub.empty:
            continue
        mae = mean_absolute_error(sub[f"{t}_факт"], sub[f"{t}_прогноз"])
        print(f"  {t}: MAE = {mae:.2f}")


if __name__ == "__main__":
    main()
