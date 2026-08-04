# AUTO-GENERATED from ЖК_Высокое_финал.ipynb — do not edit by hand
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')


# ===== NOTEBOOK CELL 1 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

# Параметры
MEDIAN_GESTATION_DAYS = 280
MEDIAN_GESTATION_MONTHS = MEDIAN_GESTATION_DAYS / 30.44
DRY_PERIODS = [45, 60]

print("="*80)
print("ЧЕСТНЫЙ РЕКУРСИВНЫЙ ПРОГНОЗ ОТЕЛОВ")
print(f"  • ДЛЯ ПРОГНОЗА ИСПОЛЬЗУЮТСЯ ТОЛЬКО ДАННЫЕ: 2022, 2023, январь-сентябрь 2024")
print(f"  • НИКАКИЕ ДАННЫЕ ПОСЛЕ СЕНТЯБРЯ 2024 НЕ ИСПОЛЬЗУЮТСЯ ДЛЯ ОБУЧЕНИЯ И ПРИЗНАКОВ!")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_semen_2022 = pd.read_excel(f"{folder}/Осеменения_2022.xlsx")
df_semen_2023 = pd.read_excel(f"{folder}/Осеменения_2023.xlsx")
df_semen_2024 = pd.read_excel(f"{folder}/Осеменения_2024.xlsx")

df_dry_2022 = pd.read_excel(f"{folder}/Запуск_2022.xlsx")
df_dry_2023 = pd.read_excel(f"{folder}/Запуск_2023.xlsx")
df_dry_2024 = pd.read_excel(f"{folder}/Запуск_2024.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")

print("\n✅ Данные загружены (ТОЛЬКО 2022-2024)")

# ============================================
# 2. ОБРАБОТКА ОСЕМЕНЕНИЙ С R='C'
# ============================================
def process_c_semen(df_semen):
    if len(df_semen) == 0:
        return df_semen
    df = df_semen.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['BDAT'] = pd.to_datetime(df['BDAT'])
    df['животное_ключ'] = df['REG'].fillna('').astype(str)
    mask_no_reg = (df['животное_ключ'] == '') | (df['животное_ключ'] == 'nan')
    df.loc[mask_no_reg, 'животное_ключ'] = df.loc[mask_no_reg, 'ID'].astype(str) + '_' + df.loc[mask_no_reg, 'BDAT'].astype(str)
    df = df.sort_values(['животное_ключ', 'Дата'])
    c_mask = df['R'].str.strip() == 'C'
    df['Дата_исправленная'] = df['Дата']
    for idx in df[c_mask].index:
        animal_key = df.loc[idx, 'животное_ключ']
        current_date = df.loc[idx, 'Дата']
        prev_semen = df[(df['животное_ключ'] == animal_key) & (df['Дата'] < current_date) & (df.index != idx)].sort_values('Дата', ascending=False)
        if len(prev_semen) > 0:
            df.loc[idx, 'Дата_исправленная'] = prev_semen.iloc[0]['Дата']
    df['тип_осеменения'] = df['R'].str.strip()
    return df

print("\n" + "="*80)
print("ОБРАБОТКА ДАННЫХ")
print("="*80)

df_semen_2022_proc = process_c_semen(df_semen_2022)
df_semen_2023_proc = process_c_semen(df_semen_2023)
df_semen_2024_proc = process_c_semen(df_semen_2024)

# ============================================
# 3. РАСЧЕТ ВЕРОЯТНОСТИ УСПЕШНОГО ОСЕМЕНЕНИЯ (ТОЛЬКО 2022-СЕН 2024)
# ============================================
print("\n" + "="*80)
print("РАСЧЕТ ВЕРОЯТНОСТИ УСПЕШНОГО ОСЕМЕНЕНИЯ")
print("="*80)

MAX_DATE_PROB = pd.Timestamp('2024-09-30')

def filter_by_date(df_semen, max_date):
    df = df_semen.copy()
    df['Дата_исправленная'] = pd.to_datetime(df['Дата_исправленная'])
    return df[df['Дата_исправленная'] <= max_date]

all_semen_historical = pd.concat([
    filter_by_date(df_semen_2022_proc, MAX_DATE_PROB),
    filter_by_date(df_semen_2023_proc, MAX_DATE_PROB),
    filter_by_date(df_semen_2024_proc, MAX_DATE_PROB)
], ignore_index=True)

def calculate_success_probability(df_semen):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['месяц', 'вероятность_успеха'])
    df = df_semen.copy()
    df['месяц'] = df['Дата_исправленная'].dt.month
    monthly_stats = df.groupby('месяц').agg(
        всего=('тип_осеменения', 'count'),
        успешные=('тип_осеменения', lambda x: (x == 'P').sum())
    ).reset_index()
    monthly_stats['вероятность_успеха'] = monthly_stats['успешные'] / monthly_stats['всего']
    return monthly_stats

success_probability = calculate_success_probability(all_semen_historical)
prob_by_month = dict(zip(success_probability['месяц'], success_probability['вероятность_успеха']))
overall_prob = all_semen_historical['тип_осеменения'].eq('P').sum() / len(all_semen_historical)

print("\nВероятность успешного осеменения по месяцам (на основе 2022-сен 2024):")
for month in range(1, 13):
    p = prob_by_month.get(month, overall_prob)
    print(f"  {months_ru[month]}: {p:.1%}")

# ============================================
# 4. АГРЕГАЦИЯ ДАННЫХ (ТОЛЬКО ДО СЕНТЯБРЯ 2024)
# ============================================
MAX_DATE = pd.Timestamp('2024-09-30')

def aggregate_calvings_monthly(df_calvings, max_date=None):
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'отелы'])
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    otel_mask = df['Событие'].str.upper().str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])
    df = df[otel_mask].copy()
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='отелы')
    return monthly

def aggregate_dry_monthly(df_dry, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'запуски'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='запуски')
    return monthly

def aggregate_culling_monthly(df_culling, max_date=None):
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'выбытия'])
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='выбытия')
    return monthly

def aggregate_semen_monthly(df_semen, max_date=None):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'всего_осеменений', 'успешные', 'процент_успешных'])
    df = df_semen.copy()
    if max_date:
        df = df[df['Дата_исправленная'] <= max_date]
    df['месяц'] = df['Дата_исправленная'].dt.month
    df['год'] = df['Дата_исправленная'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    total_semen = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='всего_осеменений')
    success = df[df['тип_осеменения'] == 'P'].groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='успешные')
    features = total_semen.merge(success, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
    features['процент_успешных'] = (features['успешные'] / features['всего_осеменений'] * 100).fillna(0)
    return features

def aggregate_dry_shifted(df_dry, shift_days, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['дата_отела'] = df['Дата'] + pd.Timedelta(days=shift_days)
    if max_date:
        df = df[df['дата_отела'] <= max_date]
    df['год_отела'] = df['дата_отела'].dt.year
    df['месяц_отела'] = df['дата_отела'].dt.month
    df['дата_месяц'] = pd.to_datetime(df['год_отела'].astype(str) + '-' + df['месяц_отела'].astype(str) + '-01')
    dry_shifted = df.groupby(['год_отела', 'месяц_отела', 'дата_месяц']).size().reset_index(name=f'запуски_{shift_days}d_shift')
    dry_shifted.columns = ['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift']
    return dry_shifted

# Собираем ОБУЧАЮЩИЕ данные (только до сентября 2024)
monthly_calvings = pd.concat([
    aggregate_calvings_monthly(df_calvings_2022, MAX_DATE),
    aggregate_calvings_monthly(df_calvings_2023, MAX_DATE),
    aggregate_calvings_monthly(df_calvings_2024, MAX_DATE)
], ignore_index=True)

monthly_dry = pd.concat([
    aggregate_dry_monthly(df_dry_2022, MAX_DATE),
    aggregate_dry_monthly(df_dry_2023, MAX_DATE),
    aggregate_dry_monthly(df_dry_2024, MAX_DATE)
], ignore_index=True)

monthly_culling = pd.concat([
    aggregate_culling_monthly(df_culling_2022, MAX_DATE),
    aggregate_culling_monthly(df_culling_2023, MAX_DATE),
    aggregate_culling_monthly(df_culling_2024, MAX_DATE)
], ignore_index=True)

semen_features = pd.concat([
    aggregate_semen_monthly(df_semen_2022_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2023_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2024_proc, MAX_DATE)
], ignore_index=True)

dry_features = {}
for period in DRY_PERIODS:
    dry_features[period] = pd.concat([
        aggregate_dry_shifted(df_dry_2022, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2023, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2024, period, MAX_DATE)
    ], ignore_index=True)

# Объединяем
train_df = monthly_calvings.rename(columns={'отелы': 'отелы'})
train_df = train_df.merge(monthly_dry, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(monthly_culling, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(semen_features, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

for period in DRY_PERIODS:
    train_df = train_df.merge(dry_features[period], on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

print(f"\nОбучающих месяцев (2022,2023,янв-сен2024): {len(train_df)}")

# ============================================
# 5. ДОБАВЛЕНИЕ ПРИЗНАКА ВЕРОЯТНОСТИ УСПЕХА ОСЕМЕНЕНИЙ
# ============================================
train_df['вероятность_успеха_осеменения'] = train_df['месяц'].map(prob_by_month).fillna(overall_prob)
train_df['ожидаемые_успешные'] = train_df['всего_осеменений'] * train_df['вероятность_успеха_осеменения']

print("\nДобавлен признак 'вероятность_успеха_осеменения'")

# ============================================
# 6. СОЗДАНИЕ ПРИЗНАКОВ (ЛАГИ ТОЛЬКО ИЗ ОБУЧАЮЩИХ ДАННЫХ)
# ============================================
def create_features(df):
    df = df.copy()

    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)

    # Лаги отелов (только из обучающих данных)
    df['отелы_lag1'] = df['отелы'].shift(1)
    df['отелы_lag2'] = df['отелы'].shift(2)
    df['отелы_lag3'] = df['отелы'].shift(3)
    df['отелы_lag6'] = df['отелы'].shift(6)

    df['успешные_lag3'] = df['успешные'].shift(3)
    df['успешные_lag6'] = df['успешные'].shift(6)

    df['ожидаемые_успешные_lag3'] = df['ожидаемые_успешные'].shift(3)
    df['ожидаемые_успешные_lag6'] = df['ожидаемые_успешные'].shift(6)

    df['запуски_lag1'] = df['запуски'].shift(1)
    df['запуски_lag2'] = df['запуски'].shift(2)
    df['запуски_lag3'] = df['запуски'].shift(3)
    df['запуски_lag6'] = df['запуски'].shift(6)

    df['выбытия_lag1'] = df['выбытия'].shift(1)
    df['выбытия_lag2'] = df['выбытия'].shift(2)
    df['выбытия_lag3'] = df['выбытия'].shift(3)

    df['отелы_ma3'] = df['отелы'].rolling(3, min_periods=1).mean()
    df['запуски_ma3'] = df['запуски'].rolling(3, min_periods=1).mean()
    df['выбытия_ma3'] = df['выбытия'].rolling(3, min_periods=1).mean()
    df['вероятность_успеха_ma3'] = df['вероятность_успеха_осеменения'].rolling(3, min_periods=1).mean()

    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        if col in df.columns:
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean()

    df['тренд'] = range(1, len(df) + 1)

    return df

train_features = create_features(train_df)

feature_cols = [col for col in train_features.columns if col not in [
    'год', 'месяц', 'дата_месяц', 'отелы', 'запуски', 'выбытия',
    'успешные', 'всего_осеменений', 'процент_успешных'
]]
train_clean = train_features.dropna()
X_train = train_clean[feature_cols]
y_train = train_clean['отелы']

print(f"\nОбучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")

# ============================================
# 7. ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ
# ============================================
print("\n" + "="*80)
print("ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ")
print("="*80)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
grid_search = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая MAE: {abs(grid_search.best_score_):.2f}")

final_model = XGBRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

# ============================================
# 8. РЕКУРСИВНЫЙ ПРОГНОЗ (ТОЛЬКО НА ОСНОВЕ ПРОГНОЗОВ!)
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ (только на основе данных до сен 2024!)")
print("="*80)

forecasts = {}
results = []

# Список месяцев для прогноза
predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")

    pred_row = pd.DataFrame({
        'год': [year],
        'месяц': [month],
        'дата_месяц': [pd.Timestamp(f'{year}-{month:02d}-01')],
        'месяц_синус': [np.sin(2 * np.pi * month / 12)],
        'месяц_косинус': [np.cos(2 * np.pi * month / 12)],
        'квартал': [(month-1)//3 + 1],
        'тренд': [len(train_features) + len(results) + 1],
        'вероятность_успеха_осеменения': [prob_by_month.get(month, overall_prob)]
    })

    # ====== ЛАГИ ОТЕЛОВ (ТОЛЬКО ИЗ ПРОГНОЗОВ!) ======
    pred_row['отелы'] = 0
    pred_row['отелы_lag1'] = forecasts.get((year, month-1), 0) if month > 1 else 0
    pred_row['отелы_lag2'] = forecasts.get((year, month-2), 0) if month > 2 else 0
    pred_row['отелы_lag3'] = forecasts.get((year, month-3), 0) if month > 3 else 0
    pred_row['отелы_lag6'] = forecasts.get((year, month-6), 0) if month > 6 else 0
    pred_row['отелы_ma3'] = (pred_row['отелы_lag1'] + pred_row['отелы_lag2'] + pred_row['отелы_lag3']) / 3 if month > 3 else 0

    # ====== ЗАПУСКИ (СРЕДНИЕ ИСТОРИЧЕСКИЕ) ======
    hist_dry = train_df[train_df['месяц'] == month]['запуски'].mean()
    pred_row['запуски'] = hist_dry
    pred_row['запуски_lag1'] = train_df[train_df['месяц'] == (month-1 if month>1 else 12)]['запуски'].mean() if month > 1 else 0
    pred_row['запуски_lag2'] = train_df[train_df['месяц'] == (month-2 if month>2 else 11)]['запуски'].mean() if month > 2 else 0
    pred_row['запуски_lag3'] = train_df[train_df['месяц'] == (month-3 if month>3 else 10)]['запуски'].mean() if month > 3 else 0
    pred_row['запуски_lag6'] = train_df[train_df['месяц'] == (month-6 if month>6 else 6)]['запуски'].mean() if month > 6 else 0
    pred_row['запуски_ma3'] = (pred_row['запуски_lag1'] + pred_row['запуски_lag2'] + pred_row['запуски_lag3']) / 3

    # ====== ВЫБЫТИЯ (ИЗ ПРОГНОЗОВ) ======
    pred_row['выбытия'] = 0
    pred_row['выбытия_lag1'] = forecasts.get((year, month-1),
                              train_df[train_df['месяц'] == (month-1 if month>1 else 12)]['выбытия'].mean()) if month > 1 else 0
    pred_row['выбытия_lag2'] = forecasts.get((year, month-2), 0) if month > 2 else 0
    pred_row['выбытия_lag3'] = forecasts.get((year, month-3), 0) if month > 3 else 0
    pred_row['выбытия_ma3'] = (pred_row['выбытия_lag1'] + pred_row['выбытия_lag2'] + pred_row['выбытия_lag3']) / 3

    # ====== ОСЕМЕНЕНИЯ (СРЕДНИЕ ИСТОРИЧЕСКИЕ) ======
    hist_total = train_df[train_df['месяц'] == month]['всего_осеменений'].mean()
    pred_row['всего_осеменений'] = hist_total
    pred_row['успешные'] = hist_total * prob_by_month.get(month, overall_prob)
    pred_row['процент_успешных'] = (pred_row['успешные'] / (hist_total + 1)) * 100
    pred_row['ожидаемые_успешные'] = pred_row['всего_осеменений'] * pred_row['вероятность_успеха_осеменения']
    pred_row['ожидаемые_успешные_lag3'] = forecasts.get((year, month-3), 0) if month > 3 else 0
    pred_row['ожидаемые_успешные_lag6'] = forecasts.get((year, month-6), 0) if month > 6 else 0
    pred_row['успешные_lag3'] = forecasts.get((year, month-3), 0) if month > 3 else 0
    pred_row['успешные_lag6'] = forecasts.get((year, month-6), 0) if month > 6 else 0

    # ====== ЗАПУСКИ СО СДВИГОМ (СРЕДНИЕ ИСТОРИЧЕСКИЕ) ======
    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        hist_val = train_df[train_df['месяц'] == month][col].mean() if col in train_df.columns else 0
        pred_row[col] = hist_val
        pred_row[f'{col}_lag2'] = train_df[train_df['месяц'] == (month-2 if month>2 else 10)][col].mean() if col in train_df.columns else 0
        pred_row[f'{col}_lag3'] = train_df[train_df['месяц'] == (month-3 if month>3 else 9)][col].mean() if col in train_df.columns else 0
        pred_row[f'{col}_ma3'] = pred_row[col]

    pred_row['вероятность_успеха_ma3'] = pred_row['вероятность_успеха_осеменения']

    # Прогнозируем
    X_pred = pred_row[[col for col in feature_cols if col in pred_row.columns]]
    X_pred = X_pred.fillna(0)

    pred_value = final_model.predict(X_pred)[0]
    pred_value = max(0, int(round(pred_value)))

    forecasts[(year, month)] = pred_value

    # Получаем факт для сравнения (ТОЛЬКО ДЛЯ ОЦЕНКИ, НЕ ДЛЯ ПРОГНОЗА!)
    if year == 2024:
        actual_df = aggregate_calvings_monthly(df_calvings_2024)
    else:
        actual_df = aggregate_calvings_monthly(df_calvings_2025)
    actual_row = actual_df[actual_df['месяц'] == month]
    actual_value = actual_row['отелы'].values[0] if len(actual_row) > 0 else 0

    results.append({
        'год': year,
        'месяц': month,
        'прогноз': pred_value,
        'факт': actual_value
    })

    error = pred_value - actual_value
    error_pct = (error / actual_value) * 100 if actual_value > 0 else 0
    status = "✅" if abs(error_pct) <= 10 else "⚠️" if abs(error_pct) <= 20 else "❌"
    print(f"  Прогноз: {pred_value}, Факт: {actual_value}, Ошибка: {error:+d} ({error_pct:+.1f}%) {status}")

# ============================================
# 9. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ПРОГНОЗ НА ОКТЯБРЬ-ДЕКАБРЬ 2024:")
print("-" * 85)
print(f"{'Месяц':<12} {'Прогноз':>8} {'Факт':>8} {'Ошибка':>8} {'Ошибка %':>10} {'Статус':<8}")
print("-" * 85)

for r in results:
    if r['год'] == 2024:
        error = r['прогноз'] - r['факт']
        error_pct = (error / r['факт']) * 100 if r['факт'] > 0 else 0
        error_sign = "+" if error > 0 else ""
        status = "✅" if abs(error_pct) <= 10 else "⚠️" if abs(error_pct) <= 20 else "❌"
        print(f"{months_ru[r['месяц']]:<12} {r['прогноз']:>8} {r['факт']:>8} "
              f"{error_sign}{error:>7} {error_sign}{error_pct:>9.1f}% {status}")

print("\n" + "="*80)
print("ПРОГНОЗ НА 2025 ГОД:")
print("-" * 85)
print(f"{'Месяц':<12} {'Прогноз':>8} {'Факт':>8} {'Ошибка':>8} {'Ошибка %':>10} {'Статус':<8}")
print("-" * 85)

for r in results:
    if r['год'] == 2025:
        error = r['прогноз'] - r['факт']
        error_pct = (error / r['факт']) * 100 if r['факт'] > 0 else 0
        error_sign = "+" if error > 0 else ""
        status = "✅" if abs(error_pct) <= 10 else "⚠️" if abs(error_pct) <= 20 else "❌"
        print(f"{months_ru[r['месяц']]:<12} {r['прогноз']:>8} {r['факт']:>8} "
              f"{error_sign}{error:>7} {error_sign}{error_pct:>9.1f}% {status}")

print("-" * 85)

total_pred = sum(r['прогноз'] for r in results)
total_actual = sum(r['факт'] for r in results)
total_error = total_pred - total_actual
total_error_pct = (total_error / total_actual) * 100 if total_actual > 0 else 0

print(f"\n📈 ИТОГО ЗА ТЕСТОВЫЙ ПЕРИОД:")
print(f"  Прогноз: {total_pred}")
print(f"  Факт: {total_actual}")
print(f"  Ошибка: {'+' if total_error > 0 else ''}{total_error} ({'+' if total_error > 0 else ''}{total_error_pct:.1f}%)")

mae = np.mean([abs(r['прогноз'] - r['факт']) for r in results])
mape = np.mean([abs((r['прогноз'] - r['факт']) / r['факт']) * 100 for r in results if r['факт'] > 0])

print(f"\n📊 МЕТРИКИ КАЧЕСТВА:")
print(f"  MAE: {mae:.1f}")
print(f"  MAPE: {mape:.1f}%")

print("\n" + "="*80)
print("ГОТОВО!")
print("ВАЖНО: Для ВСЕГО прогноза использовались ТОЛЬКО данные за 2022, 2023 и январь-сентябрь 2024!")
print("  • Лаги строятся только на обучающих данных")
print("  • Для прогноза используются только прогнозные значения предыдущих месяцев")
print("  • Данные за октябрь-декабрь 2024 и 2025 год НЕ используются")
print("="*80)

# ===== NOTEBOOK CELL 3 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

# Параметры
MEDIAN_GESTATION_DAYS = 280
MEDIAN_GESTATION_MONTHS = MEDIAN_GESTATION_DAYS / 30.44
DRY_PERIODS = [45, 60]

print("="*80)
print("ЧЕСТНЫЙ РЕКУРСИВНЫЙ ПРОГНОЗ ОТЕЛОВ (КОРОВЫ + НЕТЕЛИ)")
print(f"  • ДЛЯ ПРОГНОЗА ИСПОЛЬЗУЮТСЯ ТОЛЬКО ДАННЫЕ: 2022, 2023, январь-сентябрь 2024")
print(f"  • ОБЩИЙ ПРОГНОЗ БЕРЕТСЯ ИЗ МОДЕЛИ 1 (ГОТОВЫЙ ПРОГНОЗ)")
print(f"  • ОТДЕЛЬНАЯ МОДЕЛЬ ДЛЯ КОРОВ")
print(f"  • НЕТЕЛИ = ВСЕГО - КОРОВЫ")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ (ТОЛЬКО 2022-2024)
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_semen_2022 = pd.read_excel(f"{folder}/Осеменения_2022.xlsx")
df_semen_2023 = pd.read_excel(f"{folder}/Осеменения_2023.xlsx")
df_semen_2024 = pd.read_excel(f"{folder}/Осеменения_2024.xlsx")

df_dry_2022 = pd.read_excel(f"{folder}/Запуск_2022.xlsx")
df_dry_2023 = pd.read_excel(f"{folder}/Запуск_2023.xlsx")
df_dry_2024 = pd.read_excel(f"{folder}/Запуск_2024.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")

print("\n✅ Данные загружены (ТОЛЬКО 2022-2024)")

# ============================================
# 2. ОБРАБОТКА ОСЕМЕНЕНИЙ С R='C'
# ============================================
def process_c_semen(df_semen):
    if len(df_semen) == 0:
        return df_semen
    df = df_semen.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['BDAT'] = pd.to_datetime(df['BDAT'])
    df['животное_ключ'] = df['REG'].fillna('').astype(str)
    mask_no_reg = (df['животное_ключ'] == '') | (df['животное_ключ'] == 'nan')
    df.loc[mask_no_reg, 'животное_ключ'] = df.loc[mask_no_reg, 'ID'].astype(str) + '_' + df.loc[mask_no_reg, 'BDAT'].astype(str)
    df = df.sort_values(['животное_ключ', 'Дата'])
    c_mask = df['R'].str.strip() == 'C'
    df['Дата_исправленная'] = df['Дата']
    for idx in df[c_mask].index:
        animal_key = df.loc[idx, 'животное_ключ']
        current_date = df.loc[idx, 'Дата']
        prev_semen = df[(df['животное_ключ'] == animal_key) & (df['Дата'] < current_date) & (df.index != idx)].sort_values('Дата', ascending=False)
        if len(prev_semen) > 0:
            df.loc[idx, 'Дата_исправленная'] = prev_semen.iloc[0]['Дата']
    df['тип_осеменения'] = df['R'].str.strip()
    return df

print("\n" + "="*80)
print("ОБРАБОТКА ДАННЫХ")
print("="*80)

df_semen_2022_proc = process_c_semen(df_semen_2022)
df_semen_2023_proc = process_c_semen(df_semen_2023)
df_semen_2024_proc = process_c_semen(df_semen_2024)

# ============================================
# 3. ФУНКЦИЯ РАЗДЕЛЕНИЯ ОТЕЛОВ НА КОРОВ И НЕТЕЛЕЙ
# ============================================
def split_calvings_by_lactation(df_calvings):
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])

    otel_mask = df['Событие'].str.upper().str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])
    df = df[otel_mask].copy()

    df['LACT'] = df['LACT'].fillna(0)

    cows = df[df['LACT'] >= 2].copy()
    heifers = df[df['LACT'] == 1].copy()

    return cows, heifers

# ============================================
# 4. АГРЕГАЦИЯ ОТЕЛОВ ПО МЕСЯЦАМ (С РАЗДЕЛЕНИЕМ)
# ============================================
def aggregate_calvings_split_monthly(df_calvings, max_date=None):
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'отелы_коровы', 'отелы_нетели', 'отелы_всего'])

    cows, heifers = split_calvings_by_lactation(df_calvings)

    cows['месяц'] = cows['Дата'].dt.month
    cows['год'] = cows['Дата'].dt.year
    cows['дата_месяц'] = pd.to_datetime(cows['год'].astype(str) + '-' + cows['месяц'].astype(str) + '-01')
    if max_date:
        cows = cows[cows['Дата'] <= max_date]
    cows_agg = cows.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='отелы_коровы')

    heifers['месяц'] = heifers['Дата'].dt.month
    heifers['год'] = heifers['Дата'].dt.year
    heifers['дата_месяц'] = pd.to_datetime(heifers['год'].astype(str) + '-' + heifers['месяц'].astype(str) + '-01')
    if max_date:
        heifers = heifers[heifers['Дата'] <= max_date]
    heifers_agg = heifers.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='отелы_нетели')

    result = cows_agg.merge(heifers_agg, on=['год', 'месяц', 'дата_месяц'], how='outer').fillna(0)
    result['отелы_всего'] = result['отелы_коровы'] + result['отелы_нетели']

    return result

# ============================================
# 5. АГРЕГАЦИЯ ДРУГИХ ДАННЫХ
# ============================================
def aggregate_dry_monthly(df_dry, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'запуски'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='запуски')
    return monthly

def aggregate_culling_monthly(df_culling, max_date=None):
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'выбытия'])
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='выбытия')
    return monthly

def aggregate_semen_monthly(df_semen, max_date=None):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'всего_осеменений', 'успешные', 'процент_успешных'])
    df = df_semen.copy()
    if max_date:
        df = df[df['Дата_исправленная'] <= max_date]
    df['месяц'] = df['Дата_исправленная'].dt.month
    df['год'] = df['Дата_исправленная'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    total_semen = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='всего_осеменений')
    success = df[df['тип_осеменения'] == 'P'].groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='успешные')
    features = total_semen.merge(success, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
    features['процент_успешных'] = (features['успешные'] / features['всего_осеменений'] * 100).fillna(0)
    return features

def aggregate_dry_shifted(df_dry, shift_days, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['дата_отела'] = df['Дата'] + pd.Timedelta(days=shift_days)
    if max_date:
        df = df[df['дата_отела'] <= max_date]
    df['год_отела'] = df['дата_отела'].dt.year
    df['месяц_отела'] = df['дата_отела'].dt.month
    df['дата_месяц'] = pd.to_datetime(df['год_отела'].astype(str) + '-' + df['месяц_отела'].astype(str) + '-01')
    dry_shifted = df.groupby(['год_отела', 'месяц_отела', 'дата_месяц']).size().reset_index(name=f'запуски_{shift_days}d_shift')
    dry_shifted.columns = ['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift']
    return dry_shifted

# ============================================
# 6. РАСЧЕТ ВЕРОЯТНОСТИ УСПЕШНОГО ОСЕМЕНЕНИЯ
# ============================================
print("\n" + "="*80)
print("РАСЧЕТ ВЕРОЯТНОСТИ УСПЕШНОГО ОСЕМЕНЕНИЯ")
print("="*80)

MAX_DATE_PROB = pd.Timestamp('2024-09-30')

def filter_by_date(df_semen, max_date):
    df = df_semen.copy()
    df['Дата_исправленная'] = pd.to_datetime(df['Дата_исправленная'])
    return df[df['Дата_исправленная'] <= max_date]

all_semen_historical = pd.concat([
    filter_by_date(df_semen_2022_proc, MAX_DATE_PROB),
    filter_by_date(df_semen_2023_proc, MAX_DATE_PROB),
    filter_by_date(df_semen_2024_proc, MAX_DATE_PROB)
], ignore_index=True)

def calculate_success_probability(df_semen):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['месяц', 'вероятность_успеха'])
    df = df_semen.copy()
    df['месяц'] = df['Дата_исправленная'].dt.month
    monthly_stats = df.groupby('месяц').agg(
        всего=('тип_осеменения', 'count'),
        успешные=('тип_осеменения', lambda x: (x == 'P').sum())
    ).reset_index()
    monthly_stats['вероятность_успеха'] = monthly_stats['успешные'] / monthly_stats['всего']
    return monthly_stats

success_probability = calculate_success_probability(all_semen_historical)
prob_by_month = dict(zip(success_probability['месяц'], success_probability['вероятность_успеха']))
overall_prob = all_semen_historical['тип_осеменения'].eq('P').sum() / len(all_semen_historical)

print("\nВероятность успешного осеменения по месяцам:")
for month in range(1, 13):
    p = prob_by_month.get(month, overall_prob)
    print(f"  {months_ru[month]}: {p:.1%}")

# ============================================
# 7. ОБУЧАЮЩИЕ ДАННЫЕ
# ============================================
MAX_DATE = pd.Timestamp('2024-09-30')

# Данные для модели коров (с разделением)
train_calvings_split = pd.concat([
    aggregate_calvings_split_monthly(df_calvings_2022, MAX_DATE),
    aggregate_calvings_split_monthly(df_calvings_2023, MAX_DATE),
    aggregate_calvings_split_monthly(df_calvings_2024, MAX_DATE)
], ignore_index=True)

train_dry = pd.concat([
    aggregate_dry_monthly(df_dry_2022, MAX_DATE),
    aggregate_dry_monthly(df_dry_2023, MAX_DATE),
    aggregate_dry_monthly(df_dry_2024, MAX_DATE)
], ignore_index=True)

train_culling = pd.concat([
    aggregate_culling_monthly(df_culling_2022, MAX_DATE),
    aggregate_culling_monthly(df_culling_2023, MAX_DATE),
    aggregate_culling_monthly(df_culling_2024, MAX_DATE)
], ignore_index=True)

train_semen = pd.concat([
    aggregate_semen_monthly(df_semen_2022_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2023_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2024_proc, MAX_DATE)
], ignore_index=True)

dry_features = {}
for period in DRY_PERIODS:
    dry_features[period] = pd.concat([
        aggregate_dry_shifted(df_dry_2022, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2023, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2024, period, MAX_DATE)
    ], ignore_index=True)

# Обучающий датафрейм для модели коров
train_cows_df = train_calvings_split.rename(columns={'отелы_коровы': 'отелы_коровы', 'отелы_нетели': 'отелы_нетели', 'отелы_всего': 'отелы_всего'})
train_cows_df = train_cows_df.merge(train_dry, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_cows_df = train_cows_df.merge(train_culling, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_cows_df = train_cows_df.merge(train_semen, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

for period in DRY_PERIODS:
    train_cows_df = train_cows_df.merge(dry_features[period], on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

# Добавляем долю нетелей для модели коров
train_cows_df['доля_нетелей'] = train_cows_df['отелы_нетели'] / (train_cows_df['отелы_всего'] + 1)
train_cows_df['коэффициент_первородства'] = train_cows_df['отелы_нетели'] / (train_cows_df['отелы_коровы'] + 1)

print(f"\nОбучающих месяцев (2022,2023,янв-сен2024): {len(train_cows_df)}")

# ============================================
# 8. ПРОГНОЗ ИЗ МОДЕЛИ 1 (ГОТОВЫЕ ПРОГНОЗЫ)
# ============================================
print("\n" + "="*80)
print("ПРОГНОЗ МОДЕЛИ 1 (ОБЩИЕ ОТЕЛЫ)")
print("="*80)

# Результаты прогноза из МОДЕЛИ 1 (из вывода)
forecast_model1 = {
    (2024, 10): 256,
    (2024, 11): 245,
    (2024, 12): 281,
    (2025, 1): 245,
    (2025, 2): 249,
    (2025, 3): 268,
    (2025, 4): 273,
    (2025, 5): 271,
    (2025, 6): 272,
    (2025, 7): 274,
    (2025, 8): 272,
    (2025, 9): 280,
    (2025, 10): 260,
    (2025, 11): 246,
    (2025, 12): 276
}

# Для обучающих данных (2022-сен 2024) берем фактические значения
# Добавляем признак forecast_total в обучающий датафрейм
train_cows_df['прогноз_всего_из_модели1'] = train_cows_df.apply(
    lambda row: forecast_model1.get((row['год'], row['месяц']), row['отелы_всего']), axis=1
)

# ============================================
# 9. СОЗДАНИЕ ПРИЗНАКОВ ДЛЯ МОДЕЛИ КОРОВ
# ============================================
def create_features_cows(df):
    df = df.copy()

    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12).astype('float32')
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12).astype('float32')
    df['квартал'] = (df['месяц'].apply(lambda x: (x-1)//3 + 1)).astype('int8')

    # Лаги коров
    for lag in [1, 2, 3, 6]:
        df[f'отелы_коровы_lag{lag}'] = df['отелы_коровы'].shift(lag).fillna(0).astype('float32')

    df['отелы_коровы_ma3'] = df['отелы_коровы'].rolling(3, min_periods=1).mean().fillna(0).astype('float32')
    df['отелы_коровы_ma6'] = df['отелы_коровы'].rolling(6, min_periods=1).mean().fillna(0).astype('float32')

    # Лаги нетелей
    df['отелы_нетели_lag1'] = df['отелы_нетели'].shift(1).fillna(0).astype('float32')
    df['отелы_нетели_lag3'] = df['отелы_нетели'].shift(3).fillna(0).astype('float32')

    # Доля нетелей и коэффициент первородства
    df['доля_нетелей'] = df['отелы_нетели'] / (df['отелы_всего'] + 1)
    df['коэффициент_первородства'] = df['отелы_нетели'] / (df['отелы_коровы'] + 1)
    df['доля_нетелей_lag1'] = df['доля_нетелей'].shift(1).fillna(0).astype('float32')
    df['коэффициент_первородства_lag1'] = df['коэффициент_первородства'].shift(1).fillna(0).astype('float32')

    # Лаги успешных
    df['успешные_lag3'] = df['успешные'].shift(3).fillna(0).astype('float32')
    df['успешные_lag6'] = df['успешные'].shift(6).fillna(0).astype('float32')

    # Лаги запусков
    for lag in [1, 2, 3, 6]:
        df[f'запуски_lag{lag}'] = df['запуски'].shift(lag).fillna(0).astype('float32')

    # Лаги выбытий
    for lag in [1, 2, 3]:
        df[f'выбытия_lag{lag}'] = df['выбытия'].shift(lag).fillna(0).astype('float32')

    # Вероятность успеха
    df['вероятность_успеха_осеменения'] = df['месяц'].map(prob_by_month).fillna(overall_prob).astype('float32')
    df['ожидаемые_успешные'] = (df['всего_осеменений'] * df['вероятность_успеха_осеменения']).astype('float32')
    df['ожидаемые_успешные_lag3'] = df['ожидаемые_успешные'].shift(3).fillna(0).astype('float32')
    df['ожидаемые_успешные_lag6'] = df['ожидаемые_успешные'].shift(6).fillna(0).astype('float32')

    # Запуски со сдвигом
    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        if col in df.columns:
            df[f'{col}_lag2'] = df[col].shift(2).fillna(0).astype('float32')
            df[f'{col}_lag3'] = df[col].shift(3).fillna(0).astype('float32')
            df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean().fillna(0).astype('float32')

    # ВАЖНЫЙ ПРИЗНАК: прогноз из модели 1
    df['прогноз_всего_из_модели1'] = df['прогноз_всего_из_модели1'].astype('float32')

    # Тренд
    df['тренд'] = range(1, len(df) + 1)
    df = df.fillna(0)

    return df

train_features_cows = create_features_cows(train_cows_df)

# Список признаков для модели коров
feature_cols_cows = [col for col in train_features_cows.columns if col not in [
    'год', 'месяц', 'дата_месяц', 'отелы_коровы', 'отелы_нетели', 'отелы_всего',
    'запуски', 'выбытия', 'успешные', 'всего_осеменений', 'процент_успешных',
    'доля_нетелей', 'коэффициент_первородства'
]]

# Преобразуем в numpy массивы
X_train_cows = train_features_cows[feature_cols_cows].values.astype('float32')
y_train_cows = train_features_cows['отелы_коровы'].values.astype('float32')

print(f"\nОбучение модели 'отелы коров' на {X_train_cows.shape[0]} месяцах, признаков: {X_train_cows.shape[1]}")

# ============================================
# 10. ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ КОРОВ
# ============================================
print("\n" + "="*80)
print("ОПТИМИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ КОРОВ")
print("="*80)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

def safe_grid_search(X, y, model_name):
    try:
        if len(X) < 10:
            print(f"  {model_name}: слишком мало данных ({len(X)} месяцев), используем параметры по умолчанию")
            model = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=1)
            model.fit(X, y)
            return model, None

        tscv = TimeSeriesSplit(n_splits=min(3, len(X)-1))
        grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
                           param_grid, cv=tscv, scoring='neg_mean_absolute_error',
                           n_jobs=1, verbose=0)
        grid.fit(X, y)
        print(f"  {model_name}: лучшие параметры {grid.best_params_}, MAE: {abs(grid.best_score_):.2f}")
        model = XGBRegressor(**grid.best_params_, random_state=42, n_jobs=1)
        model.fit(X, y)
        return model, grid.best_params_
    except Exception as e:
        print(f"  {model_name}: ошибка оптимизации - {str(e)[:50]}, используем параметры по умолчанию")
        model = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=1)
        model.fit(X, y)
        return model, None

print("\nОбучение модели 'отелы коров':")
model_cows, _ = safe_grid_search(X_train_cows, y_train_cows, 'отелы коров')

# ============================================
# 11. РЕКУРСИВНЫЙ ПРОГНОЗ (С ПРОГНОЗОМ ИЗ МОДЕЛИ 1)
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ (с прогнозом из МОДЕЛИ 1)")
print("="*80)

forecasts_cows = {}
results_total = []
results_cows = []

# Список месяцев для прогноза
predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")

    # Берем прогноз общего количества из МОДЕЛИ 1
    pred_total = forecast_model1.get((year, month), 0)

    # Создаем строку для прогноза коров
    pred_row_cows_dict = {
        'год': year,
        'месяц': month,
        'дата_месяц': pd.Timestamp(f'{year}-{month:02d}-01'),
        'месяц_синус': np.sin(2 * np.pi * month / 12),
        'месяц_косинус': np.cos(2 * np.pi * month / 12),
        'квартал': (month-1)//3 + 1,
        'тренд': len(train_features_cows) + len(results_cows) + 1,
        'вероятность_успеха_осеменения': prob_by_month.get(month, overall_prob),
        'отелы_коровы': 0,
        'отелы_коровы_lag1': forecasts_cows.get((year, month-1), 0) if month > 1 else 0,
        'отелы_коровы_lag2': forecasts_cows.get((year, month-2), 0) if month > 2 else 0,
        'отелы_коровы_lag3': forecasts_cows.get((year, month-3), 0) if month > 3 else 0,
        'отелы_коровы_lag6': forecasts_cows.get((year, month-6), 0) if month > 6 else 0,
        'отелы_коровы_ma3': 0,
        'отелы_коровы_ma6': 0,
        'прогноз_всего_из_модели1': pred_total  # КЛЮЧЕВОЙ ПРИЗНАК!
    }

    # Лаги нетелей (нужны для признаков модели коров)
    prev_heifers_lag1 = 0
    prev_heifers_lag3 = 0
    if month > 1:
        prev_total_lag1 = forecast_model1.get((year, month-1), 0)
        prev_cows_lag1 = forecasts_cows.get((year, month-1), 0)
        prev_heifers_lag1 = max(0, prev_total_lag1 - prev_cows_lag1)
    if month > 3:
        prev_total_lag3 = forecast_model1.get((year, month-3), 0)
        prev_cows_lag3 = forecasts_cows.get((year, month-3), 0)
        prev_heifers_lag3 = max(0, prev_total_lag3 - prev_cows_lag3)

    pred_row_cows_dict['отелы_нетели_lag1'] = prev_heifers_lag1
    pred_row_cows_dict['отелы_нетели_lag3'] = prev_heifers_lag3

    # Доля нетелей и коэффициент первородства
    pred_row_cows_dict['доля_нетелей'] = prev_heifers_lag1 / (forecast_model1.get((year, month-1), 1) + 1) if month > 1 else 0
    pred_row_cows_dict['коэффициент_первородства'] = prev_heifers_lag1 / (pred_row_cows_dict['отелы_коровы_lag1'] + 1)
    pred_row_cows_dict['доля_нетелей_lag1'] = pred_row_cows_dict['доля_нетелей']
    pred_row_cows_dict['коэффициент_первородства_lag1'] = pred_row_cows_dict['коэффициент_первородства']

    # Скользящие средние
    lag1 = pred_row_cows_dict['отелы_коровы_lag1']
    lag2 = pred_row_cows_dict['отелы_коровы_lag2']
    lag3 = pred_row_cows_dict['отелы_коровы_lag3']
    pred_row_cows_dict['отелы_коровы_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0
    pred_row_cows_dict['отелы_коровы_ma6'] = pred_row_cows_dict['отелы_коровы_ma3']

    # Запуски (средние исторические)
    pred_row_cows_dict['запуски'] = train_cows_df[train_cows_df['месяц'] == month]['запуски'].mean()
    pred_row_cows_dict['запуски_lag1'] = train_cows_df[train_cows_df['месяц'] == (month-1 if month>1 else 12)]['запуски'].mean() if month > 1 else 0
    pred_row_cows_dict['запуски_lag2'] = train_cows_df[train_cows_df['месяц'] == (month-2 if month>2 else 11)]['запуски'].mean() if month > 2 else 0
    pred_row_cows_dict['запуски_lag3'] = train_cows_df[train_cows_df['месяц'] == (month-3 if month>3 else 10)]['запуски'].mean() if month > 3 else 0
    pred_row_cows_dict['запуски_lag6'] = train_cows_df[train_cows_df['месяц'] == (month-6 if month>6 else 6)]['запуски'].mean() if month > 6 else 0

    # Выбытия
    pred_row_cows_dict['выбытия'] = 0
    pred_row_cows_dict['выбытия_lag1'] = forecast_model1.get((year, month-1), 0) if month > 1 else 0
    pred_row_cows_dict['выбытия_lag2'] = forecast_model1.get((year, month-2), 0) if month > 2 else 0
    pred_row_cows_dict['выбытия_lag3'] = forecast_model1.get((year, month-3), 0) if month > 3 else 0

    # Осеменения
    total_semen = train_cows_df[train_cows_df['месяц'] == month]['всего_осеменений'].mean()
    pred_row_cows_dict['всего_осеменений'] = total_semen
    pred_row_cows_dict['успешные'] = total_semen * prob_by_month.get(month, overall_prob)
    pred_row_cows_dict['процент_успешных'] = (pred_row_cows_dict['успешные'] / (total_semen + 1)) * 100
    pred_row_cows_dict['ожидаемые_успешные'] = pred_row_cows_dict['всего_осеменений'] * pred_row_cows_dict['вероятность_успеха_осеменения']
    pred_row_cows_dict['ожидаемые_успешные_lag3'] = forecast_model1.get((year, month-3), 0) if month > 3 else 0
    pred_row_cows_dict['ожидаемые_успешные_lag6'] = forecast_model1.get((year, month-6), 0) if month > 6 else 0
    pred_row_cows_dict['успешные_lag3'] = forecast_model1.get((year, month-3), 0) if month > 3 else 0
    pred_row_cows_dict['успешные_lag6'] = forecast_model1.get((year, month-6), 0) if month > 6 else 0

    # Запуски со сдвигом
    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        hist_val = train_cows_df[train_cows_df['месяц'] == month][col].mean() if col in train_cows_df.columns else 0
        pred_row_cows_dict[col] = hist_val
        pred_row_cows_dict[f'{col}_lag2'] = train_cows_df[train_cows_df['месяц'] == (month-2 if month>2 else 10)][col].mean() if col in train_cows_df.columns else 0
        pred_row_cows_dict[f'{col}_lag3'] = train_cows_df[train_cows_df['месяц'] == (month-3 if month>3 else 9)][col].mean() if col in train_cows_df.columns else 0
        pred_row_cows_dict[f'{col}_ma3'] = pred_row_cows_dict[col]

    pred_row_cows_dict['вероятность_успеха_ma3'] = pred_row_cows_dict['вероятность_успеха_осеменения']

    # Прогнозируем коров
    pred_row_cows = pd.DataFrame([pred_row_cows_dict])
    X_pred_cows = pred_row_cows[[col for col in feature_cols_cows if col in pred_row_cows.columns]]
    X_pred_cows = X_pred_cows.fillna(0).values.astype('float32')

    pred_cows = max(0, int(round(model_cows.predict(X_pred_cows)[0])))
    pred_cows = min(pred_cows, pred_total)  # коровы не могут быть больше общего количества

    forecasts_cows[(year, month)] = pred_cows

    # Нетели = всего - коровы
    pred_heifers = pred_total - pred_cows

    # Факты для сравнения
    actual_split_df = aggregate_calvings_split_monthly(df_calvings_2024 if year == 2024 else df_calvings_2025)
    actual_split_row = actual_split_df[actual_split_df['месяц'] == month]
    actual_cows = actual_split_row['отелы_коровы'].values[0] if len(actual_split_row) > 0 else 0
    actual_heifers = actual_split_row['отелы_нетели'].values[0] if len(actual_split_row) > 0 else 0

    results_total.append({'год': year, 'месяц': month, 'прогноз': pred_total, 'факт': actual_split_row['отелы_всего'].values[0] if len(actual_split_row) > 0 else 0})
    results_cows.append({'год': year, 'месяц': month, 'прогноз': pred_cows, 'факт': actual_cows})

    print(f"  Всего (из Модели 1): прогноз {pred_total}, факт {actual_split_row['отелы_всего'].values[0] if len(actual_split_row) > 0 else 0}")
    print(f"  Коровы: прогноз {pred_cows}, факт {actual_cows}")
    print(f"  Нетели: прогноз {pred_heifers}, факт {actual_heifers}")

# ============================================
# 12. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ПРОГНОЗ НА ОКТЯБРЬ-ДЕКАБРЬ 2024:")
print("-" * 110)
print(f"{'Месяц':<12} {'Всего':>8} {'Всего':>8} {'Коровы':>8} {'Коровы':>8} {'Ошибка':>8} {'Ошибка %':>10} "
      f"{'Нетели':>8} {'Нетели':>8} {'Ошибка':>8} {'Ошибка %':>10} {'Статус':<8}")
print(f"{'':<12} {'прогноз':>8} {'факт':>8} {'прогноз':>8} {'факт':>8} {'абс':>8} {'отн':>10} "
      f"{'прогноз':>8} {'факт':>8} {'абс':>8} {'отн':>10} {'':<8}")
print("-" * 110)

for i, r in enumerate(results_total):
    if r['год'] == 2024:
        r_cows = results_cows[i]
        actual_split_df = aggregate_calvings_split_monthly(df_calvings_2024)
        actual_split_row = actual_split_df[actual_split_df['месяц'] == r['месяц']]
        actual_heifers = actual_split_row['отелы_нетели'].values[0] if len(actual_split_row) > 0 else 0
        pred_heifers = r['прогноз'] - r_cows['прогноз']

        err_cows = r_cows['прогноз'] - r_cows['факт']
        err_cows_pct = (err_cows / r_cows['факт']) * 100 if r_cows['факт'] > 0 else 0
        err_heifers = pred_heifers - actual_heifers
        err_heifers_pct = (err_heifers / actual_heifers) * 100 if actual_heifers > 0 else 0

        status = "✅" if abs(err_cows_pct) <= 10 else "⚠️" if abs(err_cows_pct) <= 20 else "❌"

        sign_cows = "+" if err_cows > 0 else ""
        sign_heifers = "+" if err_heifers > 0 else ""

        print(f"{months_ru[r['месяц']]:<12} "
              f"{r['прогноз']:>8} {r['факт']:>8.0f} "
              f"{r_cows['прогноз']:>8} {r_cows['факт']:>8.0f} "
              f"{sign_cows}{err_cows:>7.0f} {sign_cows}{err_cows_pct:>9.1f}% "
              f"{pred_heifers:>8} {actual_heifers:>8.0f} "
              f"{sign_heifers}{err_heifers:>7.0f} {sign_heifers}{err_heifers_pct:>9.1f}% {status}")

print("\n" + "="*80)
print("ПРОГНОЗ НА 2025 ГОД:")
print("-" * 110)
print(f"{'Месяц':<12} {'Всего':>8} {'Всего':>8} {'Коровы':>8} {'Коровы':>8} {'Ошибка':>8} {'Ошибка %':>10} "
      f"{'Нетели':>8} {'Нетели':>8} {'Ошибка':>8} {'Ошибка %':>10} {'Статус':<8}")
print(f"{'':<12} {'прогноз':>8} {'факт':>8} {'прогноз':>8} {'факт':>8} {'абс':>8} {'отн':>10} "
      f"{'прогноз':>8} {'факт':>8} {'абс':>8} {'отн':>10} {'':<8}")
print("-" * 110)

for i, r in enumerate(results_total):
    if r['год'] == 2025:
        r_cows = results_cows[i]
        actual_split_df = aggregate_calvings_split_monthly(df_calvings_2025)
        actual_split_row = actual_split_df[actual_split_df['месяц'] == r['месяц']]
        actual_heifers = actual_split_row['отелы_нетели'].values[0] if len(actual_split_row) > 0 else 0
        pred_heifers = r['прогноз'] - r_cows['прогноз']

        err_cows = r_cows['прогноз'] - r_cows['факт']
        err_cows_pct = (err_cows / r_cows['факт']) * 100 if r_cows['факт'] > 0 else 0
        err_heifers = pred_heifers - actual_heifers
        err_heifers_pct = (err_heifers / actual_heifers) * 100 if actual_heifers > 0 else 0

        status = "✅" if abs(err_cows_pct) <= 10 else "⚠️" if abs(err_cows_pct) <= 20 else "❌"

        sign_cows = "+" if err_cows > 0 else ""
        sign_heifers = "+" if err_heifers > 0 else ""

        print(f"{months_ru[r['месяц']]:<12} "
              f"{r['прогноз']:>8} {r['факт']:>8.0f} "
              f"{r_cows['прогноз']:>8} {r_cows['факт']:>8.0f} "
              f"{sign_cows}{err_cows:>7.0f} {sign_cows}{err_cows_pct:>9.1f}% "
              f"{pred_heifers:>8} {actual_heifers:>8.0f} "
              f"{sign_heifers}{err_heifers:>7.0f} {sign_heifers}{err_heifers_pct:>9.1f}% {status}")

print("-" * 110)

# ============================================
# 13. ИТОГОВАЯ СТАТИСТИКА
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА ЗА ТЕСТОВЫЙ ПЕРИОД:")
print("="*80)

total_pred = sum(r['прогноз'] for r in results_total)
total_actual = sum(r['факт'] for r in results_total)

cows_pred = sum(r['прогноз'] for r in results_cows)
cows_actual = sum(r['факт'] for r in results_cows)
cows_error = cows_pred - cows_actual
cows_error_pct = (cows_error / cows_actual) * 100 if cows_actual > 0 else 0

heifers_pred = total_pred - cows_pred
heifers_actual = total_actual - cows_actual
heifers_error = heifers_pred - heifers_actual
heifers_error_pct = (heifers_error / heifers_actual) * 100 if heifers_actual > 0 else 0

print(f"\n{'Показатель':<20} {'Прогноз':>12} {'Факт':>12} {'Ошибка':>12} {'Ошибка %':>12}")
print("-" * 68)
print(f"{'Всего отелов (Модель 1)':<20} {total_pred:>12} {total_actual:>12.0f} {'':>12} {'':>12}")
print(f"{'Коровы (LACT>=2)':<20} {cows_pred:>12} {cows_actual:>12.0f} {cows_error:>+12.0f} {cows_error_pct:>+11.1f}%")
print(f"{'Нетели (LACT=1)':<20} {heifers_pred:>12} {heifers_actual:>12.0f} {heifers_error:>+12.0f} {heifers_error_pct:>+11.1f}%")

errors_cows = [abs(r['прогноз'] - r['факт']) for r in results_cows]
mae_cows = np.mean(errors_cows)
mape_cows = np.mean([abs((r['прогноз'] - r['факт']) / r['факт']) * 100 for r in results_cows if r['факт'] > 0])

print(f"\n📊 МЕТРИКИ КАЧЕСТВА (Коровы):")
print(f"  MAE: {mae_cows:.1f}")
print(f"  MAPE: {mape_cows:.1f}%")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • ОБЩИЕ ОТЕЛЫ - ПРОГНОЗ ИЗ МОДЕЛИ 1 (MAPE 7.8%)")
print("  • КОРОВЫ - отдельная модель с использованием прогноза Модели 1")
print("  • НЕТЕЛИ = ВСЕГО - КОРОВЫ")
print("="*80)

# ===== NOTEBOOK CELL 5 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

DRY_PERIODS = [45, 60]

print("="*80)
print("ПРОГНОЗ ТЕЛОЧЕК ПО ДАННЫМ О РОЖДЕНИЯХ (СОБЫТИЕ = РОЖДЕН, GNDR = F/M)")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_semen_2022 = pd.read_excel(f"{folder}/Осеменения_2022.xlsx")
df_semen_2023 = pd.read_excel(f"{folder}/Осеменения_2023.xlsx")
df_semen_2024 = pd.read_excel(f"{folder}/Осеменения_2024.xlsx")

df_dry_2022 = pd.read_excel(f"{folder}/Запуск_2022.xlsx")
df_dry_2023 = pd.read_excel(f"{folder}/Запуск_2023.xlsx")
df_dry_2024 = pd.read_excel(f"{folder}/Запуск_2024.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")

# Загружаем таблицу быков
df_bulls = pd.read_excel('быки_полная_база.xlsx')
df_bulls['Плем'] = df_bulls['Плем'].astype(str).str.strip().str.upper()
df_bulls['тип_семени'] = df_bulls['Плем'].apply(lambda x: 'секс' if x == 'S' else 'обычное')
bull_type_dict = dict(zip(df_bulls['Бык'], df_bulls['тип_семени']))

print(f"\n✅ Данные загружены")
print(f"   Быков всего: {len(df_bulls)}")
print(f"   Сексированных (S): {len(df_bulls[df_bulls['тип_семени']=='секс'])}")
print(f"   Обычных (H): {len(df_bulls[df_bulls['тип_семени']=='обычное'])}")

# ============================================
# 2. ОБРАБОТКА ОСЕМЕНЕНИЙ
# ============================================
def process_c_semen(df_semen):
    if len(df_semen) == 0:
        return df_semen
    df = df_semen.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['BDAT'] = pd.to_datetime(df['BDAT'])
    df['животное_ключ'] = df['REG'].fillna('').astype(str)
    mask_no_reg = (df['животное_ключ'] == '') | (df['животное_ключ'] == 'nan')
    df.loc[mask_no_reg, 'животное_ключ'] = df.loc[mask_no_reg, 'ID'].astype(str) + '_' + df.loc[mask_no_reg, 'BDAT'].astype(str)
    df = df.sort_values(['животное_ключ', 'Дата'])
    c_mask = df['R'].str.strip() == 'C'
    df['Дата_исправленная'] = df['Дата']
    for idx in df[c_mask].index:
        animal_key = df.loc[idx, 'животное_ключ']
        current_date = df.loc[idx, 'Дата']
        prev_semen = df[(df['животное_ключ'] == animal_key) & (df['Дата'] < current_date) & (df.index != idx)].sort_values('Дата', ascending=False)
        if len(prev_semen) > 0:
            df.loc[idx, 'Дата_исправленная'] = prev_semen.iloc[0]['Дата']
    df['тип_осеменения'] = df['R'].str.strip()
    return df

df_semen_2022_proc = process_c_semen(df_semen_2022)
df_semen_2023_proc = process_c_semen(df_semen_2023)
df_semen_2024_proc = process_c_semen(df_semen_2024)

# ============================================
# 3. РАСЧЕТ ПРОЦЕНТА СЕКСИРОВАННЫХ ОСЕМЕНЕНИЙ
# ============================================
MAX_DATE = pd.Timestamp('2024-09-30')

all_semen = pd.concat([df_semen_2022_proc, df_semen_2023_proc, df_semen_2024_proc], ignore_index=True)
all_semen = all_semen[all_semen['Дата_исправленная'] <= MAX_DATE]

all_semen['тип_семени_быка'] = all_semen['Примечание'].apply(
    lambda x: bull_type_dict.get(str(x).strip(), 'неизвестно') if pd.notna(x) else 'неизвестно'
)

all_semen['месяц'] = all_semen['Дата_исправленная'].dt.month
all_semen['год'] = all_semen['Дата_исправленная'].dt.year

monthly_sex_stats = all_semen.groupby(['год', 'месяц']).agg(
    всего_осеменений=('тип_семени_быка', 'count'),
    секс_осеменений=('тип_семени_быка', lambda x: (x == 'секс').sum())
).reset_index()
monthly_sex_stats['процент_секс'] = monthly_sex_stats['секс_осеменений'] / monthly_sex_stats['всего_осеменений'] * 100

monthly_sex_pct = monthly_sex_stats.groupby('месяц')['процент_секс'].mean().to_dict()
overall_sex_pct = monthly_sex_stats['секс_осеменений'].sum() / monthly_sex_stats['всего_осеменений'].sum() * 100

print("\n" + "="*80)
print("ПРОЦЕНТ СЕКСИРОВАННЫХ ОСЕМЕНЕНИЙ (2022-сен 2024)")
print("="*80)
for month in range(1, 13):
    pct = monthly_sex_pct.get(month, overall_sex_pct)
    print(f"  {months_ru[month]}: {pct:.1f}%")
print(f"\n  Средний: {overall_sex_pct:.1f}%")

# ============================================
# 4. АГРЕГАЦИЯ РОЖДЕНИЙ (СОБЫТИЕ = РОЖДЕН, GNDR = F/M)
# ============================================
def aggregate_births_monthly(df_calvings, max_date=None):
    """Агрегирует рождения по месяцам (событие = Рожден)"""
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'отелы', 'телочки', 'бычки'])
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]

    # Только события "Рожден"
    birth_mask = df['Событие'].str.upper().str.strip() == 'РОЖДЕН'
    df = df[birth_mask].copy()

    # Определяем пол по GNDR (F = телочка, M = бычок)
    if 'GNDR' in df.columns:
        df['телочка'] = df['GNDR'].str.upper().str.strip().apply(lambda x: 1 if x == 'F' else 0)
        df['бычок'] = df['GNDR'].str.upper().str.strip().apply(lambda x: 1 if x == 'M' else 0)
    else:
        print("  ВНИМАНИЕ: Нет колонки GNDR, использую 50/50")
        df['телочка'] = 0.5
        df['бычок'] = 0.5

    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')

    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).agg(
        отелы=('телочка', 'count'),
        телочки=('телочка', 'sum'),
        бычки=('бычок', 'sum')
    ).reset_index()
    return monthly

def aggregate_dry_monthly(df_dry, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'запуски'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='запуски')
    return monthly

def aggregate_culling_monthly(df_culling, max_date=None):
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'выбытия'])
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='выбытия')
    return monthly

def aggregate_semen_monthly(df_semen, max_date=None):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'всего_осеменений', 'успешные'])
    df = df_semen.copy()
    if max_date:
        df = df[df['Дата_исправленная'] <= max_date]
    df['месяц'] = df['Дата_исправленная'].dt.month
    df['год'] = df['Дата_исправленная'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    total_semen = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='всего_осеменений')
    success = df[df['тип_осеменения'] == 'P'].groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='успешные')
    features = total_semen.merge(success, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
    return features

def aggregate_dry_shifted(df_dry, shift_days, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['дата_отела'] = df['Дата'] + pd.Timedelta(days=shift_days)
    if max_date:
        df = df[df['дата_отела'] <= max_date]
    df['год_отела'] = df['дата_отела'].dt.year
    df['месяц_отела'] = df['дата_отела'].dt.month
    df['дата_месяц'] = pd.to_datetime(df['год_отела'].astype(str) + '-' + df['месяц_отела'].astype(str) + '-01')
    dry_shifted = df.groupby(['год_отела', 'месяц_отела', 'дата_месяц']).size().reset_index(name=f'запуски_{shift_days}d_shift')
    dry_shifted.columns = ['год', 'месяц', 'дата_месяц', f'запуски_{shift_days}d_shift']
    return dry_shifted

# Собираем обучающие данные (используем births вместо calvings)
monthly_births = pd.concat([
    aggregate_births_monthly(df_calvings_2022, MAX_DATE),
    aggregate_births_monthly(df_calvings_2023, MAX_DATE),
    aggregate_births_monthly(df_calvings_2024, MAX_DATE)
], ignore_index=True)

monthly_dry = pd.concat([
    aggregate_dry_monthly(df_dry_2022, MAX_DATE),
    aggregate_dry_monthly(df_dry_2023, MAX_DATE),
    aggregate_dry_monthly(df_dry_2024, MAX_DATE)
], ignore_index=True)

monthly_culling = pd.concat([
    aggregate_culling_monthly(df_culling_2022, MAX_DATE),
    aggregate_culling_monthly(df_culling_2023, MAX_DATE),
    aggregate_culling_monthly(df_culling_2024, MAX_DATE)
], ignore_index=True)

semen_features = pd.concat([
    aggregate_semen_monthly(df_semen_2022_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2023_proc, MAX_DATE),
    aggregate_semen_monthly(df_semen_2024_proc, MAX_DATE)
], ignore_index=True)

dry_features = {}
for period in DRY_PERIODS:
    dry_features[period] = pd.concat([
        aggregate_dry_shifted(df_dry_2022, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2023, period, MAX_DATE),
        aggregate_dry_shifted(df_dry_2024, period, MAX_DATE)
    ], ignore_index=True)

# Объединяем
train_df = monthly_births
train_df = train_df.merge(monthly_dry, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(monthly_culling, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(semen_features, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

for period in DRY_PERIODS:
    train_df = train_df.merge(dry_features[period], on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

train_df['процент_секс_осеменений'] = train_df['месяц'].map(monthly_sex_pct).fillna(overall_sex_pct)

print(f"\nОбучающих месяцев (рождения): {len(train_df)}")
print(train_df.head(10))

# Проверяем распределение телочек и бычков в обучающих данных
print(f"\n📊 Распределение в обучающих данных (2022-сен 2024):")
print(f"  Всего рождений: {train_df['отелы'].sum()}")
print(f"  Телочки: {train_df['телочки'].sum()} ({train_df['телочки'].sum()/train_df['отелы'].sum()*100:.1f}%)")
print(f"  Бычки: {train_df['бычки'].sum()} ({train_df['бычки'].sum()/train_df['отелы'].sum()*100:.1f}%)")

# ============================================
# 5. СОЗДАНИЕ ПРИЗНАКОВ И ОБУЧЕНИЕ
# ============================================
def create_features(df, target_col='телочки'):
    df = df.copy()

    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)

    for lag in [1, 2, 3]:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        df[f'отелы_lag{lag}'] = df['отелы'].shift(lag)
        df[f'запуски_lag{lag}'] = df['запуски'].shift(lag)
        df[f'выбытия_lag{lag}'] = df['выбытия'].shift(lag)

    df[f'{target_col}_ma3'] = df[target_col].rolling(3, min_periods=1).mean()
    df['отелы_ma3'] = df['отелы'].rolling(3, min_periods=1).mean()
    df['запуски_ma3'] = df['запуски'].rolling(3, min_periods=1).mean()
    df['выбытия_ma3'] = df['выбытия'].rolling(3, min_periods=1).mean()

    df['процент_секс_осеменений_lag1'] = df['процент_секс_осеменений'].shift(1)
    df['процент_секс_осеменений_lag3'] = df['процент_секс_осеменений'].shift(3)

    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        if col in df.columns:
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean()

    df['тренд'] = range(1, len(df) + 1)

    return df

target = 'телочки'
train_features = create_features(train_df, target)

feature_cols = [col for col in train_features.columns if col not in [
    'год', 'месяц', 'дата_месяц', 'отелы', 'телочки', 'бычки',
    'запуски', 'выбытия', 'успешные', 'всего_осеменений'
]]

train_clean = train_features.dropna()
X_train = train_clean[feature_cols]
y_train = train_clean[target]

print(f"\nОбучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")

# Обучение
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
grid_search = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid,
                          cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая MAE: {abs(grid_search.best_score_):.2f}")

final_model = XGBRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

# ============================================
# 6. ПРОГНОЗ НА ОКТ 2024 - ДЕК 2025
# ============================================
print("\n" + "="*80)
print("ПРОГНОЗ ТЕЛОЧЕК НА ОКТ 2024 - ДЕК 2025")
print("="*80)

# Фактические данные для сравнения
actual_heifers = {2024: {10: 183, 11: 150, 12: 177}, 2025: {1: 232, 2: 198, 3: 195, 4: 246, 5: 177, 6: 201, 7: 172, 8: 121, 9: 203, 10: 150, 11: 140, 12: 160}}
actual_bulls = {2024: {10: 135, 11: 103, 12: 133}, 2025: {1: 56, 2: 75, 3: 119, 4: 85, 5: 72, 6: 77, 7: 106, 8: 98, 9: 85, 10: 95, 11: 90, 12: 85}}

# Прогноз отелов (из вашей модели)
calving_forecast = {
    (2024, 10): 256, (2024, 11): 245, (2024, 12): 281,
    (2025, 1): 245, (2025, 2): 249, (2025, 3): 268, (2025, 4): 273,
    (2025, 5): 271, (2025, 6): 272, (2025, 7): 274, (2025, 8): 272,
    (2025, 9): 280, (2025, 10): 260, (2025, 11): 246, (2025, 12): 276
}

forecasts_heifers = {}
results = []

for year, month in [(2024, m) for m in [10,11,12]] + [(2025, m) for m in range(1,13)]:
    total_forecast = calving_forecast.get((year, month), 0)

    pred_row = {
        'месяц_синус': np.sin(2 * np.pi * month / 12),
        'месяц_косинус': np.cos(2 * np.pi * month / 12),
        'квартал': (month-1)//3 + 1,
        'тренд': len(train_features) + len(results) + 1,
        'процент_секс_осеменений': monthly_sex_pct.get(month, overall_sex_pct),
        'отелы': total_forecast,
        'телочки': 0
    }

    # Лаги (упрощенно - берем средние исторические)
    for lag in [1, 2, 3]:
        if month > lag:
            val = train_df[train_df['месяц'] == (month - lag)]['телочки'].mean() if len(train_df[train_df['месяц'] == (month - lag)]) > 0 else 0
        else:
            val = train_df[train_df['месяц'] == (12 - (lag - month))]['телочки'].mean() if len(train_df[train_df['месяц'] == (12 - (lag - month))]) > 0 else 0
        pred_row[f'телочки_lag{lag}'] = val

        if month > lag:
            val = train_df[train_df['месяц'] == (month - lag)]['отелы'].mean() if len(train_df[train_df['месяц'] == (month - lag)]) > 0 else 0
        else:
            val = train_df[train_df['месяц'] == (12 - (lag - month))]['отелы'].mean() if len(train_df[train_df['месяц'] == (12 - (lag - month))]) > 0 else 0
        pred_row[f'отелы_lag{lag}'] = val

    for col in ['запуски', 'выбытия']:
        hist_val = train_df[train_df['месяц'] == month][col].mean() if len(train_df[train_df['месяц'] == month]) > 0 else 0
        pred_row[col] = hist_val
        for lag in [1, 2, 3]:
            if month > lag:
                lag_val = train_df[train_df['месяц'] == (month - lag)][col].mean() if len(train_df[train_df['месяц'] == (month - lag)]) > 0 else 0
            else:
                lag_val = train_df[train_df['месяц'] == (12 - (lag - month))][col].mean() if len(train_df[train_df['месяц'] == (12 - (lag - month))]) > 0 else 0
            pred_row[f'{col}_lag{lag}'] = lag_val
        pred_row[f'{col}_ma3'] = (pred_row[f'{col}_lag1'] + pred_row[f'{col}_lag2'] + pred_row[f'{col}_lag3']) / 3

    pred_row['телочки_ma3'] = (pred_row['телочки_lag1'] + pred_row['телочки_lag2'] + pred_row['телочки_lag3']) / 3
    pred_row['отелы_ma3'] = (pred_row['отелы_lag1'] + pred_row['отелы_lag2'] + pred_row['отелы_lag3']) / 3
    pred_row['процент_секс_осеменений_lag1'] = monthly_sex_pct.get(month-1 if month>1 else 12, overall_sex_pct)
    pred_row['процент_секс_осеменений_lag3'] = monthly_sex_pct.get(month-3 if month>3 else 10, overall_sex_pct)

    for period in DRY_PERIODS:
        col = f'запуски_{period}d_shift'
        hist_val = train_df[train_df['месяц'] == month][col].mean() if col in train_df.columns and len(train_df[train_df['месяц'] == month]) > 0 else 0
        pred_row[col] = hist_val
        pred_row[f'{col}_lag2'] = hist_val
        pred_row[f'{col}_lag3'] = hist_val
        pred_row[f'{col}_ma3'] = hist_val

    pred_df = pd.DataFrame([pred_row])
    missing_cols = set(feature_cols) - set(pred_df.columns)
    for col in missing_cols:
        pred_df[col] = 0

    X_pred = pred_df[feature_cols].fillna(0)
    pred_heifers = final_model.predict(X_pred)[0]
    pred_heifers = max(0, int(round(pred_heifers)))
    pred_bulls = max(0, total_forecast - pred_heifers)

    forecasts_heifers[(year, month)] = pred_heifers

    actual_h = actual_heifers.get(year, {}).get(month, 0)
    actual_b = actual_bulls.get(year, {}).get(month, 0)

    results.append({
        'год': year, 'месяц': month,
        'прогноз_телочки': pred_heifers, 'факт_телочки': actual_h,
        'прогноз_бычки': pred_bulls, 'факт_бычки': actual_b
    })

# Вывод
print(f"\n{'Месяц':<12} {'Телочки':>10} {'Телочки':>10} {'Бычки':>10} {'Бычки':>10} {'Ошибка':>10}")
print(f"{'':<12} {'прогноз':>10} {'факт':>10} {'прогноз':>10} {'факт':>10} {'телочки':>10}")
print("-" * 65)

for r in results:
    error = r['прогноз_телочки'] - r['факт_телочки']
    status = "✅" if abs(error) < 20 else "⚠️" if abs(error) < 40 else "❌"
    print(f"{months_ru[r['месяц']]} {r['год']:<4} {r['прогноз_телочки']:>10} {r['факт_телочки']:>10} "
          f"{r['прогноз_бычки']:>10} {r['факт_бычки']:>10} {error:+>10d} {status}")

# ===== NOTEBOOK CELL 7 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ ОТЕЛОВ ПО ЛАКТАЦИЯМ (КОМБИНИРОВАННЫЙ ПОДХОД)")
print("  • L0 = 0 (всегда)")
print("  • L1, L2 - ML модели")
print("  • L3+ = доля от общего количества")
print("  • обучение: 2022, 2023, январь-сентябрь 2024")
print("  • прогноз: октябрь-декабрь 2024 + весь 2025")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_semen_2022 = pd.read_excel(f"{folder}/Осеменения_2022.xlsx")
df_semen_2023 = pd.read_excel(f"{folder}/Осеменения_2023.xlsx")
df_semen_2024 = pd.read_excel(f"{folder}/Осеменения_2024.xlsx")

df_dry_2022 = pd.read_excel(f"{folder}/Запуск_2022.xlsx")
df_dry_2023 = pd.read_excel(f"{folder}/Запуск_2023.xlsx")
df_dry_2024 = pd.read_excel(f"{folder}/Запуск_2024.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")

print("\n✅ Данные загружены")

# ============================================
# 2. ОБРАБОТКА ОСЕМЕНЕНИЙ
# ============================================
def process_c_semen(df_semen):
    if len(df_semen) == 0:
        return df_semen
    df = df_semen.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['BDAT'] = pd.to_datetime(df['BDAT'])
    df['животное_ключ'] = df['REG'].fillna('').astype(str)
    mask_no_reg = (df['животное_ключ'] == '') | (df['животное_ключ'] == 'nan')
    df.loc[mask_no_reg, 'животное_ключ'] = df.loc[mask_no_reg, 'ID'].astype(str) + '_' + df.loc[mask_no_reg, 'BDAT'].astype(str)
    df = df.sort_values(['животное_ключ', 'Дата'])
    c_mask = df['R'].str.strip() == 'C'
    df['Дата_исправленная'] = df['Дата']
    for idx in df[c_mask].index:
        animal_key = df.loc[idx, 'животное_ключ']
        current_date = df.loc[idx, 'Дата']
        prev_semen = df[(df['животное_ключ'] == animal_key) & (df['Дата'] < current_date) & (df.index != idx)].sort_values('Дата', ascending=False)
        if len(prev_semen) > 0:
            df.loc[idx, 'Дата_исправленная'] = prev_semen.iloc[0]['Дата']
    df['тип_осеменения'] = df['R'].str.strip()
    return df

df_semen_2022_proc = process_c_semen(df_semen_2022)
df_semen_2023_proc = process_c_semen(df_semen_2023)
df_semen_2024_proc = process_c_semen(df_semen_2024)

# ============================================
# 3. ФУНКЦИИ ДЛЯ АГРЕГАЦИИ
# ============================================
def split_calvings_by_lactation_groups(df_calvings):
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    otel_mask = df['Событие'].str.upper().str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])
    df = df[otel_mask].copy()
    df['LACT'] = df['LACT'].fillna(0).astype(int)
    df['лактация_группа'] = df['LACT'].apply(lambda x:
        'L0' if x == 0 else
        'L1' if x == 1 else
        'L2' if x == 2 else
        'L3+' if x >= 3 else
        'L3+'  # все что >= 3 в одну группу
    )
    return df

def aggregate_calvings_by_lactation_monthly(df_calvings, max_date=None):
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'L0', 'L1', 'L2', 'L3+'])

    df = split_calvings_by_lactation_groups(df_calvings)

    if max_date:
        df = df[df['Дата'] <= max_date]

    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')

    grouped = df.groupby(['год', 'месяц', 'дата_месяц', 'лактация_группа']).size().unstack(fill_value=0).reset_index()

    for col in ['L0', 'L1', 'L2', 'L3+']:
        if col not in grouped.columns:
            grouped[col] = 0

    return grouped

def aggregate_dry_monthly(df_dry, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'запуски'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='запуски')
    return monthly

def aggregate_culling_monthly(df_culling, max_date=None):
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'выбытия'])
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='выбытия')
    return monthly

def aggregate_semen_monthly(df_semen, max_date=None):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'всего_осеменений', 'успешные'])
    df = df_semen.copy()
    if max_date:
        df = df[df['Дата_исправленная'] <= max_date]
    df['месяц'] = df['Дата_исправленная'].dt.month
    df['год'] = df['Дата_исправленная'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    total_semen = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='всего_осеменений')
    success = df[df['тип_осеменения'] == 'P'].groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='успешные')
    features = total_semen.merge(success, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
    return features

# ============================================
# 4. ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ
# ============================================
MAX_DATE_TRAIN = pd.Timestamp('2024-09-30')

train_calvings_lact = pd.concat([
    aggregate_calvings_by_lactation_monthly(df_calvings_2022, MAX_DATE_TRAIN),
    aggregate_calvings_by_lactation_monthly(df_calvings_2023, MAX_DATE_TRAIN),
    aggregate_calvings_by_lactation_monthly(df_calvings_2024, MAX_DATE_TRAIN)
], ignore_index=True)

train_dry = pd.concat([
    aggregate_dry_monthly(df_dry_2022, MAX_DATE_TRAIN),
    aggregate_dry_monthly(df_dry_2023, MAX_DATE_TRAIN),
    aggregate_dry_monthly(df_dry_2024, MAX_DATE_TRAIN)
], ignore_index=True)

train_culling = pd.concat([
    aggregate_culling_monthly(df_culling_2022, MAX_DATE_TRAIN),
    aggregate_culling_monthly(df_culling_2023, MAX_DATE_TRAIN),
    aggregate_culling_monthly(df_culling_2024, MAX_DATE_TRAIN)
], ignore_index=True)

train_semen = pd.concat([
    aggregate_semen_monthly(df_semen_2022_proc, MAX_DATE_TRAIN),
    aggregate_semen_monthly(df_semen_2023_proc, MAX_DATE_TRAIN),
    aggregate_semen_monthly(df_semen_2024_proc, MAX_DATE_TRAIN)
], ignore_index=True)

train_df = train_calvings_lact.merge(train_dry, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(train_culling, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(train_semen, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

# Общее количество отелов
lact_cols = ['L0', 'L1', 'L2', 'L3+']
train_df['всего_отелов'] = train_df[lact_cols].sum(axis=1)

# Исторические доли для L3+ (будем использовать для прогноза)
train_df['доля_L3+'] = train_df['L3+'] / (train_df['всего_отелов'] + 1)

print(f"\nОбучающих месяцев: {len(train_df)}")

# ============================================
# 5. ФУНКЦИЯ СОЗДАНИЯ ПРИЗНАКОВ
# ============================================
def create_features_calving(df):
    df = df.copy()

    # Сезонность
    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)
    df['тренд'] = range(1, len(df) + 1)

    # Лаги для L1 и L2
    for col in ['L1', 'L2']:
        for lag in [1, 2, 3, 6]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
        df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean().fillna(0)

    # Общее количество отелов
    df['всего_отелов_lag1'] = df['всего_отелов'].shift(1).fillna(0)
    df['всего_отелов_lag2'] = df['всего_отелов'].shift(2).fillna(0)
    df['всего_отелов_lag3'] = df['всего_отелов'].shift(3).fillna(0)
    df['всего_отелов_ma3'] = df['всего_отелов'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги запусков
    for lag in [1, 2, 3, 6]:
        df[f'запуски_lag{lag}'] = df['запуски'].shift(lag).fillna(0)
    df['запуски_ma3'] = df['запуски'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги выбытий
    for lag in [1, 2, 3, 6]:
        df[f'выбытия_lag{lag}'] = df['выбытия'].shift(lag).fillna(0)
    df['выбытия_ma3'] = df['выбытия'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги осеменений
    for lag in [3, 6]:
        df[f'успешные_lag{lag}'] = df['успешные'].shift(lag).fillna(0)
    df['успешные_ma3'] = df['успешные'].rolling(3, min_periods=1).mean().fillna(0)

    df['всего_осеменений_lag3'] = df['всего_осеменений'].shift(3).fillna(0)
    df['всего_осеменений_lag6'] = df['всего_осеменений'].shift(6).fillna(0)
    df['всего_осеменений_ma3'] = df['всего_осеменений'].rolling(3, min_periods=1).mean().fillna(0)

    return df

train_features = create_features_calving(train_df)
train_clean = train_features.dropna()

# Список признаков
feature_cols = [col for col in train_clean.columns if col not in [
    'год', 'месяц', 'дата_месяц', 'L0', 'L1', 'L2', 'L3+', 'всего_отелов', 'доля_L3+'
]]

X_train = train_clean[feature_cols]

print(f"\nОбучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")

# ============================================
# 6. ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ L1 И L2
# ============================================
print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ L1 и L2")
print("="*80)

models = {}
targets = ['L1', 'L2']

for target in targets:
    print(f"\nОбучение модели для {target}...")
    y_train = train_clean[target].values

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [0.5, 1.0]
    }

    try:
        tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
        grid = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            param_grid, cv=tscv, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        print(f"  Лучшие параметры: {best_params}")
    except:
        best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 
                       'subsample': 0.8, 'colsample_bytree': 0.8, 
                       'reg_alpha': 0.5, 'reg_lambda': 1.0}
        print(f"  Используем параметры по умолчанию")

    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

    pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_train, pred_train)
    print(f"  MAE на обучении: {mae:.2f}")

# ============================================
# 7. РАСЧЕТ ИСТОРИЧЕСКИХ ДОЛЕЙ ДЛЯ L3+
# ============================================
print("\n" + "="*80)
print("РАСЧЕТ ИСТОРИЧЕСКИХ ДОЛЕЙ ДЛЯ L3+")
print("="*80)

# Считаем среднюю долю L3+ за исторический период
avg_share_L3 = train_df['L3+'].sum() / train_df['всего_отелов'].sum() if train_df['всего_отелов'].sum() > 0 else 0.15
print(f"  Средняя доля L3+ за 2022-сентябрь 2024: {avg_share_L3:.1%}")

# Помесячные доли (для сезонности)
monthly_share_L3 = train_df.groupby('месяц').apply(
    lambda x: x['L3+'].sum() / (x['всего_отелов'].sum() + 1)
).to_dict()

print("\n  Помесячные доли L3+:")
for month in range(1, 13):
    print(f"    {months_ru[month]}: {monthly_share_L3.get(month, avg_share_L3):.1%}")

# ============================================
# 8. РЕКУРСИВНЫЙ ПРОГНОЗ
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ (октябрь 2024 - декабрь 2025)")
print("="*80)

forecasts = {target: {} for target in targets}
forecasts['L3+'] = {}
forecasts['L0'] = {}

monthly_avg = {}
for col in ['L1', 'L2', 'L3+']:
    monthly_avg[col] = train_df.groupby('месяц')[col].mean().to_dict()

monthly_avg_dry = train_df.groupby('месяц')['запуски'].mean().to_dict()
monthly_avg_culling = train_df.groupby('месяц')['выбытия'].mean().to_dict()
monthly_avg_semen = train_df.groupby('месяц')['успешные'].mean().to_dict()
monthly_avg_total_semen = train_df.groupby('месяц')['всего_осеменений'].mean().to_dict()

predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

def get_actual_calving_by_lact(year, month):
    if year == 2024:
        df = df_calvings_2024
    elif year == 2025:
        df = df_calvings_2025
    else:
        return {'L0': 0, 'L1': 0, 'L2': 0, 'L3+': 0}

    df = df.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df = df[(df['Дата'].dt.year == year) & (df['Дата'].dt.month == month)]

    if len(df) == 0:
        return {'L0': 0, 'L1': 0, 'L2': 0, 'L3+': 0}

    df = split_calvings_by_lactation_groups(df)
    result = df.groupby('лактация_группа').size().to_dict()

    for col in ['L0', 'L1', 'L2', 'L3+']:
        if col not in result:
            result[col] = 0

    return result

results = []

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")

    pred_row = {}
    pred_row['год'] = year
    pred_row['месяц'] = month
    pred_row['дата_месяц'] = pd.Timestamp(f'{year}-{month:02d}-01')
    pred_row['месяц_синус'] = np.sin(2 * np.pi * month / 12)
    pred_row['месяц_косинус'] = np.cos(2 * np.pi * month / 12)
    pred_row['квартал'] = (month-1)//3 + 1
    pred_row['тренд'] = len(train_features) + len(results) + 1

    # Базовые признаки
    pred_row['запуски'] = monthly_avg_dry.get(month, 0)
    pred_row['выбытия'] = monthly_avg_culling.get(month, 0)
    pred_row['успешные'] = monthly_avg_semen.get(month, 0)
    pred_row['всего_осеменений'] = monthly_avg_total_semen.get(month, 0)
    pred_row['всего_осеменений_lag3'] = monthly_avg_total_semen.get(month, 0)
    pred_row['всего_осеменений_lag6'] = monthly_avg_total_semen.get(month, 0)
    pred_row['всего_осеменений_ma3'] = monthly_avg_total_semen.get(month, 0)

    # Лаги для L1
    for target in ['L1', 'L2']:
        if month > 1:
            pred_row[f'{target}_lag1'] = forecasts[target].get((year, month-1),
                                      forecasts[target].get((year-1, 12), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag1'] = 0

        if month > 2:
            pred_row[f'{target}_lag2'] = forecasts[target].get((year, month-2),
                                      forecasts[target].get((year-1, 11), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag2'] = 0

        if month > 3:
            pred_row[f'{target}_lag3'] = forecasts[target].get((year, month-3),
                                      forecasts[target].get((year-1, 10), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag3'] = monthly_avg[target].get(month, 0)

        if month > 6:
            pred_row[f'{target}_lag6'] = forecasts[target].get((year, month-6),
                                      forecasts[target].get((year-1, 6), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag6'] = monthly_avg[target].get(month, 0)

        lag1 = pred_row[f'{target}_lag1']
        lag2 = pred_row[f'{target}_lag2']
        lag3 = pred_row[f'{target}_lag3']
        pred_row[f'{target}_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0

    # Лаги запусков
    for lag in [1, 2, 3, 6]:
        pred_row[f'запуски_lag{lag}'] = monthly_avg_dry.get(month, 0)
    pred_row['запуски_ma3'] = (pred_row['запуски_lag1'] + pred_row['запуски_lag2'] + pred_row['запуски_lag3']) / 3

    # Лаги выбытий
    for lag in [1, 2, 3, 6]:
        pred_row[f'выбытия_lag{lag}'] = monthly_avg_culling.get(month, 0)
    pred_row['выбытия_ma3'] = (pred_row['выбытия_lag1'] + pred_row['выбытия_lag2'] + pred_row['выбытия_lag3']) / 3

    # Лаги осеменений
    for lag in [3, 6]:
        pred_row[f'успешные_lag{lag}'] = monthly_avg_semen.get(month, 0)
    pred_row['успешные_ma3'] = monthly_avg_semen.get(month, 0)

    # Общее количество отелов (сумма прогнозов L1 и L2 + L3+)
    pred_row['всего_отелов_lag1'] = 0
    pred_row['всего_отелов_lag2'] = 0
    pred_row['всего_отелов_lag3'] = 0
    pred_row['всего_отелов_ma3'] = 0

    if month > 1:
        pred_row['всего_отелов_lag1'] = sum(forecasts[t].get((year, month-1), 0) for t in targets) + forecasts['L3+'].get((year, month-1), 0)
    if month > 2:
        pred_row['всего_отелов_lag2'] = sum(forecasts[t].get((year, month-2), 0) for t in targets) + forecasts['L3+'].get((year, month-2), 0)
    if month > 3:
        pred_row['всего_отелов_lag3'] = sum(forecasts[t].get((year, month-3), 0) for t in targets) + forecasts['L3+'].get((year, month-3), 0)

    pred_row['всего_отелов_ma3'] = (pred_row['всего_отелов_lag1'] + 
                                     pred_row['всего_отелов_lag2'] + 
                                     pred_row['всего_отелов_lag3']) / 3 if (pred_row['всего_отелов_lag1'] + 
                                                                            pred_row['всего_отелов_lag2'] + 
                                                                            pred_row['всего_отелов_lag3']) > 0 else 0

    # Прогнозируем L1 и L2
    pred_df = pd.DataFrame([pred_row])
    X_pred = pred_df[feature_cols]
    X_pred = X_pred.fillna(0)

    pred_L1 = max(0, int(round(models['L1'].predict(X_pred)[0])))
    pred_L2 = max(0, int(round(models['L2'].predict(X_pred)[0])))

    # L3+ = доля от общего количества
    # Сначала оцениваем общее количество (L1 + L2) / (1 - доля_L3)
    share_L3 = monthly_share_L3.get(month, avg_share_L3)
    total_est = (pred_L1 + pred_L2) / (1 - share_L3) if share_L3 < 1 else pred_L1 + pred_L2
    pred_L3 = max(0, int(round(total_est * share_L3)))

    # L0 всегда 0
    pred_L0 = 0

    # Сохраняем прогнозы
    forecasts['L0'][(year, month)] = pred_L0
    forecasts['L1'][(year, month)] = pred_L1
    forecasts['L2'][(year, month)] = pred_L2
    forecasts['L3+'][(year, month)] = pred_L3

    actual_values = get_actual_calving_by_lact(year, month)

    results.append({
        'год': year,
        'месяц': month,
        'прогноз': {'L0': pred_L0, 'L1': pred_L1, 'L2': pred_L2, 'L3+': pred_L3},
        'факт': actual_values
    })

    print(f"  Прогноз: L0={pred_L0}, L1={pred_L1}, L2={pred_L2}, L3+={pred_L3}")
    print(f"  Доля L3+: {share_L3:.1%}")

# ============================================
# 9. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗОВ ПО ЛАКТАЦИЯМ")
print("="*80)

header = f"{'Месяц':<12}"
for lact in ['L0', 'L1', 'L2', 'L3+']:
    header += f" {lact}_прогноз> {lact}_ошибка> {lact}_факт"
print(header)
print("-" * (12 + 4 * 6 * 15))

for r in results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    row = f"{month_name:<12}"

    for lact in ['L0', 'L1', 'L2', 'L3+']:
        pred = r['прогноз'][lact]
        fact = r['факт'][lact]
        error = pred - fact
        row += f" {pred:>8} {error:>+8} {fact:>8}"

    print(row)

print("-" * (12 + 4 * 6 * 15))

# ============================================
# 10. ИТОГИ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА ЗА ПЕРИОД ПРОГНОЗА")
print("="*80)

for lact in ['L0', 'L1', 'L2', 'L3+']:
    preds = [r['прогноз'][lact] for r in results]
    facts = [r['факт'][lact] for r in results]
    total_pred = sum(preds)
    total_fact = sum(facts)
    total_error = total_pred - total_fact
    total_error_pct = (total_error / total_fact * 100) if total_fact > 0 else 0
    mae = mean_absolute_error(facts, preds)

    print(f"\n{lact}:")
    print(f"  Прогноз: {total_pred}")
    print(f"  Факт: {total_fact}")
    print(f"  Ошибка: {total_error:+d} ({total_error_pct:+.1f}%)")
    print(f"  MAE: {mae:.2f}")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • L0 = 0 (всегда)")
print("  • L1, L2 - ML модели с регуляризацией")
print("  • L3+ = доля от общего количества (историческая)")
print("  • Обучение: 2022, 2023, январь-сентябрь 2024")
print("  • Прогноз: октябрь 2024 - декабрь 2025")
print("="*80)

# ===== NOTEBOOK CELL 10 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ СУХОСТОЙНЫХ КОРОВ (рекурсивный)")
print("  • обучение: 2022, 2023, январь-сентябрь 2024")
print("  • прогноз: октябрь 2024 - декабрь 2025")
print("  • для прогноза используются ТОЛЬКО предыдущие ПРОГНОЗЫ!")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_events = pd.read_excel("События-по-коровам (1).xlsx")
print(f"\n✅ Загружен файл событий: {len(df_events)} записей")

# ============================================
# 2. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================
df_events = df_events[df_events['Столбец1'] == 'ЖК Высокое'].copy()
df_events['Дата'] = pd.to_datetime(df_events['Дата'])
df_events = df_events[df_events['Дата'] <= pd.Timestamp('2025-12-31')].copy()

print(f"  После фильтрации: {len(df_events)} записей")

# Создаем столбцы для группировки
df_events['год'] = df_events['Дата'].dt.year
df_events['месяц'] = df_events['Дата'].dt.month
df_events['год_месяц'] = df_events['Дата'].dt.to_period('M')

# ============================================
# 3. ФУНКЦИИ ДЛЯ АГРЕГАЦИИ (ТОЛЬКО ДЛЯ ОБУЧЕНИЯ)
# ============================================
MAX_DATE_TRAIN = pd.Timestamp('2024-09-30')

# Фильтруем данные для обучения (янв 2022 - сен 2024)
df_events_train = df_events[(df_events['Дата'] >= pd.Timestamp('2022-01-01')) &
                            (df_events['Дата'] <= MAX_DATE_TRAIN)].copy()

print(f"  Обучающих событий (янв 2022 - сен 2024): {len(df_events_train)}")

# ============================================
# 4. РАСЧЕТ ЗАПУСКОВ ДЛЯ ОБУЧЕНИЯ
# ============================================
df_dry_train = df_events_train[df_events_train['тип_файла'] == 'ЗАПУСК'].copy()
dry_monthly_train = df_dry_train.groupby(['год', 'месяц']).size().reset_index(name='запуски')
print(f"  Запусков в обучении: {len(df_dry_train)}")

# ============================================
# 5. РАСЧЕТ ОТЕЛОВ КОРОВ (LACT >= 2) ДЛЯ ОБУЧЕНИЯ
# ============================================
df_calving_cows_train = df_events_train[(df_events_train['Событие'] == 'ОТЕЛ') &
                                        (df_events_train['LACT'] >= 2)].copy()
calving_cows_monthly_train = df_calving_cows_train.groupby(['год', 'месяц']).size().reset_index(name='отелы_коров')
print(f"  Отелов коров (LACT>=2) в обучении: {len(df_calving_cows_train)}")

# ============================================
# 6. РАСЧЕТ ВЫБЫТИЙ СУХОСТОЙНЫХ ДЛЯ ОБУЧЕНИЯ
# ============================================
def calculate_culling_suhostoynye_train(df_events):
    """Определяет выбытия сухостойных по последнему событию"""
    df_sorted = df_events.sort_values(['ID', 'Дата'])

    culling_suh_list = []

    for cow_id in df_sorted['ID'].unique():
        cow_events = df_sorted[df_sorted['ID'] == cow_id].copy()

        if len(cow_events) == 0:
            continue

        last_event = cow_events.iloc[-1]

        if last_event['тип_файла'] == 'ВЫБЫТИЕ':
            if len(cow_events) >= 2:
                prev_event = cow_events.iloc[-2]
                if prev_event['тип_файла'] == 'ЗАПУСК':
                    culling_suh_list.append({
                        'ID': cow_id,
                        'дата_выбытия': last_event['Дата'],
                        'год': last_event['год'],
                        'месяц': last_event['месяц']
                    })

    df_culling_suh = pd.DataFrame(culling_suh_list)
    if len(df_culling_suh) > 0:
        culling_suh_monthly = df_culling_suh.groupby(['год', 'месяц']).size().reset_index(name='выбытия_сухостойных')
    else:
        culling_suh_monthly = pd.DataFrame(columns=['год', 'месяц', 'выбытия_сухостойных'])

    print(f"  Выбытий сухостойных в обучении: {len(df_culling_suh)}")
    return culling_suh_monthly

culling_suh_monthly_train = calculate_culling_suhostoynye_train(df_events_train)

# ============================================
# 7. РАСЧЕТ СУХОСТОЙНЫХ ДЛЯ ОБУЧЕНИЯ (ФАКТ)
# ============================================
print("\n" + "="*80)
print("РАСЧЕТ ФАКТИЧЕСКИХ СУХОСТОЙНЫХ ДЛЯ ОБУЧЕНИЯ")
print("="*80)

# Начальное значение: февраль 2022 = 261
SUHOSTOYNYE_BASE_FEB_2022 = 261

# Создаем полный список месяцев для обучения (янв 2022 - сен 2024)
train_months = []
for year in range(2022, 2025):
    start_month = 1 if year > 2022 else 1
    end_month = 12 if year < 2024 else 9
    for month in range(start_month, end_month + 1):
        if year == 2024 and month > 9:
            break
        train_months.append((year, month))

df_train_months = pd.DataFrame(train_months, columns=['год', 'месяц'])

# Добавляем данные
df_train_months = df_train_months.merge(dry_monthly_train, on=['год', 'месяц'], how='left').fillna(0)
df_train_months = df_train_months.merge(calving_cows_monthly_train, on=['год', 'месяц'], how='left').fillna(0)

if len(culling_suh_monthly_train) > 0:
    df_train_months = df_train_months.merge(culling_suh_monthly_train, on=['год', 'месяц'], how='left').fillna(0)
else:
    df_train_months['выбытия_сухостойных'] = 0

# Сортируем
df_train_months = df_train_months.sort_values(['год', 'месяц']).reset_index(drop=True)

# Рассчитываем сухостойных рекурсивно
df_train_months['сухостойные'] = 0
suhostoynye_prev = SUHOSTOYNYE_BASE_FEB_2022

for idx, row in df_train_months.iterrows():
    suhostoynye_current = (suhostoynye_prev +
                           row['запуски'] -
                           row['отелы_коров'] -
                           row['выбытия_сухостойных'])

    df_train_months.at[idx, 'сухостойные'] = suhostoynye_current
    suhostoynye_prev = suhostoynye_current

print(f"  Всего месяцев в обучении: {len(df_train_months)}")
print(f"  Сухостойных на сентябрь 2024: {df_train_months.iloc[-1]['сухостойные']:.0f}")

# ============================================
# 8. ПОДГОТОВКА ДАННЫХ ДЛЯ ML МОДЕЛИ
# ============================================
print("\n" + "="*80)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ ML МОДЕЛИ")
print("="*80)

def create_features_for_suhostoynye(df):
    df = df.copy()

    # Сезонность
    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)
    df['тренд'] = range(1, len(df) + 1)

    # Лаги сухостойных
    for lag in [1, 2, 3, 6]:
        df[f'сухостойные_lag{lag}'] = df['сухостойные'].shift(lag).fillna(0)
    df['сухостойные_ma3'] = df['сухостойные'].rolling(3, min_periods=1).mean().fillna(0)
    df['сухостойные_ma6'] = df['сухостойные'].rolling(6, min_periods=1).mean().fillna(0)

    # Лаги запусков
    for lag in [1, 2, 3, 6]:
        df[f'запуски_lag{lag}'] = df['запуски'].shift(lag).fillna(0)
    df['запуски_ma3'] = df['запуски'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги отелов коров
    for lag in [1, 2, 3, 6]:
        df[f'отелы_коров_lag{lag}'] = df['отелы_коров'].shift(lag).fillna(0)
    df['отелы_коров_ma3'] = df['отелы_коров'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги выбытий сухостойных
    for lag in [1, 2, 3, 6]:
        df[f'выбытия_сухостойных_lag{lag}'] = df['выбытия_сухостойных'].shift(lag).fillna(0)
    df['выбытия_сухостойных_ma3'] = df['выбытия_сухостойных'].rolling(3, min_periods=1).mean().fillna(0)

    return df

# Создаем признаки для обучения
train_features = create_features_for_suhostoynye(df_train_months)

# Удаляем NaN
feature_cols = [col for col in train_features.columns if col not in [
    'год', 'месяц', 'сухостойные', 'запуски', 'отелы_коров', 'выбытия_сухостойных'
]]
train_clean = train_features.dropna()
X_train = train_clean[feature_cols]
y_train = train_clean['сухостойные']

print(f"  Обучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")

# ============================================
# 9. ОБУЧЕНИЕ МОДЕЛИ
# ============================================
print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛИ")
print("="*80)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

try:
    tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
    grid = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid, cv=tscv, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"  Лучшие параметры: {best_params}")
except:
    best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
    print(f"  Используем параметры по умолчанию")

model = XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
mae = mean_absolute_error(y_train, pred_train)
print(f"  MAE на обучении: {mae:.2f}")

# ============================================
# 10. РЕКУРСИВНЫЙ ПРОГНОЗ
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ (октябрь 2024 - декабрь 2025)")
print("="*80)

# Прогноз фуражных (из модели)
furazh_forecast = {
    (2024, 9): 2909,
    (2024, 10): 2936,
    (2024, 11): 2944,
    (2024, 12): 2918,
    (2025, 1): 2942,
    (2025, 2): 2965,
    (2025, 3): 2973,
    (2025, 4): 2991,
    (2025, 5): 2997,
    (2025, 6): 3001,
    (2025, 7): 3005,
    (2025, 8): 3005,
    (2025, 9): 3006,
    (2025, 10): 3025,
    (2025, 11): 3019,
    (2025, 12): 3009
}

# Исторические средние по месяцам (для лагов, где нет прогнозов)
monthly_avg_suh = df_train_months.groupby('месяц')['сухостойные'].mean().to_dict()
monthly_avg_dry = df_train_months.groupby('месяц')['запуски'].mean().to_dict()
monthly_avg_calving_cows = df_train_months.groupby('месяц')['отелы_коров'].mean().to_dict()
monthly_avg_culling_suh = df_train_months.groupby('месяц')['выбытия_сухостойных'].mean().to_dict()

# Прогноз сухостойных (рекурсивно)
forecasts_suhostoynye = {}
forecasts_doynye = {}

# Начальное значение: сухостойные на сентябрь 2024 (из расчета)
suhostoynye_prev = df_train_months.iloc[-1]['сухостойные']

predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

results = []

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")

    # Создаем строку для прогноза
    pred_row = {}
    pred_row['год'] = year
    pred_row['месяц'] = month
    pred_row['месяц_синус'] = np.sin(2 * np.pi * month / 12)
    pred_row['месяц_косинус'] = np.cos(2 * np.pi * month / 12)
    pred_row['квартал'] = (month-1)//3 + 1
    pred_row['тренд'] = len(train_features) + len(results) + 1

    # ====== ЛАГИ СУХОСТОЙНЫХ (только из прогнозов!) ======
    if month > 1:
        pred_row['сухостойные_lag1'] = forecasts_suhostoynye.get((year, month-1),
                                   forecasts_suhostoynye.get((year-1, 12), monthly_avg_suh.get(month, 0)))
    else:
        pred_row['сухостойные_lag1'] = 0

    if month > 2:
        pred_row['сухостойные_lag2'] = forecasts_suhostoynye.get((year, month-2),
                                   forecasts_suhostoynye.get((year-1, 11), monthly_avg_suh.get(month, 0)))
    else:
        pred_row['сухостойные_lag2'] = 0

    if month > 3:
        pred_row['сухостойные_lag3'] = forecasts_suhostoynye.get((year, month-3),
                                   forecasts_suhostoynye.get((year-1, 10), monthly_avg_suh.get(month, 0)))
    else:
        pred_row['сухостойные_lag3'] = monthly_avg_suh.get(month, 0)

    if month > 6:
        pred_row['сухостойные_lag6'] = forecasts_suhostoynye.get((year, month-6),
                                   forecasts_suhostoynye.get((year-1, 6), monthly_avg_suh.get(month, 0)))
    else:
        pred_row['сухостойные_lag6'] = monthly_avg_suh.get(month, 0)

    # Скользящие средние
    lag1 = pred_row['сухостойные_lag1']
    lag2 = pred_row['сухостойные_lag2']
    lag3 = pred_row['сухостойные_lag3']
    lag6 = pred_row['сухостойные_lag6']

    pred_row['сухостойные_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0
    pred_row['сухостойные_ma6'] = (lag1 + lag2 + lag3 + lag6 + lag6 + lag6) / 6 if (lag1 + lag2 + lag3 + lag6) > 0 else 0

    # ====== ЛАГИ ЗАПУСКОВ ======
    for lag in [1, 2, 3, 6]:
        val = monthly_avg_dry.get(month, 0)
        pred_row[f'запуски_lag{lag}'] = val
    pred_row['запуски_ma3'] = (pred_row['запуски_lag1'] + pred_row['запуски_lag2'] + pred_row['запуски_lag3']) / 3

    # ====== ЛАГИ ОТЕЛОВ КОРОВ ======
    for lag in [1, 2, 3, 6]:
        val = monthly_avg_calving_cows.get(month, 0)
        pred_row[f'отелы_коров_lag{lag}'] = val
    pred_row['отелы_коров_ma3'] = (pred_row['отелы_коров_lag1'] + pred_row['отелы_коров_lag2'] + pred_row['отелы_коров_lag3']) / 3

    # ====== ЛАГИ ВЫБЫТИЙ СУХОСТОЙНЫХ ======
    for lag in [1, 2, 3, 6]:
        val = monthly_avg_culling_suh.get(month, 0)
        pred_row[f'выбытия_сухостойных_lag{lag}'] = val
    pred_row['выбытия_сухостойных_ma3'] = (pred_row['выбытия_сухостойных_lag1'] +
                                            pred_row['выбытия_сухостойных_lag2'] +
                                            pred_row['выбытия_сухостойных_lag3']) / 3

    # Прогнозируем сухостойных
    pred_df = pd.DataFrame([pred_row])
    X_pred = pred_df[[col for col in feature_cols if col in pred_df.columns]]
    X_pred = X_pred.fillna(0)

    pred_suhostoynye = model.predict(X_pred)[0]
    pred_suhostoynye = max(0, int(round(pred_suhostoynye)))

    # Сохраняем прогноз сухостойных
    forecasts_suhostoynye[(year, month)] = pred_suhostoynye

    # Дойные = Фуражные - Сухостойные
    furazh = furazh_forecast.get((year, month), 0)
    pred_doynye = furazh - pred_suhostoynye
    pred_doynye = max(0, int(round(pred_doynye)))

    forecasts_doynye[(year, month)] = pred_doynye

    results.append({
        'год': year,
        'месяц': month,
        'сухостойные': pred_suhostoynye,
        'дойные': pred_doynye,
        'фуражные': furazh
    })

    print(f"  Сухостойные: {pred_suhostoynye}")
    print(f"  Дойные: {pred_doynye}")
    print(f"  Фуражные: {furazh}")
    print(f"  Проверка: {pred_suhostoynye} + {pred_doynye} = {pred_suhostoynye + pred_doynye}")

# ============================================
# 11. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗА")
print("="*80)

header = f"{'Месяц':<12} {'Сухостойные':>12} {'Дойные':>12} {'Фуражные':>12}"
print(header)
print("-" * 50)

for r in results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    print(f"{month_name:<12} "
          f"{r['сухостойные']:>12} "
          f"{r['дойные']:>12} "
          f"{r['фуражные']:>12}")

print("-" * 50)

# ============================================
# 12. СТАТИСТИКА
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*80)

first_suh = results[0]['сухостойные']
last_suh = results[-1]['сухостойные']
change_suh = last_suh - first_suh
change_suh_pct = (change_suh / first_suh * 100) if first_suh > 0 else 0

print(f"\nСухостойные:")
print(f"  Октябрь 2024: {first_suh}")
print(f"  Декабрь 2025: {last_suh}")
print(f"  Изменение: {change_suh:+d} ({change_suh_pct:+.1f}%)")

first_doy = results[0]['дойные']
last_doy = results[-1]['дойные']
change_doy = last_doy - first_doy
change_doy_pct = (change_doy / first_doy * 100) if first_doy > 0 else 0

print(f"\nДойные:")
print(f"  Октябрь 2024: {first_doy}")
print(f"  Декабрь 2025: {last_doy}")
print(f"  Изменение: {change_doy:+d} ({change_doy_pct:+.1f}%)")

avg_suh = np.mean([r['сухостойные'] for r in results])
avg_doy = np.mean([r['дойные'] for r in results])
avg_furazh = np.mean([r['фуражные'] for r in results])

print(f"\nСреднемесячные показатели:")
print(f"  Сухостойные: {avg_suh:.0f}")
print(f"  Дойные: {avg_doy:.0f}")
print(f"  Фуражные: {avg_furazh:.0f}")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • Обучение: 2022, 2023, январь-сентябрь 2024")
print("  • Прогноз: октябрь 2024 - декабрь 2025")
print("  • Для прогноза используются ТОЛЬКО предыдущие ПРОГНОЗЫ!")
print("  • Дойные = Фуражные - Сухостойные")
print("="*80)

# ===== NOTEBOOK CELL 12 =====
import pandas as pd
import numpy as np

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ ФУРАЖНЫХ КОРОВ")
print("="*80)

# ============================================
# 1. ВХОДНЫЕ ДАННЫЕ (ПРОГНОЗЫ ИЗ МОДЕЛЕЙ)
# ============================================

# Прогноз отелов (коровы и нетели) - из модели 2
# Месяцы: октябрь 2024 - декабрь 2025
calving_forecast = {
    # 2024 год
    (2024, 10): {'коровы': 158, 'нетели': 98},
    (2024, 11): {'коровы': 164, 'нетели': 81},
    (2024, 12): {'коровы': 202, 'нетели': 79},
    # 2025 год
    (2025, 1): {'коровы': 160, 'нетели': 85},
    (2025, 2): {'коровы': 147, 'нетели': 102},
    (2025, 3): {'коровы': 188, 'нетели': 80},
    (2025, 4): {'коровы': 183, 'нетели': 90},
    (2025, 5): {'коровы': 190, 'нетели': 81},
    (2025, 6): {'коровы': 194, 'нетели': 78},
    (2025, 7): {'коровы': 198, 'нетели': 76},
    (2025, 8): {'коровы': 195, 'нетели': 77},
    (2025, 9): {'коровы': 201, 'нетели': 79},
    (2025, 10): {'коровы': 174, 'нетели': 86},
    (2025, 11): {'коровы': 181, 'нетели': 65},
    (2025, 12): {'коровы': 203, 'нетели': 73}
}

# Прогноз выбытий по лактациям - из модели 3
culling_forecast = {
    # 2024 год
    (2024, 10): {'L0': 282, 'L1': 10, 'L2': 10, 'L3': 12, 'L4': 15, 'L5+': 24},
    (2024, 11): {'L0': 292, 'L1': 11, 'L2': 8, 'L3': 12, 'L4': 10, 'L5+': 32},
    (2024, 12): {'L0': 287, 'L1': 20, 'L2': 12, 'L3': 22, 'L4': 13, 'L5+': 38},
    # 2025 год
    (2025, 1): {'L0': 225, 'L1': 10, 'L2': 7, 'L3': 11, 'L4': 15, 'L5+': 18},
    (2025, 2): {'L0': 245, 'L1': 15, 'L2': 10, 'L3': 21, 'L4': 17, 'L5+': 16},
    (2025, 3): {'L0': 267, 'L1': 16, 'L2': 13, 'L3': 12, 'L4': 11, 'L5+': 20},
    (2025, 4): {'L0': 287, 'L1': 19, 'L2': 11, 'L3': 10, 'L4': 13, 'L5+': 19},
    (2025, 5): {'L0': 284, 'L1': 18, 'L2': 11, 'L3': 11, 'L4': 15, 'L5+': 20},
    (2025, 6): {'L0': 293, 'L1': 16, 'L2': 11, 'L3': 10, 'L4': 15, 'L5+': 22},
    (2025, 7): {'L0': 315, 'L1': 14, 'L2': 11, 'L3': 9, 'L4': 14, 'L5+': 24},
    (2025, 8): {'L0': 299, 'L1': 16, 'L2': 11, 'L3': 11, 'L4': 16, 'L5+': 23},
    (2025, 9): {'L0': 288, 'L1': 14, 'L2': 12, 'L3': 10, 'L4': 18, 'L5+': 24},
    (2025, 10): {'L0': 285, 'L1': 15, 'L2': 8, 'L3': 9, 'L4': 15, 'L5+': 20},
    (2025, 11): {'L0': 301, 'L1': 17, 'L2': 8, 'L3': 10, 'L4': 14, 'L5+': 22},
    (2025, 12): {'L0': 304, 'L1': 19, 'L2': 12, 'L3': 11, 'L4': 16, 'L5+': 25}
}

# ============================================
# 2. ФАКТИЧЕСКОЕ КОЛИЧЕСТВО ФУРАЖНЫХ НА СЕНТЯБРЬ 2024 (БАЗА)
# ============================================
# Нужно взять из ваших данных, здесь пример:
FURAZH_BASE_SEP_2024 = 2909  # ЗАМЕНИТЕ НА РЕАЛЬНОЕ ЗНАЧЕНИЕ!

# ============================================
# 3. РАСЧЕТ ФУРАЖНЫХ ПО ФОРМУЛЕ
# ============================================
print(f"\nБазовое количество фуражных на сентябрь 2024: {FURAZH_BASE_SEP_2024}")
print("\n" + "="*80)
print("РАСЧЕТ ПО МЕСЯЦАМ")
print("="*80)

# Список месяцев для прогноза
predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

# Словарь для хранения результатов
results = []

# Начальное значение (сентябрь 2024)
furazh_prev = FURAZH_BASE_SEP_2024

for year, month in predict_months:
    # Получаем прогнозы для текущего месяца
    calving = calving_forecast.get((year, month), {'коровы': 0, 'нетели': 0})
    culling = culling_forecast.get((year, month), {'L0': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0, 'L5+': 0})

    # Отелы нетелей (переход L0 -> L1, пополнение фуражных)
    otely_netelei = calving['нетели']

    # Выбытие коров (только L1 и выше, L0 - это нетели, их выбытие не учитываем)
    vybytie_korov = culling['L1'] + culling['L2'] + culling['L3'] + culling['L4'] + culling['L5+']

    # Формула: фуражные = фуражные_прошлый_месяц + отелы_нетелей - выбытие_коров
    furazh_current = furazh_prev + otely_netelei - vybytie_korov

    # Сохраняем результат
    results.append({
        'год': year,
        'месяц': month,
        'фуражные_предыдущий': furazh_prev,
        'отелы_нетелей': otely_netelei,
        'выбытие_коров': vybytie_korov,
        'фуражные_текущий': furazh_current
    })

    # Обновляем для следующего месяца
    furazh_prev = furazh_current

# ============================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗА ФУРАЖНЫХ")
print("="*80)

# Заголовок
header = f"{'Месяц':<12} {'Фуражные (n-1)':>15} {'+ Отелы нетелей':>18} {'- Выбытие коров':>18} {'= Фуражные (n)':>15}"
print(header)
print("-" * 80)

# Данные
for r in results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    print(f"{month_name:<12} "
          f"{r['фуражные_предыдущий']:>15} "
          f"{r['отелы_нетелей']:>18} "
          f"{r['выбытие_коров']:>18} "
          f"{r['фуражные_текущий']:>15}")

print("-" * 80)

# ============================================
# 5. ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА
# ============================================
print("\n" + "="*80)
print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
print("="*80)

# Общая динамика
first_val = results[0]['фуражные_текущий']
last_val = results[-1]['фуражные_текущий']
change = last_val - first_val
change_pct = (change / first_val * 100) if first_val > 0 else 0

print(f"\nНачало прогноза (октябрь 2024): {first_val}")
print(f"Конец прогноза (декабрь 2025): {last_val}")
print(f"Изменение: {change:+d} ({change_pct:+.1f}%)")

# Средние значения
avg_otely = np.mean([r['отелы_нетелей'] for r in results])
avg_vybytie = np.mean([r['выбытие_коров'] for r in results])
avg_furazh = np.mean([r['фуражные_текущий'] for r in results])

print(f"\nСреднемесячные показатели:")
print(f"  Отелы нетелей: {avg_otely:.1f}")
print(f"  Выбытие коров: {avg_vybytie:.1f}")
print(f"  Фуражные: {avg_furazh:.1f}")

# Минимальное и максимальное значение
min_furazh = min([r['фуражные_текущий'] for r in results])
max_furazh = max([r['фуражные_текущий'] for r in results])
min_month = next(f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}" for r in results if r['фуражные_текущий'] == min_furazh)
max_month = next(f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}" for r in results if r['фуражные_текущий'] == max_furazh)

print(f"\nДиапазон значений:")
print(f"  Минимум: {min_furazh} ({min_month})")
print(f"  Максимум: {max_furazh} ({max_month})")

print("\n" + "="*80)
print("ГОТОВО!")
print("Формула: Фуражные(n) = Фуражные(n-1) + Отелы_нетелей(n) - Выбытие_коров(n)")
print("  • Отелы нетелей = L0 -> L1 (пополнение)")
print("  • Выбытие коров = L1 + L2 + L3 + L4 + L5+ (убыль)")
print("  • Выбытие L0 не учитывается (нетели не входят в фуражных)")
print("="*80)

# ===== NOTEBOOK CELL 14 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ ВЫБЫТИЙ ПО ЛАКТАЦИЯМ")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_semen_2022 = pd.read_excel(f"{folder}/Осеменения_2022.xlsx")
df_semen_2023 = pd.read_excel(f"{folder}/Осеменения_2023.xlsx")
df_semen_2024 = pd.read_excel(f"{folder}/Осеменения_2024.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")
df_culling_2025 = pd.read_excel(f"{folder}/Выбытие_2025.xlsx")

df_dry_2022 = pd.read_excel(f"{folder}/Запуск_2022.xlsx")
df_dry_2023 = pd.read_excel(f"{folder}/Запуск_2023.xlsx")
df_dry_2024 = pd.read_excel(f"{folder}/Запуск_2024.xlsx")

print("\n✅ Данные загружены")

# ============================================
# 2. ФУНКЦИЯ РАЗДЕЛЕНИЯ ВЫБЫТИЙ ПО ЛАКТАЦИЯМ
# ============================================
def split_culling_by_lactation(df_culling):
    """Разделяет выбытия по лактациям (L0, L1, L2, L3, L4, L5+)"""
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])

    # Определяем лактацию
    df['LACT'] = df['LACT'].fillna(0).astype(int)

    # Категории лактаций
    df['лактация_группа'] = df['LACT'].apply(lambda x:
        'L0' if x == 0 else
        'L1' if x == 1 else
        'L2' if x == 2 else
        'L3' if x == 3 else
        'L4' if x == 4 else
        'L5+'
    )

    return df

def aggregate_culling_by_lactation_monthly(df_culling, max_date=None):
    """Агрегирует выбытия по месяцам и лактациям"""
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5+'])

    df = split_culling_by_lactation(df_culling)

    if max_date:
        df = df[df['Дата'] <= max_date]

    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')

    # Группировка по месяцам и лактациям
    grouped = df.groupby(['год', 'месяц', 'дата_месяц', 'лактация_группа']).size().unstack(fill_value=0).reset_index()

    # Убеждаемся, что все колонки есть
    for col in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']:
        if col not in grouped.columns:
            grouped[col] = 0

    return grouped

# ============================================
# 3. АГРЕГАЦИЯ ДРУГИХ ДАННЫХ (ДЛЯ ПРИЗНАКОВ)
# ============================================
def aggregate_calvings_monthly(df_calvings, max_date=None):
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'отелы'])
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    otel_mask = df['Событие'].str.upper().str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])
    df = df[otel_mask].copy()
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='отелы')
    return monthly

def process_c_semen(df_semen):
    """Обработка осеменений с R='C'"""
    if len(df_semen) == 0:
        return df_semen
    df = df_semen.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['BDAT'] = pd.to_datetime(df['BDAT'])
    df['животное_ключ'] = df['REG'].fillna('').astype(str)
    mask_no_reg = (df['животное_ключ'] == '') | (df['животное_ключ'] == 'nan')
    df.loc[mask_no_reg, 'животное_ключ'] = df.loc[mask_no_reg, 'ID'].astype(str) + '_' + df.loc[mask_no_reg, 'BDAT'].astype(str)
    df = df.sort_values(['животное_ключ', 'Дата'])
    c_mask = df['R'].str.strip() == 'C'
    df['Дата_исправленная'] = df['Дата']
    for idx in df[c_mask].index:
        animal_key = df.loc[idx, 'животное_ключ']
        current_date = df.loc[idx, 'Дата']
        prev_semen = df[(df['животное_ключ'] == animal_key) & (df['Дата'] < current_date) & (df.index != idx)].sort_values('Дата', ascending=False)
        if len(prev_semen) > 0:
            df.loc[idx, 'Дата_исправленная'] = prev_semen.iloc[0]['Дата']
    df['тип_осеменения'] = df['R'].str.strip()
    return df

def aggregate_semen_monthly(df_semen, max_date=None):
    if len(df_semen) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'всего_осеменений', 'успешные'])

    df = process_c_semen(df_semen)
    if max_date:
        df = df[df['Дата_исправленная'] <= max_date]
    df['месяц'] = df['Дата_исправленная'].dt.month
    df['год'] = df['Дата_исправленная'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    total_semen = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='всего_осеменений')
    success = df[df['тип_осеменения'] == 'P'].groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='успешные')
    features = total_semen.merge(success, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
    return features

def aggregate_dry_monthly(df_dry, max_date=None):
    if len(df_dry) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'запуски'])
    df = df_dry.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='запуски')
    return monthly

# ============================================
# 4. ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ (янв 2022 - сен 2024)
# ============================================
MAX_DATE_TRAIN = pd.Timestamp('2024-09-30')

# Выбытия по лактациям
train_culling_lact = pd.concat([
    aggregate_culling_by_lactation_monthly(df_culling_2022, MAX_DATE_TRAIN),
    aggregate_culling_by_lactation_monthly(df_culling_2023, MAX_DATE_TRAIN),
    aggregate_culling_by_lactation_monthly(df_culling_2024, MAX_DATE_TRAIN)
], ignore_index=True)

# Отелы
train_calvings = pd.concat([
    aggregate_calvings_monthly(df_calvings_2022, MAX_DATE_TRAIN),
    aggregate_calvings_monthly(df_calvings_2023, MAX_DATE_TRAIN),
    aggregate_calvings_monthly(df_calvings_2024, MAX_DATE_TRAIN)
], ignore_index=True)

# Осеменения
train_semen = pd.concat([
    aggregate_semen_monthly(df_semen_2022, MAX_DATE_TRAIN),
    aggregate_semen_monthly(df_semen_2023, MAX_DATE_TRAIN),
    aggregate_semen_monthly(df_semen_2024, MAX_DATE_TRAIN)
], ignore_index=True)

# Запуски
train_dry = pd.concat([
    aggregate_dry_monthly(df_dry_2022, MAX_DATE_TRAIN),
    aggregate_dry_monthly(df_dry_2023, MAX_DATE_TRAIN),
    aggregate_dry_monthly(df_dry_2024, MAX_DATE_TRAIN)
], ignore_index=True)

# Объединяем все
train_df = train_culling_lact.merge(train_calvings, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(train_semen, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(train_dry, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

print(f"\nОбучающих месяцев: {len(train_df)}")

# ============================================
# 5. ФУНКЦИЯ СОЗДАНИЯ ПРИЗНАКОВ
# ============================================
def create_features_culling(df):
    df = df.copy()

    # Сезонность
    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)
    df['тренд'] = range(1, len(df) + 1)

    # Лаги выбытий по лактациям
    lact_cols = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']
    for col in lact_cols:
        for lag in [1, 2, 3, 6]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
        df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean().fillna(0)
        df[f'{col}_ma6'] = df[col].rolling(6, min_periods=1).mean().fillna(0)

    # Общее выбытие
    df['всего_выбытий'] = df[lact_cols].sum(axis=1)
    df['всего_выбытий_lag1'] = df['всего_выбытий'].shift(1).fillna(0)
    df['всего_выбытий_lag2'] = df['всего_выбытий'].shift(2).fillna(0)
    df['всего_выбытий_lag3'] = df['всего_выбытий'].shift(3).fillna(0)
    df['всего_выбытий_ma3'] = df['всего_выбытий'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги отелов
    for lag in [1, 2, 3, 6]:
        df[f'отелы_lag{lag}'] = df['отелы'].shift(lag).fillna(0)
    df['отелы_ma3'] = df['отелы'].rolling(3, min_periods=1).mean().fillna(0)
    df['отелы_ma6'] = df['отелы'].rolling(6, min_periods=1).mean().fillna(0)

    # Лаги запусков
    for lag in [1, 2, 3, 6]:
        df[f'запуски_lag{lag}'] = df['запуски'].shift(lag).fillna(0)
    df['запуски_ma3'] = df['запуски'].rolling(3, min_periods=1).mean().fillna(0)

    # Лаги осеменений
    for lag in [3, 6]:
        df[f'успешные_lag{lag}'] = df['успешные'].shift(lag).fillna(0)
    df['успешные_ma3'] = df['успешные'].rolling(3, min_periods=1).mean().fillna(0)

    # Доли выбытий по лактациям
    for col in lact_cols:
        df[f'доля_{col}'] = df[col] / (df['всего_выбытий'] + 1)
        df[f'доля_{col}_lag1'] = df[f'доля_{col}'].shift(1).fillna(0)

    # Соотношение выбытий к отелам
    df['выбытия_на_отелы'] = df['всего_выбытий'] / (df['отелы'] + 1)
    df['выбытия_на_отелы_lag1'] = df['выбытия_на_отелы'].shift(1).fillna(0)

    return df

# ============================================
# 6. ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ КАЖДОЙ ЛАКТАЦИИ
# ============================================
train_features = create_features_culling(train_df)

# Удаляем строки с NaN
train_clean = train_features.dropna()

# Список признаков (все, кроме целевых и служебных)
feature_cols = [col for col in train_clean.columns if col not in [
    'год', 'месяц', 'дата_месяц', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5+',
    'всего_выбытий', 'доля_L0', 'доля_L1', 'доля_L2', 'доля_L3', 'доля_L4', 'доля_L5+'
]]

X_train = train_clean[feature_cols]
lact_cols = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']

print(f"\nОбучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")
print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ КАЖДОЙ ЛАКТАЦИИ")
print("="*80)

models = {}
for lact in lact_cols:
    print(f"\nОбучение модели для {lact}...")
    y_train = train_clean[lact].values

    # Простая оптимизация
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    try:
        tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
        grid = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            param_grid, cv=tscv, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        print(f"  Лучшие параметры: {best_params}")
    except:
        best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
        print(f"  Используем параметры по умолчанию")

    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    models[lact] = model

    # Проверка на обучающих данных
    pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_train, pred_train)
    print(f"  MAE на обучении: {mae:.2f}")

# ============================================
# 7. РЕКУРСИВНЫЙ ПРОГНОЗ
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ НА ОКТЯБРЬ 2024 - ДЕКАБРЬ 2025")
print("="*80)

# Словари для хранения прогнозов
forecasts = {lact: {} for lact in lact_cols}

# Исторические средние по месяцам (для лагов, где нет прогнозов)
monthly_avg = {}
for lact in lact_cols:
    monthly_avg[lact] = train_df.groupby('месяц')[lact].mean().to_dict()
monthly_avg_calvings = train_df.groupby('месяц')['отелы'].mean().to_dict()
monthly_avg_dry = train_df.groupby('месяц')['запуски'].mean().to_dict()
monthly_avg_semen = train_df.groupby('месяц')['успешные'].mean().to_dict()
monthly_avg_total_semen = train_df.groupby('месяц')['всего_осеменений'].mean().to_dict()

# Список месяцев для прогноза
predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

# Получаем факты для сравнения (НЕ для прогноза!)
def get_actual_culling_by_lact(year, month):
    """Получает фактические выбытия по лактациям для указанного месяца"""
    if year == 2024:
        df = df_culling_2024
    elif year == 2025:
        df = df_culling_2025
    else:
        return {lact: 0 for lact in lact_cols}

    df = df.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    df = df[(df['Дата'].dt.year == year) & (df['Дата'].dt.month == month)]

    if len(df) == 0:
        return {lact: 0 for lact in lact_cols}

    df = split_culling_by_lactation(df)
    result = df.groupby('лактация_группа').size().to_dict()

    # Заполняем недостающие лактации
    for lact in lact_cols:
        if lact not in result:
            result[lact] = 0

    return result

results = []

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")

    # Создаем строку для прогноза
    pred_row = {}
    pred_row['год'] = year
    pred_row['месяц'] = month
    pred_row['дата_месяц'] = pd.Timestamp(f'{year}-{month:02d}-01')
    pred_row['месяц_синус'] = np.sin(2 * np.pi * month / 12)
    pred_row['месяц_косинус'] = np.cos(2 * np.pi * month / 12)
    pred_row['квартал'] = (month-1)//3 + 1
    pred_row['тренд'] = len(train_features) + len(results) + 1

    # ====== БАЗОВЫЕ ПРИЗНАКИ (нужны для модели) ======
    pred_row['отелы'] = monthly_avg_calvings.get(month, 0)
    pred_row['всего_осеменений'] = monthly_avg_total_semen.get(month, 0)
    pred_row['успешные'] = monthly_avg_semen.get(month, 0)
    pred_row['запуски'] = monthly_avg_dry.get(month, 0)

    # ====== ЛАГИ ВЫБЫТИЙ (только из прогнозов!) ======
    for lact in lact_cols:
        # lag1
        if month > 1:
            pred_row[f'{lact}_lag1'] = forecasts[lact].get((year, month-1),
                                      forecasts[lact].get((year-1, 12), monthly_avg[lact].get(month, 0)))
        else:
            pred_row[f'{lact}_lag1'] = 0

        # lag2
        if month > 2:
            pred_row[f'{lact}_lag2'] = forecasts[lact].get((year, month-2),
                                      forecasts[lact].get((year-1, 11), monthly_avg[lact].get(month, 0)))
        else:
            pred_row[f'{lact}_lag2'] = 0

        # lag3
        if month > 3:
            pred_row[f'{lact}_lag3'] = forecasts[lact].get((year, month-3),
                                      forecasts[lact].get((year-1, 10), monthly_avg[lact].get(month, 0)))
        else:
            pred_row[f'{lact}_lag3'] = monthly_avg[lact].get(month, 0)

        # lag6
        if month > 6:
            pred_row[f'{lact}_lag6'] = forecasts[lact].get((year, month-6),
                                      forecasts[lact].get((year-1, 6), monthly_avg[lact].get(month, 0)))
        else:
            pred_row[f'{lact}_lag6'] = monthly_avg[lact].get(month, 0)

        # MA3
        lag1 = pred_row[f'{lact}_lag1']
        lag2 = pred_row[f'{lact}_lag2']
        lag3 = pred_row[f'{lact}_lag3']
        pred_row[f'{lact}_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0

        # MA6
        lag6 = pred_row[f'{lact}_lag6']
        pred_row[f'{lact}_ma6'] = (lag1 + lag2 + lag3 + lag6 + lag6 + lag6) / 6 if (lag1 + lag2 + lag3 + lag6) > 0 else 0

    # ====== ОБЩЕЕ ВЫБЫТИЕ (из прогнозов по лактациям) ======
    pred_row['всего_выбытий'] = sum(pred_row[f'{lact}_lag1'] for lact in lact_cols)
    pred_row['всего_выбытий_lag1'] = 0
    pred_row['всего_выбытий_lag2'] = 0
    pred_row['всего_выбытий_lag3'] = 0
    pred_row['всего_выбытий_ma3'] = 0

    if month > 1:
        prev_total = sum(forecasts[lact].get((year, month-1), 0) for lact in lact_cols)
        pred_row['всего_выбытий_lag1'] = prev_total
    if month > 2:
        prev_total = sum(forecasts[lact].get((year, month-2), 0) for lact in lact_cols)
        pred_row['всего_выбытий_lag2'] = prev_total
    if month > 3:
        prev_total = sum(forecasts[lact].get((year, month-3), 0) for lact in lact_cols)
        pred_row['всего_выбытий_lag3'] = prev_total

    pred_row['всего_выбытий_ma3'] = (pred_row['всего_выбытий_lag1'] +
                                      pred_row['всего_выбытий_lag2'] +
                                      pred_row['всего_выбытий_lag3']) / 3 if (pred_row['всего_выбытий_lag1'] +
                                                                              pred_row['всего_выбытий_lag2'] +
                                                                              pred_row['всего_выбытий_lag3']) > 0 else 0

    # ====== ЛАГИ ОТЕЛОВ ======
    for lag in [1, 2, 3, 6]:
        val = monthly_avg_calvings.get(month, 0)
        pred_row[f'отелы_lag{lag}'] = val
    pred_row['отелы_ma3'] = (pred_row['отелы_lag1'] + pred_row['отелы_lag2'] + pred_row['отелы_lag3']) / 3
    pred_row['отелы_ma6'] = pred_row['отелы_ma3']

    # ====== ЛАГИ ЗАПУСКОВ ======
    for lag in [1, 2, 3, 6]:
        pred_row[f'запуски_lag{lag}'] = monthly_avg_dry.get(month, 0)
    pred_row['запуски_ma3'] = (pred_row['запуски_lag1'] + pred_row['запуски_lag2'] + pred_row['запуски_lag3']) / 3

    # ====== ЛАГИ ОСЕМЕНЕНИЙ ======
    for lag in [3, 6]:
        pred_row[f'успешные_lag{lag}'] = monthly_avg_semen.get(month, 0)
    pred_row['успешные_ma3'] = monthly_avg_semen.get(month, 0)

    # ====== ДОЛИ ВЫБЫТИЙ ======
    total = pred_row['всего_выбытий'] + 1
    for lact in lact_cols:
        pred_row[f'доля_{lact}'] = pred_row.get(f'{lact}_lag1', 0) / total
        pred_row[f'доля_{lact}_lag1'] = pred_row[f'доля_{lact}']

    # ====== ВЫБЫТИЯ НА ОТЕЛЫ ======
    pred_row['выбытия_на_отелы'] = pred_row['всего_выбытий'] / (pred_row['отелы'] + 1)
    pred_row['выбытия_на_отелы_lag1'] = pred_row['выбытия_на_отелы']

    # Прогнозируем для каждой лактации
    pred_df = pd.DataFrame([pred_row])
    X_pred = pred_df[feature_cols]
    X_pred = X_pred.fillna(0)

    pred_values = {}
    for lact in lact_cols:
        val = models[lact].predict(X_pred)[0]
        val = max(0, int(round(val)))
        pred_values[lact] = val
        forecasts[lact][(year, month)] = val

    # Получаем факты для сравнения
    actual_values = get_actual_culling_by_lact(year, month)

    results.append({
        'год': year,
        'месяц': month,
        'прогноз': pred_values,
        'факт': actual_values
    })

    # Вывод информации
    print(f"  Прогноз: L0={pred_values['L0']}, L1={pred_values['L1']}, L2={pred_values['L2']}, "
          f"L3={pred_values['L3']}, L4={pred_values['L4']}, L5+={pred_values['L5+']}")

# ============================================
# 8. ВЫВОД РЕЗУЛЬТАТОВ В ТРЕБУЕМОМ ФОРМАТЕ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗОВ ПО ЛАКТАЦИЯМ")
print("="*80)

# Заголовок
header = f"{'Месяц':<12}"
for lact in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']:
    header += f" {lact}_прогноз> {lact}_ошибка> {lact}_факт"
print(header)
print("-" * (12 + 6 * 6 * 15))

# Данные
for r in results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    row = f"{month_name:<12}"

    for lact in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']:
        pred = r['прогноз'][lact]
        fact = r['факт'][lact]
        error = pred - fact
        row += f" {pred:>8} {error:>+8} {fact:>8}"

    print(row)

print("-" * (12 + 6 * 6 * 15))

# Итоги
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА ЗА ПЕРИОД ПРОГНОЗА")
print("="*80)

for lact in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5+']:
    preds = [r['прогноз'][lact] for r in results]
    facts = [r['факт'][lact] for r in results]
    total_pred = sum(preds)
    total_fact = sum(facts)
    total_error = total_pred - total_fact
    total_error_pct = (total_error / total_fact * 100) if total_fact > 0 else 0
    mae = mean_absolute_error(facts, preds)

    print(f"\n{lact}:")
    print(f"  Прогноз: {total_pred}")
    print(f"  Факт: {total_fact}")
    print(f"  Ошибка: {total_error:+d} ({total_error_pct:+.1f}%)")
    print(f"  MAE: {mae:.2f}")

print("\n" + "="*80)
print("ГОТОВО!")
print("ВАЖНО: Для прогноза использовались ТОЛЬКО данные за 2022, 2023 и январь-сентябрь 2024!")
print("  • Рекурсивный прогноз (только прогнозные значения предыдущих месяцев)")
print("  • Факты за октябрь-декабрь 2024 и 2025 используются ТОЛЬКО для сравнения")
print("="*80)

# ===== NOTEBOOK CELL 16 =====
import pandas as pd
import numpy as np

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("РАЗДЕЛЕНИЕ ВЫБЫТИЙ ФУРАЖНЫХ КОРОВ НА СУХОСТОЙНЫХ И ДОЙНЫХ")
print("="*80)

# ============================================
# 1. ВХОДНЫЕ ДАННЫЕ
# ============================================

# Прогноз выбытий по лактациям (из модели 3)
culling_forecast = {
    # 2024 год
    (2024, 10): {'L0': 282, 'L1': 10, 'L2': 10, 'L3': 12, 'L4': 15, 'L5+': 24},
    (2024, 11): {'L0': 292, 'L1': 11, 'L2': 8, 'L3': 12, 'L4': 10, 'L5+': 32},
    (2024, 12): {'L0': 287, 'L1': 20, 'L2': 12, 'L3': 22, 'L4': 13, 'L5+': 38},
    # 2025 год
    (2025, 1): {'L0': 225, 'L1': 10, 'L2': 7, 'L3': 11, 'L4': 15, 'L5+': 18},
    (2025, 2): {'L0': 245, 'L1': 15, 'L2': 10, 'L3': 21, 'L4': 17, 'L5+': 16},
    (2025, 3): {'L0': 267, 'L1': 16, 'L2': 13, 'L3': 12, 'L4': 11, 'L5+': 20},
    (2025, 4): {'L0': 287, 'L1': 19, 'L2': 11, 'L3': 10, 'L4': 13, 'L5+': 19},
    (2025, 5): {'L0': 284, 'L1': 18, 'L2': 11, 'L3': 11, 'L4': 15, 'L5+': 20},
    (2025, 6): {'L0': 293, 'L1': 16, 'L2': 11, 'L3': 10, 'L4': 15, 'L5+': 22},
    (2025, 7): {'L0': 315, 'L1': 14, 'L2': 11, 'L3': 9, 'L4': 14, 'L5+': 24},
    (2025, 8): {'L0': 299, 'L1': 16, 'L2': 11, 'L3': 11, 'L4': 16, 'L5+': 23},
    (2025, 9): {'L0': 288, 'L1': 14, 'L2': 12, 'L3': 10, 'L4': 18, 'L5+': 24},
    (2025, 10): {'L0': 285, 'L1': 15, 'L2': 8, 'L3': 9, 'L4': 15, 'L5+': 20},
    (2025, 11): {'L0': 301, 'L1': 17, 'L2': 8, 'L3': 10, 'L4': 14, 'L5+': 22},
    (2025, 12): {'L0': 304, 'L1': 19, 'L2': 12, 'L3': 11, 'L4': 16, 'L5+': 25}
}

# Прогноз сухостойных и дойных (из модели 4)
status_forecast = {
    (2024, 10): {'сухостойные': 1, 'дойные': 45},
    (2024, 11): {'сухостойные': 2, 'дойные': 42},
    (2024, 12): {'сухостойные': 1, 'дойные': 64},
    (2025, 1): {'сухостойные': 1, 'дойные': 40},
    (2025, 2): {'сухостойные': 1, 'дойные': 45},
    (2025, 3): {'сухостойные': 1, 'дойные': 45},
    (2025, 4): {'сухостойные': 2, 'дойные': 50},
    (2025, 5): {'сухостойные': 2, 'дойные': 49},
    (2025, 6): {'сухостойные': 2, 'дойные': 51},
    (2025, 7): {'сухостойные': 3, 'дойные': 51},
    (2025, 8): {'сухостойные': 2, 'дойные': 49},
    (2025, 9): {'сухостойные': 2, 'дойные': 48},
    (2025, 10): {'сухостойные': 2, 'дойные': 50},
    (2025, 11): {'сухостойные': 2, 'дойные': 52},
    (2025, 12): {'сухостойные': 2, 'дойные': 61}
}

# ============================================
# 2. РАСЧЕТ СКОРРЕКТИРОВАННЫХ ВЫБЫТИЙ
# ============================================
print("\n" + "="*80)
print("РАСЧЕТ СКОРРЕКТИРОВАННЫХ ВЫБЫТИЙ")
print("="*80)

# Функция для корректировки выбытий
def adjust_culling_by_status(culling_forecast, status_forecast):
    """
    Корректирует выбытия по статусу на основе прогноза по лактациям
    """
    results = []

    for (year, month), culling_data in culling_forecast.items():
        # Общее выбытие фуражных коров (L1 и выше)
        total_culling_furazh = (culling_data['L1'] + culling_data['L2'] +
                                culling_data['L3'] + culling_data['L4'] + culling_data['L5+'])

        # Прогноз статуса
        status = status_forecast.get((year, month), {'сухостойные': 1, 'дойные': 1})
        total_status = status['сухостойные'] + status['дойные']

        if total_status > 0:
            # Доли по статусу
            dolya_suh = status['сухостойные'] / total_status
            dolya_doy = status['дойные'] / total_status
        else:
            dolya_suh = 0.5
            dolya_doy = 0.5

        # Корректируем выбытия
        culling_suh = int(round(total_culling_furazh * dolya_suh))
        culling_doy = total_culling_furazh - culling_suh

        results.append({
            'год': year,
            'месяц': month,
            'всего_фуражных': total_culling_furazh,
            'доля_сухостойных': dolya_suh,
            'доля_дойных': dolya_doy,
            'сухостойные': culling_suh,
            'дойные': culling_doy
        })

    return results

# Получаем скорректированные данные
adjusted_results = adjust_culling_by_status(culling_forecast, status_forecast)

# ============================================
# 3. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("СКОРРЕКТИРОВАННЫЙ ПРОГНОЗ ВЫБЫТИЙ ПО СТАТУСУ")
print("="*80)

# Заголовок
header = f"{'Месяц':<12} {'Всего фураж':>12} {'% сух':>8} {'% дой':>8} {'Сух_прогноз':>12} {'Дой_прогноз':>12}"
print(header)
print("-" * 65)

# Данные
for r in adjusted_results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    print(f"{month_name:<12} "
          f"{r['всего_фуражных']:>12} "
          f"{r['доля_сухостойных']:>7.1%} "
          f"{r['доля_дойных']:>7.1%} "
          f"{r['сухостойные']:>12} "
          f"{r['дойные']:>12}")

print("-" * 65)

# ============================================
# 4. ИТОГИ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*80)

total_suh = sum(r['сухостойные'] for r in adjusted_results)
total_doy = sum(r['дойные'] for r in adjusted_results)
total_furazh = sum(r['всего_фуражных'] for r in adjusted_results)

print(f"\nВсего за период (октябрь 2024 - декабрь 2025):")
print(f"  Выбытия фуражных коров: {total_furazh}")
print(f"  Из них сухостойные: {total_suh} ({total_suh/total_furazh*100:.1f}%)")
print(f"  Из них дойные: {total_doy} ({total_doy/total_furazh*100:.1f}%)")

# Средние доли
avg_dolya_suh = np.mean([r['доля_сухостойных'] for r in adjusted_results])
avg_dolya_doy = np.mean([r['доля_дойных'] for r in adjusted_results])

print(f"\nСредние доли за период:")
print(f"  Сухостойные: {avg_dolya_suh:.1%}")
print(f"  Дойные: {avg_dolya_doy:.1%}")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • Использован прогноз выбытий по лактациям (модель 3)")
print("  • Использован прогноз структуры сухостойные/дойные (модель 4)")
print("  • Выбытия разделены пропорционально структуре")
print("="*80)

# ===== NOTEBOOK CELL 18 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ ПАДЕЖА: БЫЧКИ И ТЕЛОЧКИ (Lact=0)")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
folder = "фильтр_ЖК_Высокое"

df_calvings_2022 = pd.read_excel(f"{folder}/Отелы_2022.xlsx")
df_calvings_2023 = pd.read_excel(f"{folder}/Отелы_2023.xlsx")
df_calvings_2024 = pd.read_excel(f"{folder}/Отелы_2024.xlsx")
df_calvings_2025 = pd.read_excel(f"{folder}/Отелы_2025.xlsx")

df_culling_2022 = pd.read_excel(f"{folder}/Выбытие_2022.xlsx")
df_culling_2023 = pd.read_excel(f"{folder}/Выбытие_2023.xlsx")
df_culling_2024 = pd.read_excel(f"{folder}/Выбытие_2024.xlsx")

print(f"\n✅ Данные загружены")

# ============================================
# 2. ФУНКЦИЯ ПОИСКА ПАДЕЖА С ПРОВЕРКОЙ Lact=0
# ============================================
MAX_DATE = pd.Timestamp('2024-09-30')

def extract_deaths_with_gender(df_culling, df_calvings, max_date=None):
    """
    Извлекает данные о падеже с проверкой:
    1. Событие = 'ПАЛА'
    2. Lact = 0 (молодняк)
    3. Определяем пол по GNDR из данных об отелах
    """
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['ID', 'REG', 'Дата', 'пол'])

    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])

    # Фильтруем: событие ПАЛА
    death_mask = df['Событие'].str.upper().str.strip() == 'ПАЛА'
    df = df[death_mask].copy()

    # Фильтруем: Lact = 0
    if 'Lact' in df.columns:
        df = df[df['Lact'] == 0].copy()
        print(f"   Павших с Lact=0: {len(df)}")
    else:
        print("   ВНИМАНИЕ: Нет колонки Lact, пропускаем фильтр")

    if max_date:
        df = df[df['Дата'] <= max_date]

    # Создаем базу полов из отелов
    all_calvings = pd.concat([
        df_calvings_2022, df_calvings_2023, df_calvings_2024, df_calvings_2025
    ], ignore_index=True)
    all_calvings = all_calvings[all_calvings['Событие'].str.upper().str.strip() == 'РОЖДЕН']

    # Создаем ключи для поиска
    all_calvings['ключ'] = all_calvings['ID'].astype(str) + '_' + all_calvings['REG'].fillna('').astype(str)
    all_calvings['ключ_id'] = all_calvings['ID'].astype(str)

    # Определяем пол
    if 'GNDR' in all_calvings.columns:
        gender_dict = dict(zip(all_calvings['ключ'], all_calvings['GNDR'].str.upper().str.strip()))
        gender_dict_id = dict(zip(all_calvings['ключ_id'], all_calvings['GNDR'].str.upper().str.strip()))
    else:
        gender_dict = {}
        gender_dict_id = {}

    # Определяем пол для павших
    df['ключ'] = df['ID'].astype(str) + '_' + df['REG'].fillna('').astype(str)
    df['пол'] = df['ключ'].map(gender_dict)

    # Если не нашли по ключу, пробуем по ID
    df.loc[df['пол'].isna(), 'пол'] = df.loc[df['пол'].isna(), 'ID'].astype(str).map(gender_dict_id)

    # Для неизвестного пола - пробуем найти в данных об отелах по REG
    for idx in df[df['пол'].isna()].index:
        reg = df.loc[idx, 'REG']
        if pd.notna(reg):
            matching = all_calvings[all_calvings['REG'] == reg]
            if len(matching) > 0 and 'GNDR' in matching.columns:
                df.loc[idx, 'пол'] = matching.iloc[0]['GNDR'].upper().strip()

    # Если все еще неизвестно - помечаем
    df['пол'] = df['пол'].fillna('UNKNOWN')

    return df[['ID', 'REG', 'Дата', 'пол']]

def aggregate_deaths_monthly(df_deaths):
    """Агрегирует падеж по месяцам"""
    if len(df_deaths) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'падеж_телочки', 'падеж_бычки'])

    df = df_deaths.copy()
    df['телочка'] = df['пол'].apply(lambda x: 1 if x == 'F' else 0)
    df['бычок'] = df['пол'].apply(lambda x: 1 if x == 'M' else 0)

    # Неизвестный пол - делим поровну
    unknown_mask = df['пол'] == 'UNKNOWN'
    if unknown_mask.sum() > 0:
        df.loc[unknown_mask, 'телочка'] = 0.5
        df.loc[unknown_mask, 'бычок'] = 0.5

    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')

    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).agg(
        падеж_телочки=('телочка', 'sum'),
        падеж_бычки=('бычок', 'sum')
    ).reset_index()

    # Общий падеж = сумма телочек и бычков
    monthly['падеж_всего'] = monthly['падеж_телочки'] + monthly['падеж_бычки']

    return monthly

# Извлекаем данные о падеже
print("\n📊 Извлечение данных о падеже (Lact=0):")
deaths_2022 = extract_deaths_with_gender(df_culling_2022, df_calvings_2022, MAX_DATE)
deaths_2023 = extract_deaths_with_gender(df_culling_2023, df_calvings_2023, MAX_DATE)
deaths_2024 = extract_deaths_with_gender(df_culling_2024, df_calvings_2024, MAX_DATE)

# Агрегируем по месяцам
monthly_deaths = pd.concat([
    aggregate_deaths_monthly(deaths_2022),
    aggregate_deaths_monthly(deaths_2023),
    aggregate_deaths_monthly(deaths_2024)
], ignore_index=True)

print(f"\n📊 Данные по падежу (2022-сен 2024):")
print(monthly_deaths.head(10))
print(f"\nВсего месяцев: {len(monthly_deaths)}")
print(f"Всего падежей: {monthly_deaths['падеж_всего'].sum():.0f}")
print(f"  Телочки: {monthly_deaths['падеж_телочки'].sum():.0f}")
print(f"  Бычки: {monthly_deaths['падеж_бычки'].sum():.0f}")

# ============================================
# 3. ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДРУГИХ ДАННЫХ
# ============================================
def aggregate_births_monthly(df_calvings, max_date=None):
    """Агрегирует рождения по месяцам"""
    if len(df_calvings) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'отелы'])
    df = df_calvings.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    birth_mask = df['Событие'].str.upper().str.strip() == 'РОЖДЕН'
    df = df[birth_mask].copy()
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='отелы')
    return monthly

def aggregate_culling_total_monthly(df_culling, max_date=None):
    """Агрегирует все выбытия (не только падеж)"""
    if len(df_culling) == 0:
        return pd.DataFrame(columns=['год', 'месяц', 'дата_месяц', 'выбытия_всего'])
    df = df_culling.copy()
    df['Дата'] = pd.to_datetime(df['Дата'])
    if max_date:
        df = df[df['Дата'] <= max_date]
    df['месяц'] = df['Дата'].dt.month
    df['год'] = df['Дата'].dt.year
    df['дата_месяц'] = pd.to_datetime(df['год'].astype(str) + '-' + df['месяц'].astype(str) + '-01')
    monthly = df.groupby(['год', 'месяц', 'дата_месяц']).size().reset_index(name='выбытия_всего')
    return monthly

# Загружаем дополнительные данные
monthly_births = pd.concat([
    aggregate_births_monthly(df_calvings_2022, MAX_DATE),
    aggregate_births_monthly(df_calvings_2023, MAX_DATE),
    aggregate_births_monthly(df_calvings_2024, MAX_DATE)
], ignore_index=True)

monthly_culling_total = pd.concat([
    aggregate_culling_total_monthly(df_culling_2022, MAX_DATE),
    aggregate_culling_total_monthly(df_culling_2023, MAX_DATE),
    aggregate_culling_total_monthly(df_culling_2024, MAX_DATE)
], ignore_index=True)

# ============================================
# 4. ПОСТРОЕНИЕ ОБУЧАЮЩЕЙ ВЫБОРКИ
# ============================================
# Объединяем все данные
train_df = monthly_deaths.copy()
train_df = train_df.merge(monthly_births, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)
train_df = train_df.merge(monthly_culling_total, on=['год', 'месяц', 'дата_месяц'], how='left').fillna(0)

# Добавляем сезонные признаки
train_df['месяц_синус'] = np.sin(2 * np.pi * train_df['месяц'] / 12)
train_df['месяц_косинус'] = np.cos(2 * np.pi * train_df['месяц'] / 12)

# Скользящие средние для стабилизации
for col in ['падеж_телочки', 'падеж_бычки', 'падеж_всего', 'отелы']:
    train_df[f'{col}_ma3'] = train_df[col].rolling(3, min_periods=1).mean()
    train_df[f'{col}_ma6'] = train_df[col].rolling(6, min_periods=1).mean()

# Лаги (исторические значения)
for lag in [1, 2, 3, 6]:
    for col in ['падеж_телочки', 'падеж_бычки', 'падеж_всего', 'отелы']:
        train_df[f'{col}_lag{lag}'] = train_df[col].shift(lag)

# Отношения и пропорции
train_df['падеж_на_отел'] = train_df['падеж_всего'] / (train_df['отелы'] + 1)
train_df['падеж_телочек_на_отел'] = train_df['падеж_телочки'] / (train_df['отелы'] + 1)
train_df['падеж_бычков_на_отел'] = train_df['падеж_бычки'] / (train_df['отелы'] + 1)

# Доля падежа в общих выбытиях
train_df['доля_падежа'] = train_df['падеж_всего'] / (train_df['выбытия_всего'] + 1)

# Тренд
train_df['тренд'] = range(1, len(train_df) + 1)

print(f"\n📊 Обучающая выборка ({len(train_df)} месяцев):")
print(train_df[['год', 'месяц', 'падеж_всего', 'падеж_телочки', 'падеж_бычки', 'отелы']].tail(10))

# ============================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================
def train_model_for_target(train_df, target_col, feature_cols):
    """Обучает модель для указанной целевой переменной"""
    # Удаляем строки с NaN
    train_clean = train_df.dropna()

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]

    print(f"\n  Обучение для {target_col}: {len(train_clean)} месяцев, признаков: {len(feature_cols)}")

    # Grid search
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-2))
    grid_search = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"  Лучшие параметры: {grid_search.best_params_}")
    print(f"  Лучшая MAE: {abs(grid_search.best_score_):.2f}")

    # Обучаем финальную модель
    final_model = XGBRegressor(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)

    return final_model, grid_search.best_params_

# Определяем признаки для модели
feature_cols = [col for col in train_df.columns if col not in [
    'год', 'месяц', 'дата_месяц',
    'падеж_всего', 'падеж_телочки', 'падеж_бычки'
]]

print(f"\n🎯 Обучение моделей падежа...")
print(f"Всего признаков: {len(feature_cols)}")

# Обучаем модели ТОЛЬКО для телочек и бычков
models = {}
targets = {
    'падеж_телочки': 'Падеж телочек',
    'падеж_бычки': 'Падеж бычков'
}

for target_col, target_name in targets.items():
    print(f"\n{'='*50}")
    print(f"Модель: {target_name}")
    print(f"{'='*50}")
    models[target_col], _ = train_model_for_target(train_df, target_col, feature_cols)

# ============================================
# 6. ПРОГНОЗ НА ОКТ 2024 - ДЕК 2025
# ============================================
print("\n" + "="*80)
print("ПРОГНОЗ ПАДЕЖА НА ОКТ 2024 - ДЕК 2025")
print("="*80)

# Прогноз отелов (из предыдущей модели)
calving_forecast = {
    (2024, 10): 256, (2024, 11): 245, (2024, 12): 281,
    (2025, 1): 245, (2025, 2): 249, (2025, 3): 268, (2025, 4): 273,
    (2025, 5): 271, (2025, 6): 272, (2025, 7): 274, (2025, 8): 272,
    (2025, 9): 280, (2025, 10): 260, (2025, 11): 246, (2025, 12): 276
}

# Функция для создания прогнозных признаков
def create_prediction_features(month, year, train_df, feature_cols, prev_predictions=None):
    """Создает признаки для прогноза на один месяц"""
    pred_row = {}

    # Базовые признаки
    pred_row['месяц_синус'] = np.sin(2 * np.pi * month / 12)
    pred_row['месяц_косинус'] = np.cos(2 * np.pi * month / 12)
    pred_row['тренд'] = len(train_df) + (year - 2024) * 12 + month - 9

    # Прогноз отелов
    pred_row['отелы'] = calving_forecast.get((year, month), 250)

    # Выбытия всего (среднее за аналогичные месяцы)
    hist_culling = train_df[train_df['месяц'] == month]['выбытия_всего'].mean()
    pred_row['выбытия_всего'] = hist_culling if not np.isnan(hist_culling) else 300

    # Лаги (используем исторические данные или предыдущие прогнозы)
    for lag in [1, 2, 3, 6]:
        # Для падежа
        if prev_predictions and lag <= len(prev_predictions):
            # Используем предыдущие прогнозы
            for col in ['падеж_телочки', 'падеж_бычки', 'падеж_всего']:
                pred_row[f'{col}_lag{lag}'] = prev_predictions[-lag][col]
        else:
            # Используем исторические средние
            for col in ['падеж_телочки', 'падеж_бычки', 'падеж_всего']:
                hist_val = train_df[train_df['месяц'] == month][col].mean()
                pred_row[f'{col}_lag{lag}'] = hist_val if not np.isnan(hist_val) else 0

        # Для отелов
        hist_otely = train_df[train_df['месяц'] == month]['отелы'].mean()
        pred_row[f'отелы_lag{lag}'] = hist_otely if not np.isnan(hist_otely) else 250

    # Скользящие средние
    for col in ['падеж_телочки', 'падеж_бычки', 'падеж_всего', 'отелы']:
        lag1 = pred_row.get(f'{col}_lag1', 0)
        lag2 = pred_row.get(f'{col}_lag2', 0)
        lag3 = pred_row.get(f'{col}_lag3', 0)
        pred_row[f'{col}_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0

        # MA6
        lag6 = pred_row.get(f'{col}_lag6', 0)
        pred_row[f'{col}_ma6'] = (lag1 + lag2 + lag3 + lag6) / 4 if (lag1 + lag2 + lag3 + lag6) > 0 else 0

    # Отношения
    otely = pred_row['отелы'] + 1
    pred_row['падеж_на_отел'] = pred_row.get('падеж_всего_lag1', 0) / otely if pred_row.get('падеж_всего_lag1', 0) > 0 else 0.02
    pred_row['падеж_телочек_на_отел'] = pred_row.get('падеж_телочки_lag1', 0) / otely if pred_row.get('падеж_телочки_lag1', 0) > 0 else 0.01
    pred_row['падеж_бычков_на_отел'] = pred_row.get('падеж_бычки_lag1', 0) / otely if pred_row.get('падеж_бычки_lag1', 0) > 0 else 0.01

    # Доля падежа
    vybytiya = pred_row['выбытия_всего'] + 1
    pred_row['доля_падежа'] = pred_row.get('падеж_всего_lag1', 0) / vybytiya if pred_row.get('падеж_всего_lag1', 0) > 0 else 0.1

    # Добавляем недостающие признаки
    for col in feature_cols:
        if col not in pred_row:
            # Для признаков, которые мы не заполнили, используем 0 или среднее
            if col in train_df.columns:
                pred_row[col] = train_df[col].mean()
            else:
                pred_row[col] = 0

    return pred_row

# Прогнозируем
all_predictions = []
prev_preds = []

for year in [2024, 2025]:
    start_month = 10 if year == 2024 else 1
    end_month = 12

    for month in range(start_month, end_month + 1):
        # Создаем признаки
        pred_row = create_prediction_features(month, year, train_df, feature_cols, prev_preds)

        # Прогнозируем для телочек и бычков
        pred_df = pd.DataFrame([pred_row])

        # Убеждаемся, что все колонки есть
        for col in feature_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0

        X_pred = pred_df[feature_cols].fillna(0)

        # Прогнозируем телочек и бычков
        heifer_pred = models['падеж_телочки'].predict(X_pred)[0]
        bull_pred = models['падеж_бычки'].predict(X_pred)[0]

        # Округляем и приводим к неотрицательным значениям
        heifer_pred = max(0, int(round(heifer_pred)))
        bull_pred = max(0, int(round(bull_pred)))

        # Общий падеж = сумма
        total_pred = heifer_pred + bull_pred

        # Сохраняем прогноз
        pred_result = {
            'год': year,
            'месяц': month,
            'падеж_всего': total_pred,
            'падеж_телочки': heifer_pred,
            'падеж_бычки': bull_pred
        }
        all_predictions.append(pred_result)
        prev_preds.append(pred_result)

# ============================================
# 7. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ПРОГНОЗА ПАДЕЖА (ОКТ 2024 - ДЕК 2025)")
print("="*80)

print(f"\n{'Месяц':<12} {'Падеж':>8} {'Телочки':>8} {'Бычки':>8} {'Отелы':>8} {'% падежа':>10}")
print("-" * 60)

for pred in all_predictions:
    year = pred['год']
    month = pred['месяц']
    total_deaths = pred['падеж_всего']
    heifer_deaths = pred['падеж_телочки']
    bull_deaths = pred['падеж_бычки']
    otely = calving_forecast.get((year, month), 250)
    death_rate = total_deaths / otely * 100

    print(f"{months_ru[month]} {year:<4} {total_deaths:>8} {heifer_deaths:>8} {bull_deaths:>8} {otely:>8} {death_rate:>9.1f}%")

# ============================================
# 8. СТАТИСТИКА ПРОГНОЗА
# ============================================
print("\n" + "="*80)
print("СТАТИСТИКА ПРОГНОЗА")
print("="*80)

total_pred_deaths = sum(p['падеж_всего'] for p in all_predictions)
total_pred_heifers = sum(p['падеж_телочки'] for p in all_predictions)
total_pred_bulls = sum(p['падеж_бычки'] for p in all_predictions)
total_otely = sum(calving_forecast.get((p['год'], p['месяц']), 0) for p in all_predictions)

print(f"\nПрогноз на период Окт 2024 - Дек 2025 (15 месяцев):")
print(f"  Всего падежей: {total_pred_deaths}")
print(f"  Из них телочки: {total_pred_heifers} ({total_pred_heifers/total_pred_deaths*100:.1f}%)")
print(f"  Из них бычки: {total_pred_bulls} ({total_pred_bulls/total_pred_deaths*100:.1f}%)")
print(f"  Всего отелов: {total_otely}")
print(f"  Общий процент падежа: {total_pred_deaths/total_otely*100:.2f}%")

# Проверка: сумма телочек и бычков = общий падеж
print(f"\n✅ Проверка: {total_pred_heifers} + {total_pred_bulls} = {total_pred_heifers + total_pred_bulls}")
print(f"   Общий падеж: {total_pred_deaths}")
print(f"   Совпадает: {'✅ ДА' if total_pred_heifers + total_pred_bulls == total_pred_deaths else '❌ НЕТ'}")

print("\n" + "="*80)
print("✅ Прогноз падежа завершен!")
print("="*80)

# ===== NOTEBOOK CELL 20 =====
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Русские названия месяцев
months_ru = {
    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
}

print("="*80)
print("ПРОГНОЗ ПОГОЛОВЬЯ ПО ЛАКТАЦИЯМ (L1, L2, L3, L4, L5+)")
print("  • обучение: январь 2022 - сентябрь 2024")
print("  • прогноз: октябрь 2024 - декабрь 2025")
print("  • сумма L1+L2+L3+L4+L5+ = фуражные коровы")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

# 1.1 Фактическое поголовье по лактациям (из предыдущего расчета)
df_lact_actual = pd.read_excel("поголовье_по_лактациям_январь2022_декабрь2025.xlsx")
print(f"\n✅ Загружены фактические данные по лактациям: {len(df_lact_actual)} месяцев")

# 1.2 Прогноз фуражных коров
furazh_forecast = {
    (2024, 9): 2909,
    (2024, 10): 2936,
    (2024, 11): 2944,
    (2024, 12): 2918,
    (2025, 1): 2942,
    (2025, 2): 2965,
    (2025, 3): 2973,
    (2025, 4): 2991,
    (2025, 5): 2997,
    (2025, 6): 3001,
    (2025, 7): 3005,
    (2025, 8): 3005,
    (2025, 9): 3006,
    (2025, 10): 3025,
    (2025, 11): 3019,
    (2025, 12): 3009
}

print("\n" + "="*80)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
print("="*80)

# ============================================
# 2. ПОДГОТОВКА ОБУЧАЮЩИХ ДАННЫХ (январь 2022 - сентябрь 2024)
# ============================================

# Берем только данные до сентября 2024
train_df = df_lact_actual[
    ~((df_lact_actual['год'] == 2024) & (df_lact_actual['месяц'] > 9))
].copy()

# Добавляем фуражных (L1+L2+L3+L4+L5+)
train_df['фуражные'] = train_df['L1'] + train_df['L2'] + train_df['L3'] + train_df['L4'] + train_df['L5+']

print(f"Обучающих месяцев: {len(train_df)}")

# ============================================
# 3. ФУНКЦИЯ СОЗДАНИЯ ПРИЗНАКОВ
# ============================================
def create_features_lact(df):
    df = df.copy()
    
    # Сезонность
    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)
    df['тренд'] = range(1, len(df) + 1)
    
    lact_cols = ['L1', 'L2', 'L3', 'L4', 'L5+']
    
    # Лаги для каждой лактации
    for col in lact_cols:
        for lag in [1, 2, 3, 6]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
        df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean().fillna(0)
        df[f'{col}_ma6'] = df[col].rolling(6, min_periods=1).mean().fillna(0)
    
    # Общее количество фуражных
    df['фуражные_lag1'] = df['фуражные'].shift(1).fillna(0)
    df['фуражные_lag2'] = df['фуражные'].shift(2).fillna(0)
    df['фуражные_lag3'] = df['фуражные'].shift(3).fillna(0)
    df['фуражные_ma3'] = df['фуражные'].rolling(3, min_periods=1).mean().fillna(0)
    
    # Доли каждой лактации в фуражных
    for col in lact_cols:
        df[f'доля_{col}'] = df[col] / (df['фуражные'] + 1)
        df[f'доля_{col}_lag1'] = df[f'доля_{col}'].shift(1).fillna(0)
    
    return df

train_features = create_features_lact(train_df)
train_clean = train_features.dropna()

# Список признаков - исключаем L0 и всего
feature_cols = [col for col in train_clean.columns if col not in [
    'год', 'месяц', 'L0', 'всего', 'L1', 'L2', 'L3', 'L4', 'L5+', 'фуражные',
    'доля_L1', 'доля_L2', 'доля_L3', 'доля_L4', 'доля_L5+'
]]

lact_cols = ['L1', 'L2', 'L3', 'L4', 'L5+']

X_train = train_clean[feature_cols]

print(f"Обучение на {len(train_clean)} месяцах, признаков: {len(feature_cols)}")

# ============================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ КАЖДОЙ ЛАКТАЦИИ
# ============================================
print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*80)

models = {}
for target in lact_cols:
    print(f"\nОбучение модели для {target}...")
    y_train = train_clean[target].values
    
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    try:
        tscv = TimeSeriesSplit(n_splits=min(3, len(train_clean)-1))
        grid = GridSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            param_grid, cv=tscv, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        print(f"  Лучшие параметры: {best_params}")
    except:
        best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 
                       'subsample': 0.8, 'colsample_bytree': 0.8}
        print(f"  Используем параметры по умолчанию")
    
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model
    
    pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_train, pred_train)
    print(f"  MAE на обучении: {mae:.2f}")

# ============================================
# 5. РЕКУРСИВНЫЙ ПРОГНОЗ
# ============================================
print("\n" + "="*80)
print("РЕКУРСИВНЫЙ ПРОГНОЗ (октябрь 2024 - декабрь 2025)")
print("="*80)

# Словари для хранения прогнозов
forecasts = {target: {} for target in lact_cols}

# Исторические средние по месяцам
monthly_avg = {}
for target in lact_cols:
    monthly_avg[target] = train_df.groupby('месяц')[target].mean().to_dict()

predict_months = []
for month in [10, 11, 12]:
    predict_months.append((2024, month))
for month in range(1, 13):
    predict_months.append((2025, month))

results = []

# Начальные значения (сентябрь 2024) - берем из фактических данных
last_actual = train_df[train_df['год'] == 2024][train_df['месяц'] == 9]
if len(last_actual) > 0:
    prev_values = {
        'L1': last_actual['L1'].values[0],
        'L2': last_actual['L2'].values[0],
        'L3': last_actual['L3'].values[0],
        'L4': last_actual['L4'].values[0],
        'L5+': last_actual['L5+'].values[0]
    }
else:
    prev_values = {'L1': 633, 'L2': 834, 'L3': 527, 'L4': 334, 'L5+': 266}

print(f"\nБазовые значения на сентябрь 2024:")
for lact, val in prev_values.items():
    print(f"  {lact}: {val}")

print("\n" + "="*80)
print("ПРОГНОЗ ПО МЕСЯЦАМ")
print("="*80)

for year, month in predict_months:
    print(f"\nПрогноз на {months_ru[month]} {year}:")
    
    # Создаем строку для прогноза
    pred_row = {}
    pred_row['год'] = year
    pred_row['месяц'] = month
    pred_row['месяц_синус'] = np.sin(2 * np.pi * month / 12)
    pred_row['месяц_косинус'] = np.cos(2 * np.pi * month / 12)
    pred_row['квартал'] = (month-1)//3 + 1
    pred_row['тренд'] = len(train_features) + len(results) + 1
    
    # Добавляем фуражные (для признаков)
    furazh = furazh_forecast.get((year, month), 0)
    pred_row['фуражные'] = furazh
    pred_row['фуражные_lag1'] = furazh_forecast.get((year, month-1 if month > 1 else 12), 0)
    pred_row['фуражные_lag2'] = furazh_forecast.get((year, month-2 if month > 2 else 11), 0)
    pred_row['фуражные_lag3'] = furazh_forecast.get((year, month-3 if month > 3 else 10), 0)
    pred_row['фуражные_ma3'] = (pred_row['фуражные_lag1'] + pred_row['фуражные_lag2'] + pred_row['фуражные_lag3']) / 3
    
    # Добавляем лаги для каждой лактации (из прогнозов)
    for target in lact_cols:
        # lag1
        if month > 1:
            pred_row[f'{target}_lag1'] = forecasts[target].get((year, month-1),
                                      forecasts[target].get((year-1, 12), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag1'] = 0
        
        # lag2
        if month > 2:
            pred_row[f'{target}_lag2'] = forecasts[target].get((year, month-2),
                                      forecasts[target].get((year-1, 11), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag2'] = 0
        
        # lag3
        if month > 3:
            pred_row[f'{target}_lag3'] = forecasts[target].get((year, month-3),
                                      forecasts[target].get((year-1, 10), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag3'] = monthly_avg[target].get(month, 0)
        
        # lag6
        if month > 6:
            pred_row[f'{target}_lag6'] = forecasts[target].get((year, month-6),
                                      forecasts[target].get((year-1, 6), monthly_avg[target].get(month, 0)))
        else:
            pred_row[f'{target}_lag6'] = monthly_avg[target].get(month, 0)
        
        # MA3
        lag1 = pred_row[f'{target}_lag1']
        lag2 = pred_row[f'{target}_lag2']
        lag3 = pred_row[f'{target}_lag3']
        pred_row[f'{target}_ma3'] = (lag1 + lag2 + lag3) / 3 if (lag1 + lag2 + lag3) > 0 else 0
        
        # MA6
        lag6 = pred_row[f'{target}_lag6']
        pred_row[f'{target}_ma6'] = (lag1 + lag2 + lag3 + lag6 + lag6 + lag6) / 6 if (lag1 + lag2 + lag3 + lag6) > 0 else 0
        
        # Доля в фуражных
        pred_row[f'доля_{target}'] = pred_row.get(f'{target}_lag1', 0) / (furazh + 1)
        pred_row[f'доля_{target}_lag1'] = pred_row[f'доля_{target}']
    
    # Прогнозируем для каждой лактации
    pred_df = pd.DataFrame([pred_row])
    X_pred = pred_df[feature_cols]
    X_pred = X_pred.fillna(0)
    
    pred_values_raw = {}
    for target in lact_cols:
        val = models[target].predict(X_pred)[0]
        pred_values_raw[target] = max(0, int(round(val)))
    
    # ====== КОРРЕКТИРОВКА: сумма = фуражные ======
    total_pred = sum(pred_values_raw.values())
    
    if total_pred > 0 and total_pred != furazh:
        # Пропорционально корректируем
        scale = furazh / total_pred
        for target in lact_cols:
            pred_values_raw[target] = max(0, int(round(pred_values_raw[target] * scale)))
        
        # Добиваем остаток (чтобы сумма точно равнялась фуражным)
        total_adj = sum(pred_values_raw.values())
        diff = furazh - total_adj
        if diff != 0:
            # Добавляем остаток к самой большой лактации
            max_lact = max(pred_values_raw, key=pred_values_raw.get)
            pred_values_raw[max_lact] += diff
    
    # Сохраняем прогнозы
    for target in lact_cols:
        forecasts[target][(year, month)] = pred_values_raw[target]
    
    # Проверка суммы
    final_total = sum(pred_values_raw.values())
    
    results.append({
        'год': year,
        'месяц': month,
        'прогноз': pred_values_raw.copy(),
        'фуражные': furazh,
        'сумма': final_total
    })
    
    print(f"  Прогноз: L1={pred_values_raw['L1']}, L2={pred_values_raw['L2']}, "
          f"L3={pred_values_raw['L3']}, L4={pred_values_raw['L4']}, L5+={pred_values_raw['L5+']}")
    print(f"  Сумма: {final_total}, Фуражные: {furazh}, Разница: {final_total - furazh}")

# ============================================
# 6. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗА")
print("="*80)

header = f"{'Месяц':<12} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5+':>8} {'Сумма':>8} {'Фуражные':>10}"
print(header)
print("-" * 80)

for r in results:
    month_name = f"{months_ru[r['месяц']]}{str(r['год'])[-2:]}"
    p = r['прогноз']
    print(f"{month_name:<12} "
          f"{p['L1']:>8} "
          f"{p['L2']:>8} "
          f"{p['L3']:>8} "
          f"{p['L4']:>8} "
          f"{p['L5+']:>8} "
          f"{r['сумма']:>8} "
          f"{r['фуражные']:>10}")

print("-" * 80)

# ============================================
# 7. СТАТИСТИКА
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*80)

for target in lact_cols:
    values = [r['прогноз'][target] for r in results]
    print(f"\n{target}:")
    print(f"  Среднее: {np.mean(values):.0f}")
    print(f"  Минимум: {min(values)}")
    print(f"  Максимум: {max(values)}")
    print(f"  Сумма: {sum(values)}")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • Обучение: январь 2022 - сентябрь 2024")
print("  • Прогноз: октябрь 2024 - декабрь 2025")
print("  • Сумма L1+L2+L3+L4+L5+ = фуражные (скорректировано)")
print("="*80)

# ===== NOTEBOOK CELL 23 =====
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ПОЛНЫЙ ОТЧЕТ ПО ЖК ВЫСОКОМУ")
print("Период: январь 2022 - декабрь 2025")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
df = pd.read_excel('События-по-коровам.xlsx')
print(f"\n✅ Загружен файл: {len(df):,} строк")

df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
df['BDAT'] = pd.to_datetime(df['BDAT'], errors='coerce')
df = df.sort_values(['ключ_коровы', 'Дата']).reset_index(drop=True)

# ============================================
# 2. ФУНКЦИИ ДЛЯ ОПРЕДЕЛЕНИЯ ВОЗРАСТА
# ============================================
def get_age_group_exact(birth_date, event_date):
    if pd.isna(birth_date) or pd.isna(event_date):
        return None
    
    months_diff = (event_date.year - birth_date.year) * 12 + (event_date.month - birth_date.month)
    
    if event_date.day < birth_date.day:
        months_diff -= 1
    
    if months_diff < 0:
        return None
    elif months_diff < 2:
        return '0-2'
    elif months_diff < 6:
        return '2-6'
    elif months_diff < 12:
        return '6-12'
    elif months_diff < 18:
        return '12-18'
    else:
        return '18+'

def get_age_group_bulls(birth_date, event_date):
    if pd.isna(birth_date) or pd.isna(event_date):
        return None
    
    months_diff = (event_date.year - birth_date.year) * 12 + (event_date.month - birth_date.month)
    
    if event_date.day < birth_date.day:
        months_diff -= 1
    
    if months_diff < 0:
        return None
    elif months_diff < 2:
        return '0-2'
    elif months_diff < 6:
        return '2-6'
    else:
        return '0-6'

# ============================================
# 3. СОЗДАЕМ СЛОВАРЬ С ИНФОРМАЦИЕЙ О КАЖДОЙ КОРОВЕ
# ============================================
print("\n📊 Сбор информации о животных...")

cow_info = {}
for cow_key, group in df.groupby('ключ_коровы'):
    group = group.sort_values('Дата')
    if len(group) == 0:
        continue
    
    bdat = group.iloc[0]['BDAT']
    if pd.isna(bdat):
        continue
    
    has_success = False
    for _, row in group.iterrows():
        event = str(row.get('Событие', '')).strip()
        r_val = str(row.get('R', '')).strip()
        if event == 'ОСЕМЕН' and r_val == 'P':
            has_success = True
            break
    
    cow_info[cow_key] = {
        'bdat': bdat,
        'has_success': has_success
    }

print(f"  Информация о {len(cow_info):,} животных")

# ============================================
# 4. ПРОДАЖА ТЕЛОК (ТОЧНО КАК В КОДЕ1)
#    Условия: Столбец1 = 'ЖК Высокое', Куда = 'МТФ_ВЫСОКОЕ', LACT = 0
# ============================================
print("\n📊 Продажа телок (МТФ_ВЫСОКОЕ, LACT=0)...")

# Фильтруем только ЖК Высокое
df_filtered = df[df['Столбец1'] == 'ЖК Высокое'].copy()

# Фильтруем: Куда = МТФ_ВЫСОКОЕ (и его вариации)
mask_kuda = (
    df_filtered['Куда'].str.upper().str.strip().isin(['МТФ_ВЫСОКОЕ', 'МТФ ВЫСОКОЕ']) |
    df_filtered['Куда'].str.upper().str.strip().str.contains('МТФ_ВЫСОКОЕ', na=False) |
    df_filtered['Куда'].str.upper().str.strip().str.contains('МТФ ВЫСОКОЕ', na=False)
)

# Фильтруем: LACT = 0 (как в коде1!)
mask_lact = df_filtered['LACT'] == 0

df_heifers_sold = df_filtered[mask_kuda & mask_lact].copy()

print(f"  Найдено записей: {len(df_heifers_sold)}")

# Расчет возраста (как в коде1)
def get_animal_age(row):
    cow_key = row['ключ_коровы']
    event_date = row['Дата']
    
    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')
    if len(cow_events) == 0:
        return None
    
    bdat = cow_events.iloc[0]['BDAT']
    if pd.isna(bdat):
        return None
    
    return get_age_group_exact(bdat, event_date)

df_heifers_sold['возраст_группа'] = df_heifers_sold.apply(get_animal_age, axis=1)
df_heifers_sold = df_heifers_sold[df_heifers_sold['возраст_группа'].notna()].copy()
df_heifers_sold['год'] = df_heifers_sold['Дата'].dt.year
df_heifers_sold['месяц'] = df_heifers_sold['Дата'].dt.month

# ============================================
# 5. ПРОДАЖА БЫЧКОВ (из кода2)
#    Условия: Столбец1 = 'ЖК Высокое', Куда = 'БЫЧКИ', LACT = 0
# ============================================
print("\n📊 Продажа бычков (БЫЧКИ, LACT=0)...")

df_bulls_sold = df[
    (df['Столбец1'] == 'ЖК Высокое') &
    (df['Куда'].str.upper().str.contains('БЫЧКИ', na=False)) &
    (df['LACT'] == 0)
].copy()

print(f"  Найдено: {len(df_bulls_sold)}")

def get_animal_age_bulls(row):
    cow_key = row['ключ_коровы']
    event_date = row['Дата']
    
    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')
    if len(cow_events) == 0:
        return None
    
    bdat = cow_events.iloc[0]['BDAT']
    if pd.isna(bdat):
        return None
    
    return get_age_group_bulls(bdat, event_date)

df_bulls_sold['возраст_группа'] = df_bulls_sold.apply(get_animal_age_bulls, axis=1)
df_bulls_sold = df_bulls_sold[df_bulls_sold['возраст_группа'].notna()].copy()
df_bulls_sold['год'] = df_bulls_sold['Дата'].dt.year
df_bulls_sold['месяц'] = df_bulls_sold['Дата'].dt.month

# ============================================
# 6. ПОКУПКА (LACT=0, Событие = ВЫБЫТИЕ/ПРОДАНА, Куда в списке)
#    Разделяем на телок и нетелей по наличию P
# ============================================
print("\n📊 Покупка (LACT=0, выбытие, Куда в списке)...")

# Фильтруем события выбытия
df_exit = df[
    (df['LACT'] == 0) &
    (df['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА']))
].copy()

# Фильтруем по Куда
kuda_list = ['ЖК_ВЫСОК', 'ЖК_ВЫСОКОЕ', 'ЖКВЫСОК', 'ЖКВЫСОКО', 'ЖКВЫСОКОЕ']
mask_kuda_buy = df_exit['Куда'].str.upper().str.strip().apply(
    lambda x: any(v in x for v in [k.upper() for k in kuda_list]) if isinstance(x, str) else False
)
df_buy = df_exit[mask_kuda_buy].copy()

print(f"  Найдено записей: {len(df_buy)}")

# Определяем стельность
df_buy['is_pregnant'] = df_buy['ключ_коровы'].apply(
    lambda cow_key: cow_info.get(cow_key, {}).get('has_success', False)
)

# Разделяем
df_pregnant = df_buy[df_buy['is_pregnant'] == True].copy()      # Нетели
df_heifers_buy = df_buy[df_buy['is_pregnant'] == False].copy()  # Телки

print(f"  Нетелей (с P): {len(df_pregnant)}")
print(f"  Телки (без P): {len(df_heifers_buy)}")

# Расчет возраста для покупных телок
def get_animal_age_buy(row):
    cow_key = row['ключ_коровы']
    event_date = row['Дата']
    
    cow_events = df[df['ключ_коровы'] == cow_key].sort_values('Дата')
    if len(cow_events) == 0:
        return None
    
    bdat = cow_events.iloc[0]['BDAT']
    if pd.isna(bdat):
        return None
    
    return get_age_group_exact(bdat, event_date)

df_heifers_buy['возраст_группа'] = df_heifers_buy.apply(get_animal_age_buy, axis=1)
df_heifers_buy = df_heifers_buy[df_heifers_buy['возраст_группа'].notna()].copy()
df_heifers_buy['год'] = df_heifers_buy['Дата'].dt.year
df_heifers_buy['месяц'] = df_heifers_buy['Дата'].dt.month

# Для нетелей просто год/месяц
df_pregnant['год'] = df_pregnant['Дата'].dt.year
df_pregnant['месяц'] = df_pregnant['Дата'].dt.month

# ============================================
# 7. ФОРМИРУЕМ ИТОГОВУЮ ТАБЛИЦУ
# ============================================
print("\n📊 Формирование таблицы...")

all_months = []
for year in range(2022, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 12:
            break
        all_months.append((year, month))

age_groups = ['0-2', '2-6', '6-12', '12-18', '18+']
bull_groups = ['0-2', '2-6', '0-6']

data = []
for year, month in all_months:
    row = {'год': year, 'месяц': month}
    
    # 1. Продажа телок (МТФ_ВЫСОКОЕ, LACT=0)
    for age in age_groups:
        val = len(df_heifers_sold[
            (df_heifers_sold['год'] == year) & 
            (df_heifers_sold['месяц'] == month) & 
            (df_heifers_sold['возраст_группа'] == age)
        ])
        row[f'продажа_телки_{age}_внутри'] = val
    
    # 2. Продажа бычков (БЫЧКИ, LACT=0)
    for age in bull_groups:
        val = len(df_bulls_sold[
            (df_bulls_sold['год'] == year) & 
            (df_bulls_sold['месяц'] == month) & 
            (df_bulls_sold['возраст_группа'] == age)
        ])
        row[f'продажа_бычки_{age}_внутри'] = val
    
    # 3. Покупка телок (без P)
    for age in age_groups:
        val = len(df_heifers_buy[
            (df_heifers_buy['год'] == year) & 
            (df_heifers_buy['месяц'] == month) & 
            (df_heifers_buy['возраст_группа'] == age)
        ])
        row[f'покупка_телки_{age}_внутри'] = val
    
    # 4. Покупка нетелей (с P)
    val = len(df_pregnant[
        (df_pregnant['год'] == year) & 
        (df_pregnant['месяц'] == month)
    ])
    row['покупка_нетели_внутри'] = val
    
    data.append(row)

df_results = pd.DataFrame(data)

# ============================================
# 8. ИТОГОВАЯ СТАТИСТИКА
# ============================================
print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА:")
print("="*80)

print(f"\nВсего месяцев: {len(df_results)}")
print(f"Период: {df_results['год'].min()}-{df_results['месяц'].min()} по {df_results['год'].max()}-{df_results['месяц'].max()}")

print("\n📊 ПРОДАЖА ТЕЛОК (МТФ_ВЫСОКОЕ, LACT=0):")
for age in age_groups:
    col = f'продажа_телки_{age}_внутри'
    print(f"  {col}: {df_results[col].sum()}")

print("\n📊 ПРОДАЖА БЫЧКОВ (БЫЧКИ, LACT=0):")
for age in bull_groups:
    col = f'продажа_бычки_{age}_внутри'
    print(f"  {col}: {df_results[col].sum()}")

print("\n📊 ПОКУПКА ТЕЛОК (без P, LACT=0):")
for age in age_groups:
    col = f'покупка_телки_{age}_внутри'
    print(f"  {col}: {df_results[col].sum()}")

print(f"\n📊 ПОКУПКА НЕТЕЛЕЙ (с P, LACT=0):")
print(f"  покупка_нетели_внутри: {df_results['покупка_нетели_внутри'].sum()}")

# ============================================
# 9. СОХРАНЕНИЕ
# ============================================
output_file = "полный_отчет_ЖК_ВЫСОКОЕ.xlsx"
df_results.to_excel(output_file, index=False)
print(f"\n✅ Результаты сохранены в файл: {output_file}")

print("\n" + "="*80)
print("ГОТОВО!")
print("  • Продажа телок: МТФ_ВЫСОКОЕ, LACT=0 (как в коде1)")
print("  • Продажа бычков: БЫЧКИ, LACT=0 (как в коде2)")
print("  • Покупка телок: ЖК_ВЫСОК..., LACT=0, без P")
print("  • Покупка нетелей: ЖК_ВЫСОК..., LACT=0, с P")
print("="*80)

# Показываем пример таблицы
print("\n📋 ПРИМЕР ТАБЛИЦЫ (первые 5 строк):")
print(df_results.head().to_string(index=False))

# ===== NOTEBOOK CELL 25 =====
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

print("="*80)
print("ПРОГНОЗ ПАРАМЕТРОВ ЖК ВЫСОКОЕ (КОМПРОМИССНАЯ ВЕРСИЯ)")
print("  • Обучение: 2022 - сентябрь 2024")
print("  • Прогноз: октябрь 2024 - декабрь 2025")
print("  • 13 целевых параметров")
print("  • Лаги: 1,2,3,6,12 месяцев")
print("="*80)

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
df = pd.read_excel('События-по-коровам.xlsx')
print(f"\n✅ Загружен файл: {len(df):,} строк")

df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
df['BDAT'] = pd.to_datetime(df['BDAT'], errors='coerce')
df = df.sort_values(['ключ_коровы', 'Дата']).reset_index(drop=True)

# ============================================
# 2. ФУНКЦИИ ДЛЯ РАСЧЕТА ВОЗРАСТА
# ============================================
def get_age_group_exact(birth_date, event_date):
    if pd.isna(birth_date) or pd.isna(event_date):
        return None
    months_diff = (event_date.year - birth_date.year) * 12 + (event_date.month - birth_date.month)
    if event_date.day < birth_date.day:
        months_diff -= 1
    if months_diff < 0:
        return None
    elif months_diff < 2:
        return '0-2'
    elif months_diff < 6:
        return '2-6'
    elif months_diff < 12:
        return '6-12'
    elif months_diff < 18:
        return '12-18'
    else:
        return '18+'

def get_age_group_bulls(birth_date, event_date):
    if pd.isna(birth_date) or pd.isna(event_date):
        return None
    months_diff = (event_date.year - birth_date.year) * 12 + (event_date.month - birth_date.month)
    if event_date.day < birth_date.day:
        months_diff -= 1
    if months_diff < 0:
        return None
    elif months_diff < 2:
        return '0-2'
    elif months_diff < 6:
        return '2-6'
    else:
        return '0-6'

# ============================================
# 3. РАСЧЕТ ПАРАМЕТРОВ ПО МЕСЯЦАМ
# ============================================
def calculate_parameters(df):
    print("\n📊 Расчет параметров...")
    
    # Словарь с информацией о коровах
    cow_info = {}
    for cow_key, group in df.groupby('ключ_коровы'):
        group = group.sort_values('Дата')
        if len(group) == 0:
            continue
        bdat = group.iloc[0]['BDAT']
        if pd.isna(bdat):
            continue
        has_success = False
        for _, row in group.iterrows():
            event = str(row.get('Событие', '')).strip()
            r_val = str(row.get('R', '')).strip()
            if event == 'ОСЕМЕН' and r_val == 'P':
                has_success = True
                break
        cow_info[cow_key] = {'bdat': bdat, 'has_success': has_success}
    
    # Списки для фильтрации
    kuda_list = ['ЖК_ВЫСОК', 'ЖК_ВЫСОКОЕ', 'ЖКВЫСОК', 'ЖКВЫСОКО', 'ЖКВЫСОКОЕ']
    kuda_heifers = ['МТФ_ВЫСОКОЕ', 'МТФ ВЫСОКОЕ', 'МТФВЫСОКОЕ', 'МТФ_ВЫСОК', 'МТФ ВЫСОК', 'МТФВЫСОК']
    
    # Фильтруем события выбытия
    mask_exit = df['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА'])
    df_exit = df[mask_exit].copy()
    df_exit['год'] = df_exit['Дата'].dt.year
    df_exit['месяц'] = df_exit['Дата'].dt.month
    
    params = []
    
    for year in range(2022, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 12:
                break
            
            row = {'год': year, 'месяц': month}
            
            # Инициализация
            for age in ['0-2', '2-6', '6-12', '12-18', '18+']:
                row[f'продажа_телки_{age}_внутри'] = 0
                row[f'покупка_телки_{age}_внутри'] = 0
            row['продажа_бычки_0-6_внутри'] = 0
            row['покупка_нетели_внутри'] = 0
            
            month_events = df_exit[(df_exit['год'] == year) & (df_exit['месяц'] == month)]
            
            for _, event in month_events.iterrows():
                cow_key = event['ключ_коровы']
                info = cow_info.get(cow_key, {})
                bdat = info.get('bdat')
                has_success = info.get('has_success', False)
                kuda = str(event['Куда']).upper().strip()
                lact = event.get('LACT', 0)
                event_date = event['Дата']
                
                # Бычки
                if lact == 0 and 'БЫЧКИ' in kuda:
                    age = get_age_group_bulls(bdat, event_date)
                    if age == '0-6':
                        row['продажа_бычки_0-6_внутри'] += 1
                    continue
                
                # Покупка нетелей/телок
                if lact == 0 and any(v in kuda for v in [x.upper() for x in kuda_list]):
                    if has_success:
                        row['покупка_нетели_внутри'] += 1
                    else:
                        age = get_age_group_exact(bdat, event_date)
                        if age:
                            row[f'покупка_телки_{age}_внутри'] += 1
                    continue
                
                # Продажа телок
                if lact == 0 and any(v in kuda for v in [x.upper() for x in kuda_heifers]):
                    age = get_age_group_exact(bdat, event_date)
                    if age:
                        row[f'продажа_телки_{age}_внутри'] += 1
                    continue
            
            params.append(row)
    
    df_params = pd.DataFrame(params)
    print(f"  Рассчитано {len(df_params)} месяцев")
    return df_params

df_params = calculate_parameters(df)

# ============================================
# 4. ДОБАВЛЕНИЕ ПРИЗНАКОВ (КОМПРОМИССНАЯ ВЕРСИЯ)
# ============================================
def add_features(df_params, df_raw):
    print("\n📊 Добавление признаков...")
    
    df = df_params.copy()
    
    # ===== 1. БАЗОВЫЕ ПРИЗНАКИ =====
    df['месяц_синус'] = np.sin(2 * np.pi * df['месяц'] / 12)
    df['месяц_косинус'] = np.cos(2 * np.pi * df['месяц'] / 12)
    df['квартал'] = df['месяц'].apply(lambda x: (x-1)//3 + 1)
    df['год_цифра'] = df['год'] - 2021
    df['тренд'] = range(1, len(df) + 1)
    
    # ===== 2. ПРИЗНАКИ ИЗ RAW DATA =====
    df_raw['год'] = df_raw['Дата'].dt.year
    df_raw['месяц'] = df_raw['Дата'].dt.month
    
    # 2.1 Осеменения (всего и успешные)
    semen = df_raw[df_raw['Событие'].str.strip() == 'ОСЕМЕН']
    semen_monthly = semen.groupby(['год', 'месяц']).size().reset_index(name='осеменения_всего')
    semen_p = semen[semen['R'].str.strip() == 'P'].groupby(['год', 'месяц']).size().reset_index(name='осеменения_успешные')
    semen_failed = semen[~semen['R'].str.strip().isin(['P'])].groupby(['год', 'месяц']).size().reset_index(name='осеменения_неуспешные')
    
    df = df.merge(semen_monthly, on=['год', 'месяц'], how='left').fillna(0)
    df = df.merge(semen_p, on=['год', 'месяц'], how='left').fillna(0)
    df = df.merge(semen_failed, on=['год', 'месяц'], how='left').fillna(0)
    df['осеменения_процент_успеха'] = (df['осеменения_успешные'] / (df['осеменения_всего'] + 1) * 100)
    
    # 2.2 Отелы
    calvings = df_raw[df_raw['Событие'].str.strip().isin(['ОТЕЛ', 'ОТЁЛ', 'CALVING', 'ОТЕЛЕНИЕ'])]
    calvings_monthly = calvings.groupby(['год', 'месяц']).size().reset_index(name='отелы')
    df = df.merge(calvings_monthly, on=['год', 'месяц'], how='left').fillna(0)
    
    # 2.3 Запуски
    dry = df_raw[df_raw['Событие'].str.strip() == 'ЗАПУСК']
    dry_monthly = dry.groupby(['год', 'месяц']).size().reset_index(name='запуски')
    df = df.merge(dry_monthly, on=['год', 'месяц'], how='left').fillna(0)
    
    # 2.4 Выбытия
    exits = df_raw[df_raw['Событие'].str.strip().isin(['ВЫБЫТИЕ', 'ПРОДАНА'])]
    exits_monthly = exits.groupby(['год', 'месяц']).size().reset_index(name='выбытия_всего')
    df = df.merge(exits_monthly, on=['год', 'месяц'], how='left').fillna(0)
    
    exits_lact0 = exits[exits['LACT'] == 0].groupby(['год', 'месяц']).size().reset_index(name='выбытия_lact0')
    exits_lact1 = exits[exits['LACT'] == 1].groupby(['год', 'месяц']).size().reset_index(name='выбытия_lact1')
    df = df.merge(exits_lact0, on=['год', 'месяц'], how='left').fillna(0)
    df = df.merge(exits_lact1, on=['год', 'месяц'], how='left').fillna(0)
    
    # ===== 3. ЛАГИ (1,2,3,6,12) ДЛЯ ВСЕХ ПАРАМЕТРОВ =====
    target_cols = [
        'продажа_телки_0-2_внутри', 'продажа_телки_2-6_внутри', 
        'продажа_телки_6-12_внутри', 'продажа_телки_12-18_внутри', 'продажа_телки_18+_внутри',
        'продажа_бычки_0-6_внутри',
        'покупка_телки_0-2_внутри', 'покупка_телки_2-6_внутри',
        'покупка_телки_6-12_внутри', 'покупка_телки_12-18_внутри', 'покупка_телки_18+_внутри',
        'покупка_нетели_внутри'
    ]
    
    feature_cols = [
        'отелы', 'запуски', 'осеменения_всего', 'осеменения_успешные', 
        'осеменения_неуспешные', 'осеменения_процент_успеха',
        'выбытия_всего', 'выбытия_lact0', 'выбытия_lact1'
    ]
    
    # Лаги для целевых параметров
    for col in target_cols:
        for lag in [1, 2, 3, 6, 12]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for window in [3, 6, 12]:
            df[f'{col}_ma{window}'] = df[col].rolling(window, min_periods=1).mean()
    
    # Лаги для дополнительных признаков
    for col in feature_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            for window in [3, 6, 12]:
                df[f'{col}_ma{window}'] = df[col].rolling(window, min_periods=1).mean()
    
    # ===== 4. СУММАРНЫЕ ПРИЗНАКИ =====
    df['продажа_телки_всего'] = df[[f'продажа_телки_{age}_внутри' for age in ['0-2', '2-6', '6-12', '12-18', '18+']]].sum(axis=1)
    df['покупка_телки_всего'] = df[[f'покупка_телки_{age}_внутри' for age in ['0-2', '2-6', '6-12', '12-18', '18+']]].sum(axis=1)
    df['соотношение_продажи_покупки'] = df['продажа_телки_всего'] / (df['покупка_телки_всего'] + 1)
    
    for col in ['продажа_телки_всего', 'покупка_телки_всего', 'соотношение_продажи_покупки']:
        for lag in [1, 2, 3, 6, 12]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for window in [3, 6, 12]:
            df[f'{col}_ma{window}'] = df[col].rolling(window, min_periods=1).mean()
    
    print(f"  Всего признаков: {len(df.columns)}")
    return df

df_features = add_features(df_params, df)

# ============================================
# 5. РАЗДЕЛЕНИЕ НА ОБУЧЕНИЕ И ПРОГНОЗ
# ============================================
print("\n📊 Разделение данных...")

train_data = df_features[
    (df_features['год'] < 2024) | 
    ((df_features['год'] == 2024) & (df_features['месяц'] <= 9))
].copy()

test_data = df_features[
    ((df_features['год'] == 2024) & (df_features['месяц'] >= 10)) |
    (df_features['год'] == 2025)
].copy()

print(f"  Обучающих месяцев: {len(train_data)}")
print(f"  Тестовых месяцев: {len(test_data)}")

# ============================================
# 6. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================
print("\n📊 Обучение моделей...")

target_cols = [
    'продажа_телки_0-2_внутри', 'продажа_телки_2-6_внутри', 
    'продажа_телки_6-12_внутри', 'продажа_телки_12-18_внутри', 'продажа_телки_18+_внутри',
    'продажа_бычки_0-6_внутри',
    'покупка_телки_0-2_внутри', 'покупка_телки_2-6_внутри',
    'покупка_телки_6-12_внутри', 'покупка_телки_12-18_внутри', 'покупка_телки_18+_внутри',
    'покупка_нетели_внутри'
]

exclude_cols = ['год', 'месяц'] + target_cols
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

print(f"  Всего признаков: {len(feature_cols)}")

train_clean = train_data.dropna()
print(f"  Чистых обучающих строк: {len(train_clean)}")

models = {}
predictions = {}

for target in target_cols:
    print(f"\n  Обучение для: {target}")
    
    X_train = train_clean[feature_cols]
    y_train = train_clean[target]
    
    # Оптимизация гиперпараметров (упрощенная)
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    try:
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
        print(f"    Использую стандартные параметры")
    
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    models[target] = model
    
    X_test = test_data[feature_cols].fillna(0)
    pred = model.predict(X_test)
    pred = np.maximum(0, np.round(pred))
    predictions[target] = pred

# ============================================
# 7. ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
# ============================================
print("\n📊 Формирование результатов...")

results = test_data[['год', 'месяц']].copy()
for target in target_cols:
    results[f'{target}_прогноз'] = predictions[target]
    results[f'{target}_факт'] = test_data[target].values

# ============================================
# 8. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ПРОГНОЗА (суммарно за октябрь 2024 - декабрь 2025):")
print("="*80)

for target in target_cols:
    pred_total = results[f'{target}_прогноз'].sum()
    fact_total = results[f'{target}_факт'].sum()
    error = pred_total - fact_total
    error_pct = (error / fact_total * 100) if fact_total > 0 else 0
    
    status = "✅" if abs(error_pct) <= 20 else "⚠️" if abs(error_pct) <= 50 else "❌"
    print(f"\n{target}:")
    print(f"  Прогноз: {pred_total:.0f}")
    print(f"  Факт: {fact_total:.0f}")
    print(f"  Ошибка: {error:+.0f} ({error_pct:+.1f}%) {status}")

# ============================================
# 9. СОХРАНЕНИЕ
# ============================================
output_file = "прогноз_параметров_ЖК_компромисс.xlsx"
results.to_excel(output_file, index=False)
print(f"\n✅ Результаты сохранены в файл: {output_file}")

print("\n" + "="*80)
print("ГОТОВО!")
print(f"  • Обучающих месяцев: {len(train_clean)}")
print(f"  • Тестовых месяцев: {len(test_data)}")
print(f"  • Признаков: {len(feature_cols)}")
print("="*80)