-- Удалить все загруженные данные (схема таблиц остаётся).
-- Выполните в Supabase SQL Editor или: psql "$POSTGRES_DSN" -f db/wipe_all_data.sql

TRUNCATE TABLE
    tab3_calvings_farm_raw,
    tab3_inseminations_farm_raw,
    tab3_dryoff_farm_raw,
    tab3_disposals_farm_raw,
    tab3_bulls_farm_raw,
    tab3_forecast_cache,
    tab3_subdivision_farm_map,
    tab3_capacity_places,
    model_params_cache,
    calvings_births_raw,
    inseminations_raw,
    dryoff_raw,
    disposals_raw,
    bulls_raw
RESTART IDENTITY CASCADE;
