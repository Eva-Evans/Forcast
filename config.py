import os

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql+psycopg2://herd_user:herd_password@db:5432/herd_forecast",
)

# Finál ML pipeline (prognoz_vseh_parametrov) instead of forecast_dynamic simulation.
USE_FINAL_PIPELINE = os.getenv("USE_FINAL_PIPELINE", "1").strip() not in ("0", "false", "False")

# Months forward from the month of the latest event date (12 + 3).
FORECAST_HORIZON_MONTHS = int(os.getenv("FORECAST_HORIZON_MONTHS", "15"))

SHOW_TAB2_PARAMS = os.getenv("SHOW_TAB2_PARAMS", "0").strip() in ("1", "true", "True")
SHOW_TAB3_LEGACY_FARM_FORECAST = os.getenv("SHOW_TAB3_LEGACY_FARM_FORECAST", "0").strip() in (
    "1",
    "true",
    "True",
)

PIPELINE_WORK_ROOT = os.getenv("PIPELINE_WORK_ROOT", ".pipeline_runtime")
