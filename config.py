from __future__ import annotations

import os
from urllib.parse import quote_plus, urlparse


def _secrets_get(key: str) -> str | None:
    val = os.getenv(key)
    if val and val.strip():
        return val.strip()
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return None


def _build_dsn_from_supabase_parts() -> str | None:
    """Streamlit / Supabase UI sometimes exposes host, user, password separately."""
    host = _secrets_get("SUPABASE_DB_HOST") or _secrets_get("DB_HOST")
    user = _secrets_get("SUPABASE_DB_USER") or _secrets_get("DB_USER") or "postgres"
    password = _secrets_get("SUPABASE_DB_PASSWORD") or _secrets_get("DB_PASSWORD")
    database = _secrets_get("SUPABASE_DB_NAME") or _secrets_get("DB_NAME") or "postgres"
    port = _secrets_get("SUPABASE_DB_PORT") or _secrets_get("DB_PORT") or "5432"
    if not host or not password:
        return None
    safe_pw = quote_plus(password)
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{safe_pw}"
        f"@{host}:{port}/{quote_plus(database)}?sslmode=require"
    )


def normalize_postgres_dsn(dsn: str) -> str:
    dsn = dsn.strip()
    if dsn.startswith("postgres://"):
        dsn = "postgresql+psycopg2://" + dsn[len("postgres://") :]
    elif dsn.startswith("postgresql://"):
        dsn = "postgresql+psycopg2://" + dsn[len("postgresql://") :]
    elif dsn.startswith("postgresql+psycopg2://"):
        pass
    else:
        raise ValueError(
            "POSTGRES_DSN должен начинаться с postgresql:// или postgresql+psycopg2://"
        )

    parsed = urlparse(dsn)
    host = (parsed.hostname or "").lower()
    if "supabase.co" in host and "sslmode=" not in dsn:
        dsn = f"{dsn}&sslmode=require" if "?" in dsn else f"{dsn}?sslmode=require"
    return dsn


def resolve_postgres_dsn() -> str:
    raw = _secrets_get("POSTGRES_DSN")
    if not raw:
        built = _build_dsn_from_supabase_parts()
        if built:
            return normalize_postgres_dsn(built)
        return normalize_postgres_dsn(
            "postgresql+psycopg2://herd_user:herd_password@db:5432/herd_forecast"
        )
    return normalize_postgres_dsn(raw)


POSTGRES_DSN = resolve_postgres_dsn()

# Finál ML pipeline (prognoz_vseh_parametrov) instead of forecast_dynamic simulation.
USE_FINAL_PIPELINE = (_secrets_get("USE_FINAL_PIPELINE") or "1").strip() not in (
    "0",
    "false",
    "False",
)

FORECAST_HORIZON_MONTHS = int(_secrets_get("FORECAST_HORIZON_MONTHS") or "15")

SHOW_TAB2_PARAMS = (_secrets_get("SHOW_TAB2_PARAMS") or "0").strip() in ("1", "true", "True")
SHOW_TAB3_LEGACY_FARM_FORECAST = (_secrets_get("SHOW_TAB3_LEGACY_FARM_FORECAST") or "0").strip() in (
    "1",
    "true",
    "True",
)

PIPELINE_WORK_ROOT = _secrets_get("PIPELINE_WORK_ROOT") or ".pipeline_runtime"

ADMIN_KEY = _secrets_get("ADMIN_KEY")
