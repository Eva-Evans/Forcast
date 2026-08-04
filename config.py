from __future__ import annotations

import os
import re
from urllib.parse import quote_plus

from sqlalchemy.engine.url import make_url


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


def _dsn_is_placeholder(dsn: str) -> bool:
    u = dsn.upper()
    return bool(
        re.search(r"\[YOUR[-_\s]?PASSWORD\]", u)
        or "[YOUR-" in u
        or "YOUR-PASSWORD" in u
    )


def _build_dsn_from_supabase_parts() -> str | None:
    """Password/host отдельно — без URL-кодирования вручную в одной строке."""
    host = _secrets_get("SUPABASE_DB_HOST") or _secrets_get("DB_HOST")
    password = _secrets_get("SUPABASE_DB_PASSWORD") or _secrets_get("DB_PASSWORD")
    user = _secrets_get("SUPABASE_DB_USER") or _secrets_get("DB_USER") or "postgres"
    database = _secrets_get("SUPABASE_DB_NAME") or _secrets_get("DB_NAME") or "postgres"
    port = _secrets_get("SUPABASE_DB_PORT") or _secrets_get("DB_PORT") or "5432"
    if not host or not password:
        return None
    password = _strip_wrapping_brackets(password)
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}?sslmode=require"
    )


def _strip_wrapping_brackets(password: str) -> str:
    p = password.strip()
    if len(p) >= 2 and p.startswith("[") and p.endswith("]"):
        return p[1:-1]
    return p


def _fix_bracket_password_in_dsn(dsn: str) -> str:
    """Supabase docs sometimes show [password]; in URL скобки ломают парсер и auth."""
    return re.sub(
        r":\[([^\]]+)\]@",
        lambda m: f":{quote_plus(m.group(1))}@",
        dsn,
        count=1,
    )


def _manual_rebuild_dsn(dsn: str) -> str | None:
    """Fallback, если make_url падает (скобки, @ в пароле)."""
    for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if dsn.startswith(prefix):
            rest = dsn[len(prefix) :]
            break
    else:
        return None
    at = rest.rfind("@")
    if at <= 0:
        return None
    userinfo, hostpart = rest[:at], rest[at + 1 :]
    if "/" in hostpart:
        hostport, db_and_q = hostpart.split("/", 1)
    else:
        hostport, db_and_q = hostpart, "postgres"
    if "?" in db_and_q:
        database, _query = db_and_q.split("?", 1)
    else:
        database, _query = db_and_q, ""
    if ":" in userinfo:
        user, password = userinfo.split(":", 1)
    else:
        user, password = userinfo, ""
    password = _strip_wrapping_brackets(password)
    user = _strip_wrapping_brackets(user) if user.startswith("[") else user
    host = hostport
    port = "5432"
    if hostport.startswith("[") and "]" in hostport:
        host = hostport[1 : hostport.index("]")]
        tail = hostport[hostport.index("]") + 1 :]
        if tail.startswith(":"):
            port = tail[1:]
    elif ":" in hostport:
        host, port = hostport.rsplit(":", 1)
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}?sslmode=require"
    )


def normalize_postgres_dsn(dsn: str) -> str:
    dsn = dsn.strip().strip('"').strip("'")
    dsn = _fix_bracket_password_in_dsn(dsn)
    if _dsn_is_placeholder(dsn):
        raise ValueError(
            "В Secrets в POSTGRES_DSN остался шаблон [YOUR-PASSWORD]. "
            "Вставьте реальный пароль: Supabase → Project Settings → Database → Database password. "
            "Либо удалите POSTGRES_DSN и задайте SUPABASE_DB_HOST + SUPABASE_DB_PASSWORD."
        )
    if "@" not in dsn or "://" not in dsn:
        raise ValueError(
            "POSTGRES_DSN должен быть URL вида "
            "postgresql+psycopg2://postgres:ПАРОЛЬ@db....supabase.co:5432/postgres?sslmode=require"
        )

    if dsn.startswith("postgres://"):
        dsn = "postgresql+psycopg2://" + dsn[len("postgres://") :]
    elif dsn.startswith("postgresql://"):
        dsn = "postgresql+psycopg2://" + dsn[len("postgresql://") :]
    elif not dsn.startswith("postgresql+psycopg2://"):
        raise ValueError(
            "POSTGRES_DSN должен начинаться с postgresql:// или postgresql+psycopg2://"
        )

    try:
        url = make_url(dsn)
    except Exception:
        rebuilt = _manual_rebuild_dsn(dsn)
        if not rebuilt:
            raise ValueError(
                "Не удалось разобрать POSTGRES_DSN. Уберите [ ] вокруг пароля или задайте "
                "SUPABASE_DB_HOST + SUPABASE_DB_PASSWORD в Secrets."
            )
        url = make_url(rebuilt)

    host = (url.host or "").lower()
    if "supabase.co" in host or "pooler.supabase.com" in host:
        q = dict(url.query) if url.query else {}
        if q.get("sslmode") != "require":
            url = url.update_query_dict({"sslmode": "require"})

    return url.render_as_string(hide_password=False)


def resolve_postgres_dsn() -> str:
    raw = _secrets_get("POSTGRES_DSN")
    parts_dsn = _build_dsn_from_supabase_parts()

    if raw and not _dsn_is_placeholder(raw):
        try:
            return normalize_postgres_dsn(raw)
        except ValueError:
            if parts_dsn:
                return normalize_postgres_dsn(parts_dsn)
            raise

    if raw and _dsn_is_placeholder(raw):
        if parts_dsn:
            return normalize_postgres_dsn(parts_dsn)
        raise ValueError(
            "POSTGRES_DSN содержит [YOUR-PASSWORD]. Замените пароль или добавьте "
            "SUPABASE_DB_HOST и SUPABASE_DB_PASSWORD в Secrets."
        )

    if parts_dsn:
        return normalize_postgres_dsn(parts_dsn)

    return normalize_postgres_dsn(
        "postgresql+psycopg2://herd_user:herd_password@db:5432/herd_forecast"
    )


POSTGRES_DSN = resolve_postgres_dsn()

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
