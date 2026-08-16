"""Проверка подключения к Postgres (Docker локально / Supabase на Streamlit Cloud)."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from sqlalchemy.exc import OperationalError, SQLAlchemyError

_CLOUD_SETUP = """
### База данных не настроена для Streamlit Cloud

Streamlit Cloud **не** поднимает Postgres из `docker-compose`. Нужна **облачная БД** (Supabase, Neon, …).

**Secrets** приложения → добавьте один из вариантов:

**Вариант A — одна строка (Session pooler Supabase, IPv4):**
```toml
POSTGRES_DSN = "postgresql+psycopg2://postgres.ВАШ_REF:ПАРОЛЬ@aws-0-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
ADMIN_KEY = "ваш_ключ"
```

**Вариант B — отдельные поля:**
```toml
SUPABASE_DB_HOST = "aws-0-eu-central-1.pooler.supabase.com"
SUPABASE_DB_USER = "postgres.ВАШ_REF"
SUPABASE_DB_PASSWORD = "ваш_пароль"
SUPABASE_DB_NAME = "postgres"
SUPABASE_DB_PORT = "5432"
ADMIN_KEY = "ваш_ключ"
```

В Supabase → **SQL Editor** выполните скрипт `db/init.sql` из репозитория.

Подробнее: `DEPLOY.md`, раздел «Streamlit Community Cloud».
"""


def running_on_streamlit_cloud() -> bool:
    if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    if os.getcwd().startswith("/mount/src"):
        return True
    if os.getenv("STREAMLIT_SHARING_MODE"):
        return True
    return False


def is_docker_internal_dsn(dsn: str) -> bool:
    try:
        host = (urlparse(dsn.replace("postgresql+psycopg2://", "postgresql://", 1)).hostname or "").lower()
    except Exception:
        host = ""
    return host in ("db", "herd-db", "localhost") and running_on_streamlit_cloud()


def postgres_secrets_configured() -> bool:
    from config import _build_dsn_from_supabase_parts, _dsn_is_placeholder, _secrets_get

    raw = _secrets_get("POSTGRES_DSN")
    if raw and not _dsn_is_placeholder(raw):
        return True
    return _build_dsn_from_supabase_parts() is not None


def mask_dsn_host(dsn: str) -> str:
    try:
        u = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
        p = urlparse(u)
        host = p.hostname or "?"
        db = (p.path or "").lstrip("/") or "?"
        return f"{host}:{p.port or 5432}/{db}"
    except Exception:
        return re.sub(r":[^:@/]+@", ":***@", dsn)


def check_database_connection() -> tuple[bool, str]:
    """(ok, message_for_user)"""
    from config import POSTGRES_DSN
    from db import engine

    if running_on_streamlit_cloud() and not postgres_secrets_configured():
        return False, _CLOUD_SETUP

    if is_docker_internal_dsn(POSTGRES_DSN):
        return False, _CLOUD_SETUP

    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True, f"Подключение к БД: `{mask_dsn_host(POSTGRES_DSN)}`"
    except OperationalError as exc:
        hint = _diagnose_operational_error(str(exc), POSTGRES_DSN)
        return False, hint
    except SQLAlchemyError as exc:
        return False, f"Ошибка SQLAlchemy: `{exc}`"


def _diagnose_operational_error(msg: str, dsn: str) -> str:
    low = msg.lower()
    host = mask_dsn_host(dsn)
    if "could not translate host name" in low and "db" in low:
        return _CLOUD_SETUP
    if "no address associated with hostname" in low or "name or service not known" in low:
        return (
            f"Не удалось найти хост БД (`{host}`).\n\n"
            "На Streamlit Cloud используйте **Session pooler** Supabase (IPv4), не прямой `db.xxxx.supabase.co`.\n\n"
            + _CLOUD_SETUP
        )
    if "password authentication failed" in low:
        return (
            f"Неверный пароль или пользователь для `{host}`.\n\n"
            "Проверьте `POSTGRES_DSN` или `SUPABASE_DB_PASSWORD` в Secrets (без `[ ]` вокруг пароля)."
        )
    if "ssl" in low and ("required" in low or "negotiation" in low):
        return (
            f"Нужен SSL для `{host}`.\n\n"
            "Добавьте `?sslmode=require` в `POSTGRES_DSN` или используйте поля `SUPABASE_DB_*`."
        )
    return (
        f"Не удалось подключиться к Postgres (`{host}`).\n\n"
        f"Техническое сообщение: `{msg}`\n\n"
        + _CLOUD_SETUP
    )


def render_database_gate() -> bool:
    """Показать ошибку в UI; вернуть False если БД недоступна."""
    import streamlit as st

    ok, message = check_database_connection()
    if ok:
        return True
    st.error("Нет подключения к базе данных")
    st.markdown(message)
    st.stop()
    return False
