#!/usr/bin/env python3
"""Remove all uploaded herd data from Postgres (empty DB, tables kept)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from db import engine  # noqa: E402

TABLES = (
    "tab3_calvings_farm_raw",
    "tab3_inseminations_farm_raw",
    "tab3_dryoff_farm_raw",
    "tab3_disposals_farm_raw",
    "tab3_bulls_farm_raw",
    "tab3_forecast_cache",
    "tab3_subdivision_farm_map",
    "tab3_capacity_places",
    "model_params_cache",
    "calvings_births_raw",
    "inseminations_raw",
    "dryoff_raw",
    "disposals_raw",
    "bulls_raw",
)


def _count(conn, table: str) -> int | None:
    try:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe all herd upload data from Postgres")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Подтвердить удаление (без флага только показать счётчики)",
    )
    args = parser.parse_args()

    existing: list[str] = []
    with engine.connect() as conn:
        for t in TABLES:
            n = _count(conn, t)
            if n is not None:
                existing.append(t)
                print(f"  {t}: {n:,} строк")
            else:
                print(f"  {t}: (таблицы нет или нет доступа)")

        if not args.yes:
            print("\nДобавьте --yes чтобы выполнить TRUNCATE.")
            return 0

        if not existing:
            print("\nНечего очищать (нет таблиц).")
            return 0

        quoted = ", ".join(existing)
        conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))
        conn.commit()
        print(f"\n✅ TRUNCATE выполнен для {len(existing)} таблиц.")

        for t in existing:
            n = _count(conn, t)
            print(f"  {t}: {n or 0:,} строк")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAIL:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
