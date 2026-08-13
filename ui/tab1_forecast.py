from __future__ import annotations

import io
import traceback
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from config import FORECAST_HORIZON_MONTHS, USE_FINAL_PIPELINE
from core.forecast_params import MANUAL_BASELINE_FIELDS, manual_baseline_from_inputs
from core.final_forecast_service import run_final_forecast_for_subdivision
from core.subdivisions_registry import all_known_units_flat, display_label
from forecast_dynamic import latest_data_date
from ui.tab3_farm_parts.storage import (
    _farm_name_for_subdivision,
    _load_farm_tables_from_db,
    _subdivision_status_df_from_db,
)


def _subdivision_options() -> list[dict[str, str]]:
    """Fixed catalog + subdivisions present in DB (ready first)."""
    ready: set[str] = set()
    try:
        status = _subdivision_status_df_from_db()
        if isinstance(status, pd.DataFrame) and not status.empty and "Статус" in status.columns:
            ready = set(
                status.loc[status["Статус"].astype(str) == "готово", "Подразделение"].astype(str).tolist()
            )
    except Exception:
        ready = set()

    options: list[dict[str, str]] = []
    seen_units: set[str] = set()
    for farm, unit in all_known_units_flat():
        if unit in seen_units:
            continue
        seen_units.add(unit)
        options.append(
            {
                "unit": unit,
                "farm": farm,
                "label": display_label(farm, unit),
                "ready": unit in ready,
            }
        )

    try:
        status = _subdivision_status_df_from_db()
        if isinstance(status, pd.DataFrame) and not status.empty:
            for _, row in status.iterrows():
                unit = str(row.get("Подразделение", "") or "").strip()
                farm = str(row.get("Хозяйство", "") or "").strip()
                if not unit or unit in seen_units:
                    continue
                seen_units.add(unit)
                options.append(
                    {
                        "unit": unit,
                        "farm": farm,
                        "label": display_label(farm, unit) if farm else unit,
                        "ready": str(row.get("Статус", "")) == "готово",
                    }
                )
    except Exception:
        pass

    options.sort(key=lambda x: (not x["ready"], x["label"]))
    return options


def _wide_to_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "параметр" in out.columns:
        out = out.set_index("параметр")
    return out


def render_tab1_forecast() -> None:
    if not USE_FINAL_PIPELINE:
        st.error("В config.py включите USE_FINAL_PIPELINE для finál-прогноза.")
        return

    st.subheader("Прогноз по подразделению (finál + ML)")

    mode = st.radio(
        "Режим",
        options=("Прогноз (по последней дате в БД)", "Бэктест (обрезка на выбранный месяц)"),
        horizontal=True,
        key="tab1_final_mode",
    )
    backtest = mode.startswith("Бэктест")
    anchor_date: date | None = None
    if backtest:
        c_y, c_m = st.columns(2)
        bt_year = c_y.number_input("Год якоря (обучение до конца месяца)", 2022, 2030, 2024, key="tab1_bt_year")
        bt_month = c_m.selectbox(
            "Месяц якоря",
            list(range(1, 13)),
            index=8,
            format_func=lambda m: f"{m:02d}",
            key="tab1_bt_month",
        )
        anchor_date = date(int(bt_year), int(bt_month), 1)
        st.caption(
            f"События обрежутся по **{bt_year}-{bt_month:02d}**, прогноз на **{FORECAST_HORIZON_MONTHS}** месяцев вперёд "
            "(finál + лист «факт» для сравнения с прогнозом)."
        )
    else:
        st.caption(
            f"Горизонт: {FORECAST_HORIZON_MONTHS} месяцев от месяца последней даты в данных. "
            "Только прогноз (без листа «факт»). Загрузите файлы на вкладке «Загрузка данных»."
        )

    options = _subdivision_options()
    if not options:
        st.warning("Нет подразделений. Сначала загрузите данные на вкладке «Загрузка данных».")
        return

    labels = [f"{o['label']}{' ✓' if o['ready'] else ' (нет данных)'}" for o in options]
    default_ix = next((i for i, o in enumerate(options) if o["ready"]), 0)
    choice_ix = st.selectbox(
        "Подразделение",
        options=range(len(options)),
        index=default_ix,
        format_func=lambda i: labels[i],
        key="tab1_final_subdivision_ix",
    )
    chosen = options[int(choice_ix)]
    unit = chosen["unit"]
    farm_hint = chosen["farm"]

    last_date: date | None = None
    if chosen["ready"]:
        try:
            tables_probe = _load_farm_tables_from_db(unit)
            last_date = latest_data_date(tables_probe)
            st.caption(f"Последняя дата в данных: {last_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            st.warning(f"Не удалось прочитать данные подразделения «{unit}»: {e}")

    if backtest and anchor_date:
        baseline_anchor = anchor_date.strftime("%Y-%m")
        baseline_caption = f"конец {baseline_anchor} (бэктест)"
    elif last_date:
        baseline_anchor = last_date.strftime("%Y-%m")
        baseline_caption = f"конец {baseline_anchor} (последняя дата в данных)"
    else:
        baseline_anchor = "—"
        baseline_caption = "после загрузки данных"

    baseline_raw: dict[str, Any] = {}
    with st.expander(f"База поголовья на {baseline_caption}", expanded=False):
        st.caption(
            "Необязательно. Если заполнить — сухостойные, дойные, фуражные и лактации L1–L5+ "
            "на дату обучения подставятся в пайплайн вместо авторасчёта из файлов."
        )
        bcols = st.columns(3)
        for i, (key, label) in enumerate(MANUAL_BASELINE_FIELDS):
            with bcols[i % 3]:
                baseline_raw[key] = st.number_input(
                    label,
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"tab1_baseline_{key}",
                )

    run = st.button("Рассчитать прогноз", type="primary", key="tab1_final_run", use_container_width=True)

    if run:
        if not chosen["ready"]:
            st.error(f"Подразделение «{unit}» не готово — загрузите полный комплект файлов.")
            st.stop()
        try:
            tables = _load_farm_tables_from_db(unit)
            db_farm = _farm_name_for_subdivision(unit)
            farm_use = db_farm or farm_hint
        except Exception as e:
            st.error(f"Ошибка загрузки из БД: {e}")
            st.stop()

        manual_baseline = manual_baseline_from_inputs(baseline_raw)

        with st.spinner("Считаю finál-пайплайн (XGB + цепочка моделей)… это может занять несколько минут."):
            try:
                forecast_table, fact_table, meta = run_final_forecast_for_subdivision(
                    unit,
                    tables,
                    farm_hint=farm_use,
                    anchor_date=anchor_date,
                    backtest=backtest,
                    manual_baseline=manual_baseline,
                )
            except Exception as e:
                st.error(f"Ошибка расчёта: {e}")
                st.code(traceback.format_exc())
                st.stop()

        st.session_state["tab1_final_forecast"] = forecast_table
        st.session_state["tab1_final_fact"] = fact_table
        st.session_state["tab1_final_meta"] = meta
        st.success(
            f"Готово: {meta.get('farm')} / {meta.get('unit')}, "
            f"обучение до {meta.get('train_end')}, месяцы: {', '.join(meta.get('month_cols', [])[:3])}…"
        )
        st.rerun()

    forecast_table = st.session_state.get("tab1_final_forecast")
    fact_table = st.session_state.get("tab1_final_fact")

    if isinstance(forecast_table, pd.DataFrame) and not forecast_table.empty:
        st.subheader("Прогноз (строки — параметры, столбцы — месяцы)")
        st.dataframe(_wide_to_display(forecast_table), use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            forecast_table.to_excel(writer, sheet_name="прогноз", index=False)
            if isinstance(fact_table, pd.DataFrame) and not fact_table.empty:
                fact_table.to_excel(writer, sheet_name="факт", index=False)
        st.download_button(
            "Скачать Excel (прогноз + факт)",
            data=buf.getvalue(),
            file_name=f"prognoz_{unit.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="tab1_final_dl",
        )

    with st.expander("Backtesting / факт (только в режиме бэктеста)", expanded=False):
        st.caption(
            "Лист «факт» строится только при бэктесте — для сравнения прогноза с историей на выбранном якоре."
        )
        if isinstance(fact_table, pd.DataFrame) and not fact_table.empty:
            st.dataframe(_wide_to_display(fact_table), use_container_width=True)
        else:
            st.info("Запустите расчёт в режиме «Бэктест», чтобы увидеть таблицу «факт».")
