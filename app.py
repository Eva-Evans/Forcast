from __future__ import annotations

import os
import streamlit as st

from config import SHOW_TAB2_PARAMS, USE_FINAL_PIPELINE
from ui.tab1_forecast import render_tab1_forecast
from ui.tab3_farm import render_tab3_farm

if SHOW_TAB2_PARAMS:
    from ui.tab2_params import render_tab2_params

st.set_page_config(page_title="Прогноз поголовья", layout="wide")
st.title("Прогноз поголовья (finál)")

with st.expander("Справка", expanded=False):
    st.markdown(
        """
**Логика расчёта:** пайплайн `prognoz_vseh_parametrov` (XGBoost + цепочка моделей из finál).

1. На вкладке **«Загрузка данных»** загрузите Excel/txt (можно общие файлы на все подразделения).
2. На вкладке **«Прогноз»** выберите подразделение и нажмите **«Рассчитать»**.
3. Результат — широкая таблица: **строки = параметры**, **столбцы = 15 месяцев** от месяца последней даты в данных.
        """
    )

tab_labels = ["Прогноз", "Загрузка данных"]
if SHOW_TAB2_PARAMS:
    tab_labels.insert(1, "Параметры (legacy)")

tabs = st.tabs(tab_labels)
idx = 0
with tabs[idx]:
    render_tab1_forecast()
idx += 1

if SHOW_TAB2_PARAMS:
    with tabs[idx]:
        render_tab2_params()
    idx += 1

with tabs[idx]:
    render_tab3_farm()

if not USE_FINAL_PIPELINE:
    st.warning("USE_FINAL_PIPELINE=0 — включите finál-пайплайн в config.py.")
