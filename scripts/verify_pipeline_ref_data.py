#!/usr/bin/env python3
"""Run finál pipeline on reference files from ../Прогноз_стада (golden path)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = Path("/Users/sansey2/Desktop/econiva/Прогноз_стада")
sys.path.insert(0, str(ROOT))

from prognoz_vseh_parametrov import PipelineConfig, run_pipeline  # noqa: E402

D1 = S / "d1"
bulls = ROOT / ".pipeline_runtime_verify/ЖК_Высокое/bulls_full.xlsx"
cfg = PipelineConfig(
    name="ЖК Высокое (ref data)",
    work_dir=S,
    filter_folder=str(S / "фильтр_ЖК_Высокое"),
    events_path=D1 / "События-по-коровам.xlsx",
    events_aux_path=D1 / "События-по-коровам (1).xlsx",
    lactation_path=D1 / "поголовье_по_лактациям_январь2022_декабрь2025.xlsx",
    bulls_path=bulls if bulls.exists() else ROOT / "быки_полная_база.xlsx",
    output_xlsx=ROOT / ".pipeline_sibling_e2e.xlsx",
)

if __name__ == "__main__":
    fc, fact = run_pipeline(cfg)
    print("PIPELINE_OK", fc.shape, list(fc.columns[:5]))
    print("fact", None if fact is None else fact.shape)
