from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from prognoz_vseh_parametrov import SUBDIVISION_ALIASES, normalize_events_df


def _ts(s: Any) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _years_in_tables(tables: dict[str, pd.DataFrame]) -> list[int]:
    years: set[int] = set()
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        col = "event_date" if "event_date" in df.columns else None
        if not col:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        years.update(int(y) for y in dt.dt.year.dropna().astype(int).tolist())
    if not years:
        years.add(date.today().year)
    return sorted(years)


def _normalize_date_frame(df: pd.DataFrame, possible_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in possible_cols:
        if col in out.columns:
            out["Дата"] = pd.to_datetime(out[col], errors="coerce")
            out["Date"] = out["Дата"]
            return out
    out["Дата"] = pd.NaT
    out["Date"] = pd.NaT
    return out


def _ensure_id_reg_bdat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ID" not in out.columns:
        out["ID"] = ""
    if "REG" not in out.columns:
        out["REG"] = ""
    if "BDAT" not in out.columns:
        out["BDAT"] = pd.NaT
    return out


def _tab3_calvings_to_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "ID": df.get("reg", pd.Series(dtype=object)).astype(str),
            "REG": df.get("reg", pd.Series(dtype=object)).astype(str),
            "BDAT": _ts(df.get("birth_date")),
            "Date": _ts(df.get("event_date")),
            "Event": df.get("event_type", pd.Series(dtype=object)),
            "Событие": df.get("event_type", pd.Series(dtype=object)),
            "GNDR": df.get("sex", pd.Series(dtype=object)),
            "DREG": df.get("mother_reg", pd.Series(dtype=object)),
            "LACT": pd.to_numeric(df.get("lact"), errors="coerce"),
        }
    )
    out["тип_файла"] = "ОТЕЛ"
    return _normalize_date_frame(out, ["Date", "BDAT", "Дата"])


def _tab3_inseminations_to_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    reg = df.get("reg", pd.Series(dtype=object)).astype(str)
    out = pd.DataFrame(
        {
            "ID": reg,
            "REG": reg,
            "BDAT": pd.NaT,
            "Date": _ts(df.get("event_date")),
            "R": df.get("result", pd.Series(dtype=object)).astype(str).str.strip().str.upper(),
            "LACT": pd.to_numeric(df.get("lact"), errors="coerce"),
            "Lact": pd.to_numeric(df.get("lact"), errors="coerce"),
            "bull": df.get("bull", pd.Series(dtype=object)),
        }
    )
    out["Event"] = "ОСЕМЕН"
    out["Событие"] = "ОСЕМЕН"
    out["тип_файла"] = "ОСЕМЕНЕНИЕ"
    return _normalize_date_frame(out, ["Date", "BDAT", "Дата"])


def _tab3_dry_disp_to_source(dry: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for df, default_ev in (
        (dry, "ЗАПУСК"),
        (disp, "ВЫБЫТИЕ"),
    ):
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        reg = df.get("reg", pd.Series(dtype=object)).astype(str)
        reason = df.get("disposal_reason", pd.Series(dtype=object)).astype(str)
        ev = reason.where(reason.str.strip() != "", default_ev)
        block = pd.DataFrame(
            {
                "ID": reg,
                "REG": reg,
                "BDAT": pd.NaT,
                "Date": _ts(df.get("event_date")),
                "Event": ev,
                "Событие": ev,
                "REM": reason,
                "Куда": reason,
            }
        )
        block["тип_файла"] = "ЗАПУСК+ВЫБЫТИЕ"
        block = _normalize_date_frame(block, ["Date", "BDAT", "Дата"])
        parts.append(block)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def build_events_all(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Объединение отёлов, осеменений, запусков и выбытий в одну таблицу событий
    (логика «События-пo-коровам» из ноутбука Калуги).
    """
    chunks = [
        _tab3_calvings_to_source(tables.get("calv", pd.DataFrame())),
        _tab3_inseminations_to_source(tables.get("ins", pd.DataFrame())),
        _tab3_dry_disp_to_source(
            tables.get("dry", pd.DataFrame()),
            tables.get("disp", pd.DataFrame()),
        ),
    ]
    chunks = [c for c in chunks if isinstance(c, pd.DataFrame) and not c.empty]
    if not chunks:
        raise ValueError("Нет данных для объединения в «События по коровам».")

    df_all = pd.concat(chunks, ignore_index=True, sort=False)
    df_all = _ensure_id_reg_bdat(df_all)

    df_all["ID"] = df_all["ID"].astype(str).fillna("")
    df_all["REG"] = df_all["REG"].astype(str).fillna("")
    df_all["BDAT"] = pd.to_datetime(df_all["BDAT"], errors="coerce")

    mask_empty = (df_all["ID"] == "") | (df_all["ID"].str.lower() == "nan")
    df_all.loc[mask_empty, "ID"] = df_all.loc[mask_empty, "REG"]

    bdat_key = df_all["BDAT"].dt.strftime("%Y%m%d").fillna("")
    df_all["ключ_коровы"] = df_all["ID"].astype(str) + "_" + bdat_key.astype(str)
    bad_key = df_all["ключ_коровы"] == "_"
    if bad_key.any():
        df_all.loc[bad_key, "ключ_коровы"] = (
            "без_ключа_" + df_all.index[bad_key].astype(str)
        )

    if "Date" not in df_all.columns or df_all["Date"].isna().all():
        df_all["Date"] = df_all.get("Дата", pd.NaT)
    df_all = df_all.sort_values(["ключ_коровы", "Date"], kind="mergesort").reset_index(drop=True)

    priority = ["ключ_коровы", "ID", "REG", "BDAT", "тип_файла", "Date", "Дата"]
    other = [c for c in df_all.columns if c not in priority]
    df_all = df_all[priority + other]

    return normalize_events_df(df_all)


def build_events_workbook(
    tables: dict[str, pd.DataFrame],
    unit: str,
    farm: str,
    out_path: Path,
    *,
    csv_path: Path | None = None,
) -> Path:
    subdiv_names = SUBDIVISION_ALIASES.get(unit, [unit])
    primary = subdiv_names[0]

    df_all = build_events_all(tables)
    df_all["Столбец1"] = primary
    df_all["Source.Name"] = farm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_excel(out_path, index=False)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return out_path


def export_bulls_workbook(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    """
    finál: быки_полная_база.xlsx — колонки «Плем», «Бык».
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bulls = tables.get("bulls", pd.DataFrame())
    if not isinstance(bulls, pd.DataFrame) or bulls.empty:
        pd.DataFrame({"Плем": [], "Бык": []}).to_excel(path, index=False)
        return path

    code = bulls.get("bull_code", pd.Series(dtype=object)).astype(str).str.strip()
    btype = bulls.get("bull_type", pd.Series(dtype=object)).astype(str).str.strip().str.upper()

    plem = btype.copy()
    plem = plem.replace(
        {
            "SEX": "S",
            "SEXED": "S",
            "С": "S",
            "S": "S",
            "TRAD": "H",
            "TRADITIONAL": "H",
            "О": "H",
            "H": "H",
        }
    )
    mask_unknown = ~plem.isin({"S", "H"})
    plem.loc[mask_unknown & btype.str.contains("SEX|S", regex=True, na=False)] = "S"
    plem.loc[mask_unknown] = plem.loc[mask_unknown].replace("", "H").fillna("H")

    out = pd.DataFrame({"Плем": plem, "Бык": code})
    out = out.loc[(out["Бык"] != "") & (out["Бык"].str.lower() != "nan")].drop_duplicates(subset=["Бык"])
    out.to_excel(path, index=False)
    return path


# --- годовые Excel для filter_* (ячейки finál с Отелы_2022.xlsx …) ---


def _calv_to_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["REG", "Дата", "BDAT", "Событие", "GNDR", "DREG", "LACT", "R", "ID"]
        )
    out = pd.DataFrame(
        {
            "REG": df.get("reg", pd.Series(dtype=object)).astype(str),
            "Дата": _ts(df.get("event_date")),
            "BDAT": _ts(df.get("birth_date")),
            "Событие": df.get("event_type", pd.Series(dtype=object)).astype(str).str.upper(),
            "GNDR": df.get("sex", pd.Series(dtype=object)),
            "DREG": df.get("mother_reg", pd.Series(dtype=object)).astype(str),
            "LACT": pd.to_numeric(df.get("lact"), errors="coerce"),
            "R": "",
            "ID": df.get("reg", pd.Series(dtype=object)).astype(str),
        }
    )
    return out


def _ins_to_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["REG", "Дата", "BDAT", "R", "LACT", "ID", "Событие"])
    reg = df.get("reg", pd.Series(dtype=object)).astype(str)
    out = pd.DataFrame(
        {
            "REG": reg,
            "Дата": _ts(df.get("event_date")),
            "BDAT": pd.NaT,
            "R": df.get("result", pd.Series(dtype=object)).astype(str).str.strip().str.upper(),
            "LACT": pd.to_numeric(df.get("lact"), errors="coerce").fillna(0).astype(int),
            "ID": reg,
            "Событие": "ОСЕМЕН",
        }
    )
    return out


def _dry_to_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["REG", "Дата", "Событие", "ID"])
    reg = df.get("reg", pd.Series(dtype=object)).astype(str)
    return pd.DataFrame(
        {
            "REG": reg,
            "Дата": _ts(df.get("event_date")),
            "Событие": "ЗАПУСК",
            "ID": reg,
        }
    )


def _disp_to_final(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["REG", "Дата", "Событие", "REM", "Куда", "ID"])
    reg = df.get("reg", pd.Series(dtype=object)).astype(str)
    reason = df.get("disposal_reason", pd.Series(dtype=object)).astype(str)
    return pd.DataFrame(
        {
            "REG": reg,
            "Дата": _ts(df.get("event_date")),
            "Событие": reason.str.upper().where(reason.str.strip() != "", "ВЫБЫТИЕ"),
            "REM": reason,
            "Куда": reason,
            "ID": reg,
        }
    )


def _write_yearly(prefix: str, df: pd.DataFrame, out_dir: Path, years: list[int]) -> None:
    if df.empty:
        return
    work = df.copy()
    work["Дата"] = pd.to_datetime(work["Дата"], errors="coerce")
    work = normalize_events_df(work)
    for year in years:
        part = work[work["Дата"].dt.year == year]
        if not part.empty:
            part.to_excel(out_dir / f"{prefix}_{year}.xlsx", index=False)


def export_tab3_to_filter_folder(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    years = _years_in_tables(tables)
    _write_yearly("Отелы", _calv_to_final(tables.get("calv", pd.DataFrame())), out_dir, years)
    _write_yearly("Осеменения", _ins_to_final(tables.get("ins", pd.DataFrame())), out_dir, years)
    _write_yearly("Запуск", _dry_to_final(tables.get("dry", pd.DataFrame())), out_dir, years)
    _write_yearly("Выбытие", _disp_to_final(tables.get("disp", pd.DataFrame())), out_dir, years)
    return out_dir
