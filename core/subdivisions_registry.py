from __future__ import annotations

from prognoz_vseh_parametrov import KALUGA_TREE, SUBDIVISION_ALIASES, kaluga_trade_rules

DEFAULT_FARM_VYSOKOE = "ЖК Высокое"
DEFAULT_UNIT_VYSOKOE = "ЖК Высокое"


def all_known_units_flat() -> list[tuple[str, str]]:
    """(farm, unit) pairs from the fixed Kaluga tree + ЖК Высокое."""
    out: list[tuple[str, str]] = [(DEFAULT_FARM_VYSOKOE, DEFAULT_UNIT_VYSOKOE)]
    for farm, units in KALUGA_TREE.items():
        for unit in units:
            out.append((farm, unit))
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for pair in out:
        if pair in seen:
            continue
        seen.add(pair)
        uniq.append(pair)
    return uniq


def display_label(farm: str, unit: str) -> str:
    if farm == unit or farm == DEFAULT_FARM_VYSOKOE:
        return unit
    return f"{farm} / {unit}"


def resolve_farm_unit(subdivision_name: str, farm_hint: str | None = None) -> tuple[str, str]:
    sub = (subdivision_name or "").strip()
    if not sub:
        raise ValueError("Пустое имя подразделения.")
    if farm_hint and farm_hint.strip():
        farm = farm_hint.strip()
        if sub in KALUGA_TREE.get(farm, []):
            return farm, sub
        aliases = SUBDIVISION_ALIASES.get(sub, [sub])
        for unit in KALUGA_TREE.get(farm, []):
            if unit in aliases or sub in SUBDIVISION_ALIASES.get(unit, [unit]):
                return farm, unit
    for farm, units in KALUGA_TREE.items():
        if sub in units:
            return farm, sub
        for u in units:
            if sub in SUBDIVISION_ALIASES.get(u, [u]):
                return farm, u
    if sub == DEFAULT_UNIT_VYSOKOE:
        return DEFAULT_FARM_VYSOKOE, DEFAULT_UNIT_VYSOKOE
    # Unknown name: treat as standalone unit (DB-only names).
    return farm_hint.strip() if farm_hint and farm_hint.strip() else sub, sub


def trade_rules_for(farm: str, unit: str) -> dict[str, list[str]]:
    if farm in KALUGA_TREE and unit in KALUGA_TREE[farm]:
        return kaluga_trade_rules(farm, unit)
    from prognoz_vseh_parametrov import VYSOKOE_BUY, VYSOKOE_BULL_SALE, VYSOKOE_HEIFER_SALE

    return {
        "buy": list(VYSOKOE_BUY),
        "heifer_sale": list(VYSOKOE_HEIFER_SALE),
        "bull_sale": list(VYSOKOE_BULL_SALE),
    }
