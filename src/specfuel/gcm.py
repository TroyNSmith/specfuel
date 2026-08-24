"""GCM data parsers."""

import json
from dataclasses import dataclass
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / "data"

# Maps const1994.json field names to FunctionalGroup attribute names
_FIELD_MAP = {
    "order": "order",
    "tc": "t_crit",
    "pc": "p_crit",
    "vc": "v_crit",
    "tb": "t_boil",
    "tm": "t_melt",
    "w": "acentric_factor",
    "v": "molar_liquid_vol",
    "mw": "molecular_weight",
    "h": "h_form",
    "g": "g_free",
    "hv": "h_vap",
    "cpa": "heat_cap_a",
    "cpb": "heat_cap_b",
    "cpc": "heat_cap_c",
}


@dataclass
class FunctionalGroup:
    """A GCM functional group and its contribution constants."""

    name: str
    order: int

    # Physical
    t_crit: float
    p_crit: float
    v_crit: float
    t_boil: float
    t_melt: float
    acentric_factor: float
    molar_liquid_vol: float
    molecular_weight: float

    # Thermo
    h_form: float
    g_free: float
    h_vap: float
    heat_cap_a: float
    heat_cap_b: float
    heat_cap_c: float


def const1994(
    path: str | Path = DATA_DIR / "const1994.json",
) -> dict[str, FunctionalGroup]:
    """Initialize GCM definitions from Constantinou et. al. (1994).

    Loads GCM (Group Contribution Method) data from a JSON file.

    Parameters
    ----------
    path
        Path to the const1994.json file.

    Returns
    -------
        Mapping of group name to its ``FunctionalGroup`` contribution constants.

    Examples
    --------
    >>> groups = const1994()
    >>> groups["CH3"].molecular_weight
    15.0
    """
    with Path(path).open() as f:
        raw = json.load(f)

    return {
        name: FunctionalGroup(
            name=name,
            **{_FIELD_MAP[key]: value for key, value in fields.items()},
        )
        for name, fields in raw.items()
    }
