"""Tests for gcm module."""

from collections.abc import Callable
from typing import NamedTuple

import pandas as pd
import pytest
from pint import Quantity
from pint.facets.plain import PlainQuantity

from specfuel.gcm import ConstGani
from specfuel.types import INT_MATRIX
from specfuel.units import Q_

from .conftest import BASELINE_DIR, FUELS_BY_NAME

CONST_GANI_BASELINE_DIR = BASELINE_DIR / "const_gani"
BASELINE_RTOL = 1e-3


@pytest.fixture
def const_gani() -> ConstGani:
    """Fixture to create a ConstGani instance."""
    return ConstGani()


class TestConstGani:
    """Test the ConstGani class."""

    def test_num_groups(self, const_gani: ConstGani) -> None:
        """Test the num_groups property."""
        exp_groups = 121
        assert const_gani.num_groups == exp_groups


class BaselineRow(NamedTuple):
    """A single baseline row from a const_gani baseline CSV."""

    compound: str
    property: str
    temperature_c: float
    value: float
    unit: str


def _baseline_cases() -> list[tuple[str, BaselineRow]]:
    """Collect (fuel_name, row) cases from every const_gani baseline CSV.

    Returns
    -------
        List of (fuel_name, row) tuples for parametrizing baseline tests.
    """
    cases = []
    for csv_path in sorted(CONST_GANI_BASELINE_DIR.glob("*.csv")):
        fuel_name = csv_path.stem
        df = pd.read_csv(csv_path)
        cases.extend(
            (fuel_name, BaselineRow(*row))
            for row in df.itertuples(index=False, name="BaselineRow")
        )
    return cases


def _baseline_case_id(case: tuple[str, BaselineRow]) -> str:
    """Build a readable pytest ID for a baseline case.

    Parameters
    ----------
    case
        (fuel_name, row) tuple.

    Returns
    -------
        Human-readable test ID.
    """
    fuel_name, row = case
    temp = "stp" if pd.isna(row.temperature_c) else f"{row.temperature_c:g}C"
    return f"{fuel_name}-{row.compound}-{row.property}-{temp}"


CONST_GANI = ConstGani()
STP_PROPERTIES: dict[str, Callable[[INT_MATRIX], Quantity]] = {
    "molecular_weights": CONST_GANI.molecular_weights,
    "critical_temperatures": CONST_GANI.critical_temperatures,
    "critical_pressures": CONST_GANI.critical_pressures,
    "critical_volumes": CONST_GANI.critical_volumes,
    "boiling_temperatures": CONST_GANI.boiling_temperatures,
    "stp_molar_liquid_volumes": CONST_GANI.stp_molar_liquid_volumes,
    "acentric_factors": CONST_GANI.acentric_factors,
}
TEMP_PROPERTIES: dict[
    str, Callable[[INT_MATRIX, Quantity | PlainQuantity], Quantity]
] = {
    "molar_liquid_volumes": CONST_GANI.molar_liquid_volumes,
    "densities": CONST_GANI.densities,
    "kinematic_viscosities": CONST_GANI.kinematic_viscosities,
    "dynamic_viscosities": CONST_GANI.dynamic_viscosities,
}


@pytest.mark.parametrize("case", _baseline_cases(), ids=_baseline_case_id)
def test_const_gani_matches_baseline(case: tuple[str, BaselineRow]) -> None:
    """Recompute each baseline row and compare it to the recorded value."""
    fuel_name, row = case
    fuel = FUELS_BY_NAME[fuel_name]
    compound_idx = fuel.compounds.index(row.compound)

    if row.property in STP_PROPERTIES:
        values = STP_PROPERTIES[row.property](fuel.cg_decomp)
    else:
        temp = Q_(row.temperature_c, "celsius")
        values = TEMP_PROPERTIES[row.property](fuel.cg_decomp, temp)

    assert str(values.units) == row.unit
    assert values.magnitude[compound_idx] == pytest.approx(row.value, rel=BASELINE_RTOL)
