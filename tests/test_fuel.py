"""Tests for fuel module."""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from specfuel.fuel import Fuel
from specfuel.gcm import ConstGani
from specfuel.types import FLOAT_VECTOR, INT_MATRIX
from specfuel.units import Q_

from .conftest import BASELINE_DIR, FUELS_BY_NAME

FUEL_BASELINE_DIR = BASELINE_DIR / "fuel"
BASELINE_RTOL = 1e-3

GROUP_NAMES = ConstGani().group_names
NUM_GROUPS = ConstGani().num_groups


def _make_fuel(
    *,
    weights: FLOAT_VECTOR | None = None,
    cg_groups: list[str] | None = None,
    cg_decomp: INT_MATRIX | None = None,
) -> Fuel:
    """Build a minimal valid Fuel, overriding fields for validator tests.

    Returns
    -------
        A Fuel instance constructed from default single-family fields.
    """
    default_decomp = np.zeros((1, NUM_GROUPS), dtype=np.int64)
    default_decomp[0, 0] = 1
    return Fuel(
        name="test-fuel",
        families=["test-family"],
        weights=weights if weights is not None else np.array([100.0]),
        cg_groups=cg_groups if cg_groups is not None else GROUP_NAMES,
        cg_decomp=cg_decomp if cg_decomp is not None else default_decomp,
    )


class TestFromDirectory:
    """Test Fuel.from_directory."""

    def test_loads_decane_fuel(self) -> None:
        """Test that decane is loaded with the expected family data."""
        fuel = FUELS_BY_NAME["decane"]
        assert fuel.families == ["n-C10"]
        assert fuel.formulas == ["C10H22"]
        assert fuel.weights == pytest.approx([100.0])
        assert fuel.num_families == 1
        assert fuel.cg_decomp.shape == (1, NUM_GROUPS)

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        """Test that a nonexistent path raises."""
        with pytest.raises(ValueError, match="is not a directory"):
            Fuel.from_directory(tmp_path / "nonexistent")

    def test_raises_for_missing_composition_csv(self, tmp_path: Path) -> None:
        """Test that a directory missing composition.csv raises."""
        (tmp_path / "const_gani.csv").write_text("dummy")
        with pytest.raises(ValueError, match=re.escape("composition.csv")):
            Fuel.from_directory(tmp_path)

    def test_raises_for_missing_const_gani_csv(self, tmp_path: Path) -> None:
        """Test that a directory missing const_gani.csv raises."""
        (tmp_path / "composition.csv").write_text("Family,Weight %\nfoo,100\n")
        with pytest.raises(ValueError, match=re.escape("const_gani.csv")):
            Fuel.from_directory(tmp_path)


class TestValidators:
    """Test Fuel's model validators."""

    def test_validate_weights_raises_when_not_summing_to_100(self) -> None:
        """Test that weights not summing to 100% raises."""
        with pytest.raises(ValidationError, match="do not sum to 100%"):
            _make_fuel(weights=np.array([50.0]))

    def test_validate_weights_raises_on_length_mismatch(self) -> None:
        """Test that a weights/families length mismatch raises."""
        with pytest.raises(ValidationError, match="Number of weights"):
            _make_fuel(weights=np.array([50.0, 50.0]))

    def test_validate_cg_decomp_raises_on_group_name_mismatch(self) -> None:
        """Test that cg_groups not matching ConstGani group names raises."""
        with pytest.raises(ValidationError, match="group names"):
            _make_fuel(cg_groups=["not", "matching"])

    def test_validate_cg_decomp_raises_on_row_mismatch(self) -> None:
        """Test that cg_decomp rows not matching num_families raises."""
        decomp = np.zeros((2, NUM_GROUPS), dtype=np.int64)
        with pytest.raises(ValidationError, match="Rows in cg_decomp"):
            _make_fuel(cg_decomp=decomp)

    def test_validate_cg_decomp_raises_on_column_mismatch(self) -> None:
        """Test that cg_decomp columns not matching num_groups raises."""
        decomp = np.zeros((1, NUM_GROUPS - 1), dtype=np.int64)
        with pytest.raises(ValidationError, match="Columns in cg_decomp"):
            _make_fuel(cg_decomp=decomp)


class BaselineRow(NamedTuple):
    """A single baseline row from a fuel baseline CSV."""

    property: str
    correlation: str | float
    temperature_c: float
    value: float
    unit: str


def _baseline_cases() -> list[tuple[str, BaselineRow]]:
    """Collect (fuel_name, row) cases from every fuel baseline CSV.

    Returns
    -------
        List of (fuel_name, row) tuples for parametrizing baseline tests.
    """
    cases = []
    for csv_path in sorted(FUEL_BASELINE_DIR.glob("*.csv")):
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
    correlation = "" if pd.isna(row.correlation) else f"-{row.correlation}"
    return f"{fuel_name}-{row.property}{correlation}-{row.temperature_c:g}C"


@pytest.mark.parametrize("case", _baseline_cases(), ids=_baseline_case_id)
def test_fuel_property_matches_baseline(case: tuple[str, BaselineRow]) -> None:
    """Recompute each baseline row and compare it to the recorded value."""
    fuel_name, row = case
    fuel = FUELS_BY_NAME[fuel_name]
    temp = Q_(row.temperature_c, "celsius")

    if row.property == "density":
        result = fuel.density(temp)
    else:
        result = getattr(fuel, row.property)(temp, correlation=row.correlation)

    assert str(result.units) == row.unit
    assert result.magnitude == pytest.approx(row.value, rel=BASELINE_RTOL)
