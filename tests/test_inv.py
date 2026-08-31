"""Tests for inv module."""

import re
import shutil
from pathlib import Path

import pytest

from specfuel.data.examples import FUEL_DIR, ExampleFuels
from specfuel.inv import _load_constraints, solve_composition
from specfuel.units import Q_

CONSTRAINT_TEMP_C = 25.0


def _write_constraints_from_ground_truth(directory: Path) -> None:
    """Write a constraints.csv derived from the known heptane-decane composition."""
    fuel = ExampleFuels.heptane_decane
    temp = Q_(CONSTRAINT_TEMP_C, "degC")
    density = fuel.density(temp).to("kg/m^3").magnitude
    kinematic = fuel.kinematic_viscosity(temp).to("m^2/s").magnitude
    dynamic = fuel.dynamic_viscosity(temp).to("Pa*s").magnitude

    t = CONSTRAINT_TEMP_C
    rows = [
        ("density", t, density, "kg/m^3", 0.01 * density),
        ("kinematic_viscosity", t, kinematic, "m^2/s", 0.01 * kinematic),
        ("dynamic_viscosity", t, dynamic, "Pa*s", 0.01 * dynamic),
    ]
    lines = ["Property,Temperature,Target,Unit,Tolerance"]
    lines.extend(f"{p},{t},{v},{u},{tol}" for p, t, v, u, tol in rows)
    (directory / "constraints.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def heptane_decane_dir(tmp_path: Path) -> Path:
    """Directory with a copied const_gani.csv and a self-generated constraints.csv.

    Returns
    -------
        Directory containing `const_gani.csv` and `constraints.csv`.
    """
    shutil.copy(
        FUEL_DIR / "heptane-decane" / "const_gani.csv", tmp_path / "const_gani.csv"
    )
    _write_constraints_from_ground_truth(tmp_path)
    return tmp_path


class TestLoadConstraints:
    """Test _load_constraints."""

    def test_loads_valid_constraints(self, tmp_path: Path) -> None:
        """Test that a well-formed constraints.csv parses correctly."""
        path = tmp_path / "constraints.csv"
        path.write_text(
            "Property,Temperature,Target,Unit,Tolerance,Correlation\n"
            "density,25,700,kg/m^3,5,\n"
            "kinematic_viscosity,-20,5,mm^2/s,0.5,Arrhenius\n"
        )
        constraints = _load_constraints(path)
        expected_num_constraints = 2
        assert len(constraints) == expected_num_constraints
        assert constraints[0].target_property == "density"
        assert constraints[0].correlation == "Kendall-Monroe"
        assert constraints[1].correlation == "Arrhenius"

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Test that a nonexistent constraints.csv raises."""
        with pytest.raises(ValueError, match="does not exist"):
            _load_constraints(tmp_path / "constraints.csv")

    def test_raises_for_unrecognized_property(self, tmp_path: Path) -> None:
        """Test that an unrecognized Property value raises."""
        path = tmp_path / "constraints.csv"
        path.write_text(
            "Property,Temperature,Target,Unit,Tolerance\nflashpoint,25,1,K,1\n"
        )
        with pytest.raises(ValueError, match="unrecognized Property"):
            _load_constraints(path)

    def test_raises_for_incompatible_unit(self, tmp_path: Path) -> None:
        """Test that a unit incompatible with the property raises."""
        path = tmp_path / "constraints.csv"
        path.write_text(
            "Property,Temperature,Target,Unit,Tolerance\ndensity,25,1,kelvin,1\n"
        )
        with pytest.raises(ValueError, match="not compatible"):
            _load_constraints(path)


class TestSolveComposition:
    """Test solve_composition."""

    def test_recovers_known_composition(self, heptane_decane_dir: Path) -> None:
        """Test that constraints derived from a known fuel recover its composition."""
        fuel = solve_composition(heptane_decane_dir, name="recovered")
        expected = ExampleFuels.heptane_decane

        assert fuel.weights == pytest.approx(expected.weights, abs=2.0)

        temp = Q_(CONSTRAINT_TEMP_C, "degC")
        assert fuel.density(temp).magnitude == pytest.approx(
            expected.density(temp).magnitude, rel=0.02
        )
        assert fuel.kinematic_viscosity(temp).magnitude == pytest.approx(
            expected.kinematic_viscosity(temp).magnitude, rel=0.02
        )
        assert fuel.dynamic_viscosity(temp).magnitude == pytest.approx(
            expected.dynamic_viscosity(temp).magnitude, rel=0.02
        )

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        """Test that a nonexistent directory raises."""
        with pytest.raises(ValueError, match="is not a directory"):
            solve_composition(tmp_path / "nonexistent")

    def test_raises_for_missing_const_gani_csv(self, tmp_path: Path) -> None:
        """Test that a directory missing const_gani.csv raises."""
        with pytest.raises(ValueError, match=re.escape("const_gani.csv")):
            solve_composition(tmp_path)

    def test_raises_for_missing_constraints_csv(self, tmp_path: Path) -> None:
        """Test that a directory missing constraints.csv raises."""
        shutil.copy(
            FUEL_DIR / "heptane-decane" / "const_gani.csv",
            tmp_path / "const_gani.csv",
        )
        with pytest.raises(ValueError, match=re.escape("constraints.csv")):
            solve_composition(tmp_path)

    def test_raises_for_mismatched_groups(self, tmp_path: Path) -> None:
        """Test that a const_gani.csv with unexpected groups raises."""
        (tmp_path / "const_gani.csv").write_text("Family,foo,bar\nA,1,0\nB,0,1\n")
        _write_constraints_from_ground_truth(tmp_path)
        with pytest.raises(ValueError, match="do not match"):
            solve_composition(tmp_path)
