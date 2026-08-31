"""Tests for stock module."""

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from specfuel.stock import Feedstock


def _write_csv(
    path: Path,
    *,
    mw_unit: str = "g/mol",
    tc_unit: str = "kelvin",
    vm_unit: str = "L/mol",
) -> None:
    """Write a minimal valid feedstock CSV to `path`."""
    path.write_text(
        f"Family,Reference Compound,Molecular Weight ({mw_unit}),"
        f"Critical Temperature ({tc_unit}),"
        f"STP Molar Liquid Volume ({vm_unit}),Acentric Factor\n"
        "n-C10,n-decane,142.000,617.700,0.196,0.490\n"
        "n-C11,n-undecane,156.000,638.800,0.212,0.535\n"
    )


class TestFromCsv:
    """Test Feedstock.from_csv."""

    def test_loads_canonical_units(self, tmp_path: Path) -> None:
        """Test that a CSV already in canonical units loads unchanged."""
        path = tmp_path / "feedstock.csv"
        _write_csv(path)
        feedstock = Feedstock.from_csv(path)

        assert feedstock.families == ("n-C10", "n-C11")
        assert feedstock.reference_compounds == ("n-decane", "n-undecane")
        np.testing.assert_allclose(feedstock.molecular_weights, [142.0, 156.0])
        np.testing.assert_allclose(feedstock.critical_temperatures, [617.7, 638.8])
        np.testing.assert_allclose(feedstock.stp_molar_liquid_volumes, [0.196, 0.212])
        np.testing.assert_allclose(feedstock.acentric_factors, [0.490, 0.535])

    def test_converts_non_canonical_units(self, tmp_path: Path) -> None:
        """Test that units are converted to canonical g/mol and L/mol."""
        path = tmp_path / "feedstock.csv"
        _write_csv(path, mw_unit="kg/mol", tc_unit="celsius", vm_unit="mL/mol")
        feedstock = Feedstock.from_csv(path)

        np.testing.assert_allclose(feedstock.molecular_weights, [142_000.0, 156_000.0])
        np.testing.assert_allclose(feedstock.critical_temperatures, [890.85, 911.95])
        np.testing.assert_allclose(
            feedstock.stp_molar_liquid_volumes, [0.000196, 0.000212]
        )

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Test that a nonexistent path raises."""
        with pytest.raises(ValueError, match="does not exist"):
            Feedstock.from_csv(tmp_path / "nonexistent.csv")

    def test_raises_for_non_csv_suffix(self, tmp_path: Path) -> None:
        """Test that a non-.csv file raises."""
        path = tmp_path / "feedstock.txt"
        _write_csv(path)
        with pytest.raises(ValueError, match=re.escape("is not a .CSV file")):
            Feedstock.from_csv(path)

    def test_raises_for_missing_required_column(self, tmp_path: Path) -> None:
        """Test that a missing Family/Reference Compound column raises."""
        path = tmp_path / "feedstock.csv"
        path.write_text(
            "Family,Molecular Weight (g/mol),Critical Temperature (kelvin),"
            "STP Molar Liquid Volume (L/mol)\n"
            "n-C10,142.000,617.700,0.196\n"
        )
        with pytest.raises(ValueError, match="required columns"):
            Feedstock.from_csv(path)

    def test_raises_for_missing_molecular_weight_column(self, tmp_path: Path) -> None:
        """Test that a missing Molecular Weight column raises."""
        path = tmp_path / "feedstock.csv"
        path.write_text(
            "Family,Reference Compound,Critical Temperature (kelvin),"
            "STP Molar Liquid Volume (L/mol)\n"
            "n-C10,n-decane,617.700,0.196\n"
        )
        with pytest.raises(ValueError, match="Molecular Weight"):
            Feedstock.from_csv(path)

    def test_raises_for_missing_critical_temperature_column(
        self, tmp_path: Path
    ) -> None:
        """Test that a missing Critical Temperature column raises."""
        path = tmp_path / "feedstock.csv"
        path.write_text(
            "Family,Reference Compound,Molecular Weight (g/mol),"
            "STP Molar Liquid Volume (L/mol)\n"
            "n-C10,n-decane,142.000,0.196\n"
        )
        with pytest.raises(ValueError, match="Critical Temperature"):
            Feedstock.from_csv(path)

    def test_raises_for_missing_stp_molar_liquid_volume_column(
        self, tmp_path: Path
    ) -> None:
        """Test that a missing STP Molar Liquid Volume column raises."""
        path = tmp_path / "feedstock.csv"
        path.write_text(
            "Family,Reference Compound,Molecular Weight (g/mol),"
            "Critical Temperature (kelvin)\n"
            "n-C10,n-decane,142.000,617.700\n"
        )
        with pytest.raises(ValueError, match="STP Molar Liquid Volume"):
            Feedstock.from_csv(path)

    def test_raises_for_missing_acentric_factor_column(self, tmp_path: Path) -> None:
        """Test that a missing Acentric Factor column raises."""
        path = tmp_path / "feedstock.csv"
        path.write_text(
            "Family,Reference Compound,Molecular Weight (g/mol),"
            "Critical Temperature (kelvin),"
            "STP Molar Liquid Volume (L/mol)\n"
            "n-C10,n-decane,142.000,617.700,0.196\n"
        )
        with pytest.raises(ValueError, match="Acentric Factor"):
            Feedstock.from_csv(path)


class TestValidation:
    """Test Feedstock's __check_init__ validation for direct construction."""

    def test_raises_on_length_mismatch(self) -> None:
        """Test that mismatched field lengths raise."""
        with pytest.raises(ValueError, match="same length"):
            Feedstock(
                families=("A", "B"),
                reference_compounds=("a",),
                molecular_weights=jax.numpy.asarray([1.0, 2.0]),
                critical_temperatures=jax.numpy.asarray([600.0, 620.0]),
                stp_molar_liquid_volumes=jax.numpy.asarray([1.0, 2.0]),
                acentric_factors=jax.numpy.asarray([0.1, 0.2]),
            )

    def test_raises_on_non_positive_molecular_weight(self) -> None:
        """Test that a non-positive molecular weight raises."""
        with pytest.raises(ValueError, match="molecular_weights must be positive"):
            Feedstock(
                families=("A",),
                reference_compounds=("a",),
                molecular_weights=jax.numpy.asarray([0.0]),
                critical_temperatures=jax.numpy.asarray([600.0]),
                stp_molar_liquid_volumes=jax.numpy.asarray([1.0]),
                acentric_factors=jax.numpy.asarray([0.1]),
            )

    def test_raises_on_non_positive_critical_temperature(self) -> None:
        """Test that a non-positive critical temperature raises."""
        with pytest.raises(ValueError, match="critical_temperatures must be positive"):
            Feedstock(
                families=("A",),
                reference_compounds=("a",),
                molecular_weights=jax.numpy.asarray([1.0]),
                critical_temperatures=jax.numpy.asarray([0.0]),
                stp_molar_liquid_volumes=jax.numpy.asarray([1.0]),
                acentric_factors=jax.numpy.asarray([0.1]),
            )

    def test_raises_on_non_positive_stp_molar_liquid_volume(self) -> None:
        """Test that a non-positive STP molar liquid volume raises."""
        with pytest.raises(
            ValueError, match="stp_molar_liquid_volumes must be positive"
        ):
            Feedstock(
                families=("A",),
                reference_compounds=("a",),
                molecular_weights=jax.numpy.asarray([1.0]),
                critical_temperatures=jax.numpy.asarray([600.0]),
                stp_molar_liquid_volumes=jax.numpy.asarray([0.0]),
                acentric_factors=jax.numpy.asarray([0.1]),
            )


class TestJaxCompatibility:
    """Test that Feedstock is usable as a JAX pytree."""

    def test_survives_jit(self, tmp_path: Path) -> None:
        """Test that a Feedstock instance can be passed through jax.jit.

        Regression test for equinox static fields needing to be hashable
        (families/reference_compounds must be tuples, not lists).
        """
        path = tmp_path / "feedstock.csv"
        _write_csv(path)
        feedstock = Feedstock.from_csv(path)

        total_mw = jax.jit(lambda f: f.molecular_weights.sum())(feedstock)

        np.testing.assert_allclose(total_mw, 298.0)


class TestMolarLiquidVolumes:
    """Test Feedstock.molar_liquid_volumes."""

    def _expected(self, tc: float, w: float, vm_stp: float, temp: float) -> float:
        """Compute the expected value via a plain-Python port of the formula."""
        stp_term = (1 - (298.0 / tc)) ** (2.0 / 7.0)
        phi = -stp_term if temp > tc else ((1 - (temp / tc)) ** (2.0 / 7.0)) - stp_term
        z = 0.29056 - 0.08775 * w
        return vm_stp * z**phi

    def test_matches_reference_formula(self, tmp_path: Path) -> None:
        """Test that output matches a plain-Python port of the formula."""
        path = tmp_path / "feedstock.csv"
        _write_csv(path)
        feedstock = Feedstock.from_csv(path)

        expected = [
            self._expected(617.7, 0.490, 0.196, 350.0),
            self._expected(638.8, 0.535, 0.212, 350.0),
        ]
        actual = feedstock.molar_liquid_volumes(jax.numpy.asarray(350.0))

        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_is_differentiable(self, tmp_path: Path) -> None:
        """Test that molar_liquid_volumes is traceable through jax.grad."""
        path = tmp_path / "feedstock.csv"
        _write_csv(path)
        feedstock = Feedstock.from_csv(path)

        grad = jax.grad(lambda temp: feedstock.molar_liquid_volumes(temp).sum())(
            jax.numpy.asarray(350.0)
        )

        assert np.isfinite(grad)

    def test_survives_jit(self, tmp_path: Path) -> None:
        """Test that molar_liquid_volumes is traceable through jax.jit."""
        path = tmp_path / "feedstock.csv"
        _write_csv(path)
        feedstock = Feedstock.from_csv(path)

        result = jax.jit(lambda f, temp: f.molar_liquid_volumes(temp))(
            feedstock, jax.numpy.asarray(350.0)
        )

        assert result.shape == (2,)
