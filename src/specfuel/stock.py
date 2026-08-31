"""Feedstock data for SpecFuel."""

import re
from pathlib import Path
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from pandas import DataFrame

from fuellib.units import Q_

_MOLECULAR_WEIGHT_UNIT = "g/mol"
_CRITICAL_TEMPERATURE_UNIT = "kelvin"
_STP_MOLAR_LIQUID_VOLUME_UNIT = "L/mol"
_BOILING_TEMPERATURE_UNIT = "kelvin"
_ACENTRIC_FACTOR_UNIT = "dimensionless"

_MOLECULAR_WEIGHT_HEADER = re.compile(r"^Molecular Weight \((?P<unit>.+)\)$")
_CRITICAL_TEMPERATURE_HEADER = re.compile(r"^Critical Temperature \((?P<unit>.+)\)$")
_STP_MOLAR_LIQUID_VOLUME_HEADER = re.compile(
    r"^STP Molar Liquid Volume \((?P<unit>.+)\)$"
)
_BOILING_TEMPERATURE_HEADER = re.compile(r"^Boiling Temperature \((?P<unit>.+)\)$")
_ACENTRIC_FACTOR_HEADER = re.compile(r"^Acentric Factor$")


def _find_column(df: DataFrame, pattern: re.Pattern[str], path: str | Path) -> str:
    """Find the single column in `df` whose header matches `pattern`.

    Raises
    ------
        ValueError: If no column in `df` matches `pattern`.
    """
    for column in df.columns:
        if pattern.match(column):
            return column

    msg = f"'{path}' does not contain a column matching '{pattern.pattern}'."
    raise ValueError(msg)


def _load_quantity_column(
    df: DataFrame, pattern: re.Pattern[str], to_unit: str, path: str | Path
) -> Array:
    """Find a labeled column and convert its values to `to_unit`.

    Parameters
    ----------
    df
        DataFrame to search.
    pattern
        Header pattern to match; a named `unit` group is used as the
        column's source unit, otherwise `to_unit` is assumed.
    to_unit
        Unit to convert the column's values to.
    path
        Source path, used for error messages.

    Returns
    -------
        The column's values, converted to `to_unit`.
    """
    column = _find_column(df, pattern, path)
    match = pattern.match(column)
    assert match is not None  # noqa: S101 - guaranteed by _find_column

    unit = match.groupdict().get("unit", to_unit)
    quantity = Q_(df[column].to_numpy(dtype=np.float64), unit).to(to_unit)
    return quantity.magnitude


def _check_positive(values: Array, name: str) -> None:
    """Raise if any element of `values` is not positive.

    Raises
    ------
        ValueError: If any element of `values` is not positive.
    """
    if not np.all(np.asarray(values) > 0):
        msg = f"{name} must be positive."
        raise ValueError(msg)


class Feedstock(eqx.Module):
    """Catalog of feedstock components available for blending."""

    families: tuple[str, ...] = eqx.field(static=True)
    reference_compounds: tuple[str, ...] = eqx.field(static=True)
    molecular_weights: Array
    critical_temperatures: Array
    stp_molar_liquid_volumes: Array
    boiling_temperatures: Array
    acentric_factors: Array

    def __check_init__(self) -> None:
        """Validate array lengths and value signs.

        Raises
        ------
            ValueError: If the component count is inconsistent across
                fields, or if any molecular weight/critical temperature/STP
                molar liquid volume is not positive.
        """
        lengths = {
            len(self.families),
            len(self.reference_compounds),
            self.molecular_weights.shape[0],
            self.critical_temperatures.shape[0],
            self.stp_molar_liquid_volumes.shape[0],
            self.boiling_temperatures.shape[0],
            self.acentric_factors.shape[0],
        }
        if len(lengths) > 1:
            msg = (
                "families, reference_compounds, molecular_weights, "
                "critical_temperatures, stp_molar_liquid_volumes, and "
                "acentric_factors must have the same length."
            )
            raise ValueError(msg)

        _check_positive(self.molecular_weights, "molecular_weights")
        _check_positive(self.critical_temperatures, "critical_temperatures")
        _check_positive(self.stp_molar_liquid_volumes, "stp_molar_liquid_volumes")
        _check_positive(self.boiling_temperatures, "boiling_temperatures")

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        """Load a feedstock catalog from a CSV file.

        Parameters
        ----------
        path
            Path to a CSV file with `Family`, `Reference Compound`,
            `Molecular Weight (<unit>)`, `Critical Temperature (<unit>)`,
            `STP Molar Liquid Volume (<unit>)`, and `Acentric Factor` columns
            (see `tests/scratch.csv` for the format).

        Returns
        -------
            A `Feedstock` with molecular weights/critical temperatures/STP
            molar liquid volumes converted to canonical units (g/mol,
            kelvin, L/mol); acentric factors are dimensionless and stored
            as-is.

        Raises
        ------
            ValueError: If the file does not exist, is not a .CSV file, or
                is missing a required column.
        """
        path = Path(path)
        if not path.exists():
            msg = f"'{path}' does not exist."
            raise ValueError(msg)

        if path.suffix != ".csv":
            msg = f"'{path}' is not a .CSV file."
            raise ValueError(msg)

        df: DataFrame = pd.read_csv(path, header=0)

        required_columns = ["Family", "Reference Compound"]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            msg = f"'{path}' does not contain required columns: {missing}."
            raise ValueError(msg)

        molecular_weights = _load_quantity_column(
            df, _MOLECULAR_WEIGHT_HEADER, _MOLECULAR_WEIGHT_UNIT, path
        )
        critical_temperatures = _load_quantity_column(
            df, _CRITICAL_TEMPERATURE_HEADER, _CRITICAL_TEMPERATURE_UNIT, path
        )
        stp_molar_liquid_volumes = _load_quantity_column(
            df, _STP_MOLAR_LIQUID_VOLUME_HEADER, _STP_MOLAR_LIQUID_VOLUME_UNIT, path
        )
        boiling_temperatures = _load_quantity_column(
            df, _BOILING_TEMPERATURE_HEADER, _BOILING_TEMPERATURE_UNIT, path
        )
        acentric_factors = _load_quantity_column(
            df, _ACENTRIC_FACTOR_HEADER, _ACENTRIC_FACTOR_UNIT, path
        )

        return cls(
            families=tuple(df["Family"]),
            reference_compounds=tuple(df["Reference Compound"]),
            molecular_weights=molecular_weights,
            critical_temperatures=critical_temperatures,
            stp_molar_liquid_volumes=stp_molar_liquid_volumes,
            boiling_temperatures=boiling_temperatures,
            acentric_factors=acentric_factors,
        )

    def molar_liquid_volumes(self, temp: Array) -> Array:
        """Get the molar liquid volume of each component at `temp`.

        Jax-numpy implementation of the generalized Rackett equation.

        Parameters
        ----------
        temp
            Temperature, in kelvin, at which to evaluate the molar liquid
            volume.

        Returns
        -------
            Molar liquid volumes, in L/mol, one per feedstock component.
        """
        tc = self.critical_temperatures
        w = self.acentric_factors
        vm_stp = self.stp_molar_liquid_volumes

        stp_term = (1 - (298.0 / tc)) ** (2.0 / 7.0)
        phi = jnp.where(
            temp > tc,
            -stp_term,
            ((1 - (temp / tc)) ** (2.0 / 7.0)) - stp_term,
        )

        z = 0.29056 - 0.08775 * w
        return vm_stp * jnp.power(z, phi)

    def kinematic_viscosities(self, temp: Array) -> Array:
        """Get the kinematic viscosity of each component at `temp`.

        Parameters
        ----------
        temp
            Temperature, in kelvin, at which to evaluate the kinematic
            viscosity.

        Returns
        -------
            Kinematic viscosities in mm^2/s.
        """
        # convert to C for Dutt's equation
        temp = temp - 273.15
        tb = self.boiling_temperatures - 273.15
        return 1e-6 * jnp.exp(
            -3.0171 + (442.78 + 1.6452 * tb) / (temp + 239 - 0.19 * tb)
        )
