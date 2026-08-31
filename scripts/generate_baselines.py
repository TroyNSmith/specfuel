"""Generate baseline property CSVs used by the regression test suite.

Run via ``pixi run generate-baselines``. Recomputes Fuel/ConstGani properties
for every ``ExampleFuels`` fuel and overwrites the CSVs under
``tests/baseline_properties/``. These baselines are self-referential (derived
from the current implementation), not independently verified physical data.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pint import Quantity
from pint.facets.plain import PlainQuantity

from specfuel.data import ExampleFuels
from specfuel.fuel import Fuel
from specfuel.gcm import ConstGani
from specfuel.types import INT_MATRIX
from specfuel.units import Q_

BASELINE_DIR = Path(__file__).resolve().parent.parent / "tests" / "baseline_properties"
TEMPERATURES_C = np.arange(-40, 101, 20)

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
CORRELATIONS = ["Kendall-Monroe", "Arrhenius"]


def _example_fuels() -> list[Fuel]:
    """Get the list of all ExampleFuels instances.

    Returns
    -------
        List of Fuel instances.
    """
    return [
        ExampleFuels.decane,
        ExampleFuels.heptane,
        ExampleFuels.heptane_decane,
        ExampleFuels.jet_a,
        ExampleFuels.posf11498,
    ]


def generate_const_gani_baseline(fuel: Fuel) -> pd.DataFrame:
    """Compute per-family ConstGani baseline properties for a fuel.

    Parameters
    ----------
    fuel
        Fuel whose family decomposition is used to compute properties.

    Returns
    -------
        Tidy DataFrame with columns [family, property, temperature_C, value, unit].
    """
    rows = []
    for prop_name, stp_func in STP_PROPERTIES.items():
        values = stp_func(fuel.cg_decomp)
        for family, value in zip(fuel.families, values.magnitude, strict=True):
            rows.append(
                {
                    "family": family,
                    "property": prop_name,
                    "temperature_c": np.nan,
                    "value": value,
                    "unit": str(values.units),
                }
            )

    for prop_name, temp_func in TEMP_PROPERTIES.items():
        for temp_c in TEMPERATURES_C:
            values = temp_func(fuel.cg_decomp, Q_(temp_c, "celsius"))
            for family, value in zip(fuel.families, values.magnitude, strict=True):
                rows.append(
                    {
                        "family": family,
                        "property": prop_name,
                        "temperature_c": temp_c,
                        "value": value,
                        "unit": str(values.units),
                    }
                )

    return pd.DataFrame(rows)


def generate_fuel_baseline(fuel: Fuel) -> pd.DataFrame:
    """Compute fuel-level baseline properties over a temperature range.

    Parameters
    ----------
    fuel
        Fuel to evaluate.

    Returns
    -------
        Tidy DataFrame with columns [property, correlation, temperature_C, value, unit].
    """
    rows = []
    for temp_c in TEMPERATURES_C:
        temp = Q_(temp_c, "celsius")

        density = fuel.density(temp)
        rows.append(
            {
                "property": "density",
                "correlation": "",
                "temperature_c": temp_c,
                "value": density.magnitude,
                "unit": str(density.units),
            }
        )

        for correlation in CORRELATIONS:
            kinematic = fuel.kinematic_viscosity(temp, correlation=correlation)
            dynamic = fuel.dynamic_viscosity(temp, correlation=correlation)
            rows.append(
                {
                    "property": "kinematic_viscosity",
                    "correlation": correlation,
                    "temperature_c": temp_c,
                    "value": kinematic.magnitude,
                    "unit": str(kinematic.units),
                }
            )
            rows.append(
                {
                    "property": "dynamic_viscosity",
                    "correlation": correlation,
                    "temperature_c": temp_c,
                    "value": dynamic.magnitude,
                    "unit": str(dynamic.units),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and write baseline CSVs for all ExampleFuels."""
    const_gani_dir = BASELINE_DIR / "const_gani"
    fuel_dir = BASELINE_DIR / "fuel"
    const_gani_dir.mkdir(parents=True, exist_ok=True)
    fuel_dir.mkdir(parents=True, exist_ok=True)

    for fuel in _example_fuels():
        generate_const_gani_baseline(fuel).to_csv(
            const_gani_dir / f"{fuel.name}.csv", index=False
        )
        generate_fuel_baseline(fuel).to_csv(fuel_dir / f"{fuel.name}.csv", index=False)


if __name__ == "__main__":
    main()
