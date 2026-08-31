"""fuellib Fuel module."""

from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from pint import Quantity
from pint.facets.plain import PlainQuantity
from pydantic import BaseModel, ConfigDict, model_validator

from .comp import Component
from .decomp import ConstGaniDecomp
from .gcm import ConstGani
from .types import FLOAT_VECTOR, INT_MATRIX
from .units import Q_

CONST_GANI = ConstGani()


def _load_gc_data(
    path: str | Path,
) -> tuple[list[str], np.ndarray, list[str] | None, list[str] | None, list[str] | None]:
    """Load composition data from a .CSV file.

    Parameters
    ----------
    path
        Path to the .CSV file containing composition data.

    Raises
    ------
        ValueError: If the GC data file does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"'{path}' does not exist."
        raise ValueError(msg)

    if path.suffix != ".csv":
        msg = f"'{path}' is not a .CSV file."
        raise ValueError(msg)

    df: DataFrame = pd.read_csv(path, header=0)

    recognized_headers = [
        "Family",
        "Reference Compound",
        "Weight %",
        "Formula",
        "PelePhysics Key",
    ]
    if not any(header in recognized_headers for header in df.columns):
        unrecognized_headers = [
            header for header in df.columns if header not in recognized_headers
        ]
        msg = f"'{path}' contains unrecognized headers: {unrecognized_headers}."
        raise ValueError(msg)

    required_headers = ["Family", "Weight %"]
    if not all(header in df.columns for header in required_headers):
        msg = f"'{path}' does not contain required headers: {required_headers}."
        raise ValueError(msg)

    families = df["Family"].tolist()
    weights = df["Weight %"].to_numpy(dtype=np.float64)
    if len(families) != len(weights):
        msg = f"Number of families != number of weights in '{path}'."
        raise ValueError(msg)

    reference_compounds = (
        df["Reference Compound"].tolist()
        if "Reference Compound" in df.columns
        else None
    )
    if reference_compounds is not None and len(reference_compounds) != len(families):
        msg = f"Number of reference compounds != number of families in '{path}'."
        raise ValueError(msg)

    formulas = df["Formula"].tolist() if "Formula" in df.columns else None
    if formulas is not None and len(formulas) != len(families):
        msg = f"Number of formulas != number of families in '{path}'."
        raise ValueError(msg)

    pelephysics_keys = (
        df["PelePhysics Key"].tolist() if "PelePhysics Key" in df.columns else None
    )
    if pelephysics_keys is not None and len(pelephysics_keys) != len(families):
        msg = f"Number of PelePhysics keys != number of families in '{path}'."
        raise ValueError(msg)

    return families, weights, reference_compounds, formulas, pelephysics_keys


class Fuel(BaseModel):
    """Fuel class for specfuel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fuel properties
    name: str
    cg_groups: list[str]
    components: list[Component]

    @property
    def num_components(self) -> int:
        """Get the number of components in the fuel.

        Returns
        -------
            Number of components.
        """
        return len(self.components)

    @property
    def component_names(self) -> list[str]:
        """Get the names of the components in the fuel.

        Returns
        -------
            Names of the components, in order.
        """
        return [component.name for component in self.components]

    @property
    def _weights(self) -> FLOAT_VECTOR:
        """Assemble a weight vector from the fuel's components.

        Returns
        -------
            Weight % for each component, in order.
        """
        return np.array([component.weight for component in self.components])

    @property
    def _cg_decomp(self) -> INT_MATRIX:
        """Assemble a decomposition matrix from the fuel's components.

        Returns
        -------
            Decomposition matrix for the fuel, one row per component.
        """
        return np.array([component.cg_decomp for component in self.components])

    @property
    def _mass_fractions(self) -> FLOAT_VECTOR:
        """Get the mass fractions of the components in the fuel.

        Returns
        -------
            Mass fractions of the components.
        """
        return self._weights / np.sum(self._weights)

    @property
    def _mole_fractions(self) -> FLOAT_VECTOR:
        """Get the mole fractions of the components in the fuel.

        Returns
        -------
            Mole fractions of the components.
        """
        return (
            self._mass_fractions
            * (1.0 / CONST_GANI.molecular_weights(self._cg_decomp).magnitude)
        ) / np.sum(
            self._mass_fractions
            * (1.0 / CONST_GANI.molecular_weights(self._cg_decomp).magnitude)
        )

    def density(
        self, temp: Quantity | PlainQuantity, *, method: ConstGani = CONST_GANI
    ) -> Quantity:
        """Get the density of the fuel at a given temperature.

        Parameters
        ----------
        temp
            Temperature at which to calculate the density.
        method
            Method to use for calculating the density. Defaults to ConstGani.

        Returns
        -------
            Density of the fuel.
        """
        return cast(
            "Quantity",
            self._mass_fractions @ method.densities(self._cg_decomp, temp),
        )

    def kinematic_viscosity(
        self,
        temp: Quantity | PlainQuantity,
        *,
        method: ConstGani = CONST_GANI,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> Quantity:
        """Get the kinematic viscosity of the fuel at a given temperature.

        Parameters
        ----------
        temp
            Temperature at which to calculate the kinematic viscosity.
        method
            Method to use for calculating kinematic viscosity. Defaults to ConstGani.
        correlation
            Mixing model. Defaults to "Kendall-Monroe".

        Returns
        -------
            Kinematic viscosity of the fuel.
        """
        nu_i = method.kinematic_viscosities(self._cg_decomp, temp).to("m^2/s")

        if correlation == "Arrhenius":
            nu_mag = np.exp(np.sum(self._mole_fractions * np.log(nu_i.magnitude)))
            return cast("Quantity", Q_(nu_mag, "m^2/s"))

        return (np.sum(self._mass_fractions * (nu_i ** (1.0 / 3.0))) ** 3.0).to("m^2/s")

    def dynamic_viscosity(
        self,
        temp: Quantity | PlainQuantity,
        *,
        method: ConstGani = CONST_GANI,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> Quantity:
        """Get the dynamic viscosity of the fuel at a given temperature.

        Parameters
        ----------
        temp
            Temperature at which to calculate the dynamic viscosity.
        method
            Method to use for calculating the dynamic viscosity. Defaults to ConstGani.
        correlation
            Mixing model. Defaults to "Kendall-Monroe".

        Returns
        -------
            Dynamic viscosity of the fuel.
        """
        nu = self.kinematic_viscosity(temp, method=method, correlation=correlation)
        rho = self.density(temp, method=method)
        return (nu * rho).to("Pa*s")

    @classmethod
    def from_directory(cls, path: str | Path) -> "Fuel":
        """Create a Fuel instance from a directory.

        Parameters
        ----------
        path : str | Path
            Path to the directory containing fuel data.

        Returns
        -------
            An instance of the Fuel class.
        """
        # Implement logic to read data from the directory and create a Fuel instance
        path = Path(path)
        if not path.is_dir() or not path.exists():
            msg = f"Provided path '{path}' is not a directory."
            raise ValueError(msg)

        gc_data = path / "composition.csv"
        if not gc_data.exists():
            msg = f"'{path}' does not contain required file 'composition.csv'."
            raise ValueError(msg)

        families, weights, reference_compounds, formulas, pelephysics_keys = (
            _load_gc_data(gc_data)
        )

        cg_decomp = path / "const_gani.csv"
        if not cg_decomp.exists():
            msg = f"'{path}' does not contain required file 'const_gani.csv'."
            raise ValueError(msg)

        decomp = ConstGaniDecomp.from_csv(cg_decomp)
        if len(families) != len(decomp.families):
            msg = (
                f"'{gc_data}' and '{cg_decomp}' do not contain the same "
                f"number of components."
            )
            raise ValueError(msg)

        components = [
            Component(
                name=family,
                reference_compound=(
                    reference_compounds[i] if reference_compounds else None
                ),
                formula=formulas[i] if formulas else None,
                pelephysics_key=pelephysics_keys[i] if pelephysics_keys else None,
                weight=weights[i],
                cg_decomp=decomp.decomp[i],
            )
            for i, family in enumerate(families)
        ]

        return cls(
            name=path.name,
            cg_groups=decomp.groups,
            components=components,
        )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate that the weights sum to 100%."""
        if not np.isclose(np.sum(self._weights), 100.0, atol=5e-1):
            msg = f"Weights for fuel '{self.name}' do not sum to 100%."
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_cg_decomp(self) -> Self:
        """Validate that cg_groups and each component's cg_decomp match ConstGani."""
        if self.cg_groups != ConstGani().group_names:
            msg = f"groups for fuel '{self.name}' do not match ConstGani group names.\n"
            raise ValueError(msg)

        num_groups = ConstGani().num_groups
        for component in self.components:
            if len(component.cg_decomp) != num_groups:
                msg = (
                    f"cg_decomp for component '{component.name}' in fuel "
                    f"'{self.name}' does not match number of cg groups.\n"
                    f"Expected {num_groups}, got {len(component.cg_decomp)}.\n"
                )
                raise ValueError(msg)

        return self
