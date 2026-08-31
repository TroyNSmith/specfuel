"""specfuel Fuel module."""

from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from pint import Quantity
from pint.facets.plain import PlainQuantity
from pydantic import BaseModel, ConfigDict, model_validator

from .gcm import ConstGani
from .types import FLOAT_VECTOR, INT_MATRIX
from .units import Q_

CONST_GANI = ConstGani()


def _load_gc_data(
    path: str | Path,
) -> tuple[list[str], np.ndarray, list[str] | None, list[str] | None]:
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

    recognized_headers = ["Compound", "Weight %", "Formula", "PelePhysics Key"]
    if not any(header in recognized_headers for header in df.columns):
        unrecognized_headers = [
            header for header in df.columns if header not in recognized_headers
        ]
        msg = f"'{path}' contains unrecognized headers: {unrecognized_headers}."
        raise ValueError(msg)

    required_headers = ["Compound", "Weight %"]
    if not all(header in df.columns for header in required_headers):
        msg = f"'{path}' does not contain required headers: {required_headers}."
        raise ValueError(msg)

    compounds = df["Compound"].tolist()
    weights = df["Weight %"].to_numpy(dtype=np.float64)
    if len(compounds) != len(weights):
        msg = f"Number of compounds != number of weights in '{path}'."
        raise ValueError(msg)

    formulas = df["Formula"].tolist() if "Formula" in df.columns else None
    if formulas is not None and len(formulas) != len(compounds):
        msg = f"Number of formulas != number of compounds in '{path}'."
        raise ValueError(msg)

    pelephysics_keys = (
        df["PelePhysics Key"].tolist() if "PelePhysics Key" in df.columns else None
    )
    if pelephysics_keys is not None and len(pelephysics_keys) != len(compounds):
        msg = f"Number of PelePhysics keys != number of compounds in '{path}'."
        raise ValueError(msg)

    return compounds, weights, formulas, pelephysics_keys


def load_const_gani_decomp(
    path: str | Path,
) -> tuple[list[str], list[str], INT_MATRIX]:
    """Load GCM decomposition data from the const_gani.csv file.

    Returns
    -------
        Compound names (row index), group names (columns), and the
        decomposition matrix.

    Raises
    ------
        ValueError: If the const_gani.csv file does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"'{path}' does not exist."
        raise ValueError(msg)

    if path.suffix != ".csv":
        msg = f"'{path}' is not a .CSV file."
        raise ValueError(msg)

    df: DataFrame = pd.read_csv(path, header=0, index_col=0)
    if not all(df.dtypes == np.int64):
        msg = f"'{path}' contains non-integer values."
        raise ValueError(msg)

    return list(df.index), list(df.columns), df.to_numpy(dtype=np.int64)


class Fuel(BaseModel):
    """Fuel class for specfuel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fuel properties
    name: str
    compounds: list[str]
    weights: FLOAT_VECTOR
    formulas: list[str] | None = None
    pelephysics_keys: list[str] | None = None

    # Component properties
    cg_groups: list[str]
    cg_decomp: INT_MATRIX

    @property
    def num_compounds(self) -> int:
        """Get the number of compounds in the fuel.

        Returns
        -------
            Number of compounds.
        """
        return len(self.compounds)

    @property
    def _mass_fractions(self) -> FLOAT_VECTOR:
        """Get the mass fractions of the compounds in the fuel.

        Returns
        -------
            Mass fractions of the compounds.
        """
        return self.weights / np.sum(self.weights)

    @property
    def _mole_fractions(self) -> FLOAT_VECTOR:
        """Get the mole fractions of the compounds in the fuel.

        Returns
        -------
            Mole fractions of the compounds.
        """
        return (
            self._mass_fractions
            * (1.0 / CONST_GANI.molecular_weights(self.cg_decomp).magnitude)
        ) / np.sum(
            self._mass_fractions
            * (1.0 / CONST_GANI.molecular_weights(self.cg_decomp).magnitude)
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
            self._mass_fractions @ method.densities(self.cg_decomp, temp),
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
        nu_i = method.kinematic_viscosities(self.cg_decomp, temp).to("m^2/s")

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

        compounds, weights, formulas, pelephysics_keys = _load_gc_data(gc_data)

        cg_decomp = path / "const_gani.csv"
        if not cg_decomp.exists():
            msg = f"'{path}' does not contain required file 'const_gani.csv'."
            raise ValueError(msg)

        _cg_compounds, cg_groups, cg_decomp_mat = load_const_gani_decomp(cg_decomp)

        return cls(
            name=path.name,
            compounds=compounds,
            weights=weights,
            formulas=formulas,
            pelephysics_keys=pelephysics_keys,
            cg_groups=cg_groups,
            cg_decomp=cg_decomp_mat,
        )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate that the weights sum to 100%."""
        if len(self.weights) != self.num_compounds:
            msg = f"Number of weights != number of compounds for fuel '{self.name}'."
            raise ValueError(msg)

        if not np.isclose(np.sum(self.weights), 100.0, atol=5e-1):
            msg = f"Weights for fuel '{self.name}' do not sum to 100%."
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_cg_decomp(self) -> Self:
        """Validate that cg_decomp and cg_groups are consistent with ConstGani."""
        if self.cg_groups != ConstGani().group_names:
            msg = f"groups for fuel '{self.name}' do not match ConstGani group names.\n"
            raise ValueError(msg)

        if self.cg_decomp.shape[0] != self.num_compounds:
            msg = f"Rows in cg_decomp != number of compounds for fuel '{self.name}'."
            raise ValueError(msg)

        if self.cg_decomp.shape[1] != ConstGani().num_groups:
            msg = (
                f"Columns in cg_decomp != number of cg groups for fuel '{self.name}'.\n"
                f"Expected {ConstGani().num_groups}, got {self.cg_decomp.shape[1]}.\n"
            )
            raise ValueError(msg)

        return self
