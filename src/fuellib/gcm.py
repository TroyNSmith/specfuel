"""GCM method classes."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pint import Quantity
from pint.facets.plain import PlainQuantity

from .types import INT_MATRIX
from .units import Q_

PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / "data"


class ConstGani:
    """Class to parse, store, and compute GCM from Constantinou Gani."""

    group_data: pd.DataFrame

    def __init__(self) -> None:
        """Initialize the ConstGani class."""
        self.group_data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load GCM data from the CSV file.

        Returns
        -------
            pd.DataFrame: Parsed GCM data.
        """
        return pd.read_csv(self.csv_path, index_col=0, header=0)

    @property
    def csv_path(self) -> Path:
        """Get the path to the CSV file.

        Returns
        -------
            Path to the .csv data file.
        """
        return DATA_DIR / "gcm" / "const_gani.csv"

    @property
    def num_groups(self) -> int:
        """Get the number of groups in the GCM data.

        Returns
        -------
            Number of groups.
        """
        return len(self.group_data.columns)

    @property
    def group_names(self) -> list[str]:
        """Get the names of the groups in the GCM data.

        Returns
        -------
            List of group names.
        """
        return list(self.group_data.columns)

    def molecular_weights(self, decomp: INT_MATRIX) -> Quantity:
        """Get the molecular weights for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the compound.

        Returns
        -------
            Molecular weights for each compound in the fuel.
        """
        cont_term = self.group_data.loc["MW"].to_numpy(dtype=np.float64) @ decomp.T
        return cast("Quantity", Q_(cont_term, "g/mol"))

    def critical_temperatures(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard critical temperatures for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the compound.

        Returns
        -------
            Critical temperatures for each compound in the fuel.
        """
        tc0 = Q_(181.128, "kelvin")
        cont_term = self.group_data.loc["tck"].to_numpy(dtype=np.float64) @ decomp.T
        return tc0 * np.log(cont_term)

    def critical_pressures(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard critical pressures for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.

        Returns
        -------
            Critical pressures for each compound in the fuel.
        """
        pc1 = Q_(1.3705, "bar")
        pc2 = Q_(0.10022, "bar^(-1/2)")
        cont_term = Q_(
            self.group_data.loc["pck"].to_numpy(dtype=np.float64) @ decomp.T,
            "bar^(-1/2)",
        )
        return pc1 + (cont_term + pc2) ** (-2)

    def critical_volumes(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard critical volumes for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.

        Returns
        -------
            Critical volumes for each compound in the fuel.
        """
        vc0 = Q_(-0.00435, "m^3/kmol")
        cont_term = Q_(
            self.group_data.loc["vck"].to_numpy(dtype=np.float64) @ decomp.T, "m^3/kmol"
        )
        return vc0 + cont_term

    def boiling_temperatures(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard boiling points for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.

        Returns
        -------
            Boiling points for each compound in the fuel.
        """
        tb0 = Q_(204.359, "kelvin")
        cont_term = self.group_data.loc["tbk"].to_numpy(dtype=np.float64) @ decomp.T
        return tb0 * np.log(cont_term)

    def stp_molar_liquid_volumes(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard molar liquid volumes for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.

        Returns
        -------
            Molar liquid volumes for each compound in the fuel.
        """
        vm0 = Q_(0.01211, "m^3/kmol")
        cont_term = Q_(
            self.group_data.loc["vmk"].to_numpy(dtype=np.float64) @ decomp.T, "m^3/kmol"
        )
        return vm0 + cont_term

    def acentric_factors(self, decomp: INT_MATRIX) -> Quantity:
        """Get the standard acentric factors for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.

        Returns
        -------
            Acentric factors for each compound in the fuel.
        """
        w0 = Q_(0.4085, "dimensionless")
        cont_term = self.group_data.loc["wk"].to_numpy(dtype=np.float64) @ decomp.T
        return w0 * np.log(cont_term + 1.1507) ** (1.0 / 0.5050)

    def molar_liquid_volumes(
        self, decomp: INT_MATRIX, temp: Quantity | PlainQuantity
    ) -> Quantity:
        """Get the molar liquid volumes for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.
        temp
            Temperature at which to calculate the molar liquid volume.

        Returns
        -------
            Molar liquid volumes for each compound in the fuel.
        """
        temp = temp.to("kelvin")

        tc = self.critical_temperatures(decomp)
        w = self.acentric_factors(decomp)
        vm_stp = self.stp_molar_liquid_volumes(decomp)

        stp_term = (1 - (298.0 / tc.magnitude)) ** (2.0 / 7.0)
        phi = np.where(
            temp.magnitude > tc.magnitude,
            -stp_term,
            ((1 - (temp.magnitude / tc.magnitude)) ** (2.0 / 7.0)) - stp_term,
        )

        z = 0.29056 - 0.08775 * w.magnitude
        return vm_stp * np.power(z, phi)

    def densities(self, decomp: INT_MATRIX, temp: Quantity | PlainQuantity) -> Quantity:
        """Get the standard densities for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.
        temp
            Temperature at which to calculate the density.

        Returns
        -------
            Densities for each compound in the fuel.
        """
        return (
            self.molecular_weights(decomp) / self.molar_liquid_volumes(decomp, temp)
        ).to("kg/m^3")

    def kinematic_viscosities(
        self, decomp: INT_MATRIX, temp: Quantity | PlainQuantity
    ) -> Quantity:
        """Get the kinematic viscosities for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.
        temp
            Temperature at which to calculate the kinematic viscosity.

        Returns
        -------
            Kinematic viscosities for each compound in the fuel.
        """
        temp = temp.to("celsius")
        tb = self.boiling_temperatures(decomp).to("celsius")
        num = 442.78 + 1.6452 * tb.magnitude
        denom = temp.magnitude + 239 - 0.19 * tb.magnitude
        return cast("Quantity", Q_(np.exp(-3.0171 + num / denom), "mm^2/s"))

    def dynamic_viscosities(
        self, decomp: INT_MATRIX, temp: Quantity | PlainQuantity
    ) -> Quantity:
        """Get the dynamic viscosities for each compound in a fuel.

        Parameters
        ----------
        decomp
            Decomposition matrix for the fuel.
        temp
            Temperature at which to calculate the dynamic viscosity.

        Returns
        -------
            Dynamic viscosities for each compound in the fuel.
        """
        return self.kinematic_viscosities(decomp, temp) * self.densities(decomp, temp)
