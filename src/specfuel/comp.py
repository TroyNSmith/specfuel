"""Fuel components."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .gcm import FunctionalGroup, const1994

__all__ = ["Component"]


@dataclass
class Component:
    """A fuel component with composition and decomposition info."""

    name: str
    formula: str
    decomposition: dict[str, int]
    smiles: str | None = None

    gcm_method: Callable[[], dict[str, FunctionalGroup]] = const1994

    def groups(
        self, order: Literal[1, 2] | None = None
    ) -> list[tuple[FunctionalGroup, int]]:
        """Return the functional groups present in this component's decomposition.

        Parameters
        ----------
        order
            If given, restrict to first-order (1) or second-order (2) groups.
            If ``None``, return all groups.

        Returns
        -------
            One ``(FunctionalGroup, occurrence count)`` pair per matching group.

        Raises
        ------
        KeyError
            If a group in the decomposition is not present in the GCM data.
        """
        fgs = self.gcm_method()

        try:
            all_groups = [
                (fgs[group], count) for group, count in self.decomposition.items()
            ]
        except KeyError as e:
            msg = f"Unknown group in decomposition: {e}"
            raise KeyError(msg) from e

        if order is None:
            return all_groups
        return [(fg, count) for fg, count in all_groups if fg.order == order]


def molecular_weight(comp: Component) -> float:
    """Calculate molecular weight (g/mol) from group contributions."""
    return sum(n * fg.molecular_weight for fg, n in comp.groups())


def critical_temperature(comp: Component) -> float:
    """Calculate critical temperature (K) from group contributions."""
    return 181.128 * math.log(sum(n * fg.t_crit for fg, n in comp.groups()))


def critical_pressure(comp: Component) -> float:
    """Calculate critical pressure (bar) from group contributions."""
    return 1.3705 + (sum(n * fg.p_crit for fg, n in comp.groups()) + 0.10022) ** (-2)


def critical_volume(comp: Component) -> float:
    """Calculate critical volume (cm^3/mol) from group contributions."""
    return -0.00435 + sum(n * fg.v_crit for fg, n in comp.groups())


def boiling_temperature(comp: Component) -> float:
    """Calculate boiling temperature (K) from group contributions."""
    return 204.359 * math.log(sum(n * fg.t_boil for fg, n in comp.groups()))


def melting_temperature(comp: Component) -> float:
    """Calculate melting temperature (K) from group contributions."""
    return 102.425 * math.log(sum(n * fg.t_melt for fg, n in comp.groups()))


def enthalpy_of_formation(comp: Component) -> float:
    """Calculate enthalpy of formation (kJ/mol) from group contributions."""
    return 10.835 + sum(n * fg.h_form for fg, n in comp.groups())


def gibbs_free_energy(comp: Component) -> float:
    """Calculate Gibbs free energy (kJ/mol) from group contributions."""
    return -14.828 + sum(n * fg.g_free for fg, n in comp.groups())


def vaporization_enthalpy_stp(comp: Component) -> float:
    """Calculate enthalpy of vaporization at 298 K (kJ/mol)."""
    return 6.829 + sum(n * fg.h_vap for fg, n in comp.groups())


def acentric_factor(comp: Component) -> float:
    """Calculate the acentric factor (omega) from group contributions."""
    w = sum(n * fg.acentric_factor for fg, n in comp.groups())
    return 0.4085 * math.log(w + 1.1507) ** (1.0 / 0.5050)


def molar_liquid_volume_stp(comp: Component) -> float:
    """Calculate molar liquid volume at 298 K (cm^3/mol)."""
    return 0.01211 + sum(n * fg.molar_liquid_vol for fg, n in comp.groups())


def heat_capacity_stp(comp: Component) -> float:
    """Calculate molar specific heat at 298 K (J/mol/K)."""
    return sum(n * fg.heat_cap_a for fg, n in comp.groups()) - 19.7779


def heat_capacity_b(comp: Component) -> float:
    """Return the heat capacity temperature correction coefficient B."""
    return sum(n * fg.heat_cap_b for fg, n in comp.groups())


def heat_capacity_c(comp: Component) -> float:
    """Return the heat capacity temperature correction coefficient C."""
    return sum(n * fg.heat_cap_c for fg, n in comp.groups())
