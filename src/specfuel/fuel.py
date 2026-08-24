"""Fuel mixtures."""

import math
from dataclasses import dataclass

from .comp import Component

__all__ = ["Fuel"]


@dataclass
class Fuel:
    """A fuel mixture composed of components at specified percent weights."""

    components: list[Component]
    percent_weights: list[float]

    def __post_init__(self) -> None:
        """Validate that components and percent weights are consistent.

        Raises
        ------
        ValueError
            If the number of components and percent weights differ, if any
            percent weight is negative, or if the percent weights do not sum
            to 100.
        """
        if len(self.components) != len(self.percent_weights):
            msg = (
                "components and percent_weights must have the same length "
                f"(got {len(self.components)} and {len(self.percent_weights)})"
            )
            raise ValueError(msg)

        if any(w < 0 for w in self.percent_weights):
            msg = "percent_weights must be non-negative"
            raise ValueError(msg)

        total = sum(self.percent_weights)
        if not math.isclose(total, 100, abs_tol=1e-6):
            msg = f"percent_weights must sum to 100, got {total}"
            raise ValueError(msg)

    def composition(self) -> list[tuple[Component, float]]:
        """Return the fuel's composition as component/percent-weight pairs.

        Returns
        -------
            One ``(Component, percent weight)`` pair per component in the fuel.
        """
        return list(zip(self.components, self.percent_weights, strict=True))
