"""fuellib Component module."""

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from .types import INT_VECTOR


class Component(BaseModel):
    """A single compound within a `Fuel` mixture.

    Strictly a data container: holds one compound's composition/decomposition
    data without relying on positional vectors/matrices shared across a fuel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    reference_compound: str | None = None
    formula: str | None = None
    pelephysics_key: str | None = None
    weight: float
    cg_decomp: INT_VECTOR

    @model_validator(mode="after")
    def validate_weight(self) -> Self:
        """Validate that weight is non-negative."""
        if self.weight < 0:
            msg = f"Weight for component '{self.name}' must be non-negative."
            raise ValueError(msg)

        return self
