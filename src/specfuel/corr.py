"""Correlation functions for mixtures."""

import jax.numpy as jnp
from jax import Array

from .stock import Feedstock


def density(weights: Array, feedstock: Feedstock, temp: Array) -> Array:
    """Get the mixture density, in g/L, for a mass-fraction composition.

    Parameters
    ----------
    weights
        Mass fraction of each feedstock component (should sum to 1).
    feedstock
        Catalog of available components.
    temp
        Temperature, in kelvin, at which to evaluate the density.

    Returns
    -------
        Mixture density, in g/L.
    """
    return jnp.sum(
        weights * feedstock.molecular_weights / feedstock.molar_liquid_volumes(temp)
    )
