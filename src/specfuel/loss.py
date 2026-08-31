"""Loss functions for inverse problems."""

import jax
import jax.numpy as jnp
from jax import Array

from .corr import density
from .stock import Feedstock

SPARSITY_PENALTY_WEIGHT = 1


def _sparsity_penalty(weights: Array) -> Array:
    """Get a barrier that vanishes at 0/1 and peaks in between each weight."""
    return jnp.sum(weights * (1 - weights))


def loss_function(
    logits: Array, feedstock: Feedstock, temp: Array, target: Array
) -> Array:
    """Get the density squared error plus a sparsity penalty."""
    weights = jax.nn.softmax(logits)
    density_error = (density(weights, feedstock, temp) - target) ** 2
    penalty = SPARSITY_PENALTY_WEIGHT * _sparsity_penalty(weights)
    return density_error + penalty
