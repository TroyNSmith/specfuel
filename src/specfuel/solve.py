"""Solvers for inverse problems."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from .corr import density
from .loss import loss_function
from .stock import Feedstock


def solve_composition(
    feedstock: Feedstock,
    temp: Array,
    target: Array,
    *,
    n_steps: int = 20000,
    learning_rate: float = 0.1,
) -> tuple[Array, Array]:
    """Find a mass-fraction composition of `feedstock` matching a target density.

    Parameters
    ----------
    feedstock
        Catalog of available components.
    temp
        Temperature, in kelvin, at which to match the target density.
    target
        Target density, in g/L.
    n_steps
        Number of optimization steps to take.
    learning_rate
        Learning rate for the Adam optimizer.

    Returns
    -------
        A `(weights, achieved_density)` pair: the mass fraction of each
        feedstock component and the resulting mixture density.
    """
    logits = jnp.zeros(feedstock.molecular_weights.shape[0])
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(logits)

    @jax.jit
    def _step(logits: Array, opt_state: optax.OptState) -> tuple[Any, optax.OptState]:
        grad = jax.grad(loss_function)(logits, feedstock, temp, target)
        updates, opt_state = optimizer.update(grad, opt_state)
        return optax.apply_updates(logits, updates), opt_state

    for _ in range(n_steps):
        logits, opt_state = _step(logits, opt_state)

    weights = jax.nn.softmax(logits)
    return weights, density(weights, feedstock, temp)
