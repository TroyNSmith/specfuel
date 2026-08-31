"""Inverse-design module."""

from pathlib import Path
from typing import Literal, NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pandas import DataFrame
from pint.errors import DimensionalityError
from pydantic import BaseModel, ValidationError

from .fuel import Fuel, load_const_gani_decomp
from .gcm import ConstGani
from .types import INT_MATRIX
from .units import Q_

CONST_GANI = ConstGani()

TargetProperty = Literal["density", "kinematic_viscosity", "dynamic_viscosity"]
Correlation = Literal["Kendall-Monroe", "Arrhenius"]

# Units each target property is compared against internally (see gcm.py/fuel.py).
NATIVE_UNITS: dict[str, str] = {
    "density": "kg/m^3",
    "kinematic_viscosity": "m^2/s",
    "dynamic_viscosity": "Pa*s",
}


class Constraint(BaseModel):
    """A single property target used to solve for an ideal fuel composition."""

    target_property: TargetProperty
    temperature_c: float
    target: float
    unit: str
    tolerance: float
    correlation: Correlation = "Kendall-Monroe"


def _load_constraints(path: str | Path) -> list[Constraint]:
    """Load property constraints from a constraints.csv file.

    Parameters
    ----------
    path
        Path to the constraints.csv file.

    Returns
    -------
        One `Constraint` per row of the file.

    Raises
    ------
        ValueError: If the file is missing, malformed, or contains an
            unrecognized property or a unit incompatible with it.
    """
    path = Path(path)
    if not path.exists():
        msg = f"'{path}' does not exist."
        raise ValueError(msg)

    if path.suffix != ".csv":
        msg = f"'{path}' is not a .CSV file."
        raise ValueError(msg)

    df: DataFrame = pd.read_csv(path, header=0)

    required_headers = ["Property", "Temperature", "Target", "Unit", "Tolerance"]
    if not all(header in df.columns for header in required_headers):
        msg = f"'{path}' does not contain required headers: {required_headers}."
        raise ValueError(msg)

    properties = df["Property"].tolist()
    temperatures = df["Temperature"].tolist()
    targets = df["Target"].tolist()
    units = df["Unit"].tolist()
    tolerances = df["Tolerance"].tolist()
    correlations = (
        df["Correlation"].tolist() if "Correlation" in df.columns else [None] * len(df)
    )

    constraints = []
    for target_property, temp_c, target, unit, tolerance, correlation in zip(
        properties, temperatures, targets, units, tolerances, correlations, strict=True
    ):
        if target_property not in NATIVE_UNITS:
            msg = (
                f"'{path}' contains unrecognized Property '{target_property}'. "
                f"Expected one of {sorted(NATIVE_UNITS)}."
            )
            raise ValueError(msg)

        if correlation is None or (
            isinstance(correlation, float) and pd.isna(correlation)
        ):
            resolved_correlation = "Kendall-Monroe"
        else:
            resolved_correlation = correlation

        try:
            constraint = Constraint(
                target_property=target_property,
                temperature_c=temp_c,
                target=target,
                unit=str(unit),
                tolerance=tolerance,
                correlation=resolved_correlation,
            )
        except ValidationError as exc:
            msg = f"'{path}' contains an invalid constraint row: {exc}"
            raise ValueError(msg) from exc

        try:
            Q_(constraint.target, constraint.unit).to(
                NATIVE_UNITS[constraint.target_property]
            )
        except DimensionalityError as exc:
            msg = (
                f"Unit '{constraint.unit}' for property "
                f"'{constraint.target_property}' in '{path}' is not compatible "
                f"with '{NATIVE_UNITS[constraint.target_property]}'."
            )
            raise ValueError(msg) from exc

        constraints.append(constraint)

    if not constraints:
        msg = f"'{path}' does not contain any constraints."
        raise ValueError(msg)

    return constraints


class _CompositionLogits(eqx.Module):
    """Unconstrained logits parameterizing a candidate fuel composition."""

    logits: jax.Array


class _PreparedConstraint(NamedTuple):
    """A constraint with all pint/CSV setup work resolved to jax-ready values."""

    target_property: TargetProperty
    correlation: Correlation
    rho_i: jax.Array
    nu_i: jax.Array
    target: float
    tolerance: float


def _component_properties(
    decomp: INT_MATRIX, temp_c: float
) -> tuple[jax.Array, jax.Array]:
    """Precompute per-compound density and kinematic viscosity at a temperature.

    These don't depend on the composition being optimized, so they are
    computed once via `ConstGani` (numpy/pint) rather than inside the
    differentiable loss.

    Parameters
    ----------
    decomp
        Decomposition matrix for the candidate families.
    temp_c
        Temperature in degrees Celsius.

    Returns
    -------
        Per-compound (density [kg/m^3], kinematic viscosity [m^2/s]) arrays.
    """
    temp = Q_(temp_c, "degC")
    rho_i = CONST_GANI.densities(decomp, temp).to("kg/m^3").magnitude
    nu_i = CONST_GANI.kinematic_viscosities(decomp, temp).to("m^2/s").magnitude
    return jnp.asarray(rho_i), jnp.asarray(nu_i)


def _prepare_constraints(
    constraints: list[Constraint], decomp: INT_MATRIX
) -> list[_PreparedConstraint]:
    """Precompute jax-ready per-compound properties and native-unit targets.

    Parameters
    ----------
    constraints
        Constraints to prepare.
    decomp
        Decomposition matrix for the candidate families.

    Returns
    -------
        One `_PreparedConstraint` per input constraint.
    """
    prepared = []
    for c in constraints:
        rho_i, nu_i = _component_properties(decomp, c.temperature_c)
        native_unit = NATIVE_UNITS[c.target_property]
        prepared.append(
            _PreparedConstraint(
                target_property=c.target_property,
                correlation=c.correlation,
                rho_i=rho_i,
                nu_i=nu_i,
                target=Q_(c.target, c.unit).to(native_unit).magnitude,
                tolerance=Q_(c.tolerance, c.unit).to(native_unit).magnitude,
            )
        )
    return prepared


def _mass_fractions(logits: jax.Array) -> jax.Array:
    """Get mass fractions from unconstrained logits via softmax.

    Returns
    -------
        Mass fractions, non-negative and summing to 1.
    """
    return jax.nn.softmax(logits)


def _mole_fractions(mass_frac: jax.Array, molecular_weights: jax.Array) -> jax.Array:
    """Get mole fractions from mass fractions and molecular weights.

    Returns
    -------
        Mole fractions summing to 1.
    """
    x = mass_frac / molecular_weights
    return x / jnp.sum(x)


def _mixture_kinematic_viscosity(
    mass_frac: jax.Array,
    mole_frac: jax.Array,
    nu_i: jax.Array,
    correlation: Correlation,
) -> jax.Array:
    """Get the mixture kinematic viscosity (m^2/s) via a mixing correlation.

    Returns
    -------
        Mixture kinematic viscosity.
    """
    if correlation == "Arrhenius":
        return jnp.exp(mole_frac @ jnp.log(nu_i))
    return (mass_frac @ (nu_i ** (1.0 / 3.0))) ** 3.0


def _loss(
    model: _CompositionLogits,
    molecular_weights: jax.Array,
    prepared: list[_PreparedConstraint],
) -> jax.Array:
    """Sum of squared, tolerance-normalized residuals across all constraints.

    Returns
    -------
        Scalar loss (0 at a perfect fit).
    """
    mass_frac = _mass_fractions(model.logits)
    mole_frac = _mole_fractions(mass_frac, molecular_weights)

    total = jnp.asarray(0.0)
    for c in prepared:
        density = mass_frac @ c.rho_i
        kinematic = _mixture_kinematic_viscosity(
            mass_frac, mole_frac, c.nu_i, c.correlation
        )
        value = {
            "density": density,
            "kinematic_viscosity": kinematic,
            "dynamic_viscosity": kinematic * density,
        }[c.target_property]
        total = total + ((value - c.target) / c.tolerance) ** 2
    return total


def _vector_field(
    _t: float | jax.Array | np.ndarray,
    model: _CompositionLogits,
    args: tuple[jax.Array, list[_PreparedConstraint]],
) -> _CompositionLogits:
    """Gradient-flow vector field descending the constraint loss.

    Returns
    -------
        Negative gradient of the loss with respect to the composition logits.
    """
    molecular_weights, prepared = args
    grad = eqx.filter_grad(lambda m: _loss(m, molecular_weights, prepared))(model)
    return jax.tree_util.tree_map(lambda g: -g, grad)


def solve_composition(  # noqa: PLR0913 -- solver tuning knobs, all optional
    directory: str | Path,
    *,
    name: str | None = None,
    t1: float = 100.0,
    dt0: float = 0.05,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_steps: int = 200_000,
) -> Fuel:
    """Solve for a fuel composition that best satisfies property constraints.

    Reads a family x group decomposition matrix from `const_gani.csv` and a
    set of property targets from `constraints.csv` (both in `directory`), then
    finds the weight-fraction composition over those families that best
    satisfies the constraints in a least-squares sense. The composition is
    parameterized as `softmax(logits)` (so weights are always non-negative and
    sum to 100%) and solved for by integrating the gradient flow of the
    constraint loss to steady state with `diffrax`.

    This is a soft-constraint fit, not a feasibility solver: if the
    constraints are inconsistent or unreachable by any composition, the
    result is simply the best least-squares compromise.

    Parameters
    ----------
    directory
        Directory containing `const_gani.csv` and `constraints.csv`.
    name
        Name for the resulting `Fuel`. Defaults to the directory name.
    t1
        End time for the gradient-flow integration (only reached if the
        steady-state event doesn't trigger first).
    dt0
        Initial step size for the ODE solver.
    rtol
        Relative tolerance for the adaptive step size controller and
        steady-state detection.
    atol
        Absolute tolerance for the adaptive step size controller and
        steady-state detection.
    max_steps
        Maximum number of solver steps.

    Returns
    -------
        A `Fuel` with the solved composition.

    Raises
    ------
        ValueError: If `directory` or its required files are missing, or if
            `const_gani.csv`'s groups don't match `ConstGani`.
    """
    directory = Path(directory)
    if not directory.is_dir() or not directory.exists():
        msg = f"Provided path '{directory}' is not a directory."
        raise ValueError(msg)

    const_gani_path = directory / "const_gani.csv"
    if not const_gani_path.exists():
        msg = f"'{directory}' does not contain required file 'const_gani.csv'."
        raise ValueError(msg)

    families, groups, decomp = load_const_gani_decomp(const_gani_path)
    if groups != CONST_GANI.group_names:
        msg = f"Groups in '{const_gani_path}' do not match ConstGani group names.\n"
        raise ValueError(msg)

    constraints_path = directory / "constraints.csv"
    if not constraints_path.exists():
        msg = f"'{directory}' does not contain required file 'constraints.csv'."
        raise ValueError(msg)

    prepared = _prepare_constraints(_load_constraints(constraints_path), decomp)
    molecular_weights = jnp.asarray(
        CONST_GANI.molecular_weights(decomp).to("g/mol").magnitude
    )

    model0 = _CompositionLogits(logits=jnp.zeros(len(families)))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=dt0,
        y0=model0,
        args=(molecular_weights, prepared),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        event=diffrax.Event(diffrax.steady_state_event(rtol=rtol, atol=atol)),
        max_steps=max_steps,
    )

    weights = np.asarray(jax.nn.softmax(sol.ys.logits[-1])) * 100.0
    return Fuel(
        name=name or directory.name,
        families=families,
        weights=weights,
        cg_groups=groups,
        cg_decomp=decomp,
    )
