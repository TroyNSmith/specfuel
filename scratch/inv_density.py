"""Design a mixture from Feedstock that yields a target density at a temperature."""

import jax.numpy as jnp

from specfuel.solve import solve_composition
from specfuel.stock import Feedstock

# --- Fill these in ---
TARGET_DENSITY = 800.0  # g/L
TARGET_TEMPERATURE = 298.15  # K
STOCK_CSV_PATH = "scratch.csv"

if __name__ == "__main__":
    stock = Feedstock.from_csv(STOCK_CSV_PATH)
    solved_weights, achieved_density = solve_composition(
        stock, jnp.asarray(TARGET_TEMPERATURE), jnp.asarray(TARGET_DENSITY)
    )

    for family, weight in zip(stock.families, solved_weights, strict=True):
        print(f"{family}: {weight:.4f}")

    print(f"\nTarget density:   {TARGET_DENSITY:.3f} g/L")
    print(f"Achieved density: {achieved_density:.3f} g/L")
