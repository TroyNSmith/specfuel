"""Shared pytest fixtures for the specfuel test suite."""

from pathlib import Path

from fuellib.data import ExampleFuels
from fuellib.fuel import Fuel

BASELINE_DIR = Path(__file__).resolve().parent / "baseline_properties"

FUELS_BY_NAME: dict[str, Fuel] = {
    fuel.name: fuel for fuel in vars(ExampleFuels).values() if isinstance(fuel, Fuel)
}
