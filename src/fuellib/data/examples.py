"""Example data fixtures for testing and demonstration purposes."""

from pathlib import Path

from ..fuel import Fuel

DATA_DIR = Path(__file__).resolve().parent
FUEL_DIR = DATA_DIR / "fuel"


class ExampleFuels:
    """Example fuels for testing and demonstration purposes."""

    decane = Fuel.from_directory(FUEL_DIR / "decane")
    heptane = Fuel.from_directory(FUEL_DIR / "heptane")
    heptane_decane = Fuel.from_directory(FUEL_DIR / "heptane-decane")
    jet_a = Fuel.from_directory(FUEL_DIR / "jet_a")
    posf11498 = Fuel.from_directory(FUEL_DIR / "posf11498")
