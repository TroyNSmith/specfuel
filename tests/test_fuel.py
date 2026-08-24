"""Fuel module tests."""

import pytest

from specfuel.comp import Component
from specfuel.fuel import Fuel


def _component(name: str) -> Component:
    return Component(name=name, formula="CH4", decomposition={"CH3": 1, "CH2": 1})


def test__fuel_creation() -> None:
    """Test creating a Fuel instance."""
    comps = [_component("a"), _component("b")]
    fuel = Fuel(components=comps, percent_weights=[60, 40])

    assert fuel.components == comps
    assert fuel.percent_weights == [60, 40]


def test__fuel_composition() -> None:
    """Test that composition() pairs components with percent weights."""
    comps = [_component("a"), _component("b")]
    fuel = Fuel(components=comps, percent_weights=[60, 40])

    assert fuel.composition() == [(comps[0], 60), (comps[1], 40)]


def test__fuel_mismatched_lengths() -> None:
    """Test that mismatched components and percent_weights lengths raise."""
    with pytest.raises(ValueError, match="same length"):
        Fuel(components=[_component("a")], percent_weights=[50, 50])


def test__fuel_negative_weight() -> None:
    """Test that negative percent weights raise."""
    comps = [_component("a"), _component("b")]
    with pytest.raises(ValueError, match="non-negative"):
        Fuel(components=comps, percent_weights=[-10, 110])


def test__fuel_weights_not_summing_to_100() -> None:
    """Test that percent weights not summing to 100 raise."""
    comps = [_component("a"), _component("b")]
    with pytest.raises(ValueError, match="sum to 100"):
        Fuel(components=comps, percent_weights=[50, 40])
