"""Component module tests."""

import pytest

from specfuel.comp import (
    Component,
    acentric_factor,
    boiling_temperature,
    critical_pressure,
    critical_temperature,
    critical_volume,
    enthalpy_of_formation,
    gibbs_free_energy,
    heat_capacity_b,
    heat_capacity_c,
    heat_capacity_stp,
    melting_temperature,
    molar_liquid_volume_stp,
    molecular_weight,
    vaporization_enthalpy_stp,
)


def test__component_creation() -> None:
    """Test creating a Component instance."""
    comp = Component(
        name="decane",
        formula="C10H8",
        decomposition={"CH3": 2, "CH2": 8},
        smiles="CCCCCCCCCC",
    )
    assert comp.name == "decane"
    assert comp.formula == "C10H8"
    assert comp.decomposition == {"CH3": 2, "CH2": 8}
    assert comp.smiles == "CCCCCCCCCC"


def test__component_without_smiles() -> None:
    """Test Component can be created without SMILES."""
    comp = Component(
        name="toluene", formula="C7H8", decomposition={"ACH": 5, "ACCH3": 1}
    )
    assert comp.name == "toluene"
    assert comp.formula == "C7H8"
    assert comp.smiles is None


def test__functional_groups_returns_groups_with_counts() -> None:
    """Test groups() pairs each group with its occurrence count."""
    decane = Component(
        name="decane", formula="C10H22", decomposition={"CH3": 2, "CH2": 8}
    )
    assert {fg.name: count for fg, count in decane.groups()} == {
        "CH3": 2,
        "CH2": 8,
    }


def test__functional_groups_unknown_group_raises() -> None:
    """Test that an unknown group raises a KeyError."""
    bad = Component(name="bad", formula="X", decomposition={"NOT_A_GROUP": 1})
    with pytest.raises(KeyError, match="Unknown group in decomposition"):
        _ = bad.groups()


def test__first_order_groups_excludes_second_order() -> None:
    """Test groups(order=1) filters out second-order groups."""
    comp = Component(
        name="mixed", formula="X", decomposition={"CH3": 2, "CH2": 8, "CHS": 1}
    )
    names = {fg.name for fg, _ in comp.groups(order=1)}
    assert names == {"CH3", "CH2"}


def test__second_order_groups_excludes_first_order() -> None:
    """Test groups(order=2) filters out first-order groups."""
    comp = Component(
        name="mixed", formula="X", decomposition={"CH3": 2, "CH2": 8, "CHS": 1}
    )
    result = comp.groups(order=2)
    assert [(fg.name, count) for fg, count in result] == [("CHS", 1)]


def test__second_order_groups_empty_when_none_present() -> None:
    """Test groups(order=2) is empty for a component with no such groups."""
    decane = Component(
        name="decane", formula="C10H22", decomposition={"CH3": 2, "CH2": 8}
    )
    assert decane.groups(order=2) == []


def test__component_molecular_weight_function() -> None:
    """Test molecular_weight computes MW (g/mol) from group contributions."""
    decane_mw_g_per_mol = 142.0
    decane = Component(
        name="decane", formula="C10H22", decomposition={"CH3": 2, "CH2": 8}
    )
    assert molecular_weight(decane) == pytest.approx(decane_mw_g_per_mol)


def test__component_property_functions() -> None:
    """Test Constantinou-Gani property functions against precomputed decane values."""
    decane = Component(
        name="decane", formula="C10H22", decomposition={"CH3": 2, "CH2": 8}
    )
    assert critical_temperature(decane) == pytest.approx(623.6905158181833)
    assert critical_pressure(decane) == pytest.approx(21.15522932444543)
    assert critical_volume(decane) == pytest.approx(0.59205)
    assert boiling_temperature(decane) == pytest.approx(452.5969765549304)
    assert melting_temperature(decane) == pytest.approx(217.0630557584084)
    assert enthalpy_of_formation(decane) == pytest.approx(-247.163)
    assert gibbs_free_energy(decane) == pytest.approx(34.96)
    assert vaporization_enthalpy_stp(decane) == pytest.approx(52.261)
    assert acentric_factor(decane) == pytest.approx(0.46804990995603424)
    assert molar_liquid_volume_stp(decane) == pytest.approx(0.19551)
    assert heat_capacity_stp(decane) == pytest.approx(231.5293)
    assert heat_capacity_b(decane) == pytest.approx(439.931)
    assert heat_capacity_c(decane) == pytest.approx(-145.4728)
