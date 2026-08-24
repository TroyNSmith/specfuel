"""Tests for specfuel module initialization."""

import specfuel
from specfuel import comp, decode, gcm


def test__version_defined() -> None:
    """Test that __version__ is defined."""
    assert hasattr(specfuel, "__version__")
    assert specfuel.__version__ == "0.0.0"


def test__module_all_attribute() -> None:
    """Test that __all__ is properly defined."""
    assert hasattr(specfuel, "__all__")
    assert isinstance(specfuel.__all__, list)


def test__comp_module_importable() -> None:
    """Test that comp module can be imported."""
    assert hasattr(comp, "Component")


def test__decode_module_importable() -> None:
    """Test that decode module can be imported."""
    assert hasattr(decode, "ComponentRegistry")
    assert hasattr(decode, "decode_component")
    assert hasattr(decode, "decode_sff")
    assert hasattr(decode, "decode_sff_file")


def test__gcm_module_importable() -> None:
    """Test that gcm module can be imported."""
    assert hasattr(gcm, "const1994")
