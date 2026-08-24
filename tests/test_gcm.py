"""GCM module tests."""

import json
import tempfile
from pathlib import Path

import pytest

from specfuel.gcm import FunctionalGroup, const1994


def test__const1994_default_path() -> None:
    """Test loading const1994 with default path."""
    groups = const1994()
    assert isinstance(groups, dict)
    assert len(groups) > 0


def test__const1994_returns_functional_groups() -> None:
    """Test that const1994 returns FunctionalGroup instances."""
    groups = const1994()
    for group in groups.values():
        assert isinstance(group, FunctionalGroup)


def test__const1994_contains_expected_groups() -> None:
    """Test that returned mapping contains expected group names."""
    groups = const1994()
    expected_groups = ["CH3", "CH2", "CH", "C"]
    for group in expected_groups:
        assert group in groups
        assert groups[group].name == group


def test__const1994_data_integrity() -> None:
    """Test that loaded data maintains integrity."""
    groups = const1994()

    ch3_mw = 15.0
    ch2_mw = 14.0

    assert groups["CH3"].molecular_weight == ch3_mw
    assert groups["CH2"].molecular_weight == ch2_mw


def test__const1994_invalid_file() -> None:
    """Test that invalid JSON file raises appropriate error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content {")
        f.flush()
        temp_path = f.name

    try:
        with pytest.raises(json.JSONDecodeError):
            const1994(temp_path)
    finally:
        Path(temp_path).unlink()


def test__const1994_nonexistent_file() -> None:
    """Test that nonexistent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        const1994("/nonexistent/path/const1994.json")
