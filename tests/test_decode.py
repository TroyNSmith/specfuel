"""Decode module tests."""

import tempfile
from pathlib import Path

import pytest

from specfuel.comp import Component
from specfuel.decode import (
    ComponentRegistry,
    _extract_component_blocks,
    _parse_component_data,
    _parse_component_header,
    _parse_component_value,
    _parse_group_method,
    _validate_component_data,
    decode_component,
    decode_sff,
    decode_sff_file,
)


def test__extract_single_component() -> None:
    """Test extracting a single component block."""
    text = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end"""
    blocks = _extract_component_blocks(text)
    assert len(blocks) == 1
    assert 'comp "decane"' in blocks[0]


def test__extract_multiple_components() -> None:
    """Test extracting multiple component blocks."""
    text = """comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end

comp "toluene"
    formula = C7H8
    decomposition = {"ACH": 5, "ACCH3": 1}
end"""
    blocks = _extract_component_blocks(text)
    nblocks = 2

    assert len(blocks) == nblocks
    assert 'comp "decane"' in blocks[0]
    assert 'comp "toluene"' in blocks[1]


def test__extract_no_components() -> None:
    """Test extracting from text with no components."""
    text = "group_method = const1994"
    blocks = _extract_component_blocks(text)
    assert len(blocks) == 0


def test__extract_with_comments_and_spacing() -> None:
    """Test extracting components with various formatting."""
    text = """
comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end

comp "toluene"
    formula = C7H8
    decomposition = {"ACH": 5}
end
"""
    blocks = _extract_component_blocks(text)
    nblocks = 2

    assert len(blocks) == nblocks


def test__extract_preserves_content() -> None:
    """Test that extracted blocks preserve all content."""
    text = """comp "decane"
    formula = C10H8
    smiles = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end"""
    blocks = _extract_component_blocks(text)
    assert "formula" in blocks[0]
    assert "smiles" in blocks[0]
    assert "decomposition" in blocks[0]


def test__parse_group_method_basic() -> None:
    """Test parsing a valid group_method."""
    text = "group_method = const1994"
    result = _parse_group_method(text)
    assert result == "const1994"


def test__parse_group_method_with_spaces() -> None:
    """Test parsing group_method with extra spaces."""
    text = "group_method   =   const1994"
    result = _parse_group_method(text)
    assert result == "const1994"


def test__parse_group_method_multiline() -> None:
    """Test parsing group_method in multiline text."""
    text = """# Configuration file
group_method = const1994

comp "decane"
    formula = C10H8
end"""
    result = _parse_group_method(text)
    assert result == "const1994"


def test__parse_group_method_missing() -> None:
    """Test that missing group_method raises ValueError."""
    text = """comp "decane"
    formula = C10H8
end"""
    with pytest.raises(ValueError, match="group_method"):
        _parse_group_method(text)


def test__parse_group_method_different_value() -> None:
    """Test parsing different group_method values."""
    text = "group_method = ufaacii"
    result = _parse_group_method(text)
    assert result == "ufaacii"


def test__parse_valid_header() -> None:
    """Test parsing a valid component header."""
    result = _parse_component_header('comp "decane"')
    assert result == "decane"


def test__parse_header_with_spaces() -> None:
    """Test parsing header with extra spaces."""
    result = _parse_component_header('comp   "toluene"')
    assert result == "toluene"


def test__parse_header_invalid_format() -> None:
    """Test that invalid header format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid component declaration"):
        _parse_component_header('component "decane"')


def test__parse_header_missing_quotes() -> None:
    """Test that missing quotes raises ValueError."""
    with pytest.raises(ValueError, match="Invalid component declaration"):
        _parse_component_header("comp decane")


def test__parse_header_name_with_spaces() -> None:
    """Test parsing header with name containing spaces."""
    result = _parse_component_header('comp "branched decane"')
    assert result == "branched decane"


def test__parse_decomposition_json() -> None:
    """Test parsing decomposition JSON value."""
    value = '{"CH3":2, "CH2":8}'
    result = _parse_component_value("decomposition", value)
    assert result == {"CH3": 2, "CH2": 8}


def test__parse_decomposition_invalid_json() -> None:
    """Test that invalid JSON raises ValueError."""
    with pytest.raises(ValueError, match="Invalid decomposition JSON"):
        _parse_component_value("decomposition", "{invalid json}")


def test__parse_smiles() -> None:
    """Test parsing SMILES string."""
    result = _parse_component_value("smiles", '"CCCCCCCCCC"')
    assert result == "CCCCCCCCCC"


def test__parse_formula() -> None:
    """Test parsing formula string."""
    result = _parse_component_value("formula", "C10H8")
    assert result == "C10H8"


def test__parse_basic_data() -> None:
    """Test parsing component data block."""
    lines = [
        'comp "decane"',
        "    formula       = C10H8",
        '    decomposition = {"CH3":2, "CH2":8}',
        "end",
    ]
    result = _parse_component_data(lines)
    assert result["formula"] == "C10H8"
    assert result["decomposition"] == {"CH3": 2, "CH2": 8}


def test__parse_data_with_smiles() -> None:
    """Test parsing component data with SMILES."""
    lines = [
        'comp "decane"',
        "    formula       = C10H8",
        '    smiles        = "CCCCCCCCCC"',
        '    decomposition = {"CH3":2, "CH2":8}',
        "end",
    ]
    result = _parse_component_data(lines)
    assert result["formula"] == "C10H8"
    assert result["smiles"] == "CCCCCCCCCC"
    assert result["decomposition"] == {"CH3": 2, "CH2": 8}


def test__parse_data_empty_lines() -> None:
    """Test parsing data with empty lines."""
    lines = [
        'comp "decane"',
        "    formula       = C10H8",
        "",
        '    decomposition = {"CH3":2, "CH2":8}',
        "end",
    ]
    result = _parse_component_data(lines)
    assert result["formula"] == "C10H8"
    assert result["decomposition"] == {"CH3": 2, "CH2": 8}


def test__parse_data_missing_equals() -> None:
    """Test that line without '=' raises ValueError."""
    lines = ['comp "decane"', "    formula C10H8", "end"]
    with pytest.raises(ValueError, match="Invalid component line"):
        _parse_component_data(lines)


def test__parse_data_with_extra_spaces() -> None:
    """Test parsing data with extra whitespace."""
    lines = [
        'comp "decane"',
        "    formula       =     C10H8   ",
        '    decomposition = {"CH3":2, "CH2":8}',
        "end",
    ]
    result = _parse_component_data(lines)
    assert result["formula"] == "C10H8"


def test__valid_data() -> None:
    """Test validation of valid component data."""
    data = {"formula": "C10H8", "decomposition": {"CH3": 2, "CH2": 8}}
    # Should not raise
    _validate_component_data(data)


def test__missing_formula() -> None:
    """Test that missing formula raises ValueError."""
    data = {"decomposition": {"CH3": 2, "CH2": 8}}
    with pytest.raises(ValueError, match="formula"):
        _validate_component_data(data)


def test__missing_decomposition() -> None:
    """Test that missing decomposition raises ValueError."""
    data = {"formula": "C10H8"}
    with pytest.raises(ValueError, match="decomposition"):
        _validate_component_data(data)


def test__missing_both_fields() -> None:
    """Test that missing both required fields raises ValueError."""
    data = {"name": "decane"}
    with pytest.raises(ValueError, match="Component missing required field"):
        _validate_component_data(data)


def test__decode_basic_component() -> None:
    """Test decoding a basic component block."""
    text = """comp "decane"
    formula       = C10H8
    decomposition = {"CH3":2, "CH2":8}
end"""
    result = decode_component(text)
    assert result.name == "decane"
    assert result.formula == "C10H8"
    assert result.decomposition == {"CH3": 2, "CH2": 8}
    assert result.smiles is None


def test__decode_component_with_smiles() -> None:
    """Test decoding a component with SMILES."""
    text = """comp "decane"
    formula       = C10H8
    smiles        = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end"""
    result = decode_component(text)
    assert result.name == "decane"
    assert result.formula == "C10H8"
    assert result.smiles == "CCCCCCCCCC"
    assert result.decomposition == {"CH3": 2, "CH2": 8}


def test__decode_component_no_smiles() -> None:
    """Test decoding a component without SMILES."""
    text = """comp "toluene"
    formula       = C7H8
    decomposition = {"ACH": 5, "ACCH3": 1}
end"""
    result = decode_component(text)
    assert result.name == "toluene"
    assert result.formula == "C7H8"
    assert result.smiles is None
    assert result.decomposition == {"ACH": 5, "ACCH3": 1}


def test__decode_component_missing_formula() -> None:
    """Test that missing formula raises ValueError."""
    text = """comp "invalid"
    decomposition = {"CH3":2}
end"""
    with pytest.raises(ValueError, match="formula"):
        decode_component(text)


def test__decode_component_missing_decomposition() -> None:
    """Test that missing decomposition raises ValueError."""
    text = """comp "invalid"
    formula = C10H8
end"""
    with pytest.raises(ValueError, match="decomposition"):
        decode_component(text)


def test__decode_component_invalid_decomposition_json() -> None:
    """Test that invalid JSON in decomposition raises ValueError."""
    text = """comp "invalid"
    formula = C10H8
    decomposition = {invalid}
end"""
    with pytest.raises(ValueError, match="decomposition"):
        decode_component(text)


def test__decode_component_whitespace_handling() -> None:
    """Test that whitespace is properly handled."""
    text = """
            comp "decane"
                formula       = C10H8
                decomposition = {"CH3":2, "CH2":8}
            end
        """
    result = decode_component(text)
    assert result.name == "decane"
    assert result.formula == "C10H8"


def test__registry_creation() -> None:
    """Test creating a ComponentRegistry instance."""
    components = [
        Component(name="decane", formula="C10H8", decomposition={"CH3": 2, "CH2": 8})
    ]
    registry = ComponentRegistry(group_method="const1994", components=components)
    assert registry.group_method == "const1994"
    assert len(registry.components) == 1
    assert registry.components[0].name == "decane"


def test__registry_empty_components() -> None:
    """Test creating a registry with no components."""
    registry = ComponentRegistry(group_method="const1994", components=[])
    assert registry.group_method == "const1994"
    assert len(registry.components) == 0


def test__decode_basic_sff() -> None:
    """Test decoding a basic SFF content."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end"""
    result = decode_sff(content)
    assert isinstance(result, ComponentRegistry)
    assert result.group_method == "const1994"
    assert len(result.components) == 1
    assert result.components[0].name == "decane"


def test__decode_multiple_components() -> None:
    """Test decoding SFF with multiple components."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end

comp "toluene"
    formula = C7H8
    decomposition = {"ACH": 5, "ACCH3": 1}
end"""
    result = decode_sff(content)
    nblocks = 2

    assert result.group_method == "const1994"
    assert len(result.components) == nblocks
    assert result.components[0].name == "decane"
    assert result.components[1].name == "toluene"


def test__decode_with_smiles() -> None:
    """Test decoding SFF with SMILES data."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    smiles = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end"""
    result = decode_sff(content)
    assert result.components[0].smiles == "CCCCCCCCCC"


def test__decode_without_smiles() -> None:
    """Test decoding SFF without SMILES data."""
    content = """group_method = const1994

comp "toluene"
    formula = C7H8
    decomposition = {"ACH": 5}
end"""
    result = decode_sff(content)
    assert result.components[0].smiles is None


def test__decode_missing_group_method() -> None:
    """Test that missing group_method raises ValueError."""
    content = """comp "decane"
    formula = C10H8
    decomposition = {"CH3":2}
end"""
    with pytest.raises(ValueError, match="group_method"):
        decode_sff(content)


def test__decode_invalid_component_format() -> None:
    """Test that invalid component format raises ValueError."""
    content = """group_method = const1994

comp "invalid"
    formula = C10H8
end"""  # Missing decomposition
    with pytest.raises(ValueError, match="decomposition"):
        decode_sff(content)


def test__decode_preserves_all_data() -> None:
    """Test that decoding preserves all component data."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    smiles = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end"""
    result = decode_sff(content)
    comp = result.components[0]
    assert comp.name == "decane"
    assert comp.formula == "C10H8"
    assert comp.smiles == "CCCCCCCCCC"
    assert comp.decomposition == {"CH3": 2, "CH2": 8}


def test__decode_valid_file() -> None:
    """Test decoding a valid SFF file."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sff", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        result = decode_sff_file(temp_path)
        assert result.group_method == "const1994"
        assert len(result.components) == 1
        assert result.components[0].name == "decane"
    finally:
        Path(temp_path).unlink()


def test__decode_file_with_path_object() -> None:
    """Test decoding using a Path object."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sff", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        result = decode_sff_file(temp_path)
        assert result.group_method == "const1994"
        assert len(result.components) == 1
    finally:
        temp_path.unlink()


def test__decode_file_not_found() -> None:
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        decode_sff_file("/nonexistent/file.sff")


def test__decode_multiple_components_from_file() -> None:
    """Test decoding file with multiple components."""
    content = """group_method = const1994

comp "decane"
    formula = C10H8
    decomposition = {"CH3":2, "CH2":8}
end

comp "toluene"
    formula = C7H8
    decomposition = {"ACH": 5, "ACCH3": 1}
end"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sff", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        result = decode_sff_file(temp_path)
        nblocks = 2

        assert len(result.components) == nblocks
        assert result.components[0].name == "decane"
        assert result.components[1].name == "toluene"
    finally:
        Path(temp_path).unlink()
