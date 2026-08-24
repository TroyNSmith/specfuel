"""Decode input files."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .comp import Component

__all__ = ["ComponentRegistry", "decode_component", "decode_sff", "decode_sff_file"]


@dataclass
class ComponentRegistry:
    """Parsed SFF file data."""

    group_method: str
    components: list[Component]


def _extract_component_blocks(text: str) -> list[str]:
    """Extract individual component blocks from SFF text."""
    blocks = []
    lines = text.split("\n")
    current_block = []
    in_component = False

    for line in lines:
        if re.match(r"^\s*comp\s+", line):
            in_component = True
            current_block = [line]
        elif in_component:
            current_block.append(line)
            if line.strip() == "end":
                blocks.append("\n".join(current_block))
                in_component = False
                current_block = []

    return blocks


def _parse_group_method(text: str) -> str:
    """Extract the group_method value from SFF text."""
    match = re.search(r"group_method\s*=\s*(\w+)", text)
    if not match:
        msg = "SFF file missing required 'group_method' setting"
        raise ValueError(msg)
    return match.group(1)


def _parse_component_header(comp_line: str) -> str:
    """Extract component name from the component declaration line."""
    match = re.match(r'comp\s+"([^"]+)"', comp_line)
    if not match:
        msg = f"Invalid component declaration: {comp_line}"
        raise ValueError(msg)
    return match.group(1)


def _parse_component_value(key: str, value: str) -> str:
    """Parse a component property value based on its key."""
    if key == "decomposition":
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            msg = f"Invalid decomposition JSON: {value}"
            raise ValueError(msg) from e
    elif key == "smiles":
        return value.strip('"')
    else:
        return value


def _parse_component_data(lines: list[str]) -> dict:
    """Parse key-value pairs from component block."""
    data = {}
    for line in lines[1:]:
        line = line.strip()  # noqa: PLW2901
        if line == "end":
            break
        if not line:
            continue

        if "=" not in line:
            msg = f"Invalid component line (missing '='): {line}"
            raise ValueError(msg)

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        data[key] = _parse_component_value(key, value)

    return data


def _validate_component_data(data: dict) -> None:
    """Validate that required component fields are present."""
    if "formula" not in data:
        msg = "Component missing required field: 'formula'"
        raise ValueError(msg)
    if "decomposition" not in data:
        msg = "Component missing required field: 'decomposition'"
        raise ValueError(msg)


def decode_component(text: str) -> Component:
    """Decode a component block from SFF format.

    Parameters
    ----------
    text
        Component block text, e.g.
            ```
            comp "decane"
                formula       = C10H8
                smiles        = "CCCCCCCCCC"
                decomposition = {"CH3":2, "CH2":8}
            end
            ```

    Returns
    -------
        Decoded component with name, formula, smiles (optional), and decomposition.

    Raises
    ------
    ValueError
        If formula or decomposition are missing, or if the format is invalid.
    """
    lines = text.strip().split("\n")

    name = _parse_component_header(lines[0].strip())
    data = _parse_component_data(lines)
    _validate_component_data(data)

    return Component(
        name=name,
        formula=data["formula"],
        decomposition=data["decomposition"],
        smiles=data.get("smiles"),
    )


def decode_sff(content: str) -> ComponentRegistry:
    """Decode an SFF (SpecFuel Format) string.

    Parameters
    ----------
    content
        SFF file content as a string.

    Returns
    -------
        SpecFuel object with parsed group_method and components.

    Raises
    ------
    ValueError
        If the SFF format is invalid.
    """
    group_method = _parse_group_method(content)
    component_blocks = _extract_component_blocks(content)

    components = [decode_component(block) for block in component_blocks]

    return ComponentRegistry(group_method=group_method, components=components)


def decode_sff_file(path: str | Path) -> ComponentRegistry:
    """Decode an SFF file from disk.

    Parameters
    ----------
    path
        Path to the .sff file.

    Returns
    -------
        SpecFuel object with parsed group_method and components.

    Raises
    ------
    ValueError
        If the SFF format is invalid.
    FileNotFoundError
        If the file does not exist.
    """
    content = Path(path).read_text()
    return decode_sff(content)
