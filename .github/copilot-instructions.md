# SpecFuel Project Guide for Copilot

## Project Goal

SpecFuel is a Python package that accepts a list of fuel components and property specifications, then generates ideal fuel mixtures that satisfy all of the specifications.

**Core Functionality:**
- Parse fuel components with their chemical formulas, SMILES notation, and decomposition information
- Accept performance/property specifications for desired fuel mixtures
- Use optimization algorithms to identify ideal fuel mixture compositions

## Project Structure

```
specfuel/
├── src/specfuel/          # Main package source code
│   ├── comp.py            # Fuel component definitions and group-contribution property functions
│   ├── decode.py          # SFF (SpecFuel Format) file decoder
│   ├── fuel.py            # Fuel mixture definitions (components + percent weights)
│   ├── gcm.py             # Group contribution method data parsers
│   ├── data/              # Static data files (e.g., GCM constants)
│   ├── gcm/                # Reserved for future GCM method implementations (currently empty)
│   └── __init__.py        # Package initialization
├── tests/                 # Test suite
│   ├── test_comp.py
│   ├── test_decode.py
│   ├── test_gcm.py
│   └── test_specfuel.py
├── docs/                  # Sphinx documentation
├── scripts/               # Utility scripts for development
├── pyproject.toml         # Project metadata and tool configuration
├── pixi.toml              # Environment and task definitions
├── lefthook.yaml          # Git pre-commit hook configuration
└── README.md              # Project overview
```

## Module Architecture

The project follows a layered architecture (enforced by import-linter):

1. **Layer 1: `specfuel.decode`** - Entry point for parsing SFF input files
2. **Layer 2: `specfuel.fuel`** - Fuel mixture definitions (components + percent weights)
3. **Layer 3: `specfuel.comp`** - Fuel component definitions and utilities
4. **Layer 4: `specfuel.gcm`** - Group contribution method data parsers

Dependencies flow from Layer 1 → Layer 2 → Layer 3 → Layer 4 only. No circular dependencies allowed.

## Development Workflow & Test Suite

All development tasks are defined in `pixi.toml` and can be run with `pixi run <task>`.

### Pre-Commit Hooks (Automated)

The project uses **lefthook** to run automated checks before commits:

```bash
pixi run pre-commit          # Run all checks (requires local service running)
pixi run local-pre-commit    # Alternative workflow with local service
```

### Individual Quality Checks

Run these commands individually during development:

- **`pixi run fmt`** - Format code with Ruff (automatic code style fixes)
- **`pixi run lint`** - Lint code with Ruff and auto-fix issues
- **`pixi run types`** - Type checking with `ty` (strict Python type validation)
- **`pixi run imports`** - Validate import layering with `import-linter`
- **`pixi run test`** - Run pytest with coverage reporting

### Test Suite Details

- **Test Framework:** Pytest
- **Coverage:** Minimum 80% required (enforced in `pyproject.toml`)
- **Features:**
  - Module doctest execution (`--doctest-modules`)
  - Coverage reports in terminal and HTML (`htmlcov/index.html`)
  - Normalized whitespace comparison for doctest assertions

**Run tests:**
```bash
pixi run test
pixi run cov-view  # View HTML coverage report in browser
```

### Documentation

- **Tool:** Sphinx with MyST (Markdown support)
- **Build:** `pixi run docs-build`
- **View:** `pixi run docs-view`
- **Docstring Format:** NumPy style (see below)

## Code Style & Conventions

### Docstring Format: NumPy Style

All docstrings must follow the **NumPy docstring convention** (configured in `pyproject.toml` as `convention = "numpy"`). This is enforced by Ruff's pydocstyle linter.

**Example Module Docstring:**
```python
"""Fuel components."""
```

**Example Function Docstring:**
```python
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
```

**Key Points:**
- One-line summary followed by blank line for detailed description
- Use `Parameters`, `Returns`, `Raises` sections (not `Args`, `Returns`, `Raises`)
- Indent parameter descriptions one space beyond the parameter name
- End file docstrings with a period
- Include examples in docstrings when helpful (will be executed as doctests)

### Code Quality Rules

Enforced via Ruff linting:

- **Code style:** Black-compatible via Ruff formatter
- **Type hints:** Required for public APIs
- **Imports:** Organized and validated against module layering rules
- **Trailing commas:** Added automatically by formatter
- **Line length:** 88 characters (standard)
- **Docstring coverage:** All public functions/classes must have docstrings

### Naming Conventions

- **Function names:** Clear and concise, with minimal abbreviations (e.g., `critical_temperature`, not `t_crit`).
- **Variable names:** Abbreviations are fine as long as the meaning is clear from context (e.g., `fg` for a `FunctionalGroup` in a loop).

### Ruff Ignore Rules

The project ignores specific Ruff checks (see `pyproject.toml`):
- `D203, D213`: Blank line formatting conflicts (D211, D212 used instead)
- `CPY001`: No copyright notice requirement at file top
- `TID252`: Relative imports allowed
- `RUF022`: `__all__` sorting not enforced
- In tests: `S101` (assert statements allowed)

## Dependencies

### Core Dependencies
- **pandas** (≥3.0.5) - Data manipulation
- **matplotlib** (≥3.11.1) - Plotting
- **pint** (≥0.25.3) - Physical unit handling
- **JAX** (≥0.4.29) - High-performance numerical computation
- **equinox** (≥0.11.10) - PyTree utilities for JAX
- **diffrax** (≥0.6.1) - Numerical differential equation solvers

### Development Dependencies
- **ruff** - Linting and formatting
- **ty** - Python type checking
- **import-linter** - Module dependency validation
- **pytest, pytest-cov** - Testing and coverage
- **lefthook** - Git hooks manager
- **sphinx, myst-parser** - Documentation

### Development Setup

```bash
# Install dev environment with all tools
pixi install -e dev

# Run full setup (one-time)
pixi run setup
pixi run init

# Activate git hooks
pixi run hooks
```

## Key Files & Their Purpose

- **`src/specfuel/comp.py`** - `Component` data class (with a `groups()` method) plus module-level functions that calculate properties (e.g., `critical_temperature`, `enthalpy_of_formation`) from group contributions. These functions take a `Component` as their argument; they are not methods on `Component`. All property values are returned in the GCM's native units (no unit conversions are performed).
- **`src/specfuel/fuel.py`** - `Fuel` data class storing a list of `Component` objects and their corresponding percent weights (validated to be non-negative and sum to 100).
- **`src/specfuel/decode.py`** - High-level SFF file decoder and ComponentRegistry
- **`src/specfuel/gcm.py`** - GCM (Group Contribution Method) constant loaders (currently `const1994`, based on Constantinou-Gani 1994)
- **`input.sff`** - Example SFF format input file
- **`pyproject.toml`** - Project metadata, Ruff config, pytest config, coverage settings
- **`pixi.toml`** - Environment channels, dependencies, and task definitions
- **`lefthook.yaml`** - Pre-commit hook pipeline

## SFF Format

SpecFuel uses a custom "SpecFuel Format" (.sff files) for specifying components and properties:

```sff
group_method = some_method

comp "component_name"
    formula       = C10H8
    smiles        = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end

comp "another_component"
    formula       = C5H12
    decomposition = {"CH3":1, "CH2":4}
end
```

- `group_method`: Algorithm selection for property grouping
- `comp "name"`: Component declaration block
- `formula`: Chemical formula string
- `smiles`: SMILES notation (optional)
- `decomposition`: JSON object mapping fragment names to counts

## When Making Changes

1. **Before committing:**
   - Ensure all pre-commit checks pass: `pixi run pre-commit` or `pixi run local-pre-commit`
   - If any check fails, fix and re-run
   
2. **Type safety:**
   - Add type hints to all new functions/methods
   - Run `pixi run types` to validate type consistency
   
3. **Testing:**
   - Write tests in `tests/` for new functionality
   - Maintain ≥80% code coverage
   - Include doctest examples in function docstrings for common use cases
   
4. **Documentation:**
   - Update docstrings to NumPy format
   - Add module-level docstrings to new modules
   - Build and review: `pixi run docs-build && pixi run docs-view`
   - **Update `copilot-instructions.md`** when making changes to:
     - Project structure or module organization
     - Dependencies or environment setup
     - Development workflow or test procedures
     - Code style conventions or docstring formats
     - SFF format specification
   - **Always update `CHANGELOG.md`** for every user-facing or notable change
     (see below).

5. **Module boundaries:**
   - Respect the decode → comp → gcm layering
   - Check `pixi run imports` to validate layer integrity
   - Never introduce circular imports

6. **Changelog:**
   - `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
   - Add an entry to the `[Unreleased]` section for every change (feature,
     fix, or breaking change) under the appropriate `### Added` / `### Fixed`
     / `### Changed` / `### Removed` heading, creating the heading if it
     doesn't already exist.
   - Do this as part of the same change/commit — never leave the changelog
     out of date.

## Python Version & Environment

- **Minimum Python:** 3.12+
- **Environment Manager:** Pixi (replaces conda/mamba/venv)
- **Platforms:** Linux, macOS (Intel & ARM), Windows

## Useful Development Commands

| Command | Purpose |
|---------|---------|
| `pixi run fmt` | Auto-format code |
| `pixi run lint` | Fix linting issues |
| `pixi run types` | Check type annotations |
| `pixi run imports` | Validate import layering |
| `pixi run test` | Run full test suite |
| `pixi run test -k test_name` | Run specific test |
| `pixi run local start` | Start local development service |
| `pixi run local stop` | Stop local development service |
| `pixi run pre-commit --all-files` | Run all pre-commit checks |
| `pixi run docs-build` | Build Sphinx documentation |
| `pixi run hooks` | Toggle git hooks on/off |

## Maintaining This File

**`copilot-instructions.md` is the source of truth for project conventions and should be kept current.**

Update this file whenever:
- The project structure, module organization, or layering changes
- Dependencies are added, removed, or updated (especially versions)
- Development workflow, task commands, or tool configurations change
- Code style conventions, type checking, or linting rules are modified
- The SFF input format specification changes
- New documentation, testing frameworks, or deployment procedures are introduced
- Docstring conventions or examples are refined

This ensures Copilot has accurate context for code generation, refactoring suggestions, and architectural decisions throughout the development lifecycle.

## Common Issues & Solutions

- **Import layering violation:** Check that imports follow decode → comp → gcm order. Use `pixi run imports` to diagnose.
- **Type errors:** Run `pixi run types` and add type hints as needed. JAX arrays may require special handling.
- **Coverage below 80%:** Add tests to bring coverage above threshold, or use `# pragma: no cover` for uncovered edge cases.
- **Docstring format issues:** Use `pixi run lint` to auto-fix docstring formatting to NumPy style.
