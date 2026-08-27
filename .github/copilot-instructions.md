# SpecFuel Project Guide for Copilot

## Project Goal

SpecFuel is a Python package that accepts a list of fuel components and property specifications, then generates ideal fuel mixtures that satisfy all of the specifications.

**Vision (long-term):**
- Parse fuel components with their chemical formulas, SMILES notation, and decomposition information (e.g., from a SpecFuel Format `.sff` file)
- Accept performance/property specifications for desired fuel mixtures
- Use optimization algorithms to identify ideal fuel mixture compositions

**Currently implemented:**
- Load a fuel mixture (a fixed set of compounds + weight percentages + group
  decomposition) from a directory of two CSVs (see `Fuel.from_directory`)
- Compute mixture properties (density, kinematic/dynamic viscosity) at a
  given temperature using the Constantinou-Gani (1994) group-contribution
  method (`ConstGani`)
- Five bundled example fuels (`ExampleFuels`) for testing/demos

**Not yet implemented:** `.sff` file decoding, a standalone `Component`
class/group-contribution property functions, and the specification-driven
mixture optimization described in the vision above. Older docs/README
references to `specfuel.decode`, `specfuel.comp`, and `input.sff` describe
this future direction, not current code — don't assume those modules exist.

## Project Structure

```
specfuel/
├── src/specfuel/                 # Main package source code
│   ├── __init__.py               # Package init (exports data, fuel, gcm)
│   ├── fuel.py                   # Fuel model: load from directory, compute density/viscosity
│   ├── gcm.py                    # ConstGani: Constantinou-Gani 1994 group-contribution property calculators
│   ├── types.py                  # Shared numpy array type aliases (FLOAT_VECTOR, INT_MATRIX)
│   ├── units.py                  # Shared pint UnitRegistry and Q_ quantity constructor
│   └── data/                     # Bundled reference data
│       ├── __init__.py           # Exports ExampleFuels
│       ├── examples.py           # ExampleFuels: prebuilt Fuel instances for testing/demos
│       ├── gcm/
│       │   └── const_gani.csv    # Constantinou-Gani group contribution constants (121 groups)
│       └── fuel/                 # One subdirectory per example fuel
│           ├── decane/
│           │   ├── composition.csv   # Compound, Formula, Weight % (+ optional PelePhysics Key)
│           │   └── const_gani.csv    # Compound x group decomposition matrix
│           ├── heptane/
│           ├── heptane-decane/
│           ├── jet_a/
│           └── posf11498/
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Shared fixtures: BASELINE_DIR, FUELS_BY_NAME
│   ├── test_fuel.py               # Fuel validators/from_directory + baseline regression tests
│   ├── test_gcm.py                # ConstGani unit tests + baseline regression tests
│   ├── test_units.py              # Placeholder (currently empty)
│   ├── scratch.py                 # Ad-hoc manual script; not part of the test suite
│   └── baseline_properties/       # Golden CSVs used by baseline regression tests
│       ├── const_gani/<fuel_name>.csv  # Per-compound GCM properties (tidy/long format)
│       └── fuel/<fuel_name>.csv        # Fuel-level density/viscosity properties (tidy/long format)
├── docs/                          # Sphinx + MyST documentation
├── scripts/                       # Dev/CI helper scripts
│   ├── generate_baselines.py      # Regenerates tests/baseline_properties/ CSVs
│   └── *.sh                       # setup, init, hooks, local, lock, view-docs, cookiecutter-update
├── pyproject.toml                 # Project metadata; pytest/coverage/import-linter/ty config
├── .ruff.toml                     # Actual Ruff lint/format config (see Ruff Ignore Rules note below)
├── pixi.toml                      # Pixi environments, dependencies, and task definitions
├── lefthook.yaml                  # Git pre-commit hook pipeline
└── README.md                      # Project overview
```

## Module Architecture

The project follows a layered architecture (enforced by import-linter in `pyproject.toml`):

1. **Layer 1: `specfuel.data`** - Bundled example fuels (`ExampleFuels`)
2. **Layer 2: `specfuel.fuel`** - `Fuel` model: loading, validation, mixture properties
3. **Layer 3: `specfuel.gcm`** - `ConstGani` group-contribution property calculators
4. **Layer 4: `specfuel.types` | `specfuel.units`** - Shared numpy/pint primitives, no internal deps

Dependencies flow from Layer 1 → Layer 2 → Layer 3 → Layer 4 only. No circular dependencies allowed. Run `pixi run imports` to validate.

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
- **Baseline regression tests:** `tests/test_fuel.py` and `tests/test_gcm.py`
  compare `Fuel`/`ConstGani` property calculations against golden tidy/long
  CSVs in `tests/baseline_properties/{fuel,const_gani}/<fuel_name>.csv` (one
  file per `ExampleFuels` fuel, rows = property/temperature/value/unit).
  Comparisons use a loose tolerance (`rel=1e-3`) since baselines are
  self-generated from the current implementation, not independently verified
  physical data. Shared fixtures (`BASELINE_DIR`, `FUELS_BY_NAME`) live in
  `tests/conftest.py`. Regenerate baselines after an intentional formula
  change with `pixi run generate-baselines` (runs
  `scripts/generate_baselines.py`).

**Run tests:**
```bash
pixi run test
pixi run cov-view  # View HTML coverage report in browser
pixi run generate-baselines  # Regenerate baseline_properties/ CSVs
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
def critical_temperatures(self, decomp: INT_MATRIX) -> Quantity:
    """Get the standard critical temperatures for each compound in a fuel.

    Parameters
    ----------
    decomp
        Decomposition matrix for the compound.

    Returns
    -------
        Critical temperatures for each compound in the fuel.
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

> **Important:** Ruff's actual configuration lives in the root **`.ruff.toml`**,
> not `pyproject.toml`'s `[tool.ruff]` section. Ruff prefers `.ruff.toml` when
> both exist, so the `[tool.ruff]` table in `pyproject.toml` is currently
> unused/stale. Edit **`.ruff.toml`** for any lint/format config changes.

The project ignores specific Ruff checks (see `.ruff.toml`):
- `D203, D213`: Blank line formatting conflicts (D211, D212 used instead)
- `CPY001`: No copyright notice requirement at file top
- `TID252`: Relative imports allowed
- `RUF022`: `__all__` sorting not enforced
- In tests (`tests/**.py`): `S101` (assert statements allowed)
- In scripts (`scripts/**.py`): `INP001` (not a namespace package)

## Dependencies

### Core Dependencies (used by `src/specfuel`)
- **numpy** (≥2.5.2) - Array types (`FLOAT_VECTOR`, `INT_MATRIX`)
- **pandas** (≥3.0.5) - CSV data loading (fuel/GCM data)
- **pint** (≥0.25.3) - Physical unit handling (`Quantity`, shared `ureg`)
- **pydantic** (≥2.13.4) - `Fuel` data model and validation

### Present but not yet used by any code
- **matplotlib** (≥3.11.1), **JAX** (≥0.4.29), **equinox** (≥0.11.10),
  **diffrax** (≥0.6.1) - installed via `pixi.toml` for planned
  optimization/plotting work; no current `src/specfuel` module imports them.

### Development Dependencies
- **ruff** - Linting and formatting
- **ty** - Python type checking
- **import-linter** - Module dependency validation
- **pytest, pytest-cov** - Testing and coverage
- **lefthook** - Git hooks manager
- **sphinx, myst-parser, pydata-sphinx-theme, sphinx-autodoc2** - Documentation
- **tbump, keepachangelog** - Version bumping and changelog release automation

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

- **`src/specfuel/fuel.py`** - `Fuel` pydantic model: `name`, `compounds`, `weights`, optional `formulas`/`pelephysics_keys`, `cg_groups`, `cg_decomp`. Validators enforce weights sum to 100% and `cg_decomp` shape/group-name consistency with `ConstGani`. `Fuel.from_directory(path)` loads a `composition.csv` + `const_gani.csv` pair. Instance methods `density(temp)`, `kinematic_viscosity(temp, *, method=, correlation=)`, and `dynamic_viscosity(temp, *, method=, correlation=)` compute mixture properties (`correlation` is `"Kendall-Monroe"` or `"Arrhenius"`).
- **`src/specfuel/gcm.py`** - `ConstGani` class: loads `data/gcm/const_gani.csv` (Constantinou-Gani 1994 group contribution constants) and exposes per-compound property functions that take an `INT_MATRIX` decomposition. STP/temperature-independent: `molecular_weights`, `critical_temperatures`, `critical_pressures`, `critical_volumes`, `boiling_temperatures`, `stp_molar_liquid_volumes`, `acentric_factors`. Temperature-dependent (take a `Quantity` temperature): `molar_liquid_volumes`, `densities`, `kinematic_viscosities`, `dynamic_viscosities`. All values are returned in the GCM's native units (no unit conversions are performed).
- **`src/specfuel/types.py`** - `FLOAT_VECTOR`, `INT_MATRIX` numpy array type aliases shared by `fuel.py`/`gcm.py`.
- **`src/specfuel/units.py`** - Shared pint `ureg`/`Q_` (set as the application registry) so `Quantity` objects interoperate across the package.
- **`src/specfuel/data/examples.py`** - `ExampleFuels` class with class-level `Fuel` instances (`decane`, `heptane`, `heptane_decane`, `jet_a`, `posf11498`) built via `Fuel.from_directory` at import time.
- **`src/specfuel/data/fuel/<name>/composition.csv`** - `Compound`, `Formula`, `Weight %` columns (+ optional `Reference Compound`, `PelePhysics Key`).
- **`src/specfuel/data/fuel/<name>/const_gani.csv`** - Compound x Constantinou-Gani group decomposition integer matrix.
- **`src/specfuel/data/gcm/const_gani.csv`** - Constantinou-Gani 1994 group contribution constants (121 groups).
- **`tests/conftest.py`** - Shared `BASELINE_DIR`, `FUELS_BY_NAME` fixtures used by baseline regression tests.
- **`tests/baseline_properties/`** - Golden CSVs for regression tests (see Test Suite Details above).
- **`scripts/generate_baselines.py`** - Regenerates `tests/baseline_properties/` CSVs (`pixi run generate-baselines`).
- **`pyproject.toml`** - Project metadata, pytest/coverage config, import-linter contract, `ty` config. Its `[tool.ruff]` section is unused (see Ruff Ignore Rules note).
- **`.ruff.toml`** - Actual Ruff lint/format configuration.
- **`pixi.toml`** - Environments, dependencies, and task definitions.
- **`lefthook.yaml`** - Pre-commit hook pipeline.

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
   - **Always update `CHANGELOG.md`** for every user-facing or notable change
     (see below).

5. **Module boundaries:**
   - Respect the data → fuel → gcm → types/units layering
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
| `pixi run generate-baselines` | Regenerate baseline_properties/ CSVs |
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
- New documentation, testing frameworks, or deployment procedures are introduced
- Docstring conventions or examples are refined

This ensures Copilot has accurate context for code generation, refactoring suggestions, and architectural decisions throughout the development lifecycle.

## Common Issues & Solutions

- **Import layering violation:** Check that imports follow the data → fuel → gcm → types/units order. Use `pixi run imports` to diagnose.
- **Type errors:** Run `pixi run types` and add type hints as needed. `Quantity`/`PlainQuantity` (pint) mismatches are the most common source — see how `gcm.py` types temperature parameters as `Quantity | PlainQuantity`.
- **Coverage below 80%:** Add tests to bring coverage above threshold, or use `# pragma: no cover` for uncovered edge cases.
- **Docstring format issues:** Use `pixi run lint` to auto-fix docstring formatting to NumPy style.
