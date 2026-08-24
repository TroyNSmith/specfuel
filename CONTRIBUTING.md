# Contributing to SpecFuel

Thanks for your interest in contributing to SpecFuel! This document covers how to set up your environment, the project's conventions, and the checks your changes need to pass before being merged.

## Getting Started

SpecFuel uses [Pixi](https://pixi.sh) to manage environments, dependencies, and dev tasks (no conda/mamba/venv needed).

```bash
# Clone the repo, then from the project root:
pixi install -e dev

# One-time project setup
pixi run setup
pixi run init

# Enable git hooks (lefthook)
pixi run hooks
```

- **Minimum Python:** 3.12+
- **Supported platforms:** Linux, macOS (Intel & ARM), Windows

> **Note:** If `pixi install` solves to Python 3.14 on Windows, you may hit a
> broken `ssl` module in that conda-forge build. Pin `python = ">=3.12,<3.14"`
> under `[dependencies]` in `pixi.toml` if you run into
> `AttributeError: module 'ssl' has no attribute 'SSLWantReadError'`.

## Project Structure

```
specfuel/
├── src/specfuel/          # Main package source code
│   ├── comp.py             # Fuel component definitions and group-contribution property functions
│   ├── decode.py           # SFF (SpecFuel Format) file decoder
│   ├── fuel.py             # Fuel mixture definitions (components + percent weights)
│   ├── gcm.py               # Group contribution method data parsers
│   ├── data/                # Static data files (e.g., GCM constants)
│   └── gcm/                 # Reserved for future GCM method implementations
├── tests/                  # Test suite (pytest)
├── docs/                   # Sphinx documentation
└── scripts/                # Utility scripts for development
```

### Module Layering

The project enforces a strict import layering with `import-linter`:

1. **Layer 1:** `specfuel.decode` - entry point for parsing SFF input files
2. **Layer 2:** `specfuel.fuel` - fuel mixture definitions
3. **Layer 3:** `specfuel.comp` - fuel component definitions
4. **Layer 4:** `specfuel.gcm` - group contribution method data parsers

Dependencies must only flow from Layer 1 → Layer 2 → Layer 3 → Layer 4. Never introduce circular imports or reverse-direction imports. Run `pixi run imports` to validate.

## Development Workflow

Run these tasks individually while developing:

| Command | Purpose |
|---------|---------|
| `pixi run fmt` | Auto-format code with Ruff |
| `pixi run lint` | Lint code with Ruff (auto-fixes issues) |
| `pixi run types` | Type-check with `ty` |
| `pixi run imports` | Validate import layering with `import-linter` |
| `pixi run test` | Run the pytest suite with coverage |
| `pixi run test -k test_name` | Run a specific test |
| `pixi run cov-view` | View HTML coverage report in browser |
| `pixi run docs-build` | Build Sphinx documentation |
| `pixi run docs-view` | View built documentation |

Before committing, run all checks at once:

```bash
pixi run pre-commit          # requires local service running
pixi run local-pre-commit    # alternative local workflow
```

Lefthook also runs these checks automatically on commit once `pixi run hooks` has been enabled.

## Code Style

- **Formatting/linting:** [Ruff](https://docs.astral.sh/ruff/), Black-compatible, 88-character line length. Run `pixi run fmt` and `pixi run lint`.
- **Type hints:** Required on all public APIs. Checked with `ty` via `pixi run types`.
- **Docstrings:** [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), enforced by Ruff's pydocstyle rules. Use `Parameters`, `Returns`, and `Raises` sections (not `Args`). Include doctest examples where useful — they are executed as part of the test suite.

  ```python
  def decode_component(text: str) -> Component:
      """Decode a component block from SFF format.

      Parameters
      ----------
      text
          Component block text.

      Returns
      -------
          Decoded component with name, formula, smiles (optional), and decomposition.

      Raises
      ------
      ValueError
          If formula or decomposition are missing, or if the format is invalid.
      """
  ```

- **Naming:** Function names should be clear and mostly unabbreviated (e.g., `critical_temperature`, not `t_crit`). Short, context-clear abbreviations are fine for local variables (e.g., `fg` for a `FunctionalGroup` in a loop).

## Tests

- Written with **pytest** in `tests/`.
- Minimum coverage: **80%** (enforced in `pyproject.toml`).
- Module doctests are executed as part of the suite (`--doctest-modules`).
- Add tests for any new functionality, and prefer doctest examples in docstrings for common use cases.

```bash
pixi run test        # run full suite with coverage
pixi run cov-view     # view HTML coverage report
```

## Submitting Changes

1. Create a branch for your change.
2. Make your changes, following the module layering and code style conventions above.
3. Add/update tests to cover your change and keep coverage ≥80%.
4. Update docstrings and, if relevant, `docs/` content.
5. If you changed project structure, dependencies, workflow, conventions, or the SFF format, update [`.github/copilot-instructions.md`](.github/copilot-instructions.md) accordingly.
6. Run `pixi run pre-commit` (or `pixi run local-pre-commit`) and fix any failures.
7. Open a pull request describing the change and its motivation.

## Reporting Issues

When filing an issue, please include:
- A minimal reproduction (SFF input, code snippet, etc.)
- The full error/traceback if applicable
- Your OS and Python version
