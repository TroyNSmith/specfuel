# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Added
- Inverse fuel-composition design (`specfuel.inv.solve_composition`): given a
  directory with a `const_gani.csv` decomposition matrix and a
  `constraints.csv` of density/kinematic-viscosity/dynamic-viscosity targets,
  solves for the weight-fraction composition that best satisfies them
  (least-squares, via a `jax`/`diffrax`/`equinox` gradient-flow optimization
  over a `softmax`-parameterized composition), returning a `Fuel`.
- `fuellib.decomp.ConstGaniDecomp`: loads/validates a `const_gani.csv`
  decomposition matrix, auto-reordering group columns to match `ConstGani`'s
  expected group positions regardless of source column order.
- `fuellib.comp.Component`: a data container for a single compound within a
  `Fuel` mixture (`name`, `reference_compound`, `formula`, `pelephysics_key`,
  `weight`, `cg_decomp`), replacing `Fuel`'s previous parallel
  arrays/matrix.

### Changed
- `fuellib.decomp.ConstGaniDecomp` now tolerates `const_gani.csv`/direct
  construction `groups` that omit some `ConstGani` group names — missing
  groups are zero-filled rather than raising. Only group names not
  recognized by `ConstGani` still raise a `ValueError`.
- `Fuel.from_directory` and `specfuel.inv.solve_composition` now load
  `const_gani.csv` via `fuellib.decomp.ConstGaniDecomp.from_csv` instead of
  the removed `fuellib.fuel.load_const_gani_decomp` function.
- **Breaking:** `Fuel` now stores `components: list[fuellib.comp.Component]`
  instead of the parallel `families`/`reference_compounds`/`weights`/
  `formulas`/`pelephysics_keys`/`cg_decomp` fields. `num_families` is
  renamed `num_components`; `families` is replaced by `component_names`.
  `ConstGani`'s batch matrix API is unchanged — `Fuel` assembles matrices/
  vectors from `components` on demand.
- Regenerated `tests/baseline_properties/const_gani/*.csv` via
  `pixi run generate-baselines`; only the stale `compound` header (now
  `family`, matching current code) changed — no property values differ.
- SFF (SpecFuel Format) decoder (`specfuel.decode`) for parsing fuel
  component blocks (formula, SMILES, group decomposition) from `.sff` files.
- `Component` data class and group-contribution property functions
  (`specfuel.comp`), including `critical_temperature` and
  `enthalpy_of_formation`.
- Group contribution method (GCM) data loader for the Constantinou-Gani 1994
  method (`specfuel.gcm.const1994`).
- Test suite with pytest, doctest execution, and coverage reporting.
- Sphinx documentation setup with MyST support.
- Baseline regression test suite comparing `Fuel` and `ConstGani` property
  calculations against golden CSVs (`tests/baseline_properties/`) generated
  for every `ExampleFuels` fuel over a -40C to 100C temperature range (or at
  STP for temperature-independent properties). Baselines are (re)generated
  via `scripts/generate_baselines.py` (`pixi run generate-baselines`).

### Changed
- Capped `jax` to `<0.11.1` (in `pyproject.toml`/`pixi.toml`) to avoid an
  upstream circular-import bug in that release
  (`jax._src.clusters.__init__` imports `cloud_tpu_cluster` before
  `cluster`, breaking `import jax` entirely).
- Moved `fuel.py`, `gcm.py`, `types.py`, `units.py`, and `data/` out of
  `specfuel` into a new `src/fuellib` package, developed side-by-side with
  `specfuel` in this repo for now (eventually to become a separate
  dependency). `specfuel` now contains only `inv.py`, which imports the
  above from `fuellib`.

## [0.0.0] - YYYY-MM-DD

### Added
- Feature 1
- Feature 2...

### Fixed
- Fix 1
- Fix 2...

### Changed
- Change 1
- Change 2...
