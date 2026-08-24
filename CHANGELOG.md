# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Added
- SFF (SpecFuel Format) decoder (`specfuel.decode`) for parsing fuel
  component blocks (formula, SMILES, group decomposition) from `.sff` files.
- `Component` data class and group-contribution property functions
  (`specfuel.comp`), including `critical_temperature` and
  `enthalpy_of_formation`.
- Group contribution method (GCM) data loader for the Constantinou-Gani 1994
  method (`specfuel.gcm.const1994`).
- Test suite with pytest, doctest execution, and coverage reporting.
- Sphinx documentation setup with MyST support.

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
