# specfuel

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)

SpecFuel is a Python package for generating ideal fuel mixtures from a list of
fuel components and property specifications. It parses fuel components
(chemical formulas, SMILES notation, and group-contribution decompositions)
and uses optimization to find mixtures that satisfy desired
performance/property specifications.

## Installation

This project uses [Pixi](https://pixi.sh) to manage its environment and
dependencies.

```bash
git clone https://github.com/<org>/specfuel.git
cd specfuel
pixi install -e dev
```

## Usage

Fuel components and their group-contribution decompositions are described in
the SpecFuel Format (`.sff`), for example [input.sff](input.sff):

```sff
group_method = const1994

comp "decane"
    formula       = C10H8
    smiles        = "CCCCCCCCCC"
    decomposition = {"CH3":2, "CH2":8}
end

comp "toluene"
    formula       = C7H8
    decomposition = {"ACH": 5, "ACCH3": 1}
end
```

Decode the file and inspect the resulting components:

```python
from specfuel import decode

registry = decode.decode_sff_file("input.sff")
for component in registry.components:
    print(component.name, component.groups())
```

Group-contribution property functions (e.g. `critical_temperature`,
`enthalpy_of_formation`) in `specfuel.comp` take a `Component` and return its
property in the GCM's native units:

```python
from specfuel import comp

decane = registry.components[0]
print(comp.critical_temperature(decane))
```

## Development

Common tasks (see `pixi.toml` for the full list):

```bash
pixi run fmt      # format code with Ruff
pixi run lint     # lint code with Ruff
pixi run types     # type check with ty
pixi run imports   # validate import layering
pixi run test      # run the test suite with coverage
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).
