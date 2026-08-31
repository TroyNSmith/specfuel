"""fuellib units."""

import pint

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)

Q_ = ureg.Quantity
