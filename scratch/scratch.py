"""Scratch."""

from pathlib import Path

import pandas as pd

from fuellib.data import ExampleFuels
from fuellib.gcm import ConstGani

test = ExampleFuels.jet_a
cg = ConstGani()


with Path("scratch.csv").open("w") as f:
    f.write(
        "Family,Reference Compound,Critical Temperature (K),Molecular Weight (g/mol),STP Molar Liquid Volume (L/mol),Boiling Temperature (K),Acentric Factor\n"
    )

    for comp in test.components:
        family = comp.name
        reference = comp.reference_compound
        mw = cg.molecular_weights(comp.cg_decomp).to("g/mol")
        tc = cg.critical_temperatures(comp.cg_decomp).to("K")
        vm = cg.stp_molar_liquid_volumes(comp.cg_decomp).to("L/mol")
        tb = cg.boiling_temperatures(comp.cg_decomp).to("K")
        w = cg.acentric_factors(comp.cg_decomp)

        f.write(
            f"{family},{reference},{tc.magnitude:.3f},{mw.magnitude:.3f},{vm.magnitude:.3f},{tb.magnitude:.3f},{w.magnitude:.3f}\n"
        )
