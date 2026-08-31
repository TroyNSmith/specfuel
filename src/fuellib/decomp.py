"""Constantinou-Gani group decomposition loading and validation."""

from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, model_validator

from .gcm import ConstGani
from .types import INT_MATRIX

CONST_GANI = ConstGani()


class ConstGaniDecomp(BaseModel):
    """A compound x Constantinou-Gani group decomposition matrix.

    Guarantees that `groups`/`decomp` columns cover exactly
    `ConstGani.group_names`, in that order, regardless of the column order
    or subset of groups present in the source data (missing groups are
    zero-filled).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    families: list[str]
    groups: list[str]
    decomp: INT_MATRIX

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        """Load a decomposition matrix from a const_gani.csv file.

        Parameters
        ----------
        path
            Path to the const_gani.csv file.

        Returns
        -------
            A `ConstGaniDecomp` with columns reordered to match
            `ConstGani.group_names`.

        Raises
        ------
            ValueError: If the file does not exist, is not a .CSV file,
                does not have 'Family' as its first column, or contains
                non-integer decomposition values.
        """
        path = Path(path)
        if not path.exists():
            msg = f"'{path}' does not exist."
            raise ValueError(msg)

        if path.suffix != ".csv":
            msg = f"'{path}' is not a .CSV file."
            raise ValueError(msg)

        df: DataFrame = pd.read_csv(path, header=0)
        if len(df.columns) == 0 or df.columns[0] != "Family":
            msg = f"'{path}' must have 'Family' as its first column."
            raise ValueError(msg)

        families = df["Family"].tolist()
        decomp_df = df.drop(columns=["Family"])
        if "Reference Compound" in decomp_df.columns:
            decomp_df = decomp_df.drop(columns=["Reference Compound"])

        if not all(decomp_df.dtypes == np.int64):
            msg = f"'{path}' contains non-integer values."
            raise ValueError(msg)

        return cls(
            families=families,
            groups=list(decomp_df.columns),
            decomp=decomp_df.to_numpy(dtype=np.int64),
        )

    @model_validator(mode="after")
    def validate_and_reorder(self) -> Self:
        """Validate shapes/group names and reorder columns to match ConstGani.

        Groups absent from `self.groups` are treated as all-zero columns
        (e.g. a compound with no occurrences of that group).

        Raises
        ------
            ValueError: If `decomp`'s shape is inconsistent with `families`/
                `groups`, or if `groups` contains a name `ConstGani` doesn't
                recognize.
        """
        if self.decomp.shape[0] != len(self.families):
            msg = "Rows in decomp != number of families."
            raise ValueError(msg)

        if self.decomp.shape[1] != len(self.groups):
            msg = "Columns in decomp != number of groups."
            raise ValueError(msg)

        expected = CONST_GANI.group_names
        unexpected = sorted(set(self.groups) - set(expected))
        if unexpected:
            msg = f"Unrecognized groups not in ConstGani: {unexpected}\n"
            raise ValueError(msg)

        # zero-fill columns for any group missing from self.groups
        full_decomp = np.zeros((self.decomp.shape[0], len(expected)), dtype=np.int64)
        column_by_group = {group: i for i, group in enumerate(self.groups)}
        for j, group in enumerate(expected):
            if group in column_by_group:
                full_decomp[:, j] = self.decomp[:, column_by_group[group]]

        self.groups = list(expected)
        self.decomp = full_decomp

        return self
