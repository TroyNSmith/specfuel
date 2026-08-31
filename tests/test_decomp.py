"""Tests for decomp module."""

import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fuellib.data.examples import FUEL_DIR
from fuellib.decomp import ConstGaniDecomp
from fuellib.gcm import ConstGani

GROUP_NAMES = ConstGani().group_names
NUM_GROUPS = ConstGani().num_groups


class TestFromCsv:
    """Test ConstGaniDecomp.from_csv."""

    def test_loads_decane_const_gani(self) -> None:
        """Test that decane's const_gani.csv loads with canonical group order."""
        decomp = ConstGaniDecomp.from_csv(FUEL_DIR / "decane" / "const_gani.csv")
        assert decomp.families == ["NC10H22"]
        assert decomp.groups == GROUP_NAMES
        assert decomp.decomp.shape == (1, NUM_GROUPS)

    def test_reorders_swapped_columns(self) -> None:
        """Test that group columns out of order get reordered to match ConstGani."""
        canonical = ConstGaniDecomp.from_csv(FUEL_DIR / "decane" / "const_gani.csv")

        swapped_groups = list(GROUP_NAMES)
        swapped_groups[0], swapped_groups[1] = swapped_groups[1], swapped_groups[0]
        swapped = ConstGaniDecomp(
            families=canonical.families,
            groups=swapped_groups,
            decomp=canonical.decomp[:, [1, 0, *range(2, NUM_GROUPS)]],
        )

        assert swapped.groups == canonical.groups
        assert np.array_equal(swapped.decomp, canonical.decomp)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Test that a nonexistent path raises."""
        with pytest.raises(ValueError, match="does not exist"):
            ConstGaniDecomp.from_csv(tmp_path / "nonexistent.csv")

    def test_raises_for_non_csv_suffix(self, tmp_path: Path) -> None:
        """Test that a non-.csv file raises."""
        path = tmp_path / "const_gani.txt"
        path.write_text("Family,CH3\nA,1\n")
        with pytest.raises(ValueError, match=re.escape("is not a .CSV file")):
            ConstGaniDecomp.from_csv(path)

    def test_raises_for_missing_family_column(self, tmp_path: Path) -> None:
        """Test that a missing 'Family' first column raises."""
        path = tmp_path / "const_gani.csv"
        path.write_text("Compound,CH3\nA,1\n")
        with pytest.raises(ValueError, match="'Family' as its first column"):
            ConstGaniDecomp.from_csv(path)

    def test_raises_for_non_integer_values(self, tmp_path: Path) -> None:
        """Test that non-integer decomposition values raise."""
        path = tmp_path / "const_gani.csv"
        path.write_text("Family,CH3\nA,1.5\n")
        with pytest.raises(ValueError, match="non-integer values"):
            ConstGaniDecomp.from_csv(path)

    def test_raises_for_mismatched_groups(self, tmp_path: Path) -> None:
        """Test that unrecognized group names raise."""
        path = tmp_path / "const_gani.csv"
        path.write_text("Family,foo,bar\nA,1,0\n")
        with pytest.raises(ValueError, match="Unrecognized groups"):
            ConstGaniDecomp.from_csv(path)

    def test_fills_missing_columns_with_zero(self, tmp_path: Path) -> None:
        """Test that groups absent from the CSV become all-zero columns."""
        path = tmp_path / "const_gani.csv"
        path.write_text("Family,CH3,CH2\nA,3,2\n")
        decomp = ConstGaniDecomp.from_csv(path)

        n_ch3 = 3
        n_ch2 = 2
        n_acbr = 0  # missing group should be zero-filled

        assert decomp.groups == GROUP_NAMES
        assert decomp.decomp[0, GROUP_NAMES.index("CH3")] == n_ch3
        assert decomp.decomp[0, GROUP_NAMES.index("CH2")] == n_ch2
        assert decomp.decomp[0, GROUP_NAMES.index("ACBr")] == n_acbr


class TestValidators:
    """Test ConstGaniDecomp's model validators for direct construction."""

    def test_raises_on_row_mismatch(self) -> None:
        """Test that decomp rows not matching families raises."""
        decomp = np.zeros((2, NUM_GROUPS), dtype=np.int64)
        with pytest.raises(ValidationError, match="Rows in decomp"):
            ConstGaniDecomp(families=["A"], groups=GROUP_NAMES, decomp=decomp)

    def test_raises_on_column_mismatch(self) -> None:
        """Test that decomp columns not matching groups raises."""
        decomp = np.zeros((1, NUM_GROUPS - 1), dtype=np.int64)
        with pytest.raises(ValidationError, match="Columns in decomp"):
            ConstGaniDecomp(families=["A"], groups=GROUP_NAMES, decomp=decomp)
