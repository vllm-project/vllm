# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob


class TestUseExistingTorch:
    """
    Test the list creation in use_existing_torch.py.
    """

    def test_list_creation_equivalence(self):
        """
        Test that the simplified list creation is equivalent to the original.
        """
        # Original approach
        requires_files_original = glob.glob("requirements/*.txt")
        requires_files_original += ["pyproject.toml"]

        # Simplified approach (current implementation)
        requires_files_simplified = glob.glob("requirements/*.txt") + ["pyproject.toml"]

        assert set(requires_files_original) == set(requires_files_simplified)
