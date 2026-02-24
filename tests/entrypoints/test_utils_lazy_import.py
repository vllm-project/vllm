# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests to ensure entrypoints/utils.py doesn't trigger early
platform detection via accessing current_platform.
"""

import importlib
from unittest.mock import patch


def test_utils_import_no_platform_detection():
    """Test that importing utils doesn't trigger platform detection."""
    import vllm.platforms

    with patch.object(vllm.platforms, "_current_platform", None):
        # Force re-import of the module
        importlib.reload(vllm.entrypoints.utils)

        # Verify that importing utils didn't trigger platform detection
        assert vllm.platforms._current_platform is None, (
            "Importing vllm.entrypoints.utils shouldn't trigger detection"
        )
