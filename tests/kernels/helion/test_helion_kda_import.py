# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import safety test for KDA Helion kernel constants."""


def test_kda_constant_import_without_helion():
    """KDA_SMALL_VALUE_HEAD_THRESHOLD importable via the kda sub-package."""
    from vllm.kernels.helion.ops.kda import KDA_SMALL_VALUE_HEAD_THRESHOLD

    assert KDA_SMALL_VALUE_HEAD_THRESHOLD == 12
