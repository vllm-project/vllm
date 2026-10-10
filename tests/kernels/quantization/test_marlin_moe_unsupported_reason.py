# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that Marlin MoE explains why it rejects a config.

Run `pytest tests/kernels/quantization/test_marlin_moe_unsupported_reason.py`.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
    explain_moe_marlin_unsupported,
)
from vllm.platforms import current_platform


def _config(hidden_dim: int, intermediate: int):
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        intermediate_size_per_partition_unpadded=intermediate,
    )


pytestmark = pytest.mark.skipif(
    current_platform.is_rocm(), reason="Marlin MoE is not available on ROCm"
)


def test_supported_config_has_no_reason():
    assert explain_moe_marlin_unsupported(_config(2560, 640), 128) is None


@pytest.mark.parametrize("allow_tile_padding", [False, True])
def test_group_straddling_tp_shard_is_explained(allow_tile_padding):
    """640 at TP=2 gives 320, which is not a whole number of 128-wide groups.

    Tile padding cannot repair this: it is a correctness rule, not an
    alignment one. The message must say so, and name both numbers.
    """
    reason = explain_moe_marlin_unsupported(
        _config(2560, 320), 128, allow_tile_padding=allow_tile_padding
    )
    assert reason is not None
    assert "320" in reason
    assert "128" in reason
    assert "--enable-expert-parallel" in reason


def test_bad_group_size_is_explained():
    reason = explain_moe_marlin_unsupported(_config(2560, 640), 96)
    assert reason is not None
    assert "96" in reason


def test_bad_hidden_size_is_explained():
    reason = explain_moe_marlin_unsupported(_config(2570, 640), 128)
    assert reason is not None
    assert "2570" in reason


def test_tile_alignment_only_rejected_without_padding():
    """A tile-misaligned size is fine under padding, rejected without it."""
    cfg = _config(2560, 32)
    assert explain_moe_marlin_unsupported(cfg, 32, allow_tile_padding=True) is None
    reason = explain_moe_marlin_unsupported(cfg, 32, allow_tile_padding=False)
    assert reason is not None
    assert "64" in reason


@pytest.mark.parametrize(
    "hidden,intermediate,group_size,allow_tile_padding",
    [
        (2560, 640, 128, False),
        (2560, 640, 128, True),
        (2560, 320, 128, False),
        (2560, 320, 128, True),
        (2560, 640, 96, False),
        (2570, 640, 128, True),
        (2560, 32, 32, True),
        (2560, 32, 32, False),
        (2560, 640, -1, False),
        (2560, 640, -1, True),
    ],
)
def test_bool_view_agrees_with_explanation(
    hidden, intermediate, group_size, allow_tile_padding
):
    """The boolean API must stay exactly the negation of the explanation."""
    cfg = _config(hidden, intermediate)
    supported = check_moe_marlin_supports_config(cfg, group_size, allow_tile_padding)
    reason = explain_moe_marlin_unsupported(cfg, group_size, allow_tile_padding)
    assert supported == (reason is None)
