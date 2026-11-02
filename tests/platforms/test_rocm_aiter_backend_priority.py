# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The AITER backends are tried before ROCM_ATTN when ROCM_ATTN has no custom
paged-attention kernel for the model's head size."""

from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.v1.attention.backends.registry import AttentionBackendEnum

ROCM_ATTN = AttentionBackendEnum.ROCM_ATTN
AITER_FA = AttentionBackendEnum.ROCM_AITER_FA
AITER_UNIFIED = AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN


def _priorities(
    head_size: int = 256,
    *,
    aiter: bool = True,
    mha: bool = True,
    use_kv_connector: bool = False,
    disable_env: bool = False,
    gfx942: bool = True,
):
    from vllm.platforms.rocm import _get_backend_priorities

    with (
        patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=aiter),
        patch("vllm._aiter_ops.rocm_aiter_ops.is_mha_enabled", return_value=mha),
        patch(
            "vllm._aiter_ops.rocm_aiter_ops.is_rdna_aiter_enabled", return_value=False
        ),
        patch("vllm.platforms.rocm.on_gfx942", return_value=gfx942),
        patch.object(envs, "VLLM_DISABLE_AITER_BACKEND_PRIORITY", disable_env),
    ):
        return _get_backend_priorities(
            use_mla=False,
            use_sparse=False,
            use_kv_connector=use_kv_connector,
            head_size=head_size,
        )


@pytest.mark.parametrize("head_size", [256, 192, 320])
def test_head_sizes_without_custom_kernel_prefer_aiter(head_size):
    priorities = _priorities(head_size)
    assert priorities[0] == AITER_FA
    assert priorities[1] == AITER_UNIFIED
    assert ROCM_ATTN in priorities  # still a fallback


@pytest.mark.parametrize("head_size", [64, 128])
def test_head_sizes_with_custom_kernel_keep_rocm_attn_first(head_size):
    priorities = _priorities(head_size)
    assert priorities[0] == ROCM_ATTN
    assert AITER_FA in priorities


def test_without_mha_unified_is_used():
    assert _priorities(256, mha=False)[0] == AITER_UNIFIED


def test_kill_switch_restores_old_order():
    assert _priorities(256, disable_env=True)[0] == ROCM_ATTN


def test_other_architectures_keep_old_order():
    # The reordering is scoped to MI300 (gfx942), where it was measured.
    assert _priorities(256, gfx942=False)[0] == ROCM_ATTN


def test_aiter_unavailable_keeps_old_order():
    # is_mha_enabled() is False whenever AITER itself is unavailable.
    assert _priorities(256, aiter=False, mha=False)[0] == ROCM_ATTN


def test_kv_connector_skips_rocm_attn():
    priorities = _priorities(256, use_kv_connector=True)
    assert ROCM_ATTN not in priorities
    assert priorities[0] == AITER_FA


@pytest.mark.parametrize("head_size", [256, 64])
def test_no_duplicate_entries(head_size):
    priorities = _priorities(head_size)
    assert len(priorities) == len(set(priorities))
