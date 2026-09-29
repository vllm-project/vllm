# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata geometry must come from the builder's own group."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackendImpl,
    CPUAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

requires_cpu = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU attention backend"
)

# Laguna's shape: 48 query heads model-wide, 64 on its sliding layers, both
# against 8 KV heads.
MODEL_WIDE_NUM_HEADS = 48
NUM_KV_HEADS = 8


def _layers(layer_num_heads: list[int]):
    """Stand-in attention layers, one per head count, as one attention group."""
    return {
        f"layer_{i}": SimpleNamespace(
            impl=MagicMock(
                spec=CPUAttentionBackendImpl,
                num_heads=num_heads,
                sliding_window=None,
            )
        )
        for i, num_heads in enumerate(layer_num_heads)
    }


def _build(
    layer_num_heads: list[int], block_size: int = 16
) -> CPUAttentionMetadataBuilder:
    layers = _layers(layer_num_heads)
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.get_num_attention_heads.return_value = MODEL_WIDE_NUM_HEADS
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.uniform_decode_query_len = 4
    vllm_config.speculative_config = None
    kv_cache_spec = SimpleNamespace(
        num_kv_heads=NUM_KV_HEADS, head_size=64, block_size=block_size
    )

    with (
        patch(
            "vllm.v1.attention.backends.utils.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch(
            "vllm.v1.attention.backends.cpu_attn.get_layers_from_vllm_config",
            return_value=layers,
        ),
    ):
        return CPUAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=list(layers),
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )


@requires_cpu
@pytest.mark.parametrize("group_num_heads", [MODEL_WIDE_NUM_HEADS, 64, 16])
def test_num_heads_comes_from_the_group(group_num_heads):
    """The group's own count wins, even when it is not the model-wide one."""
    builder = _build([group_num_heads, group_num_heads])
    assert builder.num_heads == group_num_heads


@requires_cpu
def test_mixed_head_counts_in_one_group_are_rejected():
    """Grouping guarantees uniformity; a mixed group means that broke."""
    with pytest.raises(AssertionError, match="share num_heads"):
        _build([MODEL_WIDE_NUM_HEADS, 64])


@requires_cpu
@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(), reason="requires AMX support"
)
@pytest.mark.parametrize(
    ("is_cross_attention", "speculative_method", "expected_group"),
    [
        (False, None, 4),
        (False, "medusa", 4),
        (True, None, 1),
    ],
)
def test_cpu_builder_schedules_short_query_without_phase_marker(
    is_cross_attention, speculative_method, expected_group
):
    builder = _build([32], block_size=32)
    builder.is_cross_attention = is_cross_attention
    if speculative_method is not None:
        builder.vllm_config.speculative_config = SimpleNamespace(
            method=speculative_method
        )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=8192,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([8192], dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 256), dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        causal=True,
        is_prefilling=None,
    )

    with patch("vllm.v1.attention.backends.cpu_attn.envs.VLLM_CPU_ATTN_SPLIT_KV", True):
        metadata = builder.build(0, common).scheduler_metadata

    words = metadata.view(torch.int32)
    workitem_count = words[68 // 4].item()
    workitems = words[192 // 4 : 192 // 4 + workitem_count * 12].reshape(-1, 12)
    assert set(workitems[:, 3].tolist()) == {expected_group}


def test_flash_attention_geometry_comes_from_the_group():
    """FA3 must use group geometry for layers without an ``impl`` wrapper."""
    layers = {f"layer_{i}": SimpleNamespace(num_heads=16) for i in range(2)}
    vllm_config = MagicMock()
    vllm_config.model_config.get_num_attention_heads.return_value = MODEL_WIDE_NUM_HEADS
    vllm_config.model_config.get_num_kv_heads.return_value = NUM_KV_HEADS
    vllm_config.model_config.get_head_size.return_value = 128
    vllm_config.model_config.rswa_window = None
    vllm_config.model_config.is_mm_prefix_lm = False
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = (
        False
    )
    vllm_config.compilation_config.max_cudagraph_capture_size = None
    kv_cache_spec = SimpleNamespace(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        dcp_sharded=True,
    )

    with (
        patch(
            "vllm.v1.attention.backends.utils.get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch(
            "vllm.distributed.parallel_state.get_dcp_group",
            side_effect=AssertionError,
        ),
        patch(
            "vllm.v1.attention.backends.flash_attn.get_flash_attn_version",
            return_value=3,
        ),
    ):
        builder = FlashAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=list(layers),
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.aot_schedule
    assert builder.num_heads_q == 16
    assert builder.num_heads_kv == 2
    assert builder.headdim == 64
