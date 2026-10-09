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


def _build(layer_num_heads: list[int]) -> CPUAttentionMetadataBuilder:
    layers = _layers(layer_num_heads)
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.get_num_attention_heads.return_value = MODEL_WIDE_NUM_HEADS
    vllm_config.cache_config.cache_dtype = "auto"
    kv_cache_spec = SimpleNamespace(
        num_kv_heads=NUM_KV_HEADS, head_size=64, block_size=16
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


def test_mixed_fa3_sliding_fa4_full_graph_split_policy(monkeypatch):
    """FA4 full layers must not inherit FA3's fixed graph split count."""
    from vllm.v1.attention.backends import flash_attn

    config = MagicMock()
    config.attention_config.flash_attn_max_num_splits_for_cuda_graph = 32
    config.model_config.rswa_window = None
    config.model_config.is_mm_prefix_lm = False
    config.parallel_config.cp_kv_cache_interleave_size = 1
    config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = True
    config.compilation_config.max_cudagraph_capture_size = 256
    config.kernel_config.enable_jit_warmup = False
    config.scheduler_config.max_num_seqs = 64
    monkeypatch.setattr(
        flash_attn,
        "get_flash_attn_version",
        lambda *, head_size, **kwargs: 3 if head_size == 256 else 4,
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(flash_attn, "get_num_attention_heads_from_layers", lambda *a: 8)
    monkeypatch.setattr(flash_attn, "get_dcp_world_size_and_rank", lambda *a: (1, 0))
    schedule = MagicMock(
        side_effect=AssertionError("Mixed windows must disable FA3 AOT")
    )
    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", schedule)
    layers = {}
    builders = {}
    for name, head_size, window in (
        ("sliding", 256, (1023, 0)),
        ("full", 512, (-1, -1)),
    ):
        impl = object.__new__(flash_attn.FlashAttentionImpl)
        impl.sliding_window = window
        layers[name] = SimpleNamespace(
            impl=SimpleNamespace(get_impl_variants=lambda impl=impl: [impl])
        )
        spec = SimpleNamespace(
            head_size=head_size,
            num_kv_heads=1,
            dtype=torch.float8_e4m3fn,
            block_size=64,
            dcp_sharded=False,
        )
        builders[name] = flash_attn.FlashAttentionMetadataBuilder(
            spec, [name], config, torch.device("cpu")
        )
    monkeypatch.setattr(flash_attn, "get_layers_from_vllm_config", lambda *a: layers)
    assert builders["sliding"].aot_schedule
    assert not builders["full"].aot_schedule
    assert builders["sliding"].max_num_splits == 32
    assert builders["full"].max_num_splits == 0
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=128,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([127], dtype=torch.int64),
        causal=True,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    for name, expected_splits in (("sliding", 32), ("full", 0)):
        metadata = builders[name].build(0, common)
        assert metadata.max_num_splits == expected_splits
        assert metadata.scheduler_metadata is None
    schedule.assert_not_called()
