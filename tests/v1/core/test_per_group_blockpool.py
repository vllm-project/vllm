# SPDX-License-Identifier: Apache-2.0
"""Tests for per-group BlockPool allocation on hybrid models.

Covers two real architectures:
  - Qwen3.5 (GatedDeltaNet + full attention, every 4th layer)
  - Nemotron-3-Nano (Mamba + MLP-only + full attention, 3 types)

Verifies that O(1) groups (Mamba/GDN in none/align mode) get a small fixed
pool while O(n) groups (attention) get the bulk of memory, yielding
dramatically higher token capacity.
"""

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Spec builders for real architectures
# ---------------------------------------------------------------------------

def _qwen35_specs(
    kv_dtype=torch.bfloat16,
    mamba_dtype=torch.bfloat16,
    mamba_cache_mode="none",
    num_layers=24,
    attn_interval=4,
):
    """Qwen3.5-0.8B/27B: GDN + full attention layers.

    0.8B: 24 layers (18 GDN + 6 attn), kv_heads=2, head_dim=256
    27B:  64 layers (48 GDN + 16 attn), kv_heads=4, head_dim=256
    """
    num_kv_heads = 2 if num_layers <= 24 else 4
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=256,
        dtype=kv_dtype,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 8192), (32, 128, 128)),
        dtypes=(mamba_dtype, mamba_dtype),
        mamba_cache_mode=mamba_cache_mode,
    )
    specs = {}
    for i in range(num_layers):
        if (i + 1) % attn_interval == 0:
            specs[f"layer_{i}"] = attn_spec
        else:
            specs[f"layer_{i}"] = mamba_spec

    n_attn = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
    n_mamba = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
    return specs, attn_spec, mamba_spec, n_attn, n_mamba


def _nemotron_specs(
    kv_dtype=torch.bfloat16,
    mamba_dtype=torch.bfloat16,
    mamba_cache_mode="none",
):
    """Nemotron-3-Nano-4B: Mamba + MLP-only + full attention.

    42 layers: M=Mamba(21), -=MLP-only(17), *=full attention(4)
    Pattern: M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-
    Attention: kv_heads=8, head_dim=128
    Mamba: 96 heads, ssm_state=128
    MLP-only layers have NO KV cache spec.
    """
    pattern = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=8,
        head_size=128,
        dtype=kv_dtype,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 8192), (96, 128, 128)),
        dtypes=(mamba_dtype, mamba_dtype),
        mamba_cache_mode=mamba_cache_mode,
    )
    specs = {}
    for i, c in enumerate(pattern):
        if c == '*':
            specs[f"layer_{i}"] = attn_spec
        elif c == 'M':
            specs[f"layer_{i}"] = mamba_spec
        # '-' = MLP-only, no KV cache needed

    n_attn = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
    n_mamba = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
    return specs, attn_spec, mamba_spec, n_attn, n_mamba


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_config(max_model_len=1024, mamba_cache_mode=None, max_num_seqs=256):
    model_config = ModelConfig(max_model_len=max_model_len)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    kwargs = dict(model_config=model_config, scheduler_config=scheduler_config)
    if mamba_cache_mode is not None:
        kwargs["cache_config"] = CacheConfig(mamba_cache_mode=mamba_cache_mode)
    return VllmConfig(**kwargs)


def _total_page_bytes(specs):
    return sum(s.page_size_bytes for s in specs.values())


# ---------------------------------------------------------------------------
# Qwen3.5 tests
# ---------------------------------------------------------------------------

class TestQwen35PerGroupBlockPool:
    """Per-group BlockPool tests for Qwen3.5 architecture."""

    def test_split_allocation_active(self):
        """Compact allocation should produce per_group_num_blocks."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _qwen35_specs()
        mem = _total_page_bytes(specs) * 100

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        assert config.per_group_num_blocks is not None, (
            "per_group_num_blocks should be set for hybrid models"
        )
        assert len(config.per_group_num_blocks) == len(config.kv_cache_groups)

    def test_attention_gets_bulk_of_memory(self):
        """Attention groups should get vastly more blocks than Mamba."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _qwen35_specs()
        mem = 10 * GiB_bytes

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        if config.per_group_num_blocks is not None:
            for i, group in enumerate(config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, FullAttentionSpec):
                    attn_blocks = config.per_group_num_blocks[i]
                elif isinstance(group.kv_cache_spec, MambaSpec):
                    mamba_blocks = config.per_group_num_blocks[i]

            assert attn_blocks > mamba_blocks * 10, (
                f"Attention ({attn_blocks}) should have >> Mamba ({mamba_blocks}) blocks"
            )

    def test_capacity_improvement_over_naive(self):
        """Per-group allocation should yield >3x more attention token capacity."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _qwen35_specs()
        mem = 10 * GiB_bytes

        # Naive: all layers share same blocks
        total_page = _total_page_bytes(specs)
        naive_blocks = int(mem // total_page)
        naive_tokens = naive_blocks * BLOCK_SIZE

        # Per-group
        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]
        new_tokens = config.num_blocks * BLOCK_SIZE

        ratio = new_tokens / naive_tokens if naive_tokens > 0 else float('inf')
        assert ratio >= 3.0, (
            f"Expected >=3x improvement, got {ratio:.1f}x "
            f"(naive={naive_tokens}, new={new_tokens})"
        )

    def test_per_layer_tensors(self):
        """Each layer should get its own tensor at its natural page size."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _qwen35_specs()
        mem = _total_page_bytes(specs) * 50

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        total_layers = n_attn + n_mamba
        assert len(config.kv_cache_tensors) == total_layers, (
            f"Expected {total_layers} tensors, got {len(config.kv_cache_tensors)}"
        )
        for t in config.kv_cache_tensors:
            assert len(t.shared_by) == 1

    def test_mamba_mode_all_no_split(self):
        """When mamba_cache_mode='all', no split -- Mamba is O(n) too."""
        vllm_config = _get_config(mamba_cache_mode="all")
        specs, *_ = _qwen35_specs(mamba_cache_mode="all")
        mem = _total_page_bytes(specs) * 50

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        assert config.per_group_num_blocks is None, (
            "per_group_num_blocks should be None when mamba_cache_mode='all'"
        )

    def test_enough_blocks_for_max_model_len(self):
        """Attention pool must have enough blocks for at least one full request."""
        max_model_len = 4096
        vllm_config = _get_config(max_model_len=max_model_len)
        specs, attn_spec, *_ = _qwen35_specs()
        mem = _total_page_bytes(specs) * 500

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        blocks_needed = cdiv(max_model_len, BLOCK_SIZE)
        assert config.num_blocks >= blocks_needed, (
            f"Need {blocks_needed} blocks for max_model_len={max_model_len}, "
            f"got {config.num_blocks}"
        )

    @pytest.mark.parametrize("num_layers,attn_interval", [
        (24, 4),   # 0.8B: 18 GDN + 6 attn
        (64, 4),   # 27B: 48 GDN + 16 attn
    ])
    def test_scales_with_model_size(self, num_layers, attn_interval):
        """Both 0.8B and 27B architectures should benefit from split."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _qwen35_specs(
            num_layers=num_layers, attn_interval=attn_interval
        )
        mem = _total_page_bytes(specs) * 100

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]
        assert config.per_group_num_blocks is not None
        assert config.num_blocks > 0

    def test_pure_attention_unaffected(self):
        """Pure attention model should not trigger per-group split."""
        vllm_config = _get_config()
        attn_spec = FullAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16,
        )
        specs = {f"layer_{i}": attn_spec for i in range(32)}
        mem = attn_spec.page_size_bytes * 32 * 100

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        assert config.per_group_num_blocks is None, (
            "Pure attention should not use per-group split"
        )


# ---------------------------------------------------------------------------
# Nemotron-3-Nano tests
# ---------------------------------------------------------------------------

class TestNemotronPerGroupBlockPool:
    """Per-group BlockPool tests for Nemotron-3-Nano architecture.

    42 layers with 3 types: Mamba(21) + MLP-only(17) + attention(4).
    Only 4 attention layers -- per-group split is critical.
    """

    def test_split_allocation_active(self):
        """Should activate per-group split despite 3 layer types."""
        vllm_config = _get_config()
        specs, *_ = _nemotron_specs()
        mem = _total_page_bytes(specs) * 100

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        assert config.per_group_num_blocks is not None

    def test_only_kv_layers_have_specs(self):
        """MLP-only layers should not appear in KV cache specs."""
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _nemotron_specs()

        assert n_attn == 4, f"Expected 4 attention layers, got {n_attn}"
        assert n_mamba == 21, f"Expected 21 Mamba layers, got {n_mamba}"
        assert len(specs) == 25, (
            f"Expected 25 KV layers (no MLP-only), got {len(specs)}"
        )

    def test_massive_capacity_improvement(self):
        """Nemotron should see >10x capacity improvement from split allocation."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _nemotron_specs()
        mem = 10 * GiB_bytes

        # Naive
        total_page = _total_page_bytes(specs)
        naive_blocks = int(mem // total_page)
        naive_tokens = naive_blocks * BLOCK_SIZE

        # Per-group
        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]
        new_tokens = config.num_blocks * BLOCK_SIZE

        ratio = new_tokens / naive_tokens if naive_tokens > 0 else float('inf')
        assert ratio >= 10.0, (
            f"Expected >=10x improvement for Nemotron, got {ratio:.1f}x "
            f"(naive={naive_tokens:,}, new={new_tokens:,})"
        )

    def test_attention_dominates_allocation(self):
        """Attention blocks should vastly outnumber Mamba blocks."""
        vllm_config = _get_config()
        specs, attn_spec, mamba_spec, n_attn, n_mamba = _nemotron_specs()
        mem = 10 * GiB_bytes

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        if config.per_group_num_blocks is not None:
            for i, group in enumerate(config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, FullAttentionSpec):
                    attn_blocks = config.per_group_num_blocks[i]
                elif isinstance(group.kv_cache_spec, MambaSpec):
                    mamba_blocks = config.per_group_num_blocks[i]

            assert attn_blocks > mamba_blocks * 5, (
                f"Attention ({attn_blocks}) should have >> "
                f"Mamba ({mamba_blocks}) blocks"
            )

    def test_mamba_pool_sized_for_concurrency(self):
        """Mamba pool should be sized for max concurrent requests."""
        max_num_seqs = 128
        vllm_config = _get_config(max_num_seqs=max_num_seqs)
        specs, *_ = _nemotron_specs()
        mem = 10 * GiB_bytes

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        if config.per_group_num_blocks is not None:
            for i, group in enumerate(config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, MambaSpec):
                    mamba_blocks = config.per_group_num_blocks[i]
                    assert mamba_blocks <= max_num_seqs * 3, (
                        f"Mamba pool too large: {mamba_blocks} blocks "
                        f"(max_num_seqs={max_num_seqs})"
                    )
                    assert mamba_blocks >= max_num_seqs, (
                        f"Mamba pool too small: {mamba_blocks} blocks "
                        f"(need at least {max_num_seqs})"
                    )


# ---------------------------------------------------------------------------
# Cross-architecture tests
# ---------------------------------------------------------------------------

class TestPerGroupBlockPoolGeneral:
    """Tests that apply to any hybrid architecture."""

    @pytest.mark.parametrize("make_specs", [_qwen35_specs, _nemotron_specs],
                             ids=["qwen35", "nemotron"])
    def test_allocation_efficient(self, make_specs):
        """Total allocation should use >85% of available memory."""
        vllm_config = _get_config()
        specs, *_ = make_specs()
        mem = 10 * GiB_bytes

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        total_allocated = sum(t.size for t in config.kv_cache_tensors)
        efficiency = total_allocated / mem
        assert efficiency > 0.85, (
            f"Allocation efficiency {efficiency:.1%} < 85%"
        )

    @pytest.mark.parametrize("make_specs", [_qwen35_specs, _nemotron_specs],
                             ids=["qwen35", "nemotron"])
    def test_backward_compatible_when_no_split(self, make_specs):
        """When mamba_cache_mode='all', behaves like shared pool."""
        vllm_config = _get_config(mamba_cache_mode="all")
        specs, *_ = make_specs(mamba_cache_mode="all")
        mem = _total_page_bytes(specs) * 50

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        assert config.per_group_num_blocks is None

    @pytest.mark.parametrize("make_specs", [_qwen35_specs, _nemotron_specs],
                             ids=["qwen35", "nemotron"])
    def test_num_blocks_consistent_with_tensors(self, make_specs):
        """num_blocks should match the actual tensor sizes."""
        vllm_config = _get_config()
        specs, attn_spec, *_ = make_specs()
        mem = _total_page_bytes(specs) * 100

        config = get_kv_cache_configs(vllm_config, [specs], [mem])[0]

        for t in config.kv_cache_tensors:
            layer_name = t.shared_by[0]
            spec = specs[layer_name]
            if isinstance(spec, FullAttentionSpec):
                expected = spec.page_size_bytes * config.num_blocks
                assert t.size == expected, (
                    f"{layer_name}: tensor size {t.size} != "
                    f"page_size({spec.page_size_bytes}) * "
                    f"num_blocks({config.num_blocks}) = {expected}"
                )
