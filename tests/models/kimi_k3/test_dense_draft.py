# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 drafts on Kimi-K3.1: dense MLA drafts (EAGLE3, DSpark) on dense MLA
(FLASHINFER_MLA, ROCM_AITER_MLA on ROCm), GQA drafts (Qwen3 DSpark) in a KV
cache group of their own."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.models.kimi_k3.common import dsa_cache
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

# Kimi-K3.1's text config: DSA with 128-dim indexer keys, on a KDA hybrid.
K31_TEXT_CONFIG = SimpleNamespace(
    dsa_config=SimpleNamespace(index_head_dim=128), linear_attn_config={}
)


MLA_DRAFT = SimpleNamespace(kv_lora_rank=512)
GQA_DRAFT = SimpleNamespace(num_key_value_heads=16)


@pytest.mark.parametrize(
    ("method", "draft_text_config", "expected"),
    [
        # A dense Kimi-K3 draft reads its own fp8 MLA cache.
        ("eagle3", MLA_DRAFT, (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        ("dspark", MLA_DRAFT, (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        # A GQA draft (Qwen3 DSpark) keeps its own cache, off the sparse MLA path.
        ("dspark", GQA_DRAFT, (AttentionBackendEnum.FLASH_ATTN, "auto")),
        # Other methods are left alone: n-gram has no draft layers.
        ("ngram", None, (None, None)),
    ],
)
def test_kimi_k31_defaults_draft_attention(method, draft_text_config, expected):
    if current_platform.is_rocm() and draft_text_config is MLA_DRAFT:
        expected = (AttentionBackendEnum.ROCM_AITER_MLA, "fp8")
    spec_config = SimpleNamespace(
        method=method,
        attention_backend=None,
        kv_cache_dtype=None,
        draft_model_config=SimpleNamespace(hf_text_config=draft_text_config),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(backend=None),
        speculative_config=spec_config,
    )

    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)

    assert (spec_config.attention_backend, spec_config.kv_cache_dtype) == expected


def _k31_cache_dtype(cache_dtype, backend=None):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(backend=backend),
        speculative_config=None,
    )
    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    return vllm_config.cache_config.cache_dtype


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_kimi_k31_accepts_ds_mla_cache_dtypes(cache_dtype):
    assert _k31_cache_dtype(cache_dtype) == cache_dtype


@pytest.mark.parametrize(
    ("sm100", "backend", "cache_dtype", "expected"),
    [
        # SM10x defaults to FlashInfer sparse MLA, which reads per-tensor fp8.
        (True, None, "auto", "fp8_e4m3"),
        (True, None, "fp8", "fp8"),
        (True, AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "auto", "fp8_e4m3"),
        # FlashMLA sparse, explicit or the default elsewhere, reads fp8_ds_mla.
        (True, AttentionBackendEnum.FLASHMLA_SPARSE, "auto", "fp8_ds_mla"),
        (True, AttentionBackendEnum.FLASHMLA_SPARSE, "fp8", "fp8_ds_mla"),
        (False, None, "auto", "fp8_ds_mla"),
    ],
)
def test_kimi_k31_defaults_kv_cache_to_its_sparse_backend(
    monkeypatch, sm100, backend, cache_dtype, expected
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    assert _k31_cache_dtype(cache_dtype, backend) == expected


@pytest.mark.parametrize(
    ("block_size", "draft_kv_heads", "draft_block_size"),
    [
        # Attention TP1 (DP): 16 KV heads need 4096 B/token, over the MLA
        # page's 576 B/token, so the draft gets the largest block that fits.
        (8960, 16, 896),
        # Attention TP8: 2 KV heads per rank fit at the target's block size.
        (1280, 2, 1280),
    ],
)
def test_gqa_draft_gets_its_own_group_and_a_fitting_block(
    block_size, draft_kv_heads, draft_block_size
):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=False
            ),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    mla = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )
    kda = MambaSpec(
        block_size=block_size,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=mla.page_size_bytes,
        mamba_cache_mode="align",
    )
    target = {
        f"model.layers.{i}": mla if i % 4 == 3 or i == 92 else kda for i in range(93)
    }
    draft = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=draft_kv_heads,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )
    specs = target | {f"model.layers.{93 + i}.attn": draft for i in range(5)}

    groups = GQADraftKVCacheGrouping.get_kv_cache_groups(vllm_config, specs)

    # The target keeps its 4 groups; the 5 draft layers form a fifth.
    assert [len(g.layer_names) for g in groups] == [23, 23, 23, 24, 5]
    assert groups[-1].kv_cache_spec.block_size == draft_block_size
    # The draft's pages are padded to the target's (one page size for all groups).
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {mla.page_size_bytes}
    # Without a draft, the default grouping is unchanged.
    assert GQADraftKVCacheGrouping.get_kv_cache_groups(
        vllm_config, target
    ) == get_kv_cache_groups(vllm_config, target)


# Kimi-K3.1's layer pattern: 93 layers, DSA every 4th and the last, KDA otherwise.
K31_DSA_LAYERS = [i + 1 for i in range(93) if i % 4 == 3 or i == 92]
K31_PACKED_TEXT_CONFIG = SimpleNamespace(
    dsa_config=SimpleNamespace(index_head_dim=128),
    linear_attn_config={
        "full_attn_layers": K31_DSA_LAYERS,
        "kda_layers": [i + 1 for i in range(93) if i + 1 not in K31_DSA_LAYERS],
    },
)
# fp8_ds_mla rows and fp8 indexer keys.
MLA_BYTES, INDEX_BYTES = 656, 132


def _kda_state(tp: int, num_spec: int) -> MambaSpec:
    shapes = MambaStateShapeCalculator.kda_state_shape(
        tp, 96, 128, conv_kernel_size=4, num_spec=num_spec
    )
    return MambaSpec(
        block_size=1, shapes=shapes, dtypes=(torch.bfloat16, torch.float32)
    )


def _packed_block_size(monkeypatch, tp, state_bytes, speculative_config=None):
    monkeypatch.setattr(dsa_cache, "kimi_k3_indexer_uses_fp4", lambda _: False)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=K31_PACKED_TEXT_CONFIG),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=speculative_config,
    )
    block_size = dsa_cache.packed_dsa_block_size(
        vllm_config, MLA_BYTES, state_bytes, 64
    )
    assert block_size is not None and block_size % 64 == 0
    return block_size


@pytest.mark.parametrize("num_spec", [0, 5, 7])
def test_packed_dsa_block_fits_kda_states_tightly_and_pairs_across_tp(
    monkeypatch, num_spec
):
    """Spec decoding grows the KDA conv state just past a 1024-token row; the
    block must grow by a little, not double (half the KDA row left empty),
    while the block sizes of different TP sizes stay divisible for P/D."""
    row_bytes_per_token = len(K31_DSA_LAYERS) * (MLA_BYTES + INDEX_BYTES)
    block_sizes = {}
    for tp in (1, 2, 4, 8, 16):
        state = _kda_state(tp, num_spec).page_size_bytes
        block_size = _packed_block_size(monkeypatch, tp, state)
        # 69 KDA states in 3 groups of 23, each filling most of a block row.
        assert 23 * state <= block_size * row_bytes_per_token < 23 * state / 0.85
        block_sizes[tp] = block_size
    assert all(block_sizes[tp] == block_sizes[16] * 16 // tp for tp in block_sizes)


def test_packed_dsa_block_counts_a_dense_mla_draft_in_the_row(monkeypatch):
    """A dense MLA draft (the K3.1 DSpark) groups with the target's DSA caches,
    widening every block row. The block must be sized for that row, or the KDA
    groups fill under 80% of it."""
    num_spec, tp = 5, 8
    draft_layers = 5
    speculative_config = SimpleNamespace(
        method="dspark",
        kv_cache_dtype="fp8",
        draft_model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                kv_lora_rank=512, qk_rope_head_dim=64, num_hidden_layers=draft_layers
            )
        ),
    )
    kda = _kda_state(tp, num_spec)
    block_size = _packed_block_size(
        monkeypatch, tp, kda.page_size_bytes, speculative_config
    )

    def mla(head_size, dtype=torch.uint8, **kwargs):
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_size,
            dtype=dtype,
            **kwargs,
        )

    specs = {}
    for i in K31_DSA_LAYERS:
        specs[f"model.layers.{i}.self_attn"] = mla(
            576, cache_dtype_str="fp8_ds_mla", state_content_bytes=MLA_BYTES
        )
        specs[f"model.layers.{i}.indexer"] = mla(
            INDEX_BYTES, cache_role=SparseCacheRole.INDEXER
        )
    for i in K31_PACKED_TEXT_CONFIG.linear_attn_config["kda_layers"]:
        specs[f"model.layers.{i}.kda"] = replace(kda, block_size=block_size)
    for i in range(draft_layers):
        specs[f"draft.layers.{i}.self_attn"] = mla(
            576,
            dtype=torch.float8_e4m3fn,
            cache_dtype_str="fp8",
            non_causal_multi_token_decode=True,
        )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )

    groups = GQADraftKVCacheGrouping.get_kv_cache_groups(vllm_config, specs)
    bytes_per_block = _get_kv_cache_bytes_per_block(groups, KVCacheLayout.BLHNC)

    # The draft shares the DSA row, and the block was sized for that row.
    dsa_group = next(g for g in groups if "draft.layers.0.self_attn" in g.layer_names)
    assert len(dsa_group.layer_names) == 2 * len(K31_DSA_LAYERS) + draft_layers
    assert bytes_per_block == block_size * (
        len(K31_DSA_LAYERS) * (MLA_BYTES + INDEX_BYTES) + draft_layers * 576
    )
    kda_groups = [g for g in groups if g.layer_names[0].endswith(".kda")]
    assert [len(g.layer_names) for g in kda_groups] == [23, 23, 23]
    assert 23 * kda.page_size_bytes > 0.85 * bytes_per_block
