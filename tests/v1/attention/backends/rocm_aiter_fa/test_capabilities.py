# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA capability contract: what the selector accepts and rejects.

Each declaration is checked both ways: declared values pass the selector (and
the forward suites run them), undeclared values are refused at selection time
with a reason, instead of failing later inside a kernel.
"""

import pytest
import torch

from tests.v1.attention.model_shapes import (
    GEMMA4_31B_GLOBAL,
    MUSE_SPARK_FULL,
    QWEN3_8_27B,
    QWEN3_235B,
    shapes_rejected_by,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadataBuilder,
)
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

BACKEND = AiterFlashAttentionBackend
SUPPORTED_HEAD_SIZES = [64, 128, 256]


def _select(**overrides) -> str:
    config = AttentionSelectorConfig(
        **{
            "head_size": 128,
            "dtype": torch.bfloat16,
            "kv_cache_dtype": "auto",
            "block_size": 16,
            **overrides,
        }
    )
    return RocmPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.ROCM_AITER_FA,
        attn_selector_config=config,
    )


def test_declarations_match_flagship_model_needs():
    """Pin the declared surface; changing it must update these suites."""
    assert BACKEND.get_supported_head_sizes() == SUPPORTED_HEAD_SIZES
    assert BACKEND.supported_dtypes == [torch.float16, torch.bfloat16]
    assert set(BACKEND.supported_kv_cache_dtypes) == {
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    }
    assert BACKEND.get_supported_kernel_block_sizes() == [16, 32]
    assert BACKEND.supports_sink()
    assert BACKEND.supports_sliding_window()
    assert BACKEND.supports_non_causal()
    assert not BACKEND.forward_includes_kv_cache_update
    assert (
        AiterFlashAttentionMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize(
    "shape", [QWEN3_235B, QWEN3_8_27B, MUSE_SPARK_FULL], ids=lambda s: s.id
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_selector_accepts_declared_combinations(
    shape, dtype, kv_cache_dtype, block_size
):
    path = _select(
        head_size=shape.head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
    )
    assert path == AttentionBackendEnum.ROCM_AITER_FA.get_path()


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"head_size": GEMMA4_31B_GLOBAL.head_size}, "head_size"),
        ({"head_size": 96}, "head_size"),
        ({"dtype": torch.float32}, "dtype"),
        ({"kv_cache_dtype": "nvfp4"}, "kv_cache_dtype"),
        # ROCm reshape_and_cache_flash cannot write it.
        ({"kv_cache_dtype": "fp8_e5m2"}, "kv_cache_dtype"),
        ({"block_size": 8}, "block_size"),
        ({"use_mla": True}, "MLA"),
        ({"use_sparse": True}, "sparse"),
        ({"use_pcp": True}, "PCP"),
        ({"use_dcp": True}, "DCP"),
        ({"attn_type": AttentionType.ENCODER_DECODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_ONLY}, "attention type"),
        ({"use_mm_prefix": True}, "multimodal"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_selector_rejects_undeclared(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        _select(**overrides)


@pytest.mark.parametrize("feature", ["has_sliding_window", "has_sink"])
def test_shuffle_layout_rejects_window_and_sinks(feature, shuffle):
    """Shuffle decode has no window or sink input; the selector must refuse
    the combination instead of letting forward() assert at the first decode."""
    if shuffle:
        with pytest.raises(ValueError, match="shuffle KV cache layout"):
            _select(**{feature: True})
    else:
        assert _select(**{feature: True}) == (
            AttentionBackendEnum.ROCM_AITER_FA.get_path()
        )


def test_model_shapes_outside_head_sizes_are_rejected():
    rejected = shapes_rejected_by(SUPPORTED_HEAD_SIZES)
    assert {s.head_size for s in rejected} == {512}
    for shape in rejected:
        with pytest.raises(ValueError, match="head_size"):
            _select(head_size=shape.head_size)


def test_impl_does_not_claim_context_parallelism():
    """PCP/DCP would need LSE-returning decode; the impl provides neither."""
    assert not AiterFlashAttentionImpl.supports_pcp
    assert not AiterFlashAttentionImpl.supports_dcp
    assert not AiterFlashAttentionImpl.can_return_lse_for_decode


def test_impl_rejects_non_decoder_attention():
    with pytest.raises(NotImplementedError, match="decoder self-attention"):
        AiterFlashAttentionImpl(
            8,
            128,
            128**-0.5,
            8,
            None,
            None,
            "auto",
            attn_type=AttentionType.ENCODER_ONLY,
        )


def _spec(block_size: int = 16, head_size: int = 128) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("block_size", [1, 16, 32, 48, 64])
def test_customize_spec_accepts_multiples_of_16(block_size, shuffle):
    spec = BACKEND.customize_spec(_spec(block_size))
    assert spec.block_size == block_size
    if shuffle:
        assert spec.num_head_slots == 2
        assert spec.state_content_bytes == 4 * 128 * 2
    else:
        assert spec.num_head_slots == _spec(block_size).num_head_slots


@pytest.mark.parametrize("block_size", [8, 24])
def test_customize_spec_rejects_ungatherable_block_sizes(block_size):
    with pytest.raises(ValueError, match="multiple of 16"):
        BACKEND.customize_spec(_spec(block_size))


def test_customize_spec_is_idempotent_under_shuffle(shuffle):
    once = BACKEND.customize_spec(_spec())
    assert BACKEND.customize_spec(once) == once


def test_kv_cache_layouts(shuffle):
    layouts = BACKEND.supported_kv_cache_layouts()
    if shuffle:
        # Shuffle decode reads K and V as separate dense planes.
        assert layouts == (KVCacheLayout.LHBNC,)
    else:
        assert set(layouts) == {KVCacheLayout.LBHNC, KVCacheLayout.LHBNC}


def test_kv_update_bypass_follows_the_shuffle_toggle(monkeypatch):
    """External KV writers produce the packed layout, so bypass is only
    allowed without shuffle, including after the env is refreshed."""
    try:
        for enabled in (False, True):
            monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", str(int(enabled)))
            rocm_aiter_ops.refresh_env_variables()
            assert rocm_aiter_ops.is_shuffle_kv_cache_enabled() == enabled
            assert BACKEND.supports_kv_update_bypass() == (not enabled)
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    "aiter, mha, expected",
    [("1", "1", True), ("1", "0", False), ("0", "1", False), ("0", "0", False)],
)
def test_env_gates_backend_priority(monkeypatch, aiter, mha, expected):
    from vllm.platforms.rocm import _get_backend_priorities

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", aiter)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MHA", mha)
    rocm_aiter_ops.refresh_env_variables()
    try:
        priorities = _get_backend_priorities(use_mla=False, use_sparse=False)
        assert (AttentionBackendEnum.ROCM_AITER_FA in priorities) == expected
        if expected:
            # ROCM_ATTN stays the default; FA is chosen only when asked for.
            assert priorities[0] == AttentionBackendEnum.ROCM_ATTN
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()
