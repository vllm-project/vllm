# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA_SPARSE capability contract: what the selector accepts.

Each declaration is checked both ways: declared values pass the selector (and
the forward suites run them), undeclared values are refused at selection or
construction time with a reason, instead of failing later inside a kernel.
"""

from dataclasses import replace

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    make_case,
)
from tests.v1.attention.model_shapes import (
    DEEPSEEK_V3_2,
    GLM_5_3_FLASH,
    SPARSE_MLA_SHAPES,
    SPARSE_NO_DCP,
    cp_rejected_layouts,
)
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.kv_cache_interface import KVCacheLayout, MLAAttentionSpec

BACKEND = ROCMAiterMLASparseBackend
ENUM = AttentionBackendEnum.ROCM_AITER_MLA_SPARSE
# kv_lora_rank + rope: DeepSeek V3.2 / GLM-5.2 / GLM-5.3, then GLM-5.3-Flash.
HEAD_SIZES = sorted({s.head_size for s in SPARSE_MLA_SHAPES})


def _config(**overrides) -> AttentionSelectorConfig:
    return AttentionSelectorConfig(
        **{
            "head_size": 576,
            "dtype": torch.bfloat16,
            "kv_cache_dtype": "auto",
            "block_size": 64,
            "use_mla": True,
            "use_sparse": True,
            **overrides,
        }
    )


def _select(selected=ENUM, **overrides) -> str:
    return RocmPlatform.get_attn_backend_cls(
        selected_backend=selected, attn_selector_config=_config(**overrides)
    )


def test_declarations():
    """Pin the declared surface; changing it must update these suites."""
    assert HEAD_SIZES == [512, 576]
    assert BACKEND.get_supported_head_sizes() == []  # any; the impl checks dims
    assert BACKEND.supported_dtypes == [torch.float16, torch.bfloat16]
    assert set(BACKEND.supported_kv_cache_dtypes) == {
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    }
    assert BACKEND.get_supported_kernel_block_sizes() == [1, MultipleOf(16)]
    assert BACKEND.is_mla() and BACKEND.is_sparse()
    assert BACKEND.supported_kv_cache_layouts() == (
        KVCacheLayout.LBNHC,
        KVCacheLayout.LBHNC,
        KVCacheLayout.BLHNC,
    )
    assert BACKEND.supports_sink()  # MI3xx
    assert not BACKEND.supports_dcp()
    assert not ROCMAiterMLASparseImpl.supports_dcp
    assert (
        ROCMAiterMLASparseMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_auto_selection_picks_the_sparse_backend():
    path = _select(selected=None)
    assert path == ENUM.get_path()


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
# Kernel block size 1 divides any cache block size, so every block size is
# accepted; the runner then splits cache blocks into kernel blocks.
@pytest.mark.parametrize("block_size", [1, 8, 16, 24, 64, 128])
def test_selector_accepts_declared_combinations(
    head_size, dtype, kv_cache_dtype, block_size
):
    path = _select(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
    )
    assert path == ENUM.get_path()


def test_selector_accepts_sinks():
    assert _select(has_sink=True) == ENUM.get_path()


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"dtype": torch.float32}, "dtype"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "fp8_ds_mla"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "nvfp4"}, "kv_cache_dtype"),
        ({"use_mla": False}, "non-MLA"),
        ({"use_sparse": False}, "non-sparse"),
        ({"use_dcp": True}, "DCP"),
        ({"has_sliding_window": True}, "sliding window"),
        ({"use_non_causal": True}, "non-causal"),
        ({"attn_type": AttentionType.ENCODER_DECODER}, "attention type"),
        ({"attn_type": AttentionType.ENCODER_ONLY}, "attention type"),
        ({"use_mm_prefix": True}, "multimodal"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_selector_rejects_undeclared(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        _select(**overrides)


@pytest.mark.parametrize(
    "layout",
    cp_rejected_layouts({s.model for s in SPARSE_MLA_SHAPES}),
    ids=lambda layout: f"{layout.model.split('/')[-1]}-tp{layout.tp}"
    f"-dcp{layout.dcp}-pcp{layout.pcp}",
)
def test_deployed_cp_layouts_are_rejected(layout):
    """Every DCP layout of a sparse-MLA model is refused by the selector."""
    if layout.reason != SPARSE_NO_DCP:
        pytest.skip(f"rejected before backend selection: {layout.reason}")
    with pytest.raises(ValueError, match="DCP"):
        _select(use_dcp=layout.dcp > 1, use_pcp=layout.pcp > 1)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("shape", SPARSE_MLA_SHAPES, ids=lambda s: s.model)
def test_every_deployed_head_count_is_valid(env, shape, tp):
    case = make_case(shape.shard(tp), [64])
    assert case.layer.impl.num_heads == shape.num_heads // tp


@pytest.mark.parametrize("num_heads", [136, 200])
def test_unpaddable_head_counts_are_rejected(env, num_heads):
    with pytest.raises(AssertionError, match="multiple of 16 heads"):
        make_case(replace(DEEPSEEK_V3_2, num_heads=num_heads), [64])


def test_nope_fp8_needs_the_aiter_triton_kernel(env, monkeypatch):
    """Rope-free fp8 has no ASM kernel; without gfx950 it must be refused."""
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    with pytest.raises(ValueError, match="rope-free"):
        make_case(GLM_5_3_FLASH.shard(8), [64], kv_cache_dtype="fp8")


def test_fp16_with_fp8_cache_is_refused(env):
    """No kernel runs an fp16 model on an fp8 cache correctly: AITER's ASM
    decode returns wrong values and the Triton kernels take no KV scale."""
    with pytest.raises(ValueError, match="float16"):
        make_case(
            DEEPSEEK_V3_2.shard(8), [64], dtype=torch.float16, kv_cache_dtype="fp8"
        )


@pytest.mark.parametrize(
    "sinks, match",
    [
        (lambda h: torch.zeros(h, dtype=torch.bfloat16), "float32"),
        (lambda h: torch.zeros(h + 1, dtype=torch.float32), "shape"),
        (
            lambda h: torch.zeros(2 * h, dtype=torch.float32, device="cuda")[::2],
            "contiguous",
        ),
    ],
    ids=["dtype", "shape", "contiguous"],
)
def test_invalid_sinks_are_rejected(env, sinks, match):
    shape = DEEPSEEK_V3_2.shard(8)
    with pytest.raises(ValueError, match=match):
        ROCMAiterMLASparseImpl(
            num_heads=shape.num_heads,
            head_size=shape.head_size,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            sinks=sinks(shape.num_heads).to("cuda"),
        )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_customize_spec_aligns_blocks_to_rows(kv_cache_dtype, block_size):
    torch_dtype = torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch_dtype,
        cache_dtype_str=kv_cache_dtype,
    )
    custom = BACKEND.customize_spec(spec)
    assert custom.block_stride_alignment == spec.state_content_size_bytes
    assert custom.page_size_bytes % spec.state_content_size_bytes == 0
