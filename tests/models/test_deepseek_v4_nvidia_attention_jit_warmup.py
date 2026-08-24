# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate NVIDIA DSv4 attention JIT dispatch."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

if not has_cutedsl():
    pytest.skip("CuTeDSL is not installed", allow_module_level=True)

from cutlass import BFloat16, Float32

from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
    _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL,
    DequantGatherKCacheKernel,
)
from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
    IndexerQFp8Kernel,
    IndexerQMxFp4Kernel,
)
from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
    SparseAttnCompressC128Block8Kernel,
    SparseAttnCompressNormRopeStoreC4Kernel,
    SparseAttnCompressNormRopeStoreFullC4Kernel,
    SparseAttnNormRopeStoreFullKernel,
    SparseAttnNormRopeStoreKernel,
)

requires_cutedsl = pytest.mark.skipif(False, reason="CuTeDSL is not installed")

@requires_cutedsl
@pytest.mark.parametrize("has_gather_lens", [False, True])
def test_dequant_gather_dispatch_matches_legacy_compile_args(
    has_gather_lens: bool,
) -> None:
    kernel = _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL
    assert isinstance(kernel, DequantGatherKCacheKernel)

    assert kernel.dispatch(
        block_size=64,
        has_gather_lens=has_gather_lens,
    ) == kernel.CompileKey(
        block_size=64,
        has_gather_lens=has_gather_lens,
    )


@requires_cutedsl
@pytest.mark.parametrize("kernel_name", ["mx_fp4", "fp8"])
@pytest.mark.parametrize("coarsen", [1, 4])
def test_indexer_q_dispatch_matches_legacy_compile_args(
    kernel_name: str,
    coarsen: int,
) -> None:
    kernel_cls = {
        "mx_fp4": IndexerQMxFp4Kernel,
        "fp8": IndexerQFp8Kernel,
    }[kernel_name]
    kernel = kernel_cls()

    assert kernel.dispatch(
        head_dim=128,
        rope_dim=64,
        num_heads=64,
        cos_sin_dtype=Float32,
        coarsen=coarsen,
    ) == kernel.CompileKey(
        head_dim=128,
        rope_dim=64,
        num_heads=64,
        cos_sin_dtype=Float32,
        coarsen=coarsen,
    )


@requires_cutedsl
def test_sparse_c4_dispatch_matches_legacy_constructor_args() -> None:
    kernel = SparseAttnCompressNormRopeStoreC4Kernel()

    assert kernel.dispatch(
        compress_ratio=4,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        state_width=1024,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        compress_ratio=4,
        overlap=True,
        norm_weight_dtype=Float32,
    )


@requires_cutedsl
@pytest.mark.parametrize("store_full_fp8", [False, True])
def test_sparse_full_c4_dispatch_matches_legacy_constructor_args(
    store_full_fp8: bool,
) -> None:
    kernel = SparseAttnCompressNormRopeStoreFullC4Kernel()

    assert kernel.dispatch(
        compress_ratio=4,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        state_width=1024,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        compress_ratio=4,
        overlap=True,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
    )


@requires_cutedsl
def test_sparse_c128_compress_dispatch_matches_legacy_constructor_args() -> None:
    kernel = SparseAttnCompressC128Block8Kernel()

    assert kernel.dispatch(head_size=512, state_width=512) == kernel.CompileKey(
        head_size=512,
        state_width=512,
    )


@requires_cutedsl
@pytest.mark.parametrize(
    (
        "cache_block_size",
        "runtime_kv_block_stride",
        "kv_cache_block_size",
        "kv_block_stride",
    ),
    [(64, None, 1, 1152), (256, None, 2, 1728), (256, 39168, 2, 39168)],
)
def test_sparse_c128_store_dispatch_matches_legacy_constructor_args(
    cache_block_size: int,
    runtime_kv_block_stride: int | None,
    kv_cache_block_size: int,
    kv_block_stride: int,
) -> None:
    kernel = SparseAttnNormRopeStoreKernel()

    assert kernel.dispatch(
        compress_ratio=128,
        cache_block_size=cache_block_size,
        cache_alignment=576,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
        runtime_kv_block_stride=runtime_kv_block_stride,
    ) == kernel.CompileKey(
        head_size=512,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        kv_block_stride=kv_block_stride,
        compress_ratio=128,
        norm_weight_dtype=Float32,
        kv_cache_block_size=kv_cache_block_size,
    )


@requires_cutedsl
def test_sparse_c128_store_warmup_uses_bound_packed_cache_stride() -> None:
    kernel = SparseAttnNormRopeStoreKernel()
    packed_stride = 39168
    storage = torch.empty(packed_stride + 1168, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        storage,
        size=(2, 2, 584),
        stride=(packed_stride, 584, 1),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                head_dim=512,
                qk_rope_head_dim=64,
            ),
        ),
        cache_config=SimpleNamespace(
            block_size=256,
            cache_dtype="fp8_ds_mla",
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": SimpleNamespace(kv_cache=kv_cache)
            }
        ),
    )

    assert kernel.get_warmup_keys(
        vllm_config,
        k_cache_prefix="model.layers.0.self_attn",
        compress_ratio=128,
    ) == [
        kernel.CompileKey(
            head_size=512,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            kv_block_stride=packed_stride,
            compress_ratio=128,
            norm_weight_dtype=BFloat16,
            kv_cache_block_size=2,
        )
    ]


@requires_cutedsl
@pytest.mark.parametrize("store_full_fp8", [False, True])
def test_sparse_full_c128_store_dispatch_matches_legacy_constructor_args(
    store_full_fp8: bool,
) -> None:
    kernel = SparseAttnNormRopeStoreFullKernel()

    assert kernel.dispatch(
        compress_ratio=128,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        compress_ratio=128,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
    )
