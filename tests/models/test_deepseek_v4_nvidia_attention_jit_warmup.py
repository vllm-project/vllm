# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate NVIDIA DSv4 attention JIT dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

if not has_cutedsl():
    pytest.skip("CuTeDSL is not installed", allow_module_level=True)

from cutlass import BFloat16, Float32

from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    VllmCuTeDSLJitKernel,
)
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


def _dequant_config(
    block_size: int,
    compress_ratios: tuple[int, ...] = (4, 128),
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=compress_ratios)
        ),
        cache_config=SimpleNamespace(block_size=block_size),
    )


def _indexer_config(*, use_fp4: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                index_n_heads=64,
                index_head_dim=128,
                qk_rope_head_dim=64,
            ),
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: "mxfp4" if use_fp4 else "fp8"
        ),
    )


def _sparse_config(
    *,
    cache_dtype: str,
    compress_ratios: tuple[int, ...],
    block_size: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                head_dim=512,
                qk_rope_head_dim=64,
                compress_ratios=compress_ratios,
            ),
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype=cache_dtype,
        ),
    )


@pytest.fixture(autouse=True)
def _mock_indexer_cache_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(
        indexer,
        "dsa_indexer_uses_fp4",
        lambda config: config.attention_config.resolve_indexer_kv_dtype("fp8")
        == "mxfp4",
    )


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


def test_sparse_c128_compress_dispatch_matches_legacy_constructor_args() -> None:
    kernel = SparseAttnCompressC128Block8Kernel()

    assert kernel.dispatch(head_size=512, state_width=512) == kernel.CompileKey(
        head_size=512,
        state_width=512,
    )


def test_sparse_c128_compress_launcher_returns_allocated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = SparseAttnCompressC128Block8Kernel()
    executor = Mock(return_value=None)
    monkeypatch.setattr(kernel, "_get_or_compile", Mock(return_value=executor))
    tensor = torch.empty(1)
    output = torch.empty((1, 512))

    result = kernel(
        state_cache=tensor,
        num_actual=1,
        token_to_req_indices=tensor,
        positions=tensor,
        slot_mapping=tensor,
        block_table=tensor,
        head_dim=512,
        compressed_kv=output,
    )

    assert result is output
    executor.assert_called_once()


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


# ---------------------------------------------------------------------------
# get_warmup_keys coverage: exercise the traced dispatch + predicate expansion
# against realistic vllm_config shapes (not just dispatch(x) == CompileKey(x)).
# ---------------------------------------------------------------------------


def test_dequant_gather_warmup_keys_enumerates_block_size_variants() -> None:
    kernel = _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL

    assert kernel.get_warmup_keys(_dequant_config(block_size=256)) == [
        kernel.CompileKey(block_size=256, has_gather_lens=True),
        kernel.CompileKey(block_size=64, has_gather_lens=False),
        kernel.CompileKey(block_size=2, has_gather_lens=False),
    ]


def test_dequant_gather_warmup_keys_follow_configured_ratios() -> None:
    kernel = _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL

    assert kernel.get_warmup_keys(
        _dequant_config(block_size=256, compress_ratios=(4,))
    ) == [
        kernel.CompileKey(block_size=256, has_gather_lens=True),
        kernel.CompileKey(block_size=64, has_gather_lens=False),
    ]


def test_dequant_gather_warmup_keys_disabled_when_block_size_zero() -> None:
    kernel = _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL
    assert kernel.get_warmup_keys(_dequant_config(block_size=0)) == []


def test_indexer_mxfp4_warmup_keys_enumerate_coarsen_axis() -> None:
    kernel = IndexerQMxFp4Kernel()

    assert set(kernel.get_warmup_keys(_indexer_config(use_fp4=True))) == {
        kernel.CompileKey(
            head_dim=128,
            rope_dim=64,
            num_heads=64,
            cos_sin_dtype=cos_sin_dtype,
            coarsen=coarsen,
        )
        for cos_sin_dtype in (Float32, BFloat16)
        for coarsen in (1, 4)
    }


def test_indexer_mxfp4_warmup_keys_disabled_without_fp4_cache() -> None:
    kernel = IndexerQMxFp4Kernel()
    assert kernel.get_warmup_keys(_indexer_config(use_fp4=False)) == []


def test_indexer_fp8_warmup_keys_enumerate_coarsen_axis() -> None:
    kernel = IndexerQFp8Kernel()

    assert set(kernel.get_warmup_keys(_indexer_config(use_fp4=False))) == {
        kernel.CompileKey(
            head_dim=128,
            rope_dim=64,
            num_heads=64,
            cos_sin_dtype=cos_sin_dtype,
            coarsen=coarsen,
        )
        for cos_sin_dtype in (Float32, BFloat16)
        for coarsen in (1, 4)
    }


def test_indexer_fp8_warmup_keys_disabled_with_fp4_cache() -> None:
    kernel = IndexerQFp8Kernel()
    assert kernel.get_warmup_keys(_indexer_config(use_fp4=True)) == []


def test_sparse_c4_warmup_keys_enabled_for_fp8_ds_mla() -> None:
    kernel = SparseAttnCompressNormRopeStoreC4Kernel()

    assert kernel.get_warmup_keys(
        _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(4,))
    ) == [
        kernel.CompileKey(
            head_size=512,
            state_width=1024,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=4,
            overlap=True,
            norm_weight_dtype=BFloat16,
        )
    ]


def test_sparse_c4_warmup_keys_disabled_without_ratio_or_ds_mla() -> None:
    kernel = SparseAttnCompressNormRopeStoreC4Kernel()
    # Ratio 4 present but wrong cache dtype -> disabled.
    assert (
        kernel.get_warmup_keys(_sparse_config(cache_dtype="auto", compress_ratios=(4,)))
        == []
    )
    # Correct cache dtype but ratio 4 missing -> disabled.
    assert (
        kernel.get_warmup_keys(
            _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(128,))
        )
        == []
    )


def test_sparse_full_c4_warmup_keys_enabled_for_non_ds_mla_fp8() -> None:
    kernel = SparseAttnCompressNormRopeStoreFullC4Kernel()

    assert kernel.get_warmup_keys(
        _sparse_config(cache_dtype="fp8", compress_ratios=(4,))
    ) == [
        kernel.CompileKey(
            head_size=512,
            state_width=1024,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=4,
            overlap=True,
            store_full_fp8=True,
            norm_weight_dtype=BFloat16,
        )
    ]


def test_sparse_full_c4_warmup_keys_disabled_for_ds_mla() -> None:
    kernel = SparseAttnCompressNormRopeStoreFullC4Kernel()
    assert (
        kernel.get_warmup_keys(
            _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(4,))
        )
        == []
    )


def test_sparse_c128_compress_warmup_keys_enabled_for_ratio_128() -> None:
    kernel = SparseAttnCompressC128Block8Kernel()

    assert kernel.get_warmup_keys(
        _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(128,))
    ) == [kernel.CompileKey(head_size=512, state_width=512)]


def test_sparse_c128_compress_warmup_keys_disabled_without_ratio_128() -> None:
    kernel = SparseAttnCompressC128Block8Kernel()
    assert (
        kernel.get_warmup_keys(
            _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(4,))
        )
        == []
    )


def test_sparse_full_c128_store_warmup_keys_enabled_for_non_ds_mla_fp8() -> None:
    kernel = SparseAttnNormRopeStoreFullKernel()

    assert kernel.get_warmup_keys(
        _sparse_config(cache_dtype="fp8", compress_ratios=(128,))
    ) == [
        kernel.CompileKey(
            head_size=512,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=128,
            store_full_fp8=True,
            norm_weight_dtype=BFloat16,
        )
    ]


def test_sparse_full_c128_store_warmup_keys_disabled_for_ds_mla() -> None:
    kernel = SparseAttnNormRopeStoreFullKernel()
    assert (
        kernel.get_warmup_keys(
            _sparse_config(cache_dtype="fp8_ds_mla", compress_ratios=(128,))
        )
        == []
    )


# ---------------------------------------------------------------------------
# Integration smoke: every kernel is wired onto the shared CuTeDSL warmup base
# (PR #53564) and its warmup_inputs() builds a fake-argument tuple for a
# representative compile key. This does not invoke the GPU JIT compiler.
# ---------------------------------------------------------------------------

_WARMUP_INPUTS_CASES = [
    (
        _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL,
        dict(block_size=64, has_gather_lens=True),
    ),
    (
        IndexerQMxFp4Kernel(),
        dict(head_dim=128, rope_dim=64, num_heads=64, cos_sin_dtype=Float32, coarsen=1),
    ),
    (
        IndexerQFp8Kernel(),
        dict(head_dim=128, rope_dim=64, num_heads=64, cos_sin_dtype=Float32, coarsen=1),
    ),
    (
        SparseAttnCompressNormRopeStoreC4Kernel(),
        dict(
            head_size=512,
            state_width=1024,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=4,
            overlap=True,
            norm_weight_dtype=BFloat16,
        ),
    ),
    (
        SparseAttnCompressNormRopeStoreFullC4Kernel(),
        dict(
            head_size=512,
            state_width=1024,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=4,
            overlap=True,
            store_full_fp8=False,
            norm_weight_dtype=BFloat16,
        ),
    ),
    (
        SparseAttnCompressC128Block8Kernel(),
        dict(head_size=512, state_width=512),
    ),
    (
        SparseAttnNormRopeStoreKernel(),
        dict(
            head_size=512,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            kv_block_stride=39168,
            compress_ratio=128,
            norm_weight_dtype=BFloat16,
            kv_cache_block_size=2,
        ),
    ),
    (
        SparseAttnNormRopeStoreFullKernel(),
        dict(
            head_size=512,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=128,
            store_full_fp8=False,
            norm_weight_dtype=BFloat16,
        ),
    ),
]


@pytest.mark.parametrize(
    "kernel, key_fields",
    _WARMUP_INPUTS_CASES,
    ids=lambda value: type(value).__name__ if hasattr(value, "CompileKey") else "",
)
def test_kernel_uses_cutedsl_warmup_base_and_builds_inputs(
    kernel: VllmCuTeDSLJitKernel,
    key_fields: dict,
) -> None:
    # PR #53564 base is actually used (finding #1: no longer a dead abstraction).
    assert isinstance(kernel, VllmCuTeDSLJitKernel)

    compile_key = kernel.CompileKey(**key_fields)
    inputs = kernel.warmup_inputs(compile_key)

    assert isinstance(inputs, tuple)
    assert len(inputs) > 0
