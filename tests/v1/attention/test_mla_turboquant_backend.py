# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-level tests for the Triton-fused TurboQuant MLA backend.

These tests exercise the static surface of `TritonMLATurboQuantBackend`
(class registration, supported dtypes / head sizes / CG mode, packed-cache
shape) and the bit-pack helpers used by the write path. They do **not**
launch a vLLM engine — numerical equivalence of the fused decode kernel
is covered by `tests/kernels/attention/test_mla_turboquant_dequant.py`.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.triton_mla_tq import (
    TritonMLATurboQuantBackend,
    TritonMLATurboQuantImpl,
    TritonMLATurboQuantMetadataBuilder,
    _enumerate_packed_sizes,
    _pack_bits_rows,
    _unpack_bits_rows,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

TQ_DTYPES = [
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]


# ---------------------------------------------------------------------------
# Test 1 — Backend registration & static surface
# ---------------------------------------------------------------------------


def test_registry_resolves_to_turboquant_backend():
    """The enum entry must import and resolve to the right class."""
    cls = AttentionBackendEnum.TRITON_MLA_TURBOQUANT.get_class()
    assert cls is TritonMLATurboQuantBackend
    assert cls.get_name() == "TRITON_MLA_TURBOQUANT"
    assert cls.get_impl_cls() is TritonMLATurboQuantImpl
    assert cls.get_builder_cls() is TritonMLATurboQuantMetadataBuilder


@pytest.mark.parametrize("dtype_str", TQ_DTYPES)
def test_supports_kv_cache_dtype(dtype_str):
    """All four advertised TurboQuant cache dtypes must be claimed."""
    assert TritonMLATurboQuantBackend.supports_kv_cache_dtype(dtype_str)


def test_does_not_claim_foreign_kv_cache_dtypes():
    """Backend must not silently accept non-TQ dtypes — that would steal the
    selection from FP8 / bf16 backends in `current_platform.get_attn_backend`.
    """
    for foreign in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
        assert not TritonMLATurboQuantBackend.supports_kv_cache_dtype(foreign)


def test_builder_advertises_uniform_batch_cudagraph():
    """Loss of `UNIFORM_BATCH` would silently disable CUDA graph capture for
    the TurboQuant decode path — a major perf regression.
    """
    assert (
        TritonMLATurboQuantMetadataBuilder._cudagraph_support
        is AttentionCGSupport.UNIFORM_BATCH
    )


# ---------------------------------------------------------------------------
# Test 2 — Packed cache shape & supported head sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", TQ_DTYPES)
def test_kv_cache_shape_is_uint8_packed(dtype_str):
    """`get_kv_cache_shape` returns `(num_blocks, block_size, head_size)`
    — head_size is the packed-bytes-per-slot, not a logical dim."""
    shape = TritonMLATurboQuantBackend.get_kv_cache_shape(
        num_blocks=8,
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str=dtype_str,
    )
    assert shape == (8, 16, 576)


def test_supported_head_sizes_cover_all_presets():
    """The `get_supported_head_sizes()` enumeration must include the packed
    sizes for every (L, R, preset, kpe_fp8) combination — otherwise vLLM's
    startup head-size validation rejects the backend.
    """
    sizes = set(TritonMLATurboQuantBackend.get_supported_head_sizes())
    # Exact set is enumerated in _enumerate_packed_sizes; mirror a few
    # high-value cells to catch regressions in either direction.
    L, R = 512, 64
    for kv_c in (
        L,  # k8v4 fp8 keys
        math.ceil(L * 4 / 8) + 2,  # 4bit MSE + vec_norm
        math.ceil(L * 3 / 8) + 2,  # 3bit MSE + vec_norm
    ):
        for k_pe in (2 * R, R + 2):  # bf16 vs fp8 kpe layout
            assert kv_c + k_pe in sizes, f"missing packed size {kv_c + k_pe}"

    # _enumerate_packed_sizes is the single source of truth — make sure
    # the public method returns exactly that set, sorted.
    assert sorted(sizes) == _enumerate_packed_sizes()


def test_supported_dtypes_bf16_only():
    """The fused Triton kernel only supports bf16 — fp16 must not be claimed."""
    assert torch.bfloat16 in TritonMLATurboQuantBackend.supported_dtypes
    assert torch.float16 not in TritonMLATurboQuantBackend.supported_dtypes


# ---------------------------------------------------------------------------
# Test 3 — Bit-pack roundtrip (foundation of the MSE write path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("N", [1, 17, 64])
def test_pack_unpack_roundtrip(bits, D, N):
    """`_unpack_bits_rows(_pack_bits_rows(x)) == x` for all (bits, N, D)
    relevant to MLA decode. Catches off-by-one byte-spill bugs in the
    packer that would silently corrupt every quantized cache write.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    idx = torch.randint(0, 1 << bits, (N, D), dtype=torch.int64, device=device)

    packed = _pack_bits_rows(idx, bits)
    expected_bytes = math.ceil(D * bits / 8)
    assert packed.shape == (N, expected_bytes)
    assert packed.dtype == torch.uint8

    out = _unpack_bits_rows(packed, bits, D)
    assert out.shape == (N, D)
    assert torch.equal(out, idx)


# ---------------------------------------------------------------------------
# Test 4 — TurboQuantConfig single source of truth for k_pe_bytes
# ---------------------------------------------------------------------------


def test_tq_config_k_pe_bytes_matches_env_var():
    """TurboQuantConfig.k_pe_bytes must be the single source of truth for
    both mla_attention.get_kv_cache_spec and the TQ MLA backend __init__.
    """
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    for rope_dim in [32, 64, 128]:
        # bf16 layout: 2 * rope_dim
        cfg = TurboQuantConfig(
            head_dim=512,
            key_quant_bits=4,
            value_quant_bits=4,
            rope_head_dim=rope_dim,
            k_pe_fp8=False,
        )
        assert cfg.k_pe_bytes == 2 * rope_dim
        assert cfg.mla_packed_bytes == cfg.key_packed_size + 2 * rope_dim

        # fp8 layout: rope_dim + 2
        cfg_fp8 = TurboQuantConfig(
            head_dim=512,
            key_quant_bits=4,
            value_quant_bits=4,
            rope_head_dim=rope_dim,
            k_pe_fp8=True,
        )
        assert cfg_fp8.k_pe_bytes == rope_dim + 2
        assert cfg_fp8.mla_packed_bytes == cfg_fp8.key_packed_size + rope_dim + 2


def test_from_cache_dtype_forwards_mla_params():
    """TurboQuantConfig.from_cache_dtype must forward rope_head_dim and
    k_pe_fp8 so that the config carries the single source of truth."""
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_k3v4_nc",
        head_dim=512,
        rope_head_dim=64,
        k_pe_fp8=True,
    )
    assert cfg.rope_head_dim == 64
    assert cfg.k_pe_fp8 is True
    assert cfg.k_pe_bytes == 66  # 64 fp8 + 2 scale
    assert cfg.mla_packed_bytes == cfg.key_packed_size + 66


def test_from_cache_dtype_defaults_rope_dim_zero():
    """Non-MLA call (no rope_head_dim) should default to 0 — k_pe_bytes
    must not be called in that case."""
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    cfg = TurboQuantConfig.from_cache_dtype("turboquant_k3v4_nc", head_dim=128)
    assert cfg.rope_head_dim == 0
    assert cfg.k_pe_fp8 is False
