# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoPE equivalence tests for the fused Kimi-K3 MLA epilogues."""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    fused_mla_decode_q_concat_kv_cache_insert,
    fused_mla_key_concat_ds_mla_insert,
    fused_mla_key_concat_kv_cache_insert,
    fused_mla_kv_concat,
    fused_mla_kv_concat_quant_fp8,
    fused_mla_qkv_quant_kv_cache_fp8_insert,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Kimi-K3 fused MLA requires CUDA"
)

_DTYPE = torch.bfloat16
_NUM_TOKENS = 3
_NUM_HEADS = 4
_BLOCK_SIZE = 8
_POSITIONS = (1, 7, 13)
_SLOTS = (0, 3, 9)


def _randn(*shape: int, dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype) * 0.2


def _rope_cache(max_position: int = 32) -> torch.Tensor:
    inv_freq = 1.0 / (
        50000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    freqs = torch.outer(positions, inv_freq)
    # The fused epilogue reads the cos/sin table in fp32 (RoPE math runs in fp32).
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _apply_gptj_rope(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    for _ in range(x.ndim - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x1 = x[..., ::2].float()
    x2 = x[..., 1::2].float()
    out1 = x1 * cos.float() - x2 * sin.float()
    out2 = x2 * cos.float() + x1 * sin.float()
    return torch.stack((out1, out2), dim=-1).flatten(-2).to(x.dtype)


def _cache_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return cache.reshape(-1, cache.shape[-1]).index_select(0, slots)


def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.to(torch.float8_e4m3fn).float(),
        atol=0.03125,
        rtol=0.15,
    )


def _strided_context_inputs(
    num_tokens: int, num_heads: int, dtype: torch.dtype, k_pe_fp8: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The real prefill-context layouts: ``k_nope``/``v`` as the two strided
    halves of one kv_b_proj output and ``k_pe`` as a column slice of the gather
    workspace (left in the fp8 cache layout for a plain fp8 cache).
    """
    kv_nope = _randn(num_tokens, num_heads, 256, dtype=dtype)
    k_nope, v = kv_nope.split((128, 128), dim=-1)
    workspace = _randn(num_tokens, 576, dtype=dtype)
    if k_pe_fp8:
        workspace = workspace.to(torch.float8_e4m3fn)
    return k_nope, workspace[:, 512:].unsqueeze(1), v


@pytest.mark.parametrize("num_tokens", [0, _NUM_TOKENS])
@pytest.mark.parametrize("num_heads", [3, _NUM_HEADS])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_context_kv_concat_accepts_strided_inputs(
    num_tokens: int, num_heads: int, input_dtype: torch.dtype
) -> None:
    k_nope, k_pe, _ = _strided_context_inputs(
        num_tokens, num_heads, input_dtype, k_pe_fp8=False
    )

    k = fused_mla_kv_concat(k_nope, k_pe)

    assert k.is_contiguous()
    assert k.dtype == input_dtype
    torch.testing.assert_close(k[..., :128], k_nope, atol=0, rtol=0)
    torch.testing.assert_close(
        k[..., 128:], k_pe.expand(-1, num_heads, -1), atol=0, rtol=0
    )


@pytest.mark.parametrize("k_pe_dtype", ["input", "fp8"])
@pytest.mark.parametrize("num_tokens", [0, _NUM_TOKENS])
@pytest.mark.parametrize("num_heads", [3, _NUM_HEADS])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_context_kv_pack_quantizes_strided_inputs(
    k_pe_dtype: str,
    num_tokens: int,
    num_heads: int,
    input_dtype: torch.dtype,
) -> None:
    k_nope, k_pe, v = _strided_context_inputs(
        num_tokens, num_heads, input_dtype, k_pe_fp8=k_pe_dtype == "fp8"
    )

    k_actual, v_actual = fused_mla_kv_concat_quant_fp8(k_nope, k_pe, v)

    fp8 = torch.float8_e4m3fn
    k_expected = torch.empty_like(k_actual)
    k_expected[..., :128] = k_nope.to(fp8)
    k_expected[..., 128:] = k_pe.to(fp8)
    assert k_actual.is_contiguous()
    assert v_actual.is_contiguous()
    # The pack casts with the native pairwise converters, so it must be
    # bit-identical to torch's `.to(fp8)` rather than merely close.
    torch.testing.assert_close(k_actual.float(), k_expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(v_actual.float(), v.to(fp8).float(), atol=0, rtol=0)


@pytest.mark.parametrize("cache_kind", ["bf16", "fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_prefill_epilogue_fuses_gptj_rope(cache_kind: str) -> None:
    torch.manual_seed(0)
    positions = torch.tensor(_POSITIONS, device="cuda", dtype=torch.int64)
    slots = torch.tensor(_SLOTS, device="cuda", dtype=torch.int64)
    cos_sin_cache = _rope_cache()
    q = _randn(_NUM_TOKENS, _NUM_HEADS, 192)
    k_nope = _randn(_NUM_TOKENS, _NUM_HEADS, 128)
    k_pe = _randn(_NUM_TOKENS, 64)
    kv_c = _randn(_NUM_TOKENS, 512)
    v = _randn(_NUM_TOKENS, _NUM_HEADS, 128)

    q_expected = q.clone()
    q_expected[..., 128:] = _apply_gptj_rope(
        q_expected[..., 128:], positions, cos_sin_cache
    )
    k_pe_expected = _apply_gptj_rope(k_pe, positions, cos_sin_cache)
    k_expected = torch.cat(
        (k_nope, k_pe_expected[:, None, :].expand(-1, _NUM_HEADS, -1)), dim=-1
    )
    cache_expected = torch.cat((kv_c, k_pe_expected), dim=-1)

    if cache_kind == "bf16":
        cache = torch.zeros(2, _BLOCK_SIZE, 576, device="cuda", dtype=_DTYPE)
        q_actual = q.clone()
        k_actual = fused_mla_key_concat_kv_cache_insert(
            q_actual,
            k_nope,
            k_pe,
            kv_c,
            cache,
            slots,
            positions,
            cos_sin_cache,
        )
        torch.testing.assert_close(q_actual, q_expected)
        torch.testing.assert_close(k_actual, k_expected)
        torch.testing.assert_close(_cache_rows(cache, slots), cache_expected)
    elif cache_kind == "fp8":
        cache = torch.zeros(
            2, _BLOCK_SIZE, 576, device="cuda", dtype=torch.float8_e4m3fn
        )
        one = torch.ones(1, device="cuda", dtype=torch.float32)
        q_actual, k_actual, v_actual = fused_mla_qkv_quant_kv_cache_fp8_insert(
            q,
            k_nope,
            k_pe,
            kv_c,
            v,
            cache,
            slots,
            one,
            one,
            one,
            one,
            positions,
            cos_sin_cache,
        )
        _assert_fp8_close(q_actual, q_expected)
        _assert_fp8_close(k_actual, k_expected)
        _assert_fp8_close(v_actual, v)
        _assert_fp8_close(_cache_rows(cache, slots), cache_expected)
    else:
        cache = torch.zeros(2, _BLOCK_SIZE, 656, device="cuda", dtype=torch.uint8)
        q_actual = q.clone()
        k_actual = fused_mla_key_concat_ds_mla_insert(
            q_actual,
            k_nope,
            k_pe,
            kv_c,
            cache,
            slots,
            positions,
            cos_sin_cache,
        )
        rope_cache = _cache_rows(cache, slots)[:, 528:656].view(_DTYPE)
        torch.testing.assert_close(q_actual, q_expected)
        torch.testing.assert_close(k_actual, k_expected)
        torch.testing.assert_close(rope_cache, k_pe_expected)


@pytest.mark.parametrize("cache_kind", ["bf16", "fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_decode_epilogue_fuses_gptj_rope(cache_kind: str) -> None:
    torch.manual_seed(1)
    positions = torch.tensor(_POSITIONS, device="cuda", dtype=torch.int64)
    slots = torch.tensor(_SLOTS, device="cuda", dtype=torch.int64)
    cos_sin_cache = _rope_cache()
    ql_nope = _randn(_NUM_TOKENS, _NUM_HEADS, 512)
    q_pe = _randn(_NUM_TOKENS, _NUM_HEADS, 64)
    kv_c = _randn(_NUM_TOKENS, 512)
    k_pe = _randn(_NUM_TOKENS, 64)

    q_pe_expected = _apply_gptj_rope(q_pe, positions, cos_sin_cache)
    k_pe_expected = _apply_gptj_rope(k_pe, positions, cos_sin_cache)
    q_expected = torch.cat((ql_nope, q_pe_expected), dim=-1)
    cache_expected = torch.cat((kv_c, k_pe_expected), dim=-1)

    kwargs = {"positions": positions, "cos_sin_cache": cos_sin_cache}
    if cache_kind == "bf16":
        cache = torch.zeros(2, _BLOCK_SIZE, 576, device="cuda", dtype=_DTYPE)
        q_actual = fused_mla_decode_q_concat_kv_cache_insert(
            ql_nope, q_pe, kv_c, k_pe, cache, slots, **kwargs
        )
        torch.testing.assert_close(q_actual, q_expected)
        torch.testing.assert_close(_cache_rows(cache, slots), cache_expected)
    elif cache_kind == "fp8":
        cache = torch.zeros(
            2, _BLOCK_SIZE, 576, device="cuda", dtype=torch.float8_e4m3fn
        )
        one = torch.ones(1, device="cuda", dtype=torch.float32)
        q_actual = fused_mla_decode_q_concat_kv_cache_insert(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            cache,
            slots,
            q_scale_inv=one,
            cache_scale_inv=one,
            **kwargs,
        )
        _assert_fp8_close(q_actual, q_expected)
        _assert_fp8_close(_cache_rows(cache, slots), cache_expected)
    else:
        cache = torch.zeros(2, _BLOCK_SIZE, 656, device="cuda", dtype=torch.uint8)
        q_actual = fused_mla_decode_q_concat_kv_cache_insert(
            ql_nope, q_pe, kv_c, k_pe, cache, slots, ds_mla=True, **kwargs
        )
        rope_cache = _cache_rows(cache, slots)[:, 528:656].view(_DTYPE)
        torch.testing.assert_close(q_actual, q_expected)
        torch.testing.assert_close(rope_cache, k_pe_expected)


@torch.inference_mode()
def test_decode_epilogue_preserves_nope_path() -> None:
    torch.manual_seed(2)
    slots = torch.tensor(_SLOTS, device="cuda", dtype=torch.int64)
    ql_nope = _randn(_NUM_TOKENS, _NUM_HEADS, 512)
    q_pe = _randn(_NUM_TOKENS, _NUM_HEADS, 64)
    kv_c = _randn(_NUM_TOKENS, 512)
    k_pe = _randn(_NUM_TOKENS, 64)
    cache = torch.zeros(2, _BLOCK_SIZE, 576, device="cuda", dtype=_DTYPE)

    q_actual = fused_mla_decode_q_concat_kv_cache_insert(
        ql_nope, q_pe, kv_c, k_pe, cache, slots
    )

    torch.testing.assert_close(q_actual, torch.cat((ql_nope, q_pe), dim=-1))
    torch.testing.assert_close(
        _cache_rows(cache, slots), torch.cat((kv_c, k_pe), dim=-1)
    )
