# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Mistral (ML4) eager kernels against the unfused ops they replace."""

import pytest
import torch
import torch.nn.functional as F

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("CUDA/ROCm-only kernels", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.layernorm import LayerNorm  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (  # noqa: E402
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.sparse_attn_indexer import (  # noqa: E402
    fused_indexer_q_rope_quant,
)

if current_platform.is_rocm():
    from vllm.models.mistral.amd.ops.indexer import (  # noqa: E402
        fused_indexer_k_norm_rope,
        fused_indexer_k_norm_rope_cache_write,
        fused_indexer_q_ssmax_rope_quant,
    )
    from vllm.models.mistral.amd.ops.layernorm import layer_norm  # noqa: E402
    from vllm.models.mistral.amd.ops.ssmax import (  # noqa: E402
        fused_rope_ssmax_scaling_,
        fused_ssmax_scaling,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
        indexer_k_quant_and_cache_triton,
    )
else:
    from vllm.models.mistral.nvidia.ops.indexer import (  # noqa: E402
        fused_indexer_k_norm_rope,
        fused_indexer_k_norm_rope_cache_write,
        fused_indexer_q_ssmax_rope_quant,
    )
    from vllm.models.mistral.nvidia.ops.layernorm import layer_norm  # noqa: E402
    from vllm.models.mistral.nvidia.ops.ssmax import (  # noqa: E402
        fused_rope_ssmax_scaling_,
        fused_ssmax_scaling,
    )
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales  # noqa: E402

ROPE_DIM = 64
DEVICE = "cuda"


def _rope_ref(rope, positions, q, k):
    """Unfused RoPE reference, in place on ``q``/``k``.

    On CUDA this is the FlashInfer kernel the fused kernels reproduce bit for
    bit. On ROCm the eager RoPE rounds each bf16 product separately, so the
    reference is computed in fp32 (one final rounding, like the fused
    kernels) and compared to one bf16 ulp.
    """
    if not current_platform.is_rocm():
        q[...], k[...] = rope(positions, q, k)
        return
    q32, k32 = rope.forward_native(positions, q.float(), k.float())
    q[...] = q32.to(q.dtype)
    k[...] = k32.to(k.dtype)


def _assert_close_ulp(actual, expected, ulps=0):
    if ulps == 0 or not current_platform.is_rocm():
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return
    a, e = actual.float(), expected.float()
    tol = ulps * 2**-7 * torch.maximum(a.abs(), e.abs())
    assert ((a - e).abs() <= tol).all(), (
        f"{((a - e).abs() > tol).sum().item()} elements beyond {ulps} bf16 ulp"
    )


def _eager_config() -> VllmConfig:
    # Models in vllm/models run without torch.compile, so CustomOps take their
    # CUDA paths (e.g. FlashInfer RoPE), which are the references here.
    return VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))


@pytest.fixture(scope="module")
def rope() -> DeepseekScalingRotaryEmbedding:
    # ML4 geometry: interleaved YaRN, factor 512 over 512 original positions.
    with set_current_vllm_config(_eager_config()):
        return DeepseekScalingRotaryEmbedding(
            ROPE_DIM,
            ROPE_DIM,
            512,
            10000.0,
            is_neox_style=False,
            scaling_factor=512.0,
            dtype=torch.bfloat16,
        ).to(DEVICE)


def _inputs(num_tokens: int, num_heads: int, param_dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    positions = torch.randint(0, 200_000, (num_tokens,), device=DEVICE)
    log_positions = torch.log(positions.float() + 1)
    # SSMax parameters load in the model dtype (bf16).
    s = (0.05 * torch.randn(num_heads, device=DEVICE)).to(param_dtype)
    b = (1 + 0.1 * torch.randn(num_heads, device=DEVICE)).to(param_dtype)
    return positions, log_positions, s, b


def _ssmax_ref(q, log_positions, s, b):
    """``SSMaxScaling.forward`` (unfused)."""
    scale = (
        log_positions.to(q.dtype)[:, None, None] * s.to(q.dtype)[:, None]
        + b.to(q.dtype)[:, None]
    )
    return q * scale


@pytest.mark.parametrize("num_tokens", [1, 77, 1024])
@pytest.mark.parametrize("pad", [0, ROPE_DIM])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
def test_ssmax_scaling(num_tokens: int, pad: int, param_dtype: torch.dtype):
    _, log_positions, s, b = _inputs(num_tokens, 16, param_dtype)
    q = torch.randn(num_tokens, 16, 128, device=DEVICE, dtype=torch.bfloat16)
    ref = _ssmax_ref(F.pad(q, (0, pad)), log_positions, s, b)
    out = fused_ssmax_scaling(q, log_positions, s, b, 128 + pad if pad else None)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 77, 1024])
def test_mla_rope_ssmax(rope, num_tokens: int):
    nope_dim, num_heads = 64, 16
    positions, log_positions, s, b = _inputs(num_tokens, num_heads)
    q = torch.randn(
        num_tokens, num_heads, nope_dim + ROPE_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    # k_pe is a strided slice of the kv_a projection, as in the model.
    kv = torch.randn(num_tokens, 512 + ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)

    q_ref, kv_ref = q.clone(), kv.clone()
    k_ref = kv_ref[:, None, 512:]
    _rope_ref(rope, positions, q_ref[..., nope_dim:], k_ref)
    q_ref = _ssmax_ref(q_ref, log_positions, s, b)

    k = kv[:, None, 512:]
    fused_rope_ssmax_scaling_(
        q, k, positions, rope.cos_sin_cache, log_positions, s, b, nope_dim, False
    )
    _assert_close_ulp(q[..., :nope_dim], q_ref[..., :nope_dim])
    _assert_close_ulp(q[..., nope_dim:], q_ref[..., nope_dim:], ulps=1)
    _assert_close_ulp(k, k_ref, ulps=1)
    torch.testing.assert_close(kv[:, :512], kv_ref[:, :512], atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 77, 1024])
@pytest.mark.parametrize("use_rope", [True, False])
def test_indexer_q(rope, num_tokens: int, use_rope: bool):
    num_heads, head_dim = 32, 128
    positions, log_positions, s, b = _inputs(num_tokens, num_heads)
    q = 3 * torch.randn(
        num_tokens, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    # Per-head weights are a strided slice of the wk/weights projection.
    kw = torch.randn(
        num_tokens, head_dim + num_heads, device=DEVICE, dtype=torch.bfloat16
    )
    weights = kw[:, head_dim:]
    softmax_scale, head_scale = head_dim**-0.5, num_heads**-0.5

    # Unfused MistralIndexer.forward.
    q_ref = _ssmax_ref(q, log_positions, s, b)
    if use_rope:
        q_fp8_ref, w_ref = fused_indexer_q_rope_quant(
            positions, q_ref, rope.cos_sin_cache, weights, softmax_scale,
            head_scale, False,
        )  # fmt: skip
        scales = (softmax_scale, head_scale)
    else:
        q_fp8_ref, q_scale = per_token_group_quant_fp8(
            q_ref.reshape(-1, head_dim), 128, column_major_scales=False,
            use_ue8m0=True,
        )  # fmt: skip
        q_fp8_ref = q_fp8_ref.view(-1, num_heads, head_dim)
        weights_scale = softmax_scale * head_scale
        w_ref = weights * q_scale.view(-1, num_heads) * weights_scale
        scales = (weights_scale, 1.0)

    q_fp8, w = fused_indexer_q_ssmax_rope_quant(
        positions,
        q,
        rope.cos_sin_cache if use_rope else None,
        weights,
        log_positions,
        s,
        b,
        *scales,
        is_neox=False,
    )
    # The RoPE reference is another Triton kernel whose fma contraction may
    # differ: allow rare one-step FP8 code differences.
    code_diff = (
        q_fp8.view(torch.uint8).int() - q_fp8_ref.view(torch.uint8).int()
    ).abs()
    assert code_diff.max() <= 1
    assert code_diff.count_nonzero() <= max(1, q.numel() // 100_000)
    torch.testing.assert_close(w, w_ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 77, 1024])
@pytest.mark.parametrize("use_rope", [True, False])
def test_indexer_k(rope, num_tokens: int, use_rope: bool):
    head_dim = 128
    positions, _, _, _ = _inputs(num_tokens, 1)
    with set_current_vllm_config(_eager_config()):
        k_norm = LayerNorm(head_dim, eps=1e-6, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        k_norm.weight.normal_(1, 0.2)
        k_norm.bias.normal_(0, 0.2)
    kw = 2 * torch.randn(num_tokens, head_dim + 32, device=DEVICE).bfloat16() + 0.5

    # Unfused MistralIndexer.forward.
    k_ref = k_norm(kw[:, :head_dim])
    if use_rope:
        k_pe = k_ref[:, None, :ROPE_DIM].clone()
        _rope_ref(rope, positions, torch.empty_like(k_pe), k_pe)
        k_ref = torch.cat([k_pe.reshape(-1, ROPE_DIM), k_ref[:, ROPE_DIM:]], -1)

    k = fused_indexer_k_norm_rope(
        positions,
        kw[:, :head_dim],
        k_norm.weight,
        k_norm.bias,
        k_norm.eps,
        rope.cos_sin_cache if use_rope else None,
        is_neox=False,
    )
    # LayerNorm statistics are reduced in a different order than torch's, so
    # allow one bf16 ulp on a small fraction of elements.
    diff = (k.float() - k_ref.float()).abs()
    ulp = torch.finfo(torch.bfloat16).eps * k_ref.float().abs().clamp_min(1e-2)
    assert (diff <= ulp).all(), diff.max()
    assert (diff > 0).float().mean() < 0.01


@pytest.mark.parametrize("num_tokens", [1, 4, 77, 1024])
def test_indexer_q_input_norm(num_tokens: int):
    hidden_size = 6144
    with set_current_vllm_config(_eager_config()):
        norm = LayerNorm(hidden_size, eps=1e-6).to(DEVICE)
    with torch.no_grad():
        norm.weight.normal_(1, 0.2)
        norm.bias.normal_(0, 0.2)
    torch.manual_seed(0)
    x = (2 * torch.randn(num_tokens, hidden_size, device=DEVICE) + 0.5).bfloat16()

    out_ref = norm(x)
    out = layer_norm(x, norm.weight, norm.bias, norm.eps)
    diff = (out.float() - out_ref.float()).abs()
    ulp = torch.finfo(torch.bfloat16).eps * out_ref.float().abs().clamp_min(1e-2)
    assert (diff <= ulp).all(), diff.max()
    assert (diff > 0).float().mean() < 0.01


@pytest.mark.parametrize("num_tokens", [1, 77, 1024])
@pytest.mark.parametrize("use_rope", [True, False])
# 16: ML4's block size; 1: the flat layout; 64: several 16-token tiles per block.
@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_indexer_k_cache_write(rope, num_tokens: int, use_rope: bool, block_size: int):
    """The fused cache write stores the same bytes as quantizing the fused bf16
    K with the indexer's insert op; padded and negative slots are skipped. On
    ROCm the readers (prefill gather, paged MQA logits) see the same keys."""
    head_dim = 128
    # Room for every token's slot at any block size.
    num_blocks = max(40, -(-2 * num_tokens // block_size))
    positions, _, _, _ = _inputs(num_tokens, 1)
    with set_current_vllm_config(_eager_config()):
        k_norm = LayerNorm(head_dim, eps=1e-6, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        k_norm.weight.normal_(1, 0.2)
        k_norm.bias.normal_(0, 0.2)
    kw = 2 * torch.randn(num_tokens, head_dim + 32, device=DEVICE).bfloat16() + 0.5
    # Shorter than the batch (graph padding), with one skipped token; a
    # single-token decode keeps its one slot so the write is checked.
    num_slots = max(num_tokens - 3, 1)
    slot_mapping = torch.randperm(num_blocks * block_size, device=DEVICE)[:num_slots]
    if num_slots > 1:
        slot_mapping[num_slots // 2] = -1
    args = (
        positions,
        kw[:, :head_dim],
        k_norm.weight,
        k_norm.bias,
        k_norm.eps,
        rope.cos_sin_cache if use_rope else None,
        False,
    )

    def empty_cache():
        return torch.zeros(
            num_blocks, block_size, head_dim + 4, dtype=torch.uint8, device=DEVICE
        )

    cache_ref = empty_cache()
    k = fused_indexer_k_norm_rope(*args)
    # Same scale format and layout as the indexer's own cache insert (on ROCm
    # the Triton insert, whose 16x16 SHUFFLE layout the readers expect).
    scale_fmt = "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0"
    insert = (
        indexer_k_quant_and_cache_triton
        if current_platform.is_rocm()
        else ops.indexer_k_quant_and_cache
    )
    insert(k[:num_slots], cache_ref, slot_mapping, 128, scale_fmt)
    cache = empty_cache()
    fused_indexer_k_norm_rope_cache_write(
        *args, kv_cache=cache, slot_mapping=slot_mapping
    )
    assert cache_ref.any(), "the reference write stored nothing"
    torch.testing.assert_close(cache, cache_ref, atol=0, rtol=0)

    if current_platform.is_rocm():
        _check_indexer_cache_readers(cache, k[:num_slots], slot_mapping, block_size)


def _check_indexer_cache_readers(cache, k, slot_mapping, block_size: int):
    """The ROCm readers of the indexer cache recover the written keys: the
    prefill gather returns them (dequantized ~= the bf16 K), and the paged MQA
    logits reference reading the cache in its layout scores them like the
    gathered keys."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        FP8_DTYPE,
        cp_gather_indexer_k_quant_cache_triton,
        fp8_paged_mqa_logits_torch,
        indexer_k_cache_is_shuffled,
    )

    num_blocks, head_dim = cache.shape[0], k.shape[-1]
    capacity = num_blocks * block_size
    # One sequence spanning every block in order: its token t is slot t.
    block_table = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32)[None]
    k_fp8 = torch.empty(capacity, head_dim, device=DEVICE, dtype=FP8_DTYPE)
    k_scale = torch.empty(capacity, 4, device=DEVICE, dtype=torch.uint8)
    cp_gather_indexer_k_quant_cache_triton(
        cache,
        k_fp8,
        k_scale,
        block_table,
        torch.tensor([0, capacity], device=DEVICE, dtype=torch.int32),
        torch.zeros(capacity, device=DEVICE, dtype=torch.int32),
    )
    gathered = k_fp8.float() * k_scale.view(torch.float32)
    written = slot_mapping >= 0
    torch.testing.assert_close(
        gathered[slot_mapping[written]], k[written].float(), atol=0.25, rtol=0.07
    )

    # Decode over the whole cache, read in the cache's layout: one token
    # (next_n=1) and a speculative verify block (next_n=4, causal rows).
    for next_n in (1, 4):
        q = torch.randn(1, next_n, 4, head_dim, device=DEVICE).to(FP8_DTYPE)
        weights = torch.rand(next_n, 4, device=DEVICE)
        logits = fp8_paged_mqa_logits_torch(
            q,
            cache.unsqueeze(2),
            weights,
            torch.tensor([capacity], device=DEVICE, dtype=torch.int32),
            block_table,
            capacity,
            shuffled=indexer_k_cache_is_shuffled(block_size),
        )
        for row in range(next_n):
            last = capacity - next_n + row  # the row's own position
            ref = (q[0, row].float() @ gathered[: last + 1].T).relu()
            ref = (ref * weights[row, :, None]).sum(0)
            torch.testing.assert_close(
                logits[row, : last + 1], ref, atol=1e-3, rtol=1e-3
            )


def test_indexer_q_clears_topk(rope):
    num_tokens, num_heads, head_dim, topk = 5, 32, 128, 2048
    positions, log_positions, s, b = _inputs(num_tokens, num_heads)
    q = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE).bfloat16()
    weights = torch.randn(num_tokens, num_heads, device=DEVICE).bfloat16()
    args = (positions, q, rope.cos_sin_cache, weights, log_positions, s, b, 1.0, 1.0)
    topk_indices = torch.arange(8 * topk, dtype=torch.int32, device=DEVICE)
    topk_indices = topk_indices.view(8, topk)
    expected = topk_indices.clone()
    expected[:num_tokens] = -1

    q_fp8_ref, w_ref = fused_indexer_q_ssmax_rope_quant(*args, is_neox=False)
    q_fp8, w = fused_indexer_q_ssmax_rope_quant(
        *args, is_neox=False, topk_indices=topk_indices
    )
    torch.testing.assert_close(topk_indices, expected, atol=0, rtol=0)
    assert torch.equal(q_fp8.view(torch.uint8), q_fp8_ref.view(torch.uint8))
    torch.testing.assert_close(w, w_ref, atol=0, rtol=0)
