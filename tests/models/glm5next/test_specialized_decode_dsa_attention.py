# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's DSA attention kernels against an FP64 PyTorch
reference: no larger error than the stock kernels they replace (cuBLAS bmm,
``fused_q_kv_rmsnorm``, ``concat_and_cache_mla``, static ``scaled_fp8_quant``)."""

import pytest
import torch

from vllm.platforms import current_platform

HEADS, NOPE, LATENT, V = 16, 256, 512, 256
Q_LORA = 1536

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the DSA attention kernels are CUDA kernels"
)


def _kv_b_weights(g):
    """W_UK_T [N, P, L] and W_UV [N, L, V] as views of one kv_b_proj weight, the
    way ``MLAAttention`` derives them."""
    weight = torch.randn(HEADS * (NOPE + V), LATENT, generator=g, device="cuda")
    weight = (weight * LATENT**-0.5).bfloat16()
    w_uk, w_uv = weight.T.view(LATENT, HEADS, NOPE + V).split([NOPE, V], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)


def _errors(value, reference):
    error = (value.double() - reference).abs()
    return error.max().item(), error.mean().item()


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_value_up_matches_bmm(tokens):
    from vllm.models.glm5next.nvidia.specialized_decode.dsa_attention import (
        head_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    _, w_uv = _kv_b_weights(g)
    x = torch.randn(tokens, HEADS, LATENT, generator=g, device="cuda").bfloat16()
    out = torch.empty(tokens, HEADS, V, dtype=torch.bfloat16, device="cuda")
    head_gemv(x, w_uv.contiguous(), out)

    reference = torch.einsum("thl,hlv->thv", x.double(), w_uv.double())
    stock = torch.empty_like(out)
    torch.bmm(x.transpose(0, 1), w_uv, out=stock.transpose(0, 1))
    ours, theirs = _errors(out, reference), _errors(stock, reference)
    assert ours[0] <= theirs[0] and ours[1] <= theirs[1] * 1.02, (ours, theirs)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("q_scale", [1.0, 0.75])
def test_absorb_quant_matches_bmm_and_quant(tokens, q_scale):
    """The absorb bmm with the static query quant fused: the FP8 bytes of the
    stock bmm + ``scaled_fp8_quant`` but where the two BF16 products round
    differently, and no larger error against the FP64 reference."""
    from vllm import _custom_ops as ops
    from vllm.models.glm5next.nvidia.specialized_decode.dsa_attention import (
        head_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w_uk, _ = _kv_b_weights(g)
    q = 4 * torch.randn(tokens, HEADS, NOPE, generator=g, device="cuda")
    q = q.bfloat16()
    scale = torch.tensor([q_scale], dtype=torch.float32, device="cuda")
    out = torch.empty(tokens, HEADS, LATENT, dtype=torch.float8_e4m3fn, device="cuda")
    head_gemv(q, w_uk, out, scale)

    product = torch.empty(tokens, HEADS, LATENT, dtype=torch.bfloat16, device="cuda")
    torch.bmm(q.transpose(0, 1), w_uk, out=product.transpose(0, 1))
    stock, _ = ops.scaled_fp8_quant(product.view(tokens, -1), scale)
    stock = stock.view(tokens, HEADS, LATENT)

    reference = torch.einsum("thp,hpl->thl", q.double(), w_uk.double()) / q_scale
    ours, theirs = _errors(out.float(), reference), _errors(stock.float(), reference)
    assert ours[0] <= theirs[0] and ours[1] <= theirs[1] * 1.02, (ours, theirs)
    same = out.view(torch.uint8) == stock.view(torch.uint8)
    assert same.float().mean() > 0.99


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_norm_matches_stock(tokens):
    """The q/kv a-norm: the stock Triton kernel's outputs but where a different
    summation order flips a rounding, and no larger error."""
    from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
    from vllm.models.glm5next.nvidia.specialized_decode.dsa_attention import norm

    g = torch.Generator(device="cuda").manual_seed(tokens)
    eps = 1e-6
    qkv = 3 * torch.randn(tokens, Q_LORA + LATENT, generator=g, device="cuda")
    qkv = qkv.bfloat16()
    q_w = (1 + 0.1 * torch.randn(Q_LORA, generator=g, device="cuda")).bfloat16()
    kv_w = (1 + 0.1 * torch.randn(LATENT, generator=g, device="cuda")).bfloat16()
    outputs = norm(qkv, q_w, kv_w, eps, Q_LORA)
    stock = fused_q_kv_rmsnorm(qkv[:, :Q_LORA], qkv[:, Q_LORA:], q_w, kv_w, eps)
    x = qkv.double()
    for ours, theirs, part, weight in zip(
        outputs, stock, (x[:, :Q_LORA], x[:, Q_LORA:]), (q_w, kv_w)
    ):
        rms = part.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        exact = part * rms * weight.double()
        mine, ref = _errors(ours, exact), _errors(theirs, exact)
        assert mine[0] <= ref[0] * 1.01 and mine[1] <= ref[1] * 1.02, (mine, ref)
        assert (ours == theirs).float().mean() > 0.99


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("k_scale", [1.0, 0.5])
def test_absorb_writes_kv_cache_as_stock(tokens, k_scale):
    """The absorb kernel's cache CTAs write ``concat_and_cache_mla``'s FP8 bytes
    at the valid slots of a padded-stride cache and nothing else, and leave
    the query as it is without them."""
    from vllm import _custom_ops as ops
    from vllm.models.glm5next.nvidia.specialized_decode.dsa_attention import (
        head_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w_uk, _ = _kv_b_weights(g)
    q = torch.randn(tokens, HEADS, NOPE, generator=g, device="cuda").bfloat16()
    kv = 3 * torch.randn(tokens, LATENT, generator=g, device="cuda")
    kv = kv.bfloat16()
    blocks, block_size = 8, 64
    # A padded block stride, as the hybrid KV cache allocates it.
    cache = torch.zeros(blocks, block_size + 2, LATENT, dtype=torch.uint8)
    cache = cache.cuda()[:, :block_size]
    stock_cache = cache.clone()
    slots = torch.randperm(blocks * block_size)[:tokens].cuda()
    if tokens > 1:
        slots[-1] = -1
    scale = torch.tensor([k_scale], dtype=torch.float32, device="cuda")
    q_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    out = torch.empty(tokens, HEADS, LATENT, dtype=torch.float8_e4m3fn, device="cuda")
    head_gemv(q, w_uk, out, q_scale, (kv, cache, slots, scale))
    plain = torch.empty_like(out)
    head_gemv(q, w_uk, plain, q_scale)
    assert torch.equal(out.view(torch.uint8), plain.view(torch.uint8))

    ops.concat_and_cache_mla(
        kv, kv[:, :0], stock_cache.view(torch.float8_e4m3fn), slots, "fp8", scale
    )
    assert torch.equal(cache, stock_cache)
    untouched = torch.ones(blocks * block_size, dtype=torch.bool, device="cuda")
    untouched[slots[slots >= 0]] = False
    assert not cache.reshape(-1, LATENT)[untouched].any()


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_kda_gate_matches_linear(tokens):
    """The KDA output gate's g_b_proj (rank 128 -> 16 heads x 128) on a strided
    slice of the in_proj output, as ``head_gemv`` of one head."""
    from vllm.models.glm5next.nvidia.specialized_decode.dsa_attention import (
        head_gemv,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    rank, width, pitch = 128, 2048, 6416
    weight = torch.randn(width, rank, generator=g, device="cuda") * rank**-0.5
    weight = weight.bfloat16()
    projected = torch.randn(tokens, pitch, generator=g, device="cuda").bfloat16()
    g_a = projected[:, 6288 : 6288 + rank]
    gate = torch.empty(tokens, width, dtype=torch.bfloat16, device="cuda")
    head_gemv(g_a.unsqueeze(1), weight.t().contiguous().unsqueeze(0), gate.unsqueeze(1))

    reference = g_a.double() @ weight.double().t()
    stock = torch.nn.functional.linear(g_a, weight)
    ours, theirs = _errors(gate, reference), _errors(stock, reference)
    assert ours[0] <= theirs[0] and ours[1] <= theirs[1] * 1.02, (ours, theirs)


@pytest.mark.parametrize(
    "shape,tokens",
    [((2048, 4096), 4), ((2048, 4096), 64), ((4096, 1536), 16), ((4096, 4096), 32)],
)
def test_split_k_projection_matches_linear(shape, tokens):
    """A routed DSA projection inside a planned step runs the split-K GEMM on a
    strided input (q_b reads q_c out of the q/kv a output) with no larger error
    than cuBLAS's against the FP64 product."""
    from types import SimpleNamespace

    from vllm.models.glm5next.nvidia.specialized_decode.linear import (
        planned_step,
        route,
    )

    class Linear:
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight)

    g = torch.Generator(device="cuda").manual_seed(tokens)
    n, k = shape
    weight = torch.randn(n, k, generator=g, device="cuda") * k**-0.5
    module = SimpleNamespace(weight=weight.bfloat16(), quant_method=Linear())
    assert route(module, [tokens])
    assert tokens in module.quant_method.projection.split_k.kernels
    x = torch.randn(tokens, k + 512, generator=g, device="cuda").bfloat16()[:, :k]
    with planned_step():
        out = module.quant_method.apply(module, x)

    reference = x.double() @ module.weight.double().t()
    stock = torch.nn.functional.linear(x, module.weight)
    ours, theirs = _errors(out, reference), _errors(stock, reference)
    assert ours[0] <= theirs[0] and ours[1] <= theirs[1] * 1.02, (ours, theirs)
