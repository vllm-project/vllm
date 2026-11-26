# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weighted RMSNorm + partial NeoX RoPE with separate paged K/V caches."""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="Step-5 kernel is only built for CUDA",
)


def norm_rope_reference(x, weight, cos_sin, positions, rotary_dim, eps, bias):
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    out = (normed * (weight.float() + bias)).to(x.dtype)
    half = rotary_dim // 2
    cos, sin = cos_sin[positions].float().unsqueeze(1).chunk(2, dim=-1)
    x0, x1 = out[..., :half].float(), out[..., half:rotary_dim].float()
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:rotary_dim] = x0 * sin + x1 * cos
    return out


def check_norm_rope_and_cache(
    dtype=torch.bfloat16,
    kv_cache_dtype="auto",
    head_dim=128,
    rotary_dim=64,
    num_tokens=19,
    block_size=16,
    fp32_weights=False,
    bias=0.0,
    strided_cache=False,
    use_cuda_graph=False,
):
    """Cover rounding, 192-wide RoPE, padded rows and short/negative slots."""
    torch.manual_seed(0)
    nq, nkv, max_pos, eps = 5, 2, 64, 1e-5
    width = (nq + 2 * nkv) * head_dim
    storage = torch.randn(num_tokens, width + 8, device="cuda", dtype=dtype)
    qkv = storage[:, :width]
    original = storage.clone()
    weight_dtype = torch.float32 if fp32_weights else dtype
    qw = torch.randn(head_dim, device="cuda", dtype=weight_dtype)
    kw = torch.randn_like(qw)
    angles = torch.randn(max_pos, rotary_dim // 2, device="cuda")
    rope_dtype = torch.float32 if fp32_weights else dtype
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(rope_dtype)
    positions = torch.randint(max_pos, (num_tokens,), device="cuda")
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    shape = (num_blocks, block_size, nkv, head_dim)
    if strided_cache:
        shape = (num_blocks, nkv, block_size, head_dim)
    key_cache = torch.zeros(shape, device="cuda", dtype=cache_dtype)
    value_cache = torch.zeros_like(key_cache)
    if strided_cache:
        key_cache = key_cache.transpose(1, 2)
        value_cache = value_cache.transpose(1, 2)
    # A short mapping models graph padding; negative slots must stay unwritten.
    num_slots = max(0, num_tokens - 3) if num_tokens > 1 else num_tokens
    slots = torch.randperm(num_blocks * block_size, device="cuda")[:num_slots]
    slots[1::3] = -1
    k_scale = torch.tensor(0.37, device="cuda")
    v_scale = torch.tensor(1.83, device="cuda")

    def run():
        return ops.fused_qknorm_rope_cache(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            slots,
            key_cache,
            value_cache,
            nq,
            nkv,
            head_dim,
            rotary_dim,
            eps,
            norm_weight_bias=bias,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    q, k, v = run()
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, k, v = run()
        key_cache.zero_()
        value_cache.zero_()
        graph.replay()

    q_in, k_in, v_in = qkv.split(
        (nq * head_dim, nkv * head_dim, nkv * head_dim), dim=-1
    )
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    for actual, src, weight, heads in ((q, q_in, qw, nq), (k, k_in, kw, nkv)):
        expected = norm_rope_reference(
            src.reshape(num_tokens, heads, head_dim),
            weight,
            cos_sin,
            positions,
            rotary_dim,
            eps,
            bias,
        ).reshape_as(actual)
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(v, v_in, rtol=0, atol=0)
    torch.testing.assert_close(storage, original, rtol=0, atol=0)

    expected_k, expected_v = torch.zeros_like(key_cache), torch.zeros_like(value_cache)
    if num_slots:
        ops.reshape_and_cache_flash(
            k.reshape(num_tokens, nkv, head_dim),
            v.reshape(num_tokens, nkv, head_dim),
            expected_k,
            expected_v,
            slots,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    for actual, expected in ((key_cache, expected_k), (value_cache, expected_v)):
        if kv_cache_dtype == "fp8":
            # Reciprocal multiplication versus division may differ by one code.
            assert (actual.int() - expected.int()).abs().max() <= 1
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(dict(head_dim=64, rotary_dim=32, num_tokens=1), id="decode"),
        pytest.param(
            dict(block_size=7, fp32_weights=True, bias=1.0, strided_cache=True),
            id="strided-cache-fp32-weights",
        ),
        pytest.param(
            dict(head_dim=192, rotary_dim=128, fp32_weights=True),
            id="head192-partial-rope",
        ),
        pytest.param(
            dict(
                head_dim=192,
                rotary_dim=192,
                block_size=7,
                bias=1.0,
                strided_cache=True,
                kv_cache_dtype="fp8",
            ),
            id="head192-full-rope-fp8",
        ),
        pytest.param(
            dict(
                head_dim=256, rotary_dim=256, strided_cache=True, kv_cache_dtype="fp8"
            ),
            id="head256-full-rope-fp8",
        ),
        pytest.param(
            dict(head_dim=192, rotary_dim=0, fp32_weights=True, bias=1.0),
            id="no-rope",
        ),
        pytest.param(
            dict(head_dim=64, rotary_dim=8, fp32_weights=True),
            id="minimum-rotary-dim",
        ),
    ],
)
def test_norm_rope_and_cache(dtype, case):
    check_norm_rope_and_cache(dtype=dtype, **case)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(dict(num_tokens=0), id="empty-input"),
        pytest.param(dict(use_cuda_graph=True), id="graph-auto-cache"),
        pytest.param(
            dict(use_cuda_graph=True, kv_cache_dtype="fp8"), id="graph-fp8-cache"
        ),
    ],
)
def test_empty_input_and_graph_replay(case):
    check_norm_rope_and_cache(**case)


def test_fp8_query_store_preserves_model_dtype_kv_and_cache():
    torch.manual_seed(19)
    tokens, q_heads, kv_heads, dim = 7, 5, 2, 192
    qkv = torch.randn(
        (tokens, (q_heads + 2 * kv_heads) * dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda")
    slots = torch.tensor([0, 1, -1, 33, 34], device="cuda")
    key_cache = torch.zeros((2, 32, kv_heads, dim), device="cuda", dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    scales = [torch.tensor(v, device="cuda") for v in (0.5, 2.0)]

    def run(fp8_query):
        return ops.fused_qknorm_rope_cache(
            qkv,
            weight,
            weight,
            rope,
            positions,
            slots,
            key_cache,
            value_cache,
            q_heads,
            kv_heads,
            dim,
            128,
            1e-5,
            q_dtype=torch.float8_e4m3fn if fp8_query else None,
            kv_cache_dtype="fp8",
            q_scale=1 / 256,
            k_scale=scales[0],
            v_scale=scales[1],
        )

    native = run(False)
    stored = key_cache.clone(), value_cache.clone()
    run(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(True)
    graph.replay()
    expected_q = (native[0].float() * 256).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(actual[0].float(), expected_q.float(), atol=0, rtol=0)
    for output, expected in zip(actual[1:], native[1:]):
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    for output, expected in zip((key_cache, value_cache), stored):
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
