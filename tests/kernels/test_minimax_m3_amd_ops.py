# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-vs-optimized unit tests for the MiniMax-M3 AMD/ROCm fused kernels.

Each optimized kernel added for the ROCm port has a slow PyTorch reference; the
tests assert the two agree within tolerance:

  * Gemma RMSNorm (plain + fused-add-residual)  -> fp32 PyTorch normalize
  * SwiGLU-OAI (split layout)                    -> fp32 PyTorch elementwise
  * Fused MXFP8 activation quant (Triton)        -> _mxfp8_e4m3_quantize_torch
  * Native MXFP8 linear (dot_scaled)             -> dequant-to-bf16 @ matmul
  * Native MXFP8 MoE (dot_scaled grouped GEMM)   -> dequant-to-bf16 MoE math

The native MXFP8 GEMMs also guard the ``dot_scaled`` rhs-scale orientation: the
scale is loaded ``[N, K//32]`` and passed WITHOUT transpose; a stray ``.T``
makes the shape ``[K//32, N]`` and Triton raises before producing output, so any
regression there fails these tests loudly.

Hardware scope: the whole module is ROCm-only (these are the AMD path; NVIDIA
uses the FlashInfer kernels). The norm/activation/quant kernels run on any ROCm
arch; the native MXFP8 ``dot_scaled`` linear/MoE tests are additionally gated to
CDNA4 gfx95x (``@requires_gfx950``) since gfx942 uses the BF16 emulation path.

Run:  pytest tests/kernels/test_minimax_m3_amd_ops.py -v
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 AMD fused ops require ROCm.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_e4m3_quantize_triton,
    dequant_mxfp8_to_bf16,
)
from vllm.models.minimax_m3.amd.ops import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    swiglu_oai_split,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import _num_warps  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # pragma: no cover - no device / non-AMD
        return ""


# The pure-Triton norm/activation/quant kernels run on any ROCm arch (CDNA3
# gfx942 and CDNA4 gfx950). The native MXFP8 ``dot_scaled`` GEMMs (linear + MoE)
# use CDNA4 hardware microscaling and are gated to gfx95x in the source
# (``RocmDotScaledMxfp8LinearKernel.is_supported``; the MoE oracle routes gfx942
# to the BF16 emulation path instead) — so those tests are gfx950-only.
requires_gfx950 = pytest.mark.skipif(
    "gfx95" not in _gcn_arch(),
    reason="native MXFP8 dot_scaled is a CDNA4 (gfx95x) feature; "
    "gfx942 uses the BF16 emulation path instead.",
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


# --------------------------------------------------------------------------- #
# Gemma RMSNorm
# --------------------------------------------------------------------------- #
def _ref_gemma_rmsnorm(x, w, eps, residual=None):
    orig_dtype = x.dtype
    xf = x.float()
    res_out = None
    if residual is not None:
        xf = xf + residual.float()
        res_out = xf.to(orig_dtype)
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xf = xf * (1.0 + w.float())
    out = xf.to(orig_dtype)
    return out if residual is None else (out, res_out)


@pytest.mark.parametrize("shape", [(1, 4096), (37, 6144), (128, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 1234])
@torch.inference_mode()
def test_gemma_rmsnorm(shape, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype) * 0.1
    got = gemma_rmsnorm(x, w, EPS)
    ref = _ref_gemma_rmsnorm(x, w, EPS)
    assert got.shape == x.shape
    assert _relerr(got, ref) < 5e-3


@pytest.mark.parametrize("shape", [(1, 6144), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_fused_add_rmsnorm(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    res = torch.randn(*shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype) * 0.1
    got_out, got_res = gemma_fused_add_rmsnorm(x, res, w, EPS)
    ref_out, ref_res = _ref_gemma_rmsnorm(x, w, EPS, residual=res)
    assert _relerr(got_out, ref_out) < 5e-3
    # residual_out is the pre-norm sum (x + res): bit-for-bit identical cast.
    assert torch.equal(got_res, ref_res)


@torch.inference_mode()
def test_gemma_rmsnorm_per_head_strided():
    """q_norm/k_norm normalize a non-contiguous ``qkv.split`` slice over head_dim."""
    torch.manual_seed(0)
    T, H, D, kv = 7, 48, 128, 8
    total = (H + 2 * kv) * D
    qkv = torch.randn(T, total, device=DEVICE, dtype=torch.bfloat16)
    q = qkv[..., : H * D]  # non-contiguous view (row stride == total)
    q_by_head = q.view(T, H, D)
    assert not q_by_head.is_contiguous()
    w = torch.randn(D, device=DEVICE, dtype=torch.bfloat16) * 0.1
    got = gemma_rmsnorm(q_by_head, w, EPS)
    ref = _ref_gemma_rmsnorm(q_by_head, w, EPS)
    assert got.shape == q_by_head.shape
    assert _relerr(got, ref) < 5e-3


def test_num_warps_monotonic():
    assert _num_warps(128) <= _num_warps(2048) <= _num_warps(8192)


# --------------------------------------------------------------------------- #
# SwiGLU-OAI (split layout)
# --------------------------------------------------------------------------- #
def _ref_swiglu(gate_up, alpha, beta, limit):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(gate_up.dtype)


@pytest.mark.parametrize("m,inter", [(1, 768), (64, 1536), (128, 1024)])
@pytest.mark.parametrize("limit", [7.0, None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_swiglu_oai_split(m, inter, limit, dtype):
    torch.manual_seed(0)
    gate_up = torch.randn(m, 2 * inter, device=DEVICE, dtype=dtype)
    got = swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=limit)
    ref = _ref_swiglu(gate_up, 1.702, 1.0, limit)
    assert got.shape == (m, inter)
    assert _relerr(got, ref) < 5e-3


# --------------------------------------------------------------------------- #
# Fused MXFP8 activation quant (Triton vs torch reference)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(64, 4096), (1, 6144), (333, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_mxfp8_quant_triton_matches_torch(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    xq_t, s_t = _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout=False)
    xq_k, s_k = _mxfp8_e4m3_quantize_triton(x)
    assert s_k.shape == s_t.shape == (shape[0], shape[1] // 32)
    # E8M0 block exponents share the floor(log2(amax))+127 algorithm; allow at
    # most a 1-step difference at exact powers of two.
    assert (s_k.int() - s_t.int()).abs().max().item() <= 1
    # Dequantized values agree to fp8 granularity.
    deq_t = dequant_mxfp8_to_bf16(xq_t, s_t)
    deq_k = dequant_mxfp8_to_bf16(xq_k, s_k)
    assert _relerr(deq_k, deq_t) < 1e-2


# --------------------------------------------------------------------------- #
# Native MXFP8 linear (dot_scaled) vs dequant-to-bf16 matmul
# --------------------------------------------------------------------------- #
@requires_gfx950
@pytest.mark.parametrize("m,n,k", [(64, 256, 128), (37, 512, 256), (1, 6144, 4096)])
@torch.inference_mode()
def test_mxfp8_native_linear(m, n, k):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    torch.manual_seed(0)
    w_bf16 = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5

    got = _mxfp8_dot_scaled_linear(x, w_fp8, w_scale)
    # Reference: consume the SAME quantized weights (isolates activation-quant
    # noise) -> dequant to bf16, plain matmul.
    w_deq = dequant_mxfp8_to_bf16(w_fp8, w_scale)
    ref = torch.nn.functional.linear(x, w_deq).to(x.dtype)
    assert got.shape == (m, n)
    # Only the activation is re-quantized inside the kernel -> small MX noise.
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# Native MXFP8 MoE (dot_scaled grouped GEMM) vs dequant-to-bf16 MoE math
# --------------------------------------------------------------------------- #
def _ref_moe(x, w13, w2, topk_weights, topk_ids, alpha, beta, limit):
    T, H = x.shape
    inter = w2.shape[-1]
    top_k = topk_ids.shape[1]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            g1 = x[t].float() @ w13[e].float().T  # [2I]
            gate = g1[:inter]
            up = g1[inter:]
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            g2 = act @ w2[e].float().T  # [H]
            out[t] += topk_weights[t, j].float() * g2
    return out.to(x.dtype)


@requires_gfx950
@pytest.mark.parametrize(
    "T,H,inter,E,top_k", [(8, 256, 512, 8, 2), (1, 512, 256, 16, 4)]
)
@torch.inference_mode()
def test_mxfp8_native_moe(T, H, inter, E, top_k):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        fused_moe_mxfp8_native,
    )

    torch.manual_seed(0)
    alpha, beta, limit = 1.702, 1.0, 7.0
    w13_bf16 = torch.randn(E, 2 * inter, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, inter, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(
        w13_bf16, is_sf_swizzled_layout=False
    )
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16, is_sf_swizzled_layout=False)

    x = torch.randn(T, H, device=DEVICE, dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    got = fused_moe_mxfp8_native(
        x,
        w13_fp8,
        w13_scale,
        w2_fp8,
        w2_scale,
        topk_weights,
        topk_ids,
        alpha=alpha,
        beta=beta,
        limit=limit,
        global_num_experts=E,
        expert_map=None,
    )
    # Reference consumes the dequantized weights (same bits the kernel reads).
    w13_deq = dequant_mxfp8_to_bf16(w13_fp8, w13_scale)
    w2_deq = dequant_mxfp8_to_bf16(w2_fp8, w2_scale)
    ref = _ref_moe(x, w13_deq, w2_deq, topk_weights, topk_ids, alpha, beta, limit)
    assert got.shape == (T, H)
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# Native MXFP8 grouped GEMM (dot_scaled) vs pure-PyTorch grouped matmul
# --------------------------------------------------------------------------- #
def _ref_grouped_gemm(a_deq, w_deq, topk_ids, a_div, num_valid, mul_weight=None):
    """Pure-PyTorch reference for ``_grouped_gemm_mxfp8``.

    For each routed (expanded) token ``tid in [0, num_valid)`` the kernel writes
    ``out[tid] = a[tid // a_div] @ w[expert(tid)].T`` (fp32 accumulate), optionally
    scaled by ``mul_weight[tid]``. The expert for ``tid`` is ``topk_ids.flatten()
    [tid]`` (row-major expansion: ``tid = token*top_k + slot``). This is computed
    here with plain ``torch.matmul`` on the dequantized operands — independent of
    the Triton ``dot_scaled`` path and of the (separate) aiter backend.
    """
    eids = topk_ids.reshape(-1)
    n = w_deq.shape[1]
    out = torch.empty(num_valid, n, dtype=torch.float32, device=a_deq.device)
    for tid in range(num_valid):
        e = int(eids[tid].item())
        out[tid] = a_deq[tid // a_div].float() @ w_deq[e].float().T
        if mul_weight is not None:
            out[tid] *= float(mul_weight[tid].item())
    return out


@requires_gfx950
@pytest.mark.parametrize("T,N,K,E,top_k", [(8, 256, 128, 8, 2), (5, 512, 256, 16, 4)])
@pytest.mark.parametrize("weighted", [False, True])
@torch.inference_mode()
def test_mxfp8_grouped_gemm_native(T, N, K, E, top_k, weighted):
    """Directly exercise ``_grouped_gemm_mxfp8`` against a non-Triton reference.

    Covers both call modes used by ``fused_moe_mxfp8_native``:
      * ``weighted=False`` -> g1: ``a_div=top_k`` (a-row shared across the top_k
        expansions of a token), no per-token weight.
      * ``weighted=True``  -> g2: ``a_div=1`` (one a-row per expansion), output
        scaled by ``topk_weights``.
    """
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _grouped_gemm_mxfp8,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    block_m = 64
    a_div = 1 if weighted else top_k
    m_routed = T * top_k
    # a-rows: g1 reads one row per token (a_div=top_k); g2 one per expansion.
    a_rows = m_routed if weighted else T
    a_bf16 = torch.randn(a_rows, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w_bf16 = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    a_fp8, a_scale = _mxfp8_e4m3_quantize_torch(a_bf16, is_sf_swizzled_layout=False)
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)

    logits = torch.randn(T, E, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    mul = topk_weights.reshape(-1) if weighted else None

    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_m, E, None, ignore_invalid_experts=False
    )
    got = _grouped_gemm_mxfp8(
        a_fp8,
        a_scale,
        w_fp8,
        w_scale,
        sorted_ids,
        expert_ids,
        num_post,
        m_routed,
        top_k,
        block_m,
        torch.bfloat16,
        a_div=a_div,
        mul_weight_by=mul,
    )
    # Reference: dequant the SAME bits the kernel reads, plain torch matmul.
    a_deq = dequant_mxfp8_to_bf16(a_fp8, a_scale)
    w_deq = dequant_mxfp8_to_bf16(w_fp8, w_scale)
    ref = _ref_grouped_gemm(a_deq, w_deq, topk_ids, a_div, m_routed, mul)
    assert got.shape == (m_routed, N)
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# MXFP8 linear emulation: BF16-at-load (default) vs per-step dequant + switch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(512, 2048), (1, 6144)])
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dequant_at_load", [True, False])
@torch.inference_mode()
def test_mxfp8_linear_emulation_bf16_at_load(
    shape, act_dtype, dequant_at_load, monkeypatch
):
    """EmulationMxfp8LinearKernel load-time BF16 dequant (default) and the
    ``VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0`` per-step fallback must produce the
    same result; the dtype-match (BF16/FP16 activations) must also hold."""
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    monkeypatch.setenv(
        "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "1" if dequant_at_load else "0"
    )
    N, K = shape
    torch.manual_seed(0)
    w_bf16 = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)
    assert w_scale.shape == (N, K // 32)

    # Reference: dequant once, plain linear in the activation dtype.
    w_ref = dequant_mxfp8_to_bf16(w_fp8, w_scale).to(act_dtype)
    x = torch.randn(7, K, device=DEVICE, dtype=act_dtype)
    out_ref = torch.nn.functional.linear(x, w_ref)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp8.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_scale.clone(), requires_grad=False)

    kernel = EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    if dequant_at_load:
        # weights converted to BF16 at load (>= 2-byte)
        assert layer.weight.element_size() >= 2
    else:
        # opt-out: weights stay 1-byte MXFP8, dequant happens per-step
        assert layer.weight.element_size() == 1

    out = kernel.apply_weights(layer, x)
    assert out.dtype == act_dtype  # dtype-match preserved (no tl.dot/F.linear crash)
    assert _relerr(out.float(), out_ref.float()) < 2e-2


# ── EP expert_mask handling for the FlyDSL (AITER_MXFP8) MoE ────────────────
# The map/mask choice lives in ``RoutedExperts.expert_map``: it hands AITER
# experts (``consumes_expert_mask``) the precomputed 0/1 ``expert_mask`` and
# everyone else the canonical -1 index map, keyed on the resolved experts kernel
# rather than the global VLLM_ROCM_USE_AITER switch. ``AiterMxfp8Experts.apply``
# forwards that mask to aiter unchanged (it no longer rebuilds it via
# ``(expert_map >= 0)``, which collapsed an already-0/1 mask to all-ones and
# produced EP garbage).
def _capture_expert_mask(expert_mask, *, global_num_experts):
    """Drive the real ``AiterMxfp8Experts.apply`` mask branch and capture the
    ``expert_mask`` it forwards to ``rocm_aiter_ops.fused_moe``."""
    from types import SimpleNamespace
    from unittest import mock

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
        AiterMxfp8Experts,
    )

    experts = object.__new__(AiterMxfp8Experts)  # bypass heavy __init__
    experts.quant_config = SimpleNamespace(gemm1_clamp_limit=None)
    experts.w1_scale_val = None
    experts.w2_scale_val = None

    captured = {}

    def _fake_fused_moe(hidden_states, w1, w2, tw, ti, *, expert_mask, **kw):
        captured["expert_mask"] = expert_mask
        return torch.zeros_like(hidden_states)

    w1 = torch.zeros(1, device=DEVICE)
    w2 = torch.zeros(1, device=DEVICE)
    out = torch.zeros(4, 8, device=DEVICE, dtype=torch.bfloat16)
    hidden = torch.zeros(4, 8, device=DEVICE, dtype=torch.bfloat16)
    tw = torch.ones(4, 2, device=DEVICE)
    ti = torch.zeros(4, 2, dtype=torch.int32, device=DEVICE)

    with mock.patch.object(rocm_aiter_ops, "fused_moe", side_effect=_fake_fused_moe):
        experts.apply(
            output=out,
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=tw,
            topk_ids=ti,
            activation=None,
            global_num_experts=global_num_experts,
            expert_map=expert_mask,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
    return captured["expert_mask"]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_aiter_mxfp8_apply_forwards_expert_mask_unchanged():
    """AiterMxfp8Experts.apply forwards the precomputed 0/1 mask to aiter as-is
    (no re-derivation that would collapse an already-0/1 mask to all-ones)."""
    # 0/1 local-expert mask over global ids + trailing sentinel (rank owns 0..3).
    ep_mask = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32, device=DEVICE
    )
    got = _capture_expert_mask(ep_mask, global_num_experts=8)
    assert torch.equal(got.cpu().to(torch.int32), ep_mask.cpu().to(torch.int32))
    assert got.sum().item() == 4  # the 4 local experts, not all 9


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_routed_experts_expert_map_delegates_to_kernel():
    """RoutedExperts.expert_map returns the 0/1 mask only for kernels that set
    ``consumes_expert_mask`` (AITER), and the canonical -1 map otherwise -- keyed
    on the resolved kernel, not the global aiter switch."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    canonical = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32)

    def resolve(consumes_mask, *, has_moe_kernel=True):
        moe_kernel = (
            SimpleNamespace(
                fused_experts=SimpleNamespace(consumes_expert_mask=consumes_mask)
            )
            if has_moe_kernel
            else None
        )
        layer = SimpleNamespace(
            _expert_map=canonical,
            expert_mask=mask,
            quant_method=SimpleNamespace(moe_kernel=moe_kernel),
        )
        return RoutedExperts.expert_map.fget(layer)

    assert torch.equal(resolve(True), mask)  # AITER kernel -> 0/1 mask
    assert torch.equal(resolve(False), canonical)  # non-AITER -> canonical map
    assert torch.equal(resolve(False, has_moe_kernel=False), canonical)  # non-modular


# --------------------------------------------------------------------------- #
# Block-sparse GQA prefill attend (union path + locality-guard fallback)
# --------------------------------------------------------------------------- #
def _sparse_attn_case(q_lens, prefix_lens, topk, union_target, fp8, seed=0):
    """Paged inputs for the prefill attend, with tunable top-k locality.

    ``union_target`` bounds the pool neighbouring query tokens draw from, so it
    sets how many distinct blocks a group of consecutive tokens selects and hence
    which side of the locality guard the group lands on. It must be at least
    ``topk``: the generic kernel derives its per-token block count from the query
    position rather than the -1 sentinel, so every row has to stay filled to
    ``min(topk, causally available)`` for it to be a valid reference.
    """
    from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

    assert union_target >= topk
    torch.manual_seed(seed)
    num_kv_heads, gqa, head_dim = 1, 16, 128
    batch = len(q_lens)
    seq_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
    pages = [(s + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE for s in seq_lens]
    num_pages = sum(pages)
    # Non-identity page order, so a kernel that confuses logical block ids with
    # physical pages cannot pass.
    physical = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32)
    block_table = torch.zeros(batch, max(pages), device=DEVICE, dtype=torch.int32)
    base = 0
    for r, n in enumerate(pages):
        block_table[r, :n] = physical[base : base + n]
        base += n

    total_q = sum(q_lens)
    q = torch.randn(
        total_q, num_kv_heads * gqa, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    kv = torch.randn(
        num_pages,
        num_kv_heads,
        SPARSE_BLOCK_SIZE,
        2 * head_dim,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    if fp8:
        kv = kv.to(current_platform.fp8_dtype())

    cu_seqlens_q = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.tensor(q_lens, device=DEVICE, dtype=torch.int32).cumsum(0)

    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, device=DEVICE, dtype=torch.int32
    )
    qs = 0
    for q_len, prefix_len in zip(q_lens, prefix_lens):
        for g0 in range(0, q_len, 4):
            # Pool drawn strictly below the group's earliest current block, so
            # adding each token's own current block stays causal for all of them.
            pool_hi = (prefix_len + g0) // SPARSE_BLOCK_SIZE
            pool = torch.randperm(pool_hi, device=DEVICE, dtype=torch.int32)
            pool = pool[: max(union_target - 1, 1)]
            for tok in range(g0, min(g0 + 4, q_len)):
                cur = (prefix_len + tok) // SPARSE_BLOCK_SIZE
                sel = torch.tensor([cur], device=DEVICE, dtype=torch.int32)
                if pool_hi:
                    perm = torch.randperm(pool.numel(), device=DEVICE)
                    sel = torch.cat([sel, pool[perm][: topk - 1]])
                topk_idx[:, qs + tok, : sel.numel()] = sel
        qs += q_len

    return (
        q,
        kv,
        topk_idx,
        block_table,
        cu_seqlens_q,
        torch.tensor(seq_lens, device=DEVICE, dtype=torch.int32),
        torch.tensor(prefix_lens, device=DEVICE, dtype=torch.int32),
        max(q_lens),
        num_kv_heads,
        head_dim**-0.5,
    )


def _run_sparse_attn(fn, args):
    q = args[0]
    # NaN-filled, so any query row the kernel fails to write is caught rather
    # than silently reading as a plausible value.
    out = torch.full_like(q, float("nan"))
    kv = args[1]
    scales = {}
    if kv.dtype not in (torch.bfloat16, torch.float16):
        one = torch.ones((), device=DEVICE, dtype=torch.float32)
        scales = {"k_scale": one, "v_scale": one}
    fn(*args, out, **scales)
    return out


def _union_sizes(topk_idx, cu_seqlens_q, group_q):
    """Distinct blocks per group of `group_q` consecutive tokens, host side."""
    cu = cu_seqlens_q.tolist()
    sizes = []
    for b in range(len(cu) - 1):
        req = topk_idx[0, cu[b] : cu[b + 1]]
        for g in range(0, req.shape[0], group_q):
            grp = req[g : g + group_q]
            sizes.append(int(torch.unique(grp[grp >= 0]).numel()))
    return sizes


# Geometries: one single-request, one ragged batch whose requests have unaligned
# q_start and tails that are not a multiple of the group size. The ragged one is
# what exercises the two kernels agreeing on a union-buffer row.
SPARSE_GEOMETRIES = [
    ((512,), (8192,), 16),
    ((131, 17, 4), (1024, 0, 4096), 16),
]


@requires_gfx950
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("route", ["union", "fallback", "mixed"])
@pytest.mark.parametrize(("q_lens", "prefix_lens", "topk"), SPARSE_GEOMETRIES)
@torch.inference_mode()
def test_sparse_prefill_attend_matches_generic(
    monkeypatch, q_lens, prefix_lens, topk, route, fp8
):
    """The gfx950 grouped attend agrees with the generic per-token kernel.

    The guard threshold is forced rather than inferred from the top-k locality,
    because which side a group lands on otherwise depends on the random pool and
    is easy to lose silently. ``mixed`` additionally asserts that both kernels
    really did write into the same output tensor.
    """
    from vllm.models.minimax_m3.amd.ops import sparse_attn as amd
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as generic_sparse_attn,
    )

    args = _sparse_attn_case(q_lens, prefix_lens, topk, union_target=26, fp8=fp8)
    sizes = _union_sizes(args[2], args[4], amd._SPARSE_ATTN_BLOCK_Q)
    candidates = amd._SPARSE_ATTN_BLOCK_Q * topk
    if route == "union":
        frac = 1.0
    elif route == "fallback":
        frac = 0.0
    else:
        # Halfway through the observed spread, so neither side can be empty.
        frac = ((min(sizes) + max(sizes)) // 2) / candidates
    for name in (
        "_SPARSE_ATTN_UNION_MAX_FRAC_FP8",
        "_SPARSE_ATTN_UNION_MAX_FRAC_DEFAULT",
    ):
        monkeypatch.setattr(amd, name, frac)

    union_max = int(candidates * frac)
    on_union = sum(1 for s in sizes if s <= union_max)
    if route == "union":
        assert on_union == len(sizes)
    elif route == "fallback":
        assert on_union == 0
    else:
        assert 0 < on_union < len(sizes), f"routing not mixed: {on_union}/{len(sizes)}"

    got = _run_sparse_attn(amd.minimax_m3_sparse_attn, args)
    expected = _run_sparse_attn(generic_sparse_attn, args)

    assert not got.isnan().any(), "query rows left unwritten by the attend"
    # 2e-2 covers bf16 rounding: the union path accumulates the softmax
    # normalizer linearly where the per-token path uses a log-space LSE.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    rel = (got.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 5e-3, f"relative Frobenius error {rel:.2e} too large"


@pytest.mark.parametrize("topk", [16, 64])
@torch.inference_mode()
def test_sparse_prefill_attend_group_size_one(monkeypatch, topk):
    """Group size 1 is the off-switch and the non-gfx950 configuration.

    Not gfx950-gated: every other ROCm arch runs this path. It must attend every
    query whatever topk is, since the guard threshold scales with the candidate
    count and no group may be dropped by both kernels.
    """
    from vllm.models.minimax_m3.amd.ops import sparse_attn as amd
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as generic_sparse_attn,
    )

    monkeypatch.setattr(amd, "_SPARSE_ATTN_BLOCK_Q", 1)
    args = _sparse_attn_case((256,), (16384,), topk, union_target=topk, fp8=False)
    got = _run_sparse_attn(amd.minimax_m3_sparse_attn, args)
    expected = _run_sparse_attn(generic_sparse_attn, args)

    assert not got.isnan().any(), "query rows left unwritten by the attend"
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
