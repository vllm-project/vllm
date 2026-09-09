# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit-test DeepGEMM FP8 and FP4 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import importlib
import math

import pytest
import torch

# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    build_psum_group_end,
    deepgemm_moe_permute,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    lock_workspace,
    unlock_workspace,
)

BLOCK_SIZE = [128, 128]


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_moe_permute_initializes_padding_scales(workspace_init):
    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    activations, scales = per_token_group_quant_fp8(
        hidden_states,
        group_size=128,
        use_ue8m0=True,
    )
    topk_ids = torch.tensor([[0], [1]], device="cuda", dtype=torch.int64)

    _, permuted_scales, expert_ids, _, _, _ = deepgemm_moe_permute(
        aq=activations,
        aq_scale=scales,
        topk_ids=topk_ids,
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    padding = expert_ids < 0
    assert padding.any()
    torch.testing.assert_close(
        permuted_scales[padding],
        torch.zeros_like(permuted_scales[padding]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("align_m", [64, 128])
def test_build_psum_group_end(align_m):
    """group_end[i] must equal aligned_start[i] + count[i].

    aligned_start is the exclusive prefix sum of the per-expert counts rounded
    up to align_m — the same cumsum ep_scatter uses for expert_start_loc. This
    is what DeepGEMM's psum scheduler reads to skip padding blocks, so a wrong
    offset silently corrupts every downstream expert. Empty experts must
    collapse to a zero-length group (group_end == aligned_start).
    """
    counts = torch.tensor([130, 0, 5, 256], dtype=torch.int64)

    group_end = build_psum_group_end(counts, align_m)

    aligned = ((counts.to(torch.int32) + align_m - 1) // align_m) * align_m
    # torch.cumsum promotes int32 -> int64; build_psum_group_end casts back.
    aligned_start = torch.cumsum(aligned, dim=0) - aligned
    expected = (aligned_start + counts).to(torch.int32)

    assert group_end.dtype == torch.int32
    assert group_end.is_contiguous()
    torch.testing.assert_close(group_end, expected, rtol=0, atol=0)
    # Empty expert (index 1) contributes no blocks: its end sits at its start.
    assert group_end[1].item() == aligned_start[1].item()
    # Each group's real end never exceeds the next group's aligned start.
    next_start = torch.cat([aligned_start[1:], aligned_start[-1:] + aligned[-1:]])
    assert torch.all(group_end <= next_start)


def test_deepgemm_fp4_use_psum_layout_gate():
    """psum layout is selected iff DeepEP v2 supplies psum_recv_per_rank.

    That field is populated only by DeepEP v2's cudagraph/decode dispatch (the
    mode with large worst-case padding), so the gate turns psum on exactly
    there and leaves the prefill / non-EP paths on the per-row m_indices layout.
    """
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    gate = DeepGemmFP4Experts._use_psum_layout

    empty_meta = mk.ExpertTokensMetadata(
        expert_num_tokens=None, expert_num_tokens_cpu=None
    )

    assert gate(None) is False
    assert gate(empty_meta) is False
    assert (
        gate(
            mk.ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=None,
                psum_recv_per_rank=torch.tensor([4], dtype=torch.int32),
            )
        )
        is True
    )


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
):
    """
    Generate (w1, w2) expert weights and their per-block scale tensors
    in FP8 block-quantized format.

      w1 shape: (E, 2N, K)
      w2 shape: (E, K, N)
    """
    dtype = torch.bfloat16
    fp8_max, fp8_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )

    # bf16 reference weights
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    w1_bf16.clamp_(fp8_min, fp8_max)
    w2_bf16.clamp_(fp8_min, fp8_max)

    block_n, block_k = block_size
    n_tiles_w1 = math.ceil((2 * n) / block_n)
    k_tiles_w1 = math.ceil(k / block_k)
    n_tiles_w2 = math.ceil(k / block_n)
    k_tiles_w2 = math.ceil(n / block_k)

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(e, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32)
    w2_s = torch.empty(e, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(
            w1_bf16[i], block_size=block_size, use_ue8m0=True
        )
        w2[i], w2_s[i] = per_block_cast_to_fp8(
            w2_bf16[i], block_size=block_size, use_ue8m0=True
        )

    return w1, w2, w1_s, w2_s


def run_single_case(m, n, k, topk, num_experts, block_size):
    """
    Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k, block_size)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
    )
    moe_config = make_dummy_moe_config()

    deep_gemm_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=TritonOrDeepGemmExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
    )

    # DeepGemm
    out_deepgemm = deep_gemm_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: N <= 512 will disable the deepgemm path due to performance issues.
MNKs = [
    (1024, 768, 128),
    (2048, 768, 512),
    (512, 1024, 1024),
    (4096, 4096, 1024),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_vs_triton(m, n, k, topk, num_experts, monkeypatch, workspace_init):
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmExperts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmExperts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmExperts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmExperts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
            block_size=BLOCK_SIZE,
        )

        # ensure that the DeepGEMM path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


# ---------------------------------------------------------------------------
# FP4 weight tests (DeepGEMM m_grouped_fp8_fp4_gemm_nt_contiguous)
# ---------------------------------------------------------------------------


def make_mxfp4_weights(
    e: int,
    n: int,
    k: int,
):
    """
    Generate (w1, w2) expert weights in MXFP4 packed format with float32 scales,
    plus BF16 reference weights for validation.

      w1 shape: (E, 2N, K//2) uint8    — packed FP4
      w2 shape: (E, K, N//2)  uint8    — packed FP4
      w1_s shape: (E, 2N, K//32) float32  — per-row block-32 scales
      w2_s shape: (E, K, N//32)  float32  — per-row block-32 scales
      w1_bf16: (E, 2N, K)   — original BF16 for reference
      w2_bf16: (E, K, N)    — original BF16 for reference
    """
    from deep_gemm.utils.math import per_token_cast_to_fp4

    dtype = torch.bfloat16
    gran_k = 32  # MXFP4 block size

    # bf16 reference weights — scale by 1/sqrt(dim) for numerical stability
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) * (k**-0.5)
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) * (n**-0.5)

    # Quantize per-expert to FP4
    w1 = torch.empty(e, 2 * n, k // 2, device="cuda", dtype=torch.uint8)
    w2 = torch.empty(e, k, n // 2, device="cuda", dtype=torch.uint8)
    w1_s = torch.empty(
        e, 2 * n, math.ceil(k / gran_k), device="cuda", dtype=torch.float32
    )
    w2_s = torch.empty(e, k, math.ceil(n / gran_k), device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_token_cast_to_fp4(
            w1_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )
        w2[i], w2_s[i] = per_token_cast_to_fp4(
            w2_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )

    return w1, w2, w1_s, w2_s, w1_bf16, w2_bf16


def _bf16_moe_reference(x, w1, w2, topk_weights, topk_ids):
    """BF16 token-loop MoE reference for correctness testing."""
    import torch.nn.functional as F

    num_tokens, hidden_size = x.shape
    intermediate = w1.shape[1] // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)
    for t in range(num_tokens):
        for kk in range(top_k):
            e = topk_ids[t, kk].item()
            w = topk_weights[t, kk].item()
            fc1 = x[t : t + 1].float() @ w1[e].float().T
            linear = fc1[:, :intermediate]
            gate = fc1[:, intermediate:]
            act = F.silu(gate) * linear
            fc2 = act @ w2[e].float().T
            output[t] += w * fc2[0]
    return output.to(torch.bfloat16)


def run_single_fp4_case(m, n, k, topk, num_experts):
    """
    Run one (M,N,K) configuration with FP4 weights on DeepGEMM and assert
    DeepGEMM FP4 == BF16 reference within tolerance.
    """
    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)

    # FP4 expert weight tensors + BF16 originals for reference
    w1, w2, w1_s, w2_s, w1_bf16, w2_bf16 = make_mxfp4_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
    )
    from vllm.platforms import current_platform

    _fp8_dtype = current_platform.fp8_dtype()
    _block_shape = GroupShape(128, 128)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _a2=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    moe_config = make_dummy_moe_config()

    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    deep_gemm_fp4_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmFP4Experts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # DeepGEMM FP4 path
    out_deepgemm_fp4 = deep_gemm_fp4_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    # BF16 reference using the same original weights
    out_ref = _bf16_moe_reference(tokens_bf16, w1_bf16, w2_bf16, topk_weights, topk_ids)

    # FP4 vs BF16 reference: quantization error from FP4 weights + FP8 activations
    diff = calc_diff(out_deepgemm_fp4, out_ref)
    assert diff < 0.05, f"FP4 diff exceeded 5%: {diff}"


# DeepSeek V4 dims: H=4096, I=2048, so N=2*I=4096, K=H=4096.
# FP4 quantization with block_k=32 needs large K for good accuracy.
FP4_MNKs = [
    (128, 4096, 4096),  # DeepSeek V4 shape
    (256, 2048, 2048),  # Half-size variant
    (128, 384, 3584),  # Kimi-K3 TP8 latent-MoE shape
]

FP4_TOPKS = [2]
FP4_NUM_EXPERTS = [8]


@pytest.mark.parametrize(("m", "n", "k"), FP4_MNKs)
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_fp4_vs_triton(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmFP4Experts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmFP4Experts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmFP4Experts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmFP4Experts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_fp4_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
        )

        # ensure that the DeepGEMM FP4 path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM FP4 path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


# ---------------------------------------------------------------------------
# FP4 psum-layout tests (DeepEP v2 decode/cudagraph path)
# ---------------------------------------------------------------------------
#
# The psum layout is what lets the SM100 grouped GEMM *skip* the worst-case
# row padding DeepEP v2 emits in its cudagraph/decode dispatch, instead of
# computing over it (non-psum m_indices path). It is selected at runtime by
# DeepGemmFP4Experts whenever expert_tokens_meta carries psum_recv_per_rank.
# The standalone (non-EP) harness never sets that field, so these tests drive
# DeepGemmFP4Experts.apply directly with a crafted decode-style metadata.


def _psum_supported() -> bool:
    from vllm.platforms import current_platform

    # psum block-skipping is the SM100 scheduler behaviour; that is also the
    # only SM the nvfp4 grouped GEMM runs on in practice.
    return is_deep_gemm_supported() and current_platform.is_device_capability_family(
        100
    )


def _make_decode_meta(m: int, device: torch.device) -> mk.ExpertTokensMetadata:
    """A DeepEP v2 cudagraph/decode carrier.

    Only psum_recv_per_rank is populated (exact per-expert counts are not
    synced in that mode), so M_sum falls back to the worst-case bound and
    DeepGemmFP4Experts._use_psum_layout() turns on. Its value is never read on
    the FP4 path — only its presence gates psum — so any device int32 works.
    """
    return mk.ExpertTokensMetadata(
        expert_num_tokens=None,
        expert_num_tokens_cpu=None,
        psum_recv_per_rank=torch.tensor([m], dtype=torch.int32, device=device),
    )


def _build_fp4_kernel(m, n, k, topk, num_experts):
    """Build a DeepGemmFP4Experts kernel plus quantized-ready inputs."""
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)
    w1, w2, w1_s, w2_s, w1_bf16, w2_bf16 = make_mxfp4_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    _fp8_dtype = current_platform.fp8_dtype()
    _block_shape = GroupShape(128, 128)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _a2=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    moe_config = make_dummy_moe_config()
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmFP4Experts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )
    return kernel, tokens_bf16, topk_weights, topk_ids, (w1, w2), (w1_bf16, w2_bf16)


def _fp4_apply(kernel, tokens_bf16, topk_weights, topk_ids, w1, w2, num_experts, meta):
    """Run prepare -> DeepGemmFP4Experts.apply with an explicit metadata.

    Returns the final (M, K) output. The metadata drives the psum gate: a
    decode-style meta forces the psum layout, real prepare metadata (no
    psum_recv_per_rank) leaves it on the non-psum m_indices path.
    """
    impl = kernel.impl
    experts = kernel.fused_experts

    a1q, a1q_scale, _real_meta, tk_ids, tk_w = impl._prepare(
        hidden_states=tokens_bf16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    if meta == "real":
        meta = _real_meta

    _, m_full, n_dim, k_dim, top_k = experts.moe_problem_size(a1q, w1, w2, tk_ids)
    ws13, ws2, out = impl._allocate_buffers(
        tokens_bf16.dtype,
        a1q.device,
        m_full,
        m_full,
        n_dim,
        k_dim,
        top_k,
        num_experts,
        num_experts,
        meta,
        MoEActivation.SILU,
    )
    experts.apply(
        output=out,
        hidden_states=a1q,
        w1=w1,
        w2=w2,
        topk_weights=tk_w,
        topk_ids=tk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=experts.a2_scale,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=meta,
        apply_router_weight_on_input=False,
    )
    return out.clone()


@pytest.mark.parametrize(("m", "n", "k"), FP4_MNKs)
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not _psum_supported(), reason="Requires SM100 deep_gemm kernels")
def test_deepgemm_fp4_psum_layout_numeric(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    """psum layout must match both the BF16 reference and the non-psum path.

    Skipping padding blocks (psum) versus computing and discarding them
    (non-psum) is an efficiency difference only; the reduced (M, K) output
    must be numerically identical up to fp8/fp4 accumulation noise.
    """
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        kernel, tokens, tw, ti, (w1, w2), (w1_bf16, w2_bf16) = _build_fp4_kernel(
            m, n, k, topk, num_experts
        )
        experts = kernel.fused_experts

        decode_meta = _make_decode_meta(m, tokens.device)
        assert experts._use_psum_layout(decode_meta) is True

        out_psum = _fp4_apply(kernel, tokens, tw, ti, w1, w2, num_experts, decode_meta)
        out_nonpsum = _fp4_apply(kernel, tokens, tw, ti, w1, w2, num_experts, "real")
        out_ref = _bf16_moe_reference(tokens, w1_bf16, w2_bf16, tw, ti)

        diff_ref = calc_diff(out_psum, out_ref)
        diff_layout = calc_diff(out_psum, out_nonpsum)
        assert diff_ref < 0.05, f"psum vs BF16 ref diff too high: {diff_ref}"
        assert diff_layout < 0.02, f"psum vs non-psum diff too high: {diff_layout}"


def _force_psum_layout(mp, kernel, m):
    """Make prepare() hand back a DeepEP v2 decode-style carrier.

    psum_recv_per_rank is the sole gate for DeepGemmFP4Experts._use_psum_layout
    and the standalone (non-EP) prepare/finalize never sets it, so it has to be
    injected. Patching the prepare result is the smallest seam that turns the
    psum path on while leaving the production apply path -- shape derivation,
    buffer allocation, finalize -- running exactly as it does in deployment.
    """
    pf = kernel.prepare_finalize
    assert not pf.supports_async(), "async prepare would bypass this patch"
    orig_prepare = pf.prepare

    def prepare_with_psum(*args, **kwargs):
        a1q, a1q_scale, _real_meta, tk_ids, tk_w = orig_prepare(*args, **kwargs)
        return a1q, a1q_scale, _make_decode_meta(m, a1q.device), tk_ids, tk_w

    mp.setattr(pf, "prepare", prepare_with_psum)


def _fill_workspace_arena(byte: int) -> None:
    """Byte-fill every allocated workspace buffer.

    The modular kernel carves its workspaces out of this arena and does not
    zero them, so whatever is left here is exactly what a padding row the
    scatter never writes will contain at kernel time. 0xFF is a NaN bit pattern
    in every float dtype the workspaces can take (bf16/fp16/fp32/fp8-e4m3), so
    it poisons without the test having to know the workspace dtype.

    Reaches into the manager's buffer list because poisoning uninitialized
    memory is inherently white-box; there is no public accessor.
    """
    for ws in current_workspace_manager()._current_workspaces:
        if ws is not None:
            ws.fill_(byte)


@pytest.mark.parametrize(("m", "n", "k"), [(128, 4096, 4096)])
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not _psum_supported(), reason="Requires SM100 deep_gemm kernels")
def test_deepgemm_fp4_psum_layout_padding_robust(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    """psum output must be unaffected by garbage in the padding rows.

    In decode mode the permuted activation buffer is sized to the worst-case
    M_sum and only the real-token prefix of each expert slot is written by the
    scatter; the alignment gaps and worst-case tail are left uninitialized.
    Poisoning the workspace arena with NaN reproduces that condition
    deterministically, and the reduced output must stay finite and match the
    zero-padding run bit-for-bit.

    Note this proves robustness/correctness under garbage padding, not physical
    block-skipping: MoE rows are independent and the unpermute reads only valid
    rows, so skipping and compute-then-discard yield identical outputs
    (skipping is a compute-savings property, covered by source review + a
    kernel microbench).
    """
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        kernel, tokens, tw, ti, (w1, w2), _ = _build_fp4_kernel(
            m, n, k, topk, num_experts
        )
        _force_psum_layout(mp, kernel, m)

        def run():
            return kernel.apply(
                hidden_states=tokens,
                w1=w1,
                w2=w2,
                topk_weights=tw,
                topk_ids=ti,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            ).clone()

        run()  # size the arena so the fills below are not dropped by a resize
        lock_workspace()  # a resize now raises rather than silently un-poisoning
        try:
            _fill_workspace_arena(0x00)
            out_clean = run()
            _fill_workspace_arena(0xFF)
            out_poison = run()
        finally:
            unlock_workspace()

        assert torch.isfinite(out_poison).all(), "padding garbage leaked into output"
        torch.testing.assert_close(out_poison, out_clean, rtol=0, atol=0)


@pytest.mark.skipif(not _psum_supported(), reason="Requires SM100 deep_gemm kernels")
def test_deepgemm_fp4_psum_layout_cudagraph(monkeypatch, workspace_init):
    """psum layout is CUDA-graph safe with varying on-device group boundaries.

    expected_m_for_psum_layout=M_sum fixes the launch grid at capture time
    (M_sum is a host-side worst-case bound in decode mode), while the per-group
    prefix-sum boundaries are read on device. Replaying the same graph after
    changing the routing in place must recompute correct results for the new
    boundaries without re-capturing.
    """
    pytest.importorskip("deep_gemm.utils.math")
    m, n, k, topk, num_experts = 128, 4096, 4096, 2, 8

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        kernel, tokens, tw, ti, (w1, w2), (w1_bf16, w2_bf16) = _build_fp4_kernel(
            m, n, k, topk, num_experts
        )
        impl = kernel.impl
        experts = kernel.fused_experts

        # Quantize once; capture with fixed tensor handles so replays after an
        # in-place routing update re-read the new topk on device.
        a1q, a1q_scale, _meta, tk_ids, tk_w = impl._prepare(
            hidden_states=tokens,
            topk_weights=tw,
            topk_ids=ti,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        meta = _make_decode_meta(m, tokens.device)
        assert experts._use_psum_layout(meta) is True

        _, m_full, n_dim, k_dim, top_k = experts.moe_problem_size(a1q, w1, w2, tk_ids)
        ws13, ws2, out = impl._allocate_buffers(
            tokens.dtype,
            a1q.device,
            m_full,
            m_full,
            n_dim,
            k_dim,
            top_k,
            num_experts,
            num_experts,
            meta,
            MoEActivation.SILU,
        )

        def run():
            experts.apply(
                output=out,
                hidden_states=a1q,
                w1=w1,
                w2=w2,
                topk_weights=tk_w,
                topk_ids=tk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                a1q_scale=a1q_scale,
                a2_scale=experts.a2_scale,
                workspace13=ws13,
                workspace2=ws2,
                expert_tokens_meta=meta,
                apply_router_weight_on_input=False,
            )

        # Warm up (JIT + allocator) on a side stream before capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()

        # Replay with the captured routing.
        g.replay()
        torch.accelerator.synchronize()
        out1 = out.clone()
        ref1 = _bf16_moe_reference(tokens, w1_bf16, w2_bf16, tw, ti)
        assert calc_diff(out1, ref1) < 0.05

        # Change routing in place -> different per-group boundaries on device.
        new_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
        new_tw, new_ti = torch.topk(new_logits, k=topk, dim=-1)
        new_tw = torch.nn.functional.softmax(new_tw, dim=-1)
        tk_ids.copy_(new_ti.to(tk_ids.dtype))
        tk_w.copy_(new_tw.to(tk_w.dtype))

        g.replay()
        torch.accelerator.synchronize()
        out2 = out.clone()
        ref2 = _bf16_moe_reference(tokens, w1_bf16, w2_bf16, new_tw, new_ti)
        assert calc_diff(out2, ref2) < 0.05

        # The routing change must actually move the result (boundaries varied).
        assert calc_diff(out1, out2) > 1e-3
