# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the sonic-moe fused-experts backend.

Two tests:
1. ``test_metadata_is_permutation`` — verifies invariants of the copied
   ``TC_topk_router_metadata`` Triton kernel (frequency, offsets, and the
   scatter/reverse-scatter permutations) without touching the GEMM path.
2. ``test_apply_matches_reference`` — runs the full ``SonicMoEExperts.apply``
   pipeline (metadata + grouped ``gemm_gated`` + grouped ``gemm`` + inline
   weighted reduce) and compares against the ``iterative_moe`` reference.

Skip-if guards: Hopper sm_90 or Blackwell sm_100/sm_103.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.platforms import current_platform


def _backend_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    if not (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(100)
        )
    ):
        return False, "sonic-moe requires Hopper sm_90 or Blackwell sm_100/sm_103"
    try:
        from vllm.model_executor.layers.fused_moe.experts.sonic_moe import (  # noqa: F401
            SonicMoEExperts,
        )
    except Exception as e:  # pragma: no cover
        return False, f"import failed: {e}"
    if not SonicMoEExperts._supports_current_device():
        return False, "SonicMoEExperts._supports_current_device() returned False"
    return True, ""


_ok, _skip_reason = _backend_available()
pytestmark = pytest.mark.skipif(not _ok, reason=_skip_reason)


_SEED = 0xC0FFEE


def _iterative_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Slow, obviously-correct reference for SwiGLU MoE.

    Mirrors ``iterative_moe`` in tests/kernels/moe/test_moe.py but takes
    pre-computed topk_weights/topk_ids.

    Layouts match vLLM convention:
        w1: (E_local, 2*I, H)  with rows packed [gate_block | up_block]
        w2: (E_local, H, I)
    """
    num_local_experts = w1.shape[0]
    intermediate = w2.shape[-1]

    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    for e_local in range(num_local_experts):
        mask = topk_ids == e_local  # (T, K_topk)
        gate_w = (topk_weights * mask).sum(dim=-1, keepdim=True).to(hidden_states.dtype)
        x = F.linear(hidden_states, w1[e_local])  # (T, 2*I)
        gate = F.silu(x[:, :intermediate])
        x = x[:, intermediate:] * gate  # (T, I)
        x = F.linear(x, w2[e_local])  # (T, H)
        out = out + (x * gate_w).to(torch.float32)
    return out.to(hidden_states.dtype)


def _build_experts(num_experts: int, top_k: int, hidden: int, intermediate: int):
    """Construct a SonicMoEExperts instance bound to a minimal MoE config."""
    from vllm.model_executor.layers.fused_moe.experts.sonic_moe import (
        SonicMoEExperts,
    )

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size_per_partition=intermediate,
        in_dtype=torch.bfloat16,
    )
    # Ensure the dummy config advertises a single-shard layout.
    assert moe_config.moe_parallel_config == FusedMoEParallelConfig.make_no_parallel()
    quant_config = FusedMoEQuantConfig.make(None)
    return SonicMoEExperts(moe_config=moe_config, quant_config=quant_config)


def _run_sonic_moe(
    experts,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Drive SonicMoEExperts.apply() directly with manual workspace allocation.

    ``apply()`` writes the already-reduced (M, K) result into ``output``;
    ``finalize_weight_and_reduce_impl`` is a NoOP, so the framework's finalize
    step is just a copy. We skip the framework here — this is unit-level.

    Inputs use the vLLM weight layout (w1: (E, 2I, H), w2: (E, H, I)); we
    apply the same interleave + permute that
    ``convert_to_unquantized_kernel_format`` performs for SONIC_MOE.
    """
    E_, two_I, H_in = w1.shape
    I_ = two_I // 2
    gate, up = w1[:, :I_, :], w1[:, I_:, :]
    w1 = (
        torch.stack([gate, up], dim=2)  # (E, I, 2, H)
        .reshape(E_, 2 * I_, H_in)  # (E, 2I, H) interleaved
        .permute(0, 2, 1)  # (E, H, 2I)
        .contiguous()
    )
    w2 = w2.permute(0, 2, 1).contiguous()  # (E, I, H)
    M, H = hidden_states.shape
    top_k = topk_ids.size(1)
    E_local, _, two_I = w1.shape
    N = two_I  # un-activated w1 dim (gated activations halve to I in apply)

    workspace1_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        M,
        N,
        H,
        top_k,
        E_local,
        E_local,
        None,
        MoEActivation.SILU,
    )

    device = hidden_states.device
    dtype = hidden_states.dtype
    workspace13 = torch.empty(workspace1_shape, dtype=dtype, device=device)
    workspace2 = torch.empty(workspace2_shape, dtype=dtype, device=device)
    output = torch.empty(output_shape, dtype=dtype, device=device)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E_local,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    return output


# ─────────────────────────────────────────────────────────────────────────────
# 1) routing-metadata kernel sanity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("T,K,E", [(64, 2, 8), (256, 4, 16), (128, 2, 8)])
def test_metadata_is_permutation(T: int, K: int, E: int) -> None:
    """Routing metadata kernel must produce inverse permutations + valid offsets."""
    from vllm.model_executor.layers.fused_moe.experts.sonic_moe.sonic_moe_experts import (  # noqa: E501
        TC_topk_router_metadata_triton,
    )

    torch.manual_seed(_SEED)
    device = torch.device("cuda")

    # Random topk_ids with values in [0, E).
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=device)

    TK = T * K
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_ids,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    # frequency totals
    assert int(expert_frequency.sum().item()) == TK
    # offsets: monotonic, starts at 0, ends at TK
    assert int(expert_frequency_offset[0].item()) == 0
    assert int(expert_frequency_offset[-1].item()) == TK
    diffs = expert_frequency_offset.diff()
    torch.testing.assert_close(diffs, expert_frequency)

    # scatter / reverse-scatter must be inverse permutations on [0, TK)
    arange = torch.arange(TK, dtype=torch.int32, device=device)
    composed = s_reverse_scatter_idx[s_scatter_idx.long()]
    torch.testing.assert_close(composed, arange)
    composed2 = s_scatter_idx[s_reverse_scatter_idx.long()]
    torch.testing.assert_close(composed2, arange)

    # Each grouped slot must correspond to a (token, k) entry whose expert matches
    # the slot's expert bucket implied by the offsets.
    cpu_offs = expert_frequency_offset.cpu().tolist()
    flat_topk = topk_ids.view(-1)
    for e in range(E):
        start, end = cpu_offs[e], cpu_offs[e + 1]
        if start == end:
            continue
        entry_idxs = s_scatter_idx[start:end].long()
        assert torch.all(flat_topk[entry_idxs] == e), (
            f"expert {e} grouped slots [{start}, {end}) reference non-matching "
            f"entries in topk_ids"
        )

    # x_gather_idx must equal s_scatter_idx // K (token index of the entry).
    torch.testing.assert_close(x_gather_idx, s_scatter_idx // K)


# ─────────────────────────────────────────────────────────────────────────────
# 2) full apply() vs iterative reference
# ─────────────────────────────────────────────────────────────────────────────


_SHAPES = [
    # (T, H, I, E_local, K)
    (64, 512, 1024, 8, 2),
    (128, 1024, 1408, 16, 4),
]


def _random_problem(T, H, I, E_local, K, *, dtype=torch.bfloat16, seed=_SEED):  # noqa: E741
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden_states = 0.02 * torch.randn(T, H, dtype=dtype, device=device)
    w1 = 0.02 * torch.randn(E_local, 2 * I, H, dtype=dtype, device=device)
    w2 = 0.02 * torch.randn(E_local, H, I, dtype=dtype, device=device)
    score = torch.randn(T, E_local, dtype=torch.float32, device=device)
    return hidden_states, w1, w2, score


def _topk_route(score: torch.Tensor, K: int, dtype: torch.dtype):
    topk_w, topk_ids = score.softmax(dim=-1).topk(K, dim=-1)
    return topk_w.to(dtype).contiguous(), topk_ids.to(torch.int32).contiguous()


@pytest.mark.parametrize("T,H,I,E_local,K", _SHAPES)
def test_apply_matches_reference(T, H, I, E_local, K) -> None:  # noqa: E741
    """Baseline: weights applied during finalize."""
    hidden_states, w1, w2, score = _random_problem(T, H, I, E_local, K)
    topk_w, topk_ids = _topk_route(score, K, hidden_states.dtype)

    experts = _build_experts(E_local, K, H, I)
    out = _run_sonic_moe(
        experts,
        hidden_states,
        w1,
        w2,
        topk_w,
        topk_ids,
        apply_router_weight_on_input=False,
    )
    ref = _iterative_moe_reference(hidden_states, w1, w2, topk_w, topk_ids)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_apply_router_weight_on_input() -> None:
    """topk=1 path: router weights are pre-applied to the input."""
    T, H, I, E_local, K = 64, 512, 1024, 8, 1  # noqa: E741
    hidden_states, w1, w2, score = _random_problem(T, H, I, E_local, K)
    topk_w, topk_ids = _topk_route(score, K, hidden_states.dtype)

    # Apply router weights upstream, matching MoEPrepareAndFinalizeNoDPEPModular.
    pre_weighted = hidden_states * topk_w.to(hidden_states.dtype)

    experts = _build_experts(E_local, K, H, I)
    out = _run_sonic_moe(
        experts,
        pre_weighted,
        w1,
        w2,
        topk_w,
        topk_ids,
        apply_router_weight_on_input=True,
    )
    # Reference: weight is applied on input, then summed (no second weighting).
    ones = torch.ones_like(topk_w)
    ref = _iterative_moe_reference(pre_weighted, w1, w2, ones, topk_ids)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
