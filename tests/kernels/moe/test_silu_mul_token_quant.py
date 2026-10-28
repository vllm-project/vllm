# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused SiLU-and-mul + per-token FP8 quant used by `TritonExperts` must match
`silu_and_mul` followed by `scaled_fp8_quant(use_per_token_if_dynamic=True)` on
every row routed to a local expert."""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.triton_moe as triton_moe
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.silu_mul_token_quant import (
    silu_mul_token_quant,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() and current_platform.supports_fp8()),
    reason="needs a GPU with FP8 support",
)

FP8 = torch.finfo(current_platform.fp8_dtype())


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    act = torch.empty((x.size(0), x.size(1) // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act, x)
    return ops.scaled_fp8_quant(act, None, use_per_token_if_dynamic=True)


def _assert_matches(q, s, ref_q, ref_s):
    torch.testing.assert_close(s, ref_s, rtol=1e-6, atol=0)
    match = (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean()
    assert match > 0.999, f"only {match:.4%} of FP8 values match"
    # Any mismatch must be a single FP8 rounding step: `eps` relative for normals,
    # one subnormal step (`smallest_normal * eps`) near zero.
    torch.testing.assert_close(
        q.float(),
        ref_q.float(),
        rtol=FP8.eps,
        atol=FP8.smallest_normal * FP8.eps,
    )


@pytest.mark.parametrize("tokens", [1, 7, 64, 1000])
@pytest.mark.parametrize("hidden", [256, 512, 2048])
@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_mul_token_quant(tokens, hidden, ep_size, dtype):
    set_random_seed(0)
    top_k, num_experts = 8, 64
    rows = tokens * top_k
    x = torch.randn(rows, 2 * hidden, device="cuda", dtype=dtype) * 3
    x[: rows // 4] *= 1e-3  # small rows exercise the minimum scale
    topk_ids = torch.randint(
        0, num_experts, (tokens, top_k), device="cuda", dtype=torch.int32
    )
    expert_map = None
    if ep_size > 1:
        local = num_experts // ep_size
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[local : 2 * local] = torch.arange(
            local, dtype=torch.int32, device="cuda"
        )

    ref_q, ref_s = _reference(x)
    q, s = silu_mul_token_quant(
        x, topk_ids if expert_map is not None else None, expert_map
    )
    assert q.shape == ref_q.shape and q.dtype == ref_q.dtype
    assert s.shape == ref_s.shape and s.dtype == torch.float32

    keep = torch.ones(rows, dtype=torch.bool, device="cuda")
    if expert_map is not None:
        keep = expert_map[topk_ids.reshape(-1).long()] >= 0
    _assert_matches(q[keep], s[keep], ref_q[keep], ref_s[keep])


def test_silu_mul_token_quant_skips_out_of_range_ids():
    """Ids outside [0, num_experts) are dropped like `moe_align_block_size` does,
    without reading `expert_map` past `num_experts` (for example fused shared-expert
    slots that the alignment does not count)."""
    set_random_seed(0)
    num_experts, hidden = 8, 256
    # Every slot past `num_experts` maps to a local expert, so reading it would make
    # the kernel process the row.
    expert_map = torch.zeros(num_experts + 4, dtype=torch.int32, device="cuda")
    expert_map[1::2] = -1
    ids = [0, 1, 2, -1, num_experts, num_experts + 2, 7, 6]
    topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    x = torch.randn(len(ids), 2 * hidden, device="cuda", dtype=torch.bfloat16)
    q = torch.full(
        (len(ids), hidden), 1.0, device="cuda", dtype=current_platform.fp8_dtype()
    )
    s = torch.full((len(ids), 1), -1.0, device="cuda", dtype=torch.float32)

    silu_mul_token_quant(x, topk_ids, expert_map, num_experts, out=(q, s))

    written = torch.tensor(
        [0 <= i < num_experts and i % 2 == 0 for i in ids], device="cuda"
    )
    assert (s[~written] == -1.0).all() and (q[~written].float() == 1.0).all()
    ref_q, ref_s = _reference(x)
    _assert_matches(q[written], s[written], ref_q[written], ref_s[written])


@pytest.mark.parametrize("use_ep", [False, True])
@torch.inference_mode()
def test_triton_experts_takes_fused_path(monkeypatch, use_ep):
    """`TritonExperts.apply` calls the fused kernel for PTPC-FP8 SiLU on ROCm only,
    with the alignment's expert count under EP, and the output matches the unfused
    `silu_and_mul` + `scaled_fp8_quant` path."""
    set_random_seed(0)
    m, k, n, topk = 33, 256, 512, 4
    global_experts = 16
    local_experts = global_experts // 2 if use_ep else global_experts
    fp8 = current_platform.fp8_dtype()

    w1 = (torch.randn(local_experts, n, k, device="cuda") / k**0.5).to(fp8)
    w2 = (torch.randn(local_experts, k, n // 2, device="cuda") / n**0.5).to(fp8)
    w1_scale = torch.rand(local_experts, n, 1, device="cuda") * 0.1 + 0.95
    w2_scale = torch.rand(local_experts, k, 1, device="cuda") * 0.1 + 0.95
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a1q, a1q_scale = ops.scaled_fp8_quant(a, None, use_per_token_if_dynamic=True)
    topk_weights = torch.rand(m, topk, device="cuda").softmax(-1)
    topk_ids = torch.stack(
        [torch.randperm(global_experts, device="cuda")[:topk] for _ in range(m)]
    ).to(torch.int32)
    expert_map = None
    if use_ep:
        expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[local_experts:] = torch.arange(
            local_experts, dtype=torch.int32, device="cuda"
        )

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(
            num_experts=global_experts,
            num_local_experts=local_experts,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n // 2,
        ),
        quant_config=fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            per_act_token_quant=True,
            per_out_ch_quant=True,
        ),
    )

    def run() -> torch.Tensor:
        ws13, ws2, out_shape = experts.workspace_shapes(
            m, n, k, topk, global_experts, local_experts, None, MoEActivation.SILU
        )
        output = torch.zeros(out_shape, device="cuda", dtype=torch.bfloat16)
        experts.apply(
            output=output,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=global_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=None,
            workspace13=torch.empty(ws13, device="cuda", dtype=torch.bfloat16),
            workspace2=torch.empty(ws2, device="cuda", dtype=torch.bfloat16),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output

    calls = []

    def fused(x, ids, emap, num_experts):
        calls.append((ids, emap, num_experts))
        return silu_mul_token_quant(x, ids, emap, num_experts)

    def unfused(x, ids, emap, num_experts):
        calls.append(None)
        return _reference(x)

    monkeypatch.setattr(triton_moe, "silu_mul_token_quant", fused)
    out_fused = run()
    if not current_platform.is_rocm():
        assert calls == []
        return
    assert len(calls) == 1
    ids, emap, num_experts = calls[0]
    if use_ep:
        assert ids is topk_ids and emap is expert_map
        assert num_experts == global_experts
    else:
        assert ids is None and emap is None

    monkeypatch.setattr(triton_moe, "silu_mul_token_quant", unfused)
    out_unfused = run()
    assert len(calls) == 2
    torch.testing.assert_close(out_fused, out_unfused, rtol=0, atol=0)
