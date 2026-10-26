# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Bailing MoE blocks pass FusedMoE the router logits in ``router_dtype``.

Ling-2.0 / Ring-2.5 configs set ``router_dtype: "fp32"``, and the reference
router runs the gate GEMM and the sigmoid in fp32. The blocks are built around
stub gate and experts modules, so the tests need no checkpoint or GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models import bailing_moe, bailing_moe_linear


class _Gate(torch.nn.Module):
    """GateLinear interface: returns ``(logits, None)`` in ``out_dtype``."""

    def __init__(
        self, input_size, output_size, out_dtype=None, params_dtype=None, prefix=""
    ):
        super().__init__()
        self.out_dtype = out_dtype or torch.get_default_dtype()
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size, dtype=params_dtype)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x.float(), self.weight.float()).to(self.out_dtype), None


class _Experts(torch.nn.Module):
    """Records the router logits it is called with."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.router_logits: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        self.router_logits = router_logits
        return hidden_states


def _build(monkeypatch, module, router_dtype: str):
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "GateLinear", _Gate)
    monkeypatch.setattr(module, "FusedMoEFactory", _Experts)
    config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_size=16,
        moe_intermediate_size=8,
        num_shared_experts=0,
        score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        router_dtype=router_dtype,
    )
    if module is bailing_moe:
        return bailing_moe.BailingMoE(intermediate_size=8, config=config)
    return bailing_moe_linear.BailingMoeV25(config=config)


@pytest.mark.parametrize(
    "module", [bailing_moe, bailing_moe_linear], ids=["v2", "v2.5-linear"]
)
@pytest.mark.parametrize(
    ("router_dtype", "logits_dtype"),
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
    ids=["fp32", "bf16"],
)
def test_router_logits_keep_router_dtype(
    monkeypatch, module, router_dtype: str, logits_dtype: torch.dtype
) -> None:
    moe = _build(monkeypatch, module, router_dtype)
    hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)

    moe(hidden_states)

    router_logits = moe.experts.router_logits
    assert router_logits.dtype == logits_dtype
    assert moe.experts.kwargs["router_logits_dtype"] == logits_dtype
    torch.testing.assert_close(
        router_logits, moe.gate(hidden_states)[0], rtol=0, atol=0
    )
