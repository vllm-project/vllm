# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer (routed experts only, MXFP8).

Experts are sharded contiguously over the TP group (EP size = TP size); each
rank returns the full all-reduced bf16 output. Checkpoint names under
`model.llm.layers.N.mlp` map 1:1 to parameters, kept in the checkpoint's
layout.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from .ops import triton_bmm
from .ops.activation import swiglu_quantize
from .ops.moe import NUM_COUNTERS, combine, route


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
        # Completion counters of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32), persistent=False
        )


class Inkling2Experts(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter
        for name, (n, k) in (("w13", (2 * inter, hidden)), ("w2", (hidden, inter))):
            self._register(f"{name}_weight", (n, k), torch.float8_e4m3fn)
            self._register(f"{name}_weight_scale", (n, k // 32), torch.uint8)

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        # Rows per local expert of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counts", torch.zeros(self.num_experts, dtype=torch.int32), persistent=False
        )

    def _register(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        param = nn.Parameter(
            torch.empty((self.num_experts, *shape), dtype=dtype), requires_grad=False
        )
        set_weight_attrs(param, {"weight_loader": self._load_expert_shard})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        loaded = loaded[start : start + self.num_experts]
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def plan(self, tokens: int) -> triton_bmm.ExpertPlan:
        return triton_bmm.expert_plan(tokens, self.topk, self.num_experts)

    def _gemm(self, quantized, name: str, tiles, plan) -> torch.Tensor:
        act, act_scale = quantized
        return triton_bmm.bmm(
            act,
            act_scale,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            tiles,
            plan,
        )

    def forward(
        self, x: torch.Tensor, routing, quantized, plan: triton_bmm.ExpertPlan
    ) -> torch.Tensor:
        """`routing` and `quantized` (`x` in the expert layout) are from
        `ops.moe.route`. The routing weight is applied inside the SwiGLU,
        before the down projection."""
        src2dst, gammas, rows, tiles = routing
        gate_up = self._gemm(quantized, "w13", tiles, plan)
        activated = swiglu_quantize(
            gate_up,
            gammas,
            self.gate_cap,
            self.up_cap,
            (rows, tiles[2 * plan.max_ctas + 2 :]),
            plan.tile,
        )
        del gate_up
        down = self._gemm(activated, "w2", tiles, plan)
        # Each rank's fp32 combine is rounded once to bf16 before the all-reduce.
        out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)


class Inkling2MoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel group.
        Padding tokens (`is_padding` of the forward context) are routed to
        no expert and come out as 0."""
        x = x.contiguous()
        gate, experts = self.gate, self.experts
        plan = experts.plan(x.shape[0])
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        routing, quantized = route(
            x,
            gate.weight,
            gate.bias,
            gate.counters,
            experts.counts,
            gate.topk,
            gate.route_scale,
            experts.expert_bounds,
            plan,
            is_padding,
        )
        return experts(x, routing, quantized, plan)
