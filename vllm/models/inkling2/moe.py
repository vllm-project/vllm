# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer (routed experts only, NVFP4 or MXFP8).

Experts are sharded contiguously over the TP group (EP size = TP size); each
rank returns the full all-reduced bf16 output. Checkpoint names under
`model.llm.layers.N.mlp` map 1:1 to parameters.
"""

from typing import Literal, Protocol

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

from .ops.activation import swiglu_quantize
from .ops.block_scales import shuffle_rows, swizzle_block_scales
from .ops.moe import NUM_COUNTERS, combine, combine_push, route
from .ops.seq_parallel import SeqParallel
from .ops.trtllm_bmm import bmm, expert_plan, load_kernels

QuantFormat = Literal["nvfp4", "mxfp8"]


class ExpertsLoRA(Protocol):
    """Adds the routed experts' LoRA deltas (`lora.Inkling2MoEWithLoRA`),
    for an experts implementation over expert-sorted rows: row `r` is token
    `perm[r] // topk`, and this rank's experts own rows
    `[offsets[0], offsets[-1])`. The NVFP4 / MXFP8 experts here (tile-padded
    rows, quantized activations) do not support it; the BF16 experts a model
    may plug in instead (SkyRL's `Inkling2BF16Experts`) do.
    """

    def add_w13(
        self,
        gate_up: torch.Tensor,
        x: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...

    def add_w2(
        self,
        down: torch.Tensor,
        activated: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...


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
    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            # Rows are shuffled and block scales swizzled (``block_scales``).
            self._register(
                f"{name}_weight",
                (n, k // pack),
                weight_dtype,
                loader=self._load_weight,
            )
            self._register(
                f"{name}_weight_scale",
                (n, k // group),
                scale_dtype,
                loader=self._load_block_scales,
            )
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

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
        self.register_buffer(
            "ones", torch.ones(self.num_experts, dtype=torch.float32), persistent=False
        )
        load_kernels(
            (self.w13_weight, self.w2_weight),
            (self.w13_weight_scale, self.w2_weight_scale),
            self.topk,
        )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
        loader=None,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if loader is None:
            loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        param.data.copy_(shuffle_rows(param.data))

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def plan(self, tokens: int):
        return expert_plan(tokens, self.topk, self.num_experts, self.nvfp4, self.shapes)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Outer scale amax and denominator of a projection's NVFP4 weights."""
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def _gemm(self, quantized, name: str, tiles, plan, projection: int):
        act, act_scale, row_scale = quantized
        return bmm(
            act,
            act_scale,
            row_scale,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            self.ones,
            tiles,
            plan,
            projection,
        )

    def forward(
        self,
        x: torch.Tensor,
        routing,
        quantized,
        plan,
        reduce: bool = True,
        push: SeqParallel | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """`routing` and `quantized` (`x` in the expert layout) are from
        `ops.moe.route`. The routing weight is applied inside the SwiGLU,
        before the down projection. Without `reduce`, returns the local
        expert rows and `src2dst` for the consumer to combine and reduce,
        or pushes their combine to the token shard owners."""
        src2dst, gammas, rows, tiles = routing
        gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
        m = plan.max_ctas
        activated = swiglu_quantize(
            gate_up,
            gammas,
            self.gate_cap,
            self.up_cap,
            (rows, tiles[2 * m + 2 :], tiles[:m]),
            plan.tile,
            *self.scales("w2"),
        )
        del gate_up
        down = self._gemm(activated, "w2", tiles, plan, 1)
        if push is not None:
            # Each rank's fp32 combine is rounded once to bf16.
            combine_push(down, src2dst, self.topk, push)
            return None
        if not reduce:
            return down, src2dst
        # Each rank's fp32 combine is rounded once to bf16 before the all-reduce.
        out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)


class Inkling2MoE(nn.Module):
    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push: SeqParallel | None = None,
        lora: ExpertsLoRA | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel group;
        without `reduce`, the unreduced `(down, src2dst)` of
        `Inkling2Experts`, or (with `push`) pushes this rank's partial sums
        to the token shard owners.
        Padding tokens (`is_padding` of the forward context) are routed to
        no expert and come out as 0. `lora` (`Inkling2MoEWithLoRA`) adds
        per-expert LoRA deltas inside the expert GEMMs, before any reduce or
        push; the NVFP4 / MXFP8 experts here do not support it."""
        if lora is not None:
            raise NotImplementedError(
                "Inkling2 expert LoRA needs BF16 experts; the NVFP4 / MXFP8 "
                "experts do not support it"
            )
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
            *experts.scales("w13"),
            is_padding,
        )
        return experts(x, routing, quantized, plan, reduce, push)
