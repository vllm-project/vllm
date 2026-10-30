# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode path of a softmax top-k MoE block with a gated shared expert and
Octave-W3 routed experts on RDNA3 (gfx1100).

Per layer it runs one GEMM for the router and the shared gate_up, one
``moe_decode_tail_rdna3`` launch (top-k, shared expert SiLU / down / gate,
workspace zeroing) and the two expert kernels, which accumulate in fp32 on top
of the shared output and write it back in fp16. That replaces the top-k,
SiLU, down, gate, zero fill and shared + routed add of the generic path.
"""

import os
from typing import TYPE_CHECKING

import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = init_logger(__name__)

MAX_TOKENS = 32
NUM_EXPERTS = 384
MAX_TOP_K = 16


def _fp16_unquantized(layer: torch.nn.Module | None) -> bool:
    return (
        layer is not None
        and isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod)
        and getattr(layer, "bias", None) is None
        and layer.weight.dtype == torch.float16
    )


class Rdna3MoEDecodeTail:
    def __init__(self, runner: "MoERunner", shared: torch.nn.Module):
        self.runner = runner
        self.shared = shared
        self.top_k = runner.router.top_k
        self.renormalize = runner.router.renormalize
        self._empty_f32: torch.Tensor | None = None
        self._w_router_gate_up: torch.Tensor | None = None
        self._counters: torch.Tensor | None = None

    @staticmethod
    def maybe_create(runner: "MoERunner") -> "Rdna3MoEDecodeTail | None":
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            FusedTopKRouter,
        )

        if os.environ.get("VLLM_RDNA3_MOE_DECODE_TAIL", "1") != "1":
            return None
        if not current_platform.is_rocm() or not hasattr(
            torch.ops._rocm_C, "moe_decode_tail_rdna3"
        ):
            return None
        from vllm.platforms.rocm import on_gfx1100

        shared = (
            runner._shared_experts._layer
            if runner._shared_experts is not None
            else None
        )
        cfg = runner.moe_config
        router = runner.router
        ok = (
            on_gfx1100()
            and type(router) is FusedTopKRouter
            and router.scoring_func == "softmax"
            and router.eplb_state is None
            and router.global_num_experts == NUM_EXPERTS
            and router.top_k <= MAX_TOP_K
            and _fp16_unquantized(runner.gate)
            and not runner._fse_fuse_gate
            and shared is not None
            and type(shared).__name__ == "Qwen2MoeMLP"
            and _fp16_unquantized(getattr(shared, "gate_up_proj", None))
            and _fp16_unquantized(getattr(shared, "down_proj", None))
            and _fp16_unquantized(getattr(shared, "expert_gate", None))
            and shared.expert_gate.weight.shape[0] == 1
            and not cfg.is_sequence_parallel
            and cfg.dp_size == 1
            and cfg.pcp_size == 1
            and not cfg.use_ep
            and runner.routed_scaling_factor == 1.0
            and runner.routed_input_transform is None
            and runner.routed_output_transform is None
        )
        if not ok:
            return None
        logger.info_once("RDNA3 MoE decode tail enabled.")
        return Rdna3MoEDecodeTail(runner, shared)

    def ensure_fused_weight(self) -> None:
        """Store the router and shared gate_up weights in one buffer, so both
        GEMMs run as one; the layers keep views into it."""
        if self._w_router_gate_up is not None:
            return
        gate, gate_up = self.runner.gate, self.shared.gate_up_proj
        w = torch.cat([gate.weight.data, gate_up.weight.data])
        e = gate.weight.shape[0]
        gate.weight.data = w[:e]
        gate_up.weight.data = w[e:]
        self._w_router_gate_up = w
        self._counters = torch.zeros(8, dtype=torch.int32, device=w.device)

    def supports(self, x: torch.Tensor) -> bool:
        from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
            Rdna3OctaveW3Experts,
        )

        routed = self.runner.routed_experts
        kernel = getattr(routed.quant_method, "moe_kernel", None)
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= MAX_TOKENS
            and x.dtype == torch.float16
            and x.is_contiguous()
            and self.runner.router.capture_fn is None
            and kernel is not None
            and isinstance(kernel.fused_experts, Rdna3OctaveW3Experts)
            and not kernel.output_is_reduced()
            and routed.expert_map is None
            and not routed.apply_router_weight_on_input
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            _identity_routing,
        )

        runner, shared = self.runner, self.shared
        routed = runner.routed_experts
        experts = routed.quant_method.moe_kernel.fused_experts
        M, top_k = x.shape[0], self.top_k

        assert self._w_router_gate_up is not None
        y = dispatch_unquantized_gemm()(runner.gate, x, self._w_router_gate_up, None)
        logits, gate_up = y[:, :NUM_EXPERTS], y[:, NUM_EXPERTS:]
        n = routed.w2_weight.size(1) * 32
        rows = M * top_k
        out = torch.empty(M, x.shape[1], dtype=torch.float32, device=x.device)
        topk_w = torch.empty(M, top_k, dtype=torch.float32, device=x.device)
        topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=x.device)
        experts_gate_up = torch.empty(rows, 2 * n, dtype=x.dtype, device=x.device)
        torch.ops._rocm_C.moe_decode_tail_rdna3(
            logits,
            gate_up,
            x,
            shared.expert_gate.weight,
            shared.down_proj.weight,
            out,
            topk_w,
            topk_ids,
            experts_gate_up,
            self.renormalize,
        )

        if self._empty_f32 is None:
            self._empty_f32 = torch.empty(0, dtype=torch.float32, device=x.device)
        sorted_ids, n_post_pad = _identity_routing(rows, x.device)
        expert_ids = topk_ids.view(-1)
        ops.moe_octw3_gemm_rdna3(
            x,
            experts_gate_up,
            routed.w13_weight,
            routed.w13_weight_scale,
            self._empty_f32,
            sorted_ids,
            expert_ids,
            n_post_pad,
            top_k,
            1,
            False,
            0,
            experts.had_w13,
        )
        out16 = torch.empty_like(x)
        torch.ops._rocm_C.moe_octw3_gemm_f32_rdna3(
            experts_gate_up,
            out,
            out16,
            self._counters,
            routed.w2_weight,
            routed.w2_weight_scale,
            topk_w.view(-1),
            sorted_ids,
            expert_ids,
            n_post_pad,
            1,
            True,
            top_k,
            experts.had_w2,
        )
        return out16
