# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused RMSNorm + GateLinear for DeepSeek V4 MoE routing."""

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

DSV4_PRO_NUM_EXPERTS = 384
DSV4_PRO_HIDDEN_SIZE = 7168
DSV4_PRO_MAX_NUM_TOKENS = 16


def _dsv4_pro_norm_gate(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime dispatcher: fused ``dsv4_norm_router_gemm`` (M<=16) vs the
    unfused ``rms_norm + dsv3_router_gemm`` fallback (M>16).

    """
    if x.shape[0] <= DSV4_PRO_MAX_NUM_TOKENS:
        return ops.dsv4_norm_router_gemm(x, norm_weight, gate_weight, rms_eps)

    normed = torch.empty_like(x)
    # Call `_C::rms_norm` here to avoid select the path of native rms
    torch.ops._C.rms_norm(normed, x, norm_weight, rms_eps)
    logits = torch.mm(normed, gate_weight.t(), out_dtype=torch.float32)
    return normed, logits


def _dsv4_pro_norm_gate_fake(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    num_experts = gate_weight.shape[0]
    return (
        torch.empty_like(x),
        torch.empty(num_tokens, num_experts, dtype=torch.float32, device=x.device),
    )


direct_register_custom_op(
    op_name="dsv4_pro_norm_gate",
    op_func=_dsv4_pro_norm_gate,
    mutates_args=[],
    fake_impl=_dsv4_pro_norm_gate_fake,
)


@PluggableLayer.register("norm_gated_linear")
class NormGateLinear(nn.Module):
    """RMSNorm + GateLinear, fused on DSV4-Pro only."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        rms_eps: float = 1e-6,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.rms_eps = rms_eps

        self.norm = RMSNorm(hidden_size, eps=rms_eps, dtype=params_dtype)
        self.gate = GateLinear(
            hidden_size,
            num_experts,
            bias=False,
            out_dtype=torch.float32,  # DSV4 router output is fp32
            params_dtype=params_dtype,
            prefix=f"{prefix}.gate" if prefix else "gate",
        )

        self.e_score_correction_bias = None
        self.tid2eid = None

        self._fused_kernel_supported = (
            hidden_size == DSV4_PRO_HIDDEN_SIZE
            and num_experts == DSV4_PRO_NUM_EXPERTS
            and self.gate.allow_dsv3_router_gemm  # cuda platform
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fused_kernel_supported:
            assert x.shape[1] == DSV4_PRO_HIDDEN_SIZE
            assert self.gate.weight.shape == (
                DSV4_PRO_NUM_EXPERTS,
                DSV4_PRO_HIDDEN_SIZE,
            )
            # This must be wrapped in a custom op because our torch.compile integration
            # does not support runtime dispatching on num_tokens.
            return torch.ops.vllm.dsv4_pro_norm_gate(
                x, self.norm.weight, self.gate.weight, self.rms_eps
            )

        # Non-Pro fallback (e.g. DSV4-Flash with hidden_size=4096):

        normed_x = self.norm(x)
        logits, _ = self.gate(normed_x)
        return normed_x, logits
