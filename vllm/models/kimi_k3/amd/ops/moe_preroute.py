# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi-K3 AMD pre-route projection fusion."""

import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def supports_kimi_k3_preroute_bf16(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
) -> bool:
    """Return whether AITER supports the exact-BF16 B1 projection cluster."""

    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
            supports_kimi_k3_moe_preroute_bf16,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    return supports_kimi_k3_moe_preroute_bf16(
        hidden_states,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )


def _kimi_k3_preroute_bf16_impl(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
        kimi_k3_moe_preroute_bf16,
    )

    return kimi_k3_moe_preroute_bf16(
        hidden_states,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
        situ_beta,
        situ_linear_beta,
    )


def _kimi_k3_preroute_bf16_fake(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shared_gate_up_weight, shared_down_weight, situ_beta, situ_linear_beta
    return (
        hidden_states.new_empty((hidden_states.shape[0], routed_weight.shape[0])),
        hidden_states.new_empty(hidden_states.shape),
    )


direct_register_custom_op(
    op_name="kimi_k3_preroute_bf16",
    op_func=_kimi_k3_preroute_bf16_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_preroute_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


class KimiK3PrerouteBf16(nn.Module):
    """Compile-safe wrapper for the fixed-shape AITER fusion."""

    def __init__(
        self,
        situ_beta: float,
        situ_linear_beta: float,
    ) -> None:
        super().__init__()
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

    @staticmethod
    def is_backend_available() -> bool:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return False
        try:
            from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
                is_kimi_k3_moe_preroute_bf16_available,
            )
        except (ImportError, ModuleNotFoundError):
            return False
        return is_kimi_k3_moe_preroute_bf16_available()

    @classmethod
    def create_if_supported(
        cls,
        *,
        use_latent_moe: bool,
        tensor_parallel_size: int,
        shared_experts: object | None,
        routed_projection: object | None,
        situ_beta: float | None,
        situ_linear_beta: float | None,
        lora_enabled: bool,
    ) -> "KimiK3PrerouteBf16 | None":
        """Create the specialization only for its complete model contract."""

        if not (
            cls.is_backend_available()
            and use_latent_moe
            and tensor_parallel_size == 8
            and shared_experts is not None
            and routed_projection is not None
            and situ_beta is not None
            and situ_linear_beta is not None
            and not lora_enabled
        ):
            return None
        return cls(situ_beta, situ_linear_beta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routed_weight: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        shared_down_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not supports_kimi_k3_preroute_bf16(
            hidden_states,
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
        ):
            return None
        return torch.ops.vllm.kimi_k3_preroute_bf16(
            hidden_states,
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
            self.situ_beta,
            self.situ_linear_beta,
        )
