# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi-K3 AMD pre-route projection fusion."""

import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_FP8_MAX = 448.0


def _quantize_weight_rows(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 matrix to OCP E4M3 with one FP32 scale per row."""

    if not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise ValueError("source weight must be a CUDA BF16 matrix")
    weight_f32 = weight.float()
    amax = weight_f32.abs().amax(dim=1)
    scale = torch.where(
        amax > 0,
        amax / _FP8_MAX,
        torch.ones_like(amax),
    )
    quantized = (
        (weight_f32 / scale[:, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return quantized, scale.contiguous()


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


def supports_kimi_k3_preroute_fp8(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the fixed-shape AITER FP8 primitives are available."""

    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
            supports_kimi_k3_moe_tri_projection_fp8,
            supports_kimi_k3_shared_down_fp8_weight,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    return supports_kimi_k3_moe_tri_projection_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        router_weight,
    ) and supports_kimi_k3_shared_down_fp8_weight(
        shared_down_weight,
        shared_down_scale,
        device=hidden_states.device,
    )


def _kimi_k3_preroute_fp8_impl(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
        kimi_k3_moe_tri_projection_fp8,
        kimi_k3_shared_down_fp8,
    )

    routed, shared_gate_up, router_logits = kimi_k3_moe_tri_projection_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        router_weight,
    )
    shared_output = kimi_k3_shared_down_fp8(
        shared_gate_up,
        shared_down_weight,
        shared_down_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return routed, shared_output, router_logits


def _kimi_k3_preroute_fp8_fake(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
        situ_beta,
        situ_linear_beta,
    )
    return (
        hidden_states.new_empty((hidden_states.shape[0], routed_weight.shape[0])),
        hidden_states.new_empty(hidden_states.shape),
        hidden_states.new_empty(
            (hidden_states.shape[0], router_weight.shape[0]),
            dtype=torch.float32,
        ),
    )


direct_register_custom_op(
    op_name="kimi_k3_preroute_fp8",
    op_func=_kimi_k3_preroute_fp8_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_preroute_fp8_fake,
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


class KimiK3PrerouteFp8Weights(nn.Module):
    """Model-load-time FP8 representation for the fused B1 decode path."""

    def __init__(
        self,
        routed_weight: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        shared_down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        routed_weight, routed_scale = _quantize_weight_rows(routed_weight)
        shared_gate_up_weight, shared_gate_up_scale = _quantize_weight_rows(
            shared_gate_up_weight
        )
        shared_down_weight, shared_down_scale = _quantize_weight_rows(
            shared_down_weight
        )
        self.register_buffer("routed_weight", routed_weight, persistent=False)
        self.register_buffer("routed_scale", routed_scale, persistent=False)
        self.register_buffer(
            "shared_gate_up_weight",
            shared_gate_up_weight,
            persistent=False,
        )
        self.register_buffer(
            "shared_gate_up_scale",
            shared_gate_up_scale,
            persistent=False,
        )
        self.register_buffer(
            "shared_down_weight",
            shared_down_weight,
            persistent=False,
        )
        self.register_buffer(
            "shared_down_scale",
            shared_down_scale,
            persistent=False,
        )

    @staticmethod
    def is_backend_available() -> bool:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return False
        try:
            from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
                is_kimi_k3_moe_preroute_fp8_available,
            )
        except (ImportError, ModuleNotFoundError):
            return False
        return is_kimi_k3_moe_preroute_fp8_available()

    @staticmethod
    def supports_weights(
        routed_weight: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        shared_down_weight: torch.Tensor,
    ) -> bool:
        weights = (
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
        return (
            tuple(weight.shape for weight in weights)
            == ((3584, 7168), (1536, 7168), (7168, 768))
            and all(weight.is_cuda for weight in weights)
            and all(weight.dtype == torch.bfloat16 for weight in weights)
            and all(weight.is_contiguous() for weight in weights)
            and len({weight.device for weight in weights}) == 1
        )

    @classmethod
    def create_if_supported(
        cls,
        *,
        use_latent_moe: bool,
        tensor_parallel_size: int,
        source_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        situ_beta: float | None,
        situ_linear_beta: float | None,
        lora_enabled: bool,
    ) -> "KimiK3PrerouteFp8Weights | None":
        """Create FP8 weights only for the complete fixed-shape contract."""

        if not (
            use_latent_moe
            and tensor_parallel_size == 8
            and source_weights is not None
            and situ_beta is not None
            and situ_linear_beta is not None
            and not lora_enabled
            and cls.is_backend_available()
            and cls.supports_weights(*source_weights)
        ):
            return None
        return cls(*source_weights)

    def supports_inputs(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
    ) -> bool:
        return supports_kimi_k3_preroute_fp8(
            hidden_states,
            self.routed_weight,
            self.routed_scale,
            self.shared_gate_up_weight,
            self.shared_gate_up_scale,
            self.shared_down_weight,
            self.shared_down_scale,
            router_weight,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        situ_beta: float,
        situ_linear_beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.kimi_k3_preroute_fp8(
            hidden_states,
            self.routed_weight,
            self.routed_scale,
            self.shared_gate_up_weight,
            self.shared_gate_up_scale,
            self.shared_down_weight,
            self.shared_down_scale,
            router_weight,
            situ_beta,
            situ_linear_beta,
        )
