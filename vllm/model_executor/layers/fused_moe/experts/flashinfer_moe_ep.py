# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MoE-EP megakernels as a modular-kernel experts implementation.

The megakernel routes on vLLM's top-k output and then dispatches, computes and
combines in one launch, so it is paired with a pass-through prepare/finalize.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpEpilogue,
    FlashInferMoeEpWeightFormat,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
    flashinfer_moe_ep_unsupported_reasons,
    is_sm107_device,
    supports_current_device,
    validate_flashinfer_moe_ep_layer,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
    require_finite_positive_scales,
)

if TYPE_CHECKING:
    pass

# Layer parameters the megakernel reads. After ``process_weights_after_loading``
# they alias the megakernel's own tensors, so EPLB permutes exactly what it uses.
KERNEL_WEIGHT_NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


_supported_weight_quant_schemes: frozenset[QuantKey] = frozenset(
    (kNvfp4Static, kMxfp4Static)
)

_supported_activations: frozenset[MoEActivation] = frozenset((MoEActivation.SILU,))


class FlashInferMoeEpExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._adapter: FlashInferMoeEp | None = None

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        reasons = flashinfer_moe_ep_unsupported_reasons(
            moe_config, weight_key, activation_key
        )
        if reasons:
            return False, f"kernel does not support {', '.join(reasons)}"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return supports_current_device()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key in _supported_weight_quant_schemes

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _supported_activations

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.use_ep
            and not moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Build the megakernel from the layer's canonical weights.

        The oracle has already put the weights in kernel format; the per-expert
        weight global scales arrive through the quant config and become the
        epilogue alphas. On SM107 the checkpoint's activation input scales also
        set the activation and FC1-output quantization of NVFP4 experts
        (``sm107_nvfp4_epilogue``); MXFP4 experts have no such scales.
        """
        validate_flashinfer_moe_ep_layer(layer)  # type: ignore[arg-type]
        epilogue = epilogue_from_quant_config(self.quant_config)
        weight_format: FlashInferMoeEpWeightFormat | None = None
        if is_sm107_device():
            # Every rank refuses a reload here, before any collective.
            if getattr(layer, "_experts_sm107_loaded", False):
                raise NotImplementedError(
                    "reloading weights is not supported with "
                    f"{self.moe_config.moe_backend} on SM107"
                )
            weight_format = cast(
                FlashInferMoeEpWeightFormat, self.quant_config.weight_quant_dtype
            )
            if weight_format == "nvfp4":
                epilogue = sm107_nvfp4_epilogue(
                    epilogue,
                    getattr(layer, "w13_input_scale", None),
                    getattr(layer, "w2_input_scale", None),
                )
            layer._experts_sm107_loaded = True
        adapter = FlashInferMoeEp(
            self.moe_config,
            FlashInferMoeEpWeights(
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                w13_scale=getattr(layer, "w13_weight_scale", None),
                w2_scale=getattr(layer, "w2_weight_scale", None),
            ),
            epilogue,
            apply_topk_in_fc1=apply_topk_in_fc1(
                self.moe_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            ),
            weight_format=weight_format,
        )
        adapter.warmup()
        for name, tensor in zip(KERNEL_WEIGHT_NAMES, adapter.kernel_weights()):
            # Rebind, never copy: the params must alias the megakernel buffers (EPLB).
            getattr(layer, name).data = tensor
        self._adapter = adapter

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._adapter is None:
            raise RuntimeError("FlashInfer MoE-EP weights have not been processed")
        output.copy_(
            self._adapter(hidden_states, topk_ids.to(torch.int32), topk_weights)
        )


def epilogue_from_quant_config(
    quant_config: FusedMoEQuantConfig,
) -> FlashInferMoeEpEpilogue:
    """Per-expert epilogue constants from the canonical quant config.

    The weight global scales (``g1_alphas``/``g2_alphas``) become the fc1/fc2
    alphas. Activations are quantized dynamically inside the megakernel, so on
    SM100/SM103 the checkpoint's input scales drop out; SM107 uses them
    (``sm107_nvfp4_epilogue``). MXFP4 checkpoints carry no global scales
    and keep the default epilogue. The alphas alias the layer's parameters, so
    EPLB permutations reach the kernel.
    """
    fc1_alpha, fc2_alpha = quant_config.g1_alphas, quant_config.g2_alphas
    if fc1_alpha is None and fc2_alpha is None:
        return FlashInferMoeEpEpilogue()
    assert fc1_alpha is not None and fc2_alpha is not None
    require_finite_positive_scales("g1_alphas", fc1_alpha)
    require_finite_positive_scales("g2_alphas", fc2_alpha)
    if fc1_alpha.dim() != 1 or fc2_alpha.dim() != 1:
        raise ValueError("the megakernel needs one weight global scale per expert")
    return FlashInferMoeEpEpilogue(
        fc1_alpha=fc1_alpha.float().contiguous(),
        fc2_alpha=fc2_alpha.float().contiguous(),
    )


def sm107_nvfp4_epilogue(
    epilogue: FlashInferMoeEpEpilogue,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> FlashInferMoeEpEpilogue:
    """NVFP4 epilogue of the SM107 megakernel from the checkpoint's activation
    scales.

    With unit norm constants small activations can underflow the E4M3 block scales.
    Tokens are quantized on their source rank, so the activation grid is the
    layer maximum over every EP rank. The alphas are scaled in place and keep
    aliasing the layer's weight global scales. Collective over the EP group:
    every EP rank must call it for the same MoE layers in the same order.
    """
    if w13_input_scale is None or w2_input_scale is None:
        raise ValueError(
            "the SM107 megakernel needs the checkpoint's NVFP4 activation input scales"
        )
    assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
    amax = torch.stack((w13_input_scale.amax(), w2_input_scale.amax())).float()
    finite = torch.stack(
        (torch.isfinite(w13_input_scale).all(), torch.isfinite(w2_input_scale).all())
    )
    amax = torch.where(finite, amax, torch.full_like(amax, math.inf))
    ep_group = get_ep_group()
    if ep_group.world_size > 1:
        torch.distributed.all_reduce(
            amax, op=torch.distributed.ReduceOp.MAX, group=ep_group.device_group
        )
    w13_amax, w2_amax = amax.tolist()
    if not (0.0 < w13_amax < math.inf and 0.0 < w2_amax < math.inf):
        raise ValueError(
            "the SM107 megakernel needs finite NVFP4 activation input scales with "
            f"a positive maximum per MoE layer over the EP group; got {w13_amax} "
            f"(w13) and {w2_amax} (w2); inf means a non-finite entry on some rank"
        )
    epilogue.fc1_alpha.mul_(w13_amax)
    epilogue.fc2_alpha.mul_(w2_amax)
    return FlashInferMoeEpEpilogue(
        input_norm_const=1.0 / w13_amax,
        fc1_alpha=epilogue.fc1_alpha,
        fc2_alpha=epilogue.fc2_alpha,
        fc1_norm_const=torch.full_like(epilogue.fc2_alpha, 1.0 / w2_amax),
    )
