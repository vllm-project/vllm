# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer experts that implement Nemotron Shortcut's clamped ReLU².

- BF16: FlashInfer's CUTLASS fused MoE with `ActivationType.ClampedRelu2`.
- MXFP8: FlashInfer's CuTe-DSL fused MoE with `quant_mode="mxfp8"` and
  `ActivationType.ClampedRelu2`.
- NVFP4: FlashInfer's CuTe-DSL fused MoE, applying the routing weight
  before the down projection as in training.

Each needs a FlashInfer build that exposes the clamp. Without one, BF16 and
MXFP8 fall back to the Triton experts (`experts.py`); NVFP4 has no Triton
experts and refuses to start.
"""

import functools
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.models.nemotron_shortcut.nvidia.moe.config import activation_clamp
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe, has_flashinfer_cutedsl
from vllm.utils.func_utils import supports_kw


@functools.cache
def _activation_type() -> Any:
    try:
        from flashinfer.fused_moe.core import ActivationType
    except ImportError:
        return None
    return ActivationType


@functools.cache
def _cute_dsl_fused_moe() -> Any:
    if not has_flashinfer_cutedsl():
        return None
    try:
        from flashinfer import cute_dsl_fused_moe
    except ImportError:
        return None
    return cute_dsl_fused_moe


def has_cutlass_clamped_relu2() -> bool:
    """FlashInfer's CUTLASS fused MoE implements `ClampedRelu2`."""
    return hasattr(_activation_type(), "ClampedRelu2")


def has_cute_dsl_options(*options: str) -> bool:
    """FlashInfer's CuTe-DSL fused MoE takes every keyword in `options`."""
    fused_moe = _cute_dsl_fused_moe()
    return fused_moe is not None and all(
        supports_kw(fused_moe, option, allow_var_kwargs=False) for option in options
    )


@functools.cache
def has_cute_dsl_mxfp8() -> bool:
    """FlashInfer's CuTe-DSL fused MoE takes MXFP8 weights and the clamp."""
    if not has_cute_dsl_options("quant_mode", "clamped_relu2_limit"):
        return False
    try:
        from flashinfer.cute_dsl.utils import (  # noqa: F401
            convert_row_major_sf_to_mma_layout,
        )
    except ImportError:
        return False
    return True


def _clamp_limit(moe_config: FusedMoEConfig) -> torch.Tensor | None:
    """The clamp as FlashInfer's ClampedRelu2 takes it: a one-element FP32
    tensor, or None for a plain ReLU²."""
    clamp = activation_clamp(moe_config)
    if clamp is None:
        return None
    return torch.tensor([clamp], dtype=torch.float32, device=moe_config.device)


def _check_aligned(moe_config: FusedMoEConfig) -> tuple[bool, str | None]:
    # The CuTe-DSL grouped GEMMs tile K and N in blocks of 128.
    for name, dim in (
        ("hidden", moe_config.hidden_dim),
        ("intermediate per partition", moe_config.intermediate_size_per_partition),
    ):
        if dim % 128 != 0:
            return False, f"the {name} size {dim} is not a multiple of 128"
    return True, None


def _tokens_replicated(parallel: FusedMoEParallelConfig) -> bool:
    # The kernels take every token and an expert interval, so tokens must be
    # replicated across EP ranks: no dispatcher, or all-gather/reduce-scatter.
    return not parallel.use_all2all_kernels or parallel.use_ag_rs_all2all_kernels


class NemotronShortcutFlashInferExperts(FlashInferExperts):
    """BF16 experts on FlashInfer's CUTLASS fused MoE."""

    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.clamp_limit = _clamp_limit(moe_config)

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            FlashInferExperts._supports_current_device()
            and current_platform.is_device_capability_family(100)
            and has_cutlass_clamped_relu2()
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.RELU2_NO_MUL

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        if self.clamp_limit is None:
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        assert not apply_router_weight_on_input
        flashinfer_cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w1,
            fc2_expert_weights=w2,
            output_dtype=self.out_dtype,
            quant_scales=[],
            output=output,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            activation_type=_activation_type().ClampedRelu2,
            clamped_relu2_limit=self.clamp_limit,
        )


class NemotronShortcutCuteDslMxfp8Experts(mk.FusedMoEExpertsModular):
    """MXFP8 x MXFP8 experts on FlashInfer's CuTe-DSL fused MoE.

    Weights stay as in the checkpoint (E4M3 with row-major E8M0 block-32
    scales); the scales are re-laid out for the kernel once after loading.
    Activations are quantized once per call. The kernel routes, runs the
    experts and sums the top-k in one pass.
    """

    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None:
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.clamp = activation_clamp(moe_config)
        self.clamp_limit = _clamp_limit(moe_config)
        self.hidden_dim = moe_config.hidden_dim
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.local_expert_offset = (
            moe_config.moe_parallel_config.ep_rank * self.local_num_experts
        )
        self._w1_sf: torch.Tensor | None = None
        self._w2_sf: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer.cute_dsl.utils import convert_row_major_sf_to_mma_layout

        # Checkpoint scales are row-major [E, rows, K // 32]; the kernel wants
        # the swizzled MMA layout. The originals stay on the layer.
        num_experts, intermediate, hidden = layer.w13_weight.shape
        self._w1_sf = convert_row_major_sf_to_mma_layout(
            layer.w13_weight_scale.data,
            m=intermediate,
            k=hidden,
            num_groups=num_experts,
            sf_vec_size=MXFP8_BLOCK_SIZE,
        )
        _, out_dim, intermediate = layer.w2_weight.shape
        self._w2_sf = convert_row_major_sf_to_mma_layout(
            layer.w2_weight_scale.data,
            m=out_dim,
            k=intermediate,
            num_groups=num_experts,
            sf_vec_size=MXFP8_BLOCK_SIZE,
        )
        # MXFP8 has no global scales; the kernel takes a per-expert alpha of 1.
        self._alpha = torch.ones(
            num_experts, dtype=torch.float32, device=layer.w13_weight.device
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
            return supported, reason
        return _check_aligned(moe_config)

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_cute_dsl_mxfp8()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return _tokens_replicated(moe_parallel_config)

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
        return (0,), (0,), (M, self.hidden_dim)

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._w1_sf is not None and self._w2_sf is not None
        assert self._alpha is not None, "process_weights_after_loading was not run"
        assert not apply_router_weight_on_input
        # The kernel takes global expert ids and the local expert offset.
        assert self.moe_config.moe_parallel_config.use_ep or expert_map is None

        num_tokens = hidden_states.shape[0]
        x_q, x_sf = mxfp8_e4m3_quantize(
            hidden_states.contiguous(), is_sf_swizzled_layout=False
        )
        if self.clamp_limit is None:
            activation_type = activation_to_flashinfer_int(MoEActivation.RELU2_NO_MUL)
            clamp = {}
        else:
            activation_type = int(_activation_type().ClampedRelu2)
            clamp = {"clamped_relu2_limit": self.clamp_limit}
        _cute_dsl_fused_moe()(
            x=x_q,
            x_sf=x_sf.view(torch.uint8).reshape(num_tokens, -1),
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.float(),
            w1_weight=w1,
            w1_weight_sf=self._w1_sf,
            w1_alpha=self._alpha,
            fc2_input_scale=None,
            w2_weight=w2,
            w2_weight_sf=self._w2_sf,
            w2_alpha=self._alpha,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=activation_type,
            quant_mode="mxfp8",
            **clamp,
        )


class NemotronShortcutCuteDslNvFp4Experts(FlashInferCuteDSLExperts):
    """NVFP4 experts on FlashInfer's CuTe-DSL fused MoE.

    The routing weight scales the ReLU² output before the down projection,
    which matches training; weighting the expert output instead changes where
    the result is rounded.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        per_token_activation: bool = False,
    ) -> None:
        if per_token_activation:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts do not support per-token activations"
            )
        if not has_cute_dsl_options(
            "routing_weights_before_fc2", "clamped_relu2_limit"
        ):
            raise RuntimeError(
                "Nemotron Shortcut NVFP4 experts need a FlashInfer whose "
                "cute_dsl_fused_moe takes routing_weights_before_fc2 and "
                "clamped_relu2_limit"
            )
        # FlashInfer weights the routing before FC2 only with ClampedRelu2.
        if activation_clamp(moe_config) is None:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts require activation_clamp_value"
            )
        parallel = moe_config.moe_parallel_config
        if moe_config.in_dtype != torch.bfloat16:
            raise NotImplementedError("NVFP4 experts require BF16 activations")
        if not _tokens_replicated(parallel):
            raise NotImplementedError(
                "NVFP4 experts with DP require the allgather_reducescatter backend"
            )
        if moe_config.num_experts % parallel.ep_size:
            raise NotImplementedError(
                "NVFP4 experts require the experts to split evenly over EP ranks"
            )
        aligned, reason = _check_aligned(moe_config)
        if not aligned:
            raise NotImplementedError(f"NVFP4 experts: {reason}")
        super().__init__(moe_config, quant_config, per_token_activation)
        self.clamp = activation_clamp(moe_config)
        self.clamp_limit = _clamp_limit(moe_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if self._localized_kwargs:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts do not support localized MoE shards"
            )

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.quant_dtype == "nvfp4"
        assert self.w1_scale is not None and self.w2_scale is not None
        assert a1q_scale is not None
        assert not apply_router_weight_on_input
        _cute_dsl_fused_moe()(
            x=hidden_states,
            # (M, K // 16) block scales; the functional API wants (M, K // 16, 1).
            x_sf=a1q_scale.unsqueeze(-1),
            token_selected_experts=topk_ids.to(torch.int32).contiguous(),
            token_final_scales=topk_weights.float().contiguous(),
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self.a2_gscale,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self.g2_alphas,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=int(_activation_type().ClampedRelu2),
            clamped_relu2_limit=self.clamp_limit,
            routing_weights_before_fc2=True,
        )
