# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class Mxfp4Config(QuantizationConfig):
    """Canonical base config for MXFP4 quantization.

    Subclasses override get_name() and override_quantization_method() to
    register themselves as the handler for a specific checkpoint format.
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        """MoE method for RoutedExperts. Subclasses override to pick a
        checkpoint-specific kernel family."""
        return Mxfp4MoEMethod(moe)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return self._make_moe_method(layer.moe_config)
        elif isinstance(layer, Attention):
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
            )
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        """MXFP4 config always uses MXFP4 quantization."""
        return True


class GptOssMxfp4Config(Mxfp4Config):
    """MXFP4 config for GPT-OSS checkpoints.

    Checkpoints carry ``"quant_method": "mxfp4"`` in their JSON config.
    override_quantization_method() maps that to the canonical internal name
    so that the rest of the loading path uses "gpt_oss_mxfp4" consistently.
    """

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gpt_oss_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # Match both "mxfp4" (original checkpoint value) and "gpt_oss_mxfp4"
        # (already normalized by verify_and_update_model_config) so that
        # explicit --quantization mxfp4 from the user doesn't cause a mismatch.
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        # Require explicit confirmation that this is a GPT-OSS model.
        # Do NOT fall back to returning the override when hf_config is None,
        # as that would silently claim all mxfp4 checkpoints.
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        return GptOssMxfp4MoEMethod(moe)


class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            swiglu_limit=7.0,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )


def _use_k3_situ_aiter(moe: FusedMoEConfig) -> bool:
    """Whether Kimi-K3's SiTU MXFP4 MoE should use the AITER A16W4 kernel.

    K3 is weight-only MXFP4 (W4A16) with SiTU activation, which the generic
    MXFP4 backend selector does not cover; route it to AITER on gfx950.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx950

    return (
        rocm_aiter_ops.is_fused_moe_enabled()
        and on_gfx950()
        and moe.activation == MoEActivation.SITU
        and moe.activation_situ_linear_beta is not None
        and rocm_aiter_ops.get_aiter_activation_type("situ") is not None
    )


def _use_k3_situ_int4_gfx942(moe: FusedMoEConfig) -> bool:
    """Whether Kimi-K3's SiTU MXFP4 experts should be requantized to int4 on gfx942.

    gfx942 has no scaled MXFP4 MFMA, so the native MXFP4 kernels do not compile
    there. Requantizing to groupwise int4 lets AITER's bf16 x int4 FlyDSL path
    serve the model, at the cost of a lossy weight conversion. That trade is the
    user's to make, so it is opt-in through
    ``--quantization-config.moe.weight int4_per_group_32`` rather than inferred from the
    hardware.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx942

    return (
        rocm_aiter_ops.is_fused_moe_enabled()
        and on_gfx942()
        and moe.activation == MoEActivation.SITU
        and moe.activation_situ_linear_beta is not None
        and _moe_weight_override_is_int4()
    )


def _moe_weight_override_is_int4() -> bool:
    """True when the user asked for int4 MoE weights on the command line.

    Set with ``--quantization-config.moe.weight int4_per_group_32``. The
    requantization is lossy, so it never happens unless it was requested.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    if vllm_config is None:
        return False
    quant_args = getattr(vllm_config.model_config, "quantization_config", None)
    moe_spec = getattr(quant_args, "moe", None)
    weight = getattr(moe_spec, "weight", None)
    if weight is None:
        return False
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static32,
    )

    return weight == kInt4Static32


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        self.is_k3_situ_aiter = _use_k3_situ_aiter(moe)
        self.is_k3_situ_int4_gfx942 = _use_k3_situ_int4_gfx942(moe)
        self.experts_cls: type[mk.FusedMoEExperts] | None
        if self.is_k3_situ_aiter or self.is_k3_situ_int4_gfx942:
            self.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_BF16
            self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]
            if self.is_k3_situ_int4_gfx942:
                logger.warning_once(
                    "Requantizing Kimi-K3 MXFP4 experts to groupwise int4 "
                    "for gfx942. This conversion is lossy and was explicitly "
                    "enabled by --quantization-config.moe.weight "
                    "int4_per_group_32."
                )
            else:
                logger.info_once("Using AITER_MXFP4_BF16 for Kimi-K3 SiTU MXFP4 MoE.")
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled():
                # AITER keeps bf16 activations below this token count, which
                # would not match the fp8 a8w4 kernels the interleaved SiTU
                # path is tuned for. The a16w4 path never reads it.
                # TODO: Remove once AITER takes this as a kernel argument.
                os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
        else:
            self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(
                moe
            )

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend,
            hidden_size,
            intermediate_size_per_partition,
            activation=self.moe.activation,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        if self.is_k3_situ_aiter:
            w13, w2, w13_scale, w2_scale = (
                self._convert_k3_situ_weight_to_kernel_format(layer)
            )
        else:
            w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
                convert_weight_to_mxfp4_moe_kernel_format(
                    mxfp4_backend=self.mxfp4_backend,
                    layer=layer,
                    w13_weight=w13,
                    w2_weight=w2,
                    w13_weight_scale=w13_scale,
                    w2_weight_scale=w2_scale,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                    _cache_permute_indices=self._cache_permute_indices,
                )
            )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        from vllm.platforms.rocm import on_gfx1250

        uses_triton_weight_format = self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and on_gfx1250()
        )
        if not uses_triton_weight_format:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16
            and not uses_triton_weight_format
        ):
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def _convert_k3_situ_weight_to_kernel_format(
        self, layer: RoutedExperts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # K3's AITER A16W4 kernel wants the separated ([gate_all, up_all])
        # stage-1 layout, unlike the interleaved gpt-oss/DeepSeek path in
        # convert_weight_to_mxfp4_moe_kernel_format. Preshuffle once here.
        from aiter.utility.fp4_utils import e8m0_shuffle

        from vllm._aiter_ops import rocm_aiter_ops

        fp4_dtype = torch.float4_e2m1fn_x2
        e8m0_dtype = torch.float8_e8m0fnu
        num_experts = layer.w13_weight.shape[0]

        # a8w4 (VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1) uses the gate/up-
        # interleaved (_gui_) fp8 flydsl kernels, which need w13 weight+scale
        # in interleave layout. Default a16w4 keeps the separated layout.
        guinterleave = rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled()
        w13 = rocm_aiter_ops.shuffle_weight_a16w4(
            layer.w13_weight.data.view(fp4_dtype), 16, guinterleave
        )
        w2 = rocm_aiter_ops.shuffle_weight_a16w4(
            layer.w2_weight.data.view(fp4_dtype), 16, False
        )
        w13_scale_raw = layer.w13_weight_scale.data.view(e8m0_dtype)
        w2_scale_raw = layer.w2_weight_scale.data.view(e8m0_dtype)
        w13_scale = rocm_aiter_ops.shuffle_scale_a16w4(
            w13_scale_raw.view(-1, w13_scale_raw.shape[-1]), num_experts, guinterleave
        )
        w2_scale = e8m0_shuffle(w2_scale_raw.view(-1, w2_scale_raw.shape[-1]))
        return w13, w2, w13_scale, w2_scale

    def _setup_kernel_k3_situ_gfx942(self, layer: RoutedExperts) -> None:
        # gfx942 has no native MXFP4 matmul. Convert the weights to groupwise
        # int4 once at load time for AITER's existing bf16 x int4 FlyDSL path.
        import inspect

        from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_moe_gemm1

        # Before ROCm/aiter#4471 the packed-int4 stage1 dropped the requested
        # activation and hardcoded SiLU, so K3 served fluent text while
        # computing the wrong function. Refuse instead of repeating that.
        if "act" not in inspect.signature(compile_moe_gemm1).parameters:
            raise RuntimeError(
                "This AITER build ignores the SiTUv2 activation on the "
                "packed-int4 MoE path and would silently compute SiLU. "
                "Rebuild with an AITER that includes ROCm/aiter#4471."
            )

        from aiter import dtypes as aiter_dtypes
        from aiter.ops.quant import per_1x32_i4_quant
        from aiter.ops.shuffle import (
            pack_int8_to_packed_int4,
            shuffle_scale_for_int4,
            shuffle_weight,
        )
        from aiter.utility import fp4_utils

        fp4_dtype = torch.float4_e2m1fn_x2
        e8m0_dtype = torch.float8_e8m0fnu

        # The dequant chain cannot run in place: mxfp4_to_f32 does a
        # repeat_interleave to split the packed nibbles, then an f32 LUT
        # gather, so the working tensor grows 8x over the packed weight
        # before per_1x32_i4_quant shrinks it again. Materializing that for
        # a whole expert tensor peaks well above 20 GiB per rank, which does
        # not fit once the weights are resident. Convert a slice of experts
        # at a time and free each slice before the next, so the transient is
        # bounded by _CONVERT_CHUNK / num_experts of the full tensor.
        _CONVERT_CHUNK = 8

        def convert(
            weight: torch.nn.Parameter,
            scale: torch.nn.Parameter,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Releases its source parameter before returning: the packed int4
            # output is the same size as the packed MXFP4 source, so holding
            # both doubles the expert footprint. Under expert parallel each
            # rank owns 1/ep_size of the experts and both fit, but under pure
            # tensor parallel every rank holds all experts and the second copy
            # does not. The caller must install each result before converting
            # the next tensor.
            w_all = weight.data.view(fp4_dtype)
            s_all = scale.data.view(e8m0_dtype)
            num_experts = w_all.shape[0]

            # Preallocate the outputs and write each chunk into its slice.
            # Accumulating chunks in a list and torch.cat-ing at the end holds
            # the whole result twice at the join, which is what the transient
            # chunking was meant to avoid in the first place.
            out_packed: torch.Tensor | None = None
            out_scale: torch.Tensor | None = None
            scale_stride = 0
            for lo in range(0, num_experts, _CONVERT_CHUNK):
                hi = min(lo + _CONVERT_CHUNK, num_experts)
                weight_f32 = fp4_utils.mxfp4_to_f32(w_all[lo:hi])
                scale_f32 = fp4_utils.e8m0_to_f32(s_all[lo:hi])
                chunk_experts, output_size, input_size = weight_f32.shape
                weight_bf16 = (
                    (
                        weight_f32.view(
                            chunk_experts, output_size, input_size // 32, 32
                        )
                        * scale_f32.view(
                            chunk_experts, output_size, input_size // 32, 1
                        )
                    )
                    .view(chunk_experts, output_size, input_size)
                    .to(torch.bfloat16)
                )
                del weight_f32, scale_f32

                weight_int4, weight_scale = per_1x32_i4_quant(weight_bf16)
                del weight_bf16
                weight_int4 = weight_int4.view(aiter_dtypes.i4x2).view(
                    chunk_experts, output_size, input_size
                )
                weight_packed = pack_int8_to_packed_int4(
                    shuffle_weight(weight_int4.view(aiter_dtypes.i8), (16, 16))
                )
                del weight_int4
                chunk_packed = weight_packed.view(
                    chunk_experts, output_size, input_size // 2
                ).view(aiter_dtypes.i4x2)
                chunk_scale = (
                    shuffle_scale_for_int4(weight_scale, group_size=32)
                    .view(-1)
                    .contiguous()
                )
                if out_packed is None:
                    out_packed = torch.empty(
                        (num_experts,) + tuple(chunk_packed.shape[1:]),
                        dtype=chunk_packed.dtype,
                        device=chunk_packed.device,
                    )
                    scale_stride = chunk_scale.numel() // chunk_experts
                    out_scale = torch.empty(
                        num_experts * scale_stride,
                        dtype=chunk_scale.dtype,
                        device=chunk_scale.device,
                    )
                assert out_packed is not None
                assert out_scale is not None
                out_packed[lo:hi].copy_(chunk_packed)
                out_scale[lo * scale_stride : hi * scale_stride].copy_(chunk_scale)
                del weight_packed, weight_scale, chunk_packed, chunk_scale
                torch.accelerator.empty_cache()

            # Every chunk has been read, so let the source storage go before
            # the caller converts the next tensor.
            del w_all, s_all
            assert out_packed is not None
            assert out_scale is not None
            weight.data = torch.empty(0, dtype=torch.uint8, device=out_packed.device)
            scale.data = torch.empty(0, dtype=torch.uint8, device=out_packed.device)
            torch.accelerator.empty_cache()
            return out_packed, out_scale

        # Install each result before converting the next tensor so only one
        # source is ever live alongside its output.
        w13, w13_scale = convert(layer.w13_weight, layer.w13_weight_scale)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        del w13, w13_scale
        torch.accelerator.empty_cache()

        w2, w2_scale = convert(layer.w2_weight, layer.w2_weight_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        del w2, w2_scale
        torch.accelerator.empty_cache()
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer):
        if self.is_k3_situ_int4_gfx942:
            self._setup_kernel_k3_situ_gfx942(layer)
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)

        from vllm.platforms.rocm import on_gfx1250

        if self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and on_gfx1250()
        ):
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        if self.mxfp4_backend == Mxfp4MoeBackend.EMULATION:
            # Canonical ``mxfp4`` checkpoints are weight-only W4A16. The
            # generic EMULATION config is W4A4, so preserve BF16 activations
            # while the fallback dequantizes only the weights.
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                gemm1_clamp_limit=swiglu_limit,
            )

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
