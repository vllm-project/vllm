# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Mxfp4Config(QuantizationConfig):
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
            # TODO: Add support for MXFP4 Linear Method.
            # MXFP4 LinearMethod is available in AMD-Quark, refer to that implementation
            # if you are interested in enabling MXFP4 here.
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
                scope="local",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            if current_platform.is_xpu():
                return XpuMxfp4MoEMethod(layer.moe_config)
            else:
                quant_method = Mxfp4MoEMethod(layer.moe_config)
                return quant_method
        elif isinstance(layer, Attention):
            # TODO: Add support for MXFP4 Attention.
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
                scope="local",
            )
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        """MXFP4 config always uses MXFP4 quantization."""
        return True


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        )

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        # Initialized in process_weights_after_loading for CUTLASS/SM90 backends
        self.moe_mk: mk.FusedMoEModularKernel | None = None

        # we round up the parameter here to ensure a2a kernel allocate the correct memory size # noqa: E501
        self.moe.hidden_dim, self.moe.intermediate_size_per_partition = (
            mxfp4_round_up_hidden_size_and_intermediate_size(
                self.mxfp4_backend,
                self.moe.hidden_dim,
                self.moe.intermediate_size_per_partition,
            )
        )

        # used for triton kernel
        self.w13_precision_config = None
        self.w2_precision_config = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        # Directly use padded size from config
        self.intermediate_size = intermediate_size_per_partition_after_pad = (
            self.moe.intermediate_size_per_partition
        )
        self.hidden_size = hidden_size = self.moe.hidden_dim

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
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
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
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
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition_after_pad,
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
        layer: FusedMoE,
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
        sf_block_size = 32  # mxfp4 block size

        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
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
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_to_mxfp4_moe_kernel_format(
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
        # For TRITON backends, weights and scales are wrapped tensors from
        # triton_kernels that don't support .detach(). The weights have already
        # been deleted and precision_config has been set inside
        # convert_to_mxfp4_moe_kernel_format, manually assign parameters
        if self.mxfp4_backend not in (
            Mxfp4MoeBackend.TRITON,
            Mxfp4MoeBackend.TRITON_MONOLITHIC,
            Mxfp4MoeBackend.TRITON_UNFUSED,
        ):
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        # TRITON_MONILITHIC need configs but can't make mk
        if (
            self.moe_quant_config
            and self.mxfp4_backend != Mxfp4MoeBackend.TRITON_MONOLITHIC
        ):
            assert self.experts_cls is not None
            self.moe_mk = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def process_weights_after_loading(self, layer):
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_bias,
            w2_bias,
        )

        layer._already_called_process_weights_after_loading = True

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        if self.mxfp4_backend in (
            Mxfp4MoeBackend.TRITON,
            Mxfp4MoeBackend.TRITON_UNFUSED,
            Mxfp4MoeBackend.TRITON_MONOLITHIC,
        ):
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None

            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    @property
    def is_monolithic(self) -> bool:
        return self.mxfp4_backend in (
            Mxfp4MoeBackend.TRITON_MONOLITHIC,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        if layer.enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        assert _can_support_mxfp4(
            layer.use_grouped_topk,
            layer.topk_group,
            layer.num_expert_group,
            layer.expert_map,
            layer.custom_routing_function,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.scoring_func,
            layer.activation,
            layer.eplb_state.expert_load_view,
            layer.eplb_state.logical_to_physical_map,
            layer.eplb_state.logical_replica_count,
        ), "MXFP4 are not supported with this configuration."

        assert self.moe_mk is not None
        return self.moe_mk(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.is_monolithic

        if layer.enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        assert _can_support_mxfp4(
            layer.use_grouped_topk,
            layer.topk_group,
            layer.num_expert_group,
            layer.expert_map,
            layer.custom_routing_function,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.scoring_func,
            layer.activation,
            layer.eplb_state.expert_load_view,
            layer.eplb_state.logical_to_physical_map,
            layer.eplb_state.logical_replica_count,
        ), "MXFP4 are not supported with this configuration."

        if self.mxfp4_backend in (
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        ):
            from flashinfer import trtllm_fp4_block_scale_moe

            if (
                self.mxfp4_backend
                == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC
            ):
                assert x.dtype == torch.bfloat16
                x_quant = x
                x_scale = None
            elif (
                self.mxfp4_backend
                == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC
            ):
                from flashinfer import mxfp8_quantize

                x_quant, x_scale = mxfp8_quantize(x, False)  # to mxfp8
                x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)

            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                routing_logits=router_logits.to(torch.bfloat16),
                routing_bias=None,
                hidden_states=x_quant,
                hidden_states_scale=x_scale,
                gemm1_weights=layer.w13_weight,  # uint8 (e2m1 x 2)
                gemm1_weights_scale=layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                gemm1_bias=layer.w13_bias,  # fp32 per expert per channel
                gemm1_alpha=layer.gemm1_alpha,  # fp32 per expert
                gemm1_beta=layer.gemm1_beta,  # fp32 per expert
                gemm1_clamp_limit=layer.gemm1_clamp_limit,  # fp32 per expert
                gemm2_weights=layer.w2_weight,  # uint8 (e2m1 x 2)
                gemm2_weights_scale=layer.w2_weight_scale,  # ue8m0
                gemm2_bias=layer.w2_bias,  # fp32 per expert per channel
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=layer.global_num_experts,
                top_k=layer.top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=self.intermediate_size,  # padded to multiple of 256
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=self.num_experts,
                routed_scaling_factor=None,
                routing_method_type=1 if layer.renormalize else 0,
                do_finalize=True,
                tune_max_num_tokens=max(self.max_capture_size, 1),
            )[0]
            return trtllm_gen_output
        elif self.mxfp4_backend == Mxfp4MoeBackend.TRITON_MONOLITHIC:
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward,
            )

            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                gating_output=router_logits,
                topk=layer.top_k,
                renormalize=layer.renormalize,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")


class XpuMxfp4MoEMethod(Mxfp4MoEMethod):
    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self.moe_config = moe_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.original_hidden_size = hidden_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert layer.activation == MoEActivation.SWIGLUOAI, (
            "Only swiglu_oai activation is supported for "
            f"XPU MXFP4 MoE, not {layer.activation}."
        )
        from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

        M, _ = x.size()
        routing_weights = torch.empty(
            M, layer.top_k, dtype=torch.float32, device=x.device
        )
        selected_experts = torch.empty(
            M, layer.top_k, dtype=torch.int32, device=x.device
        )
        token_expert_indices = torch.empty(
            M, layer.top_k, dtype=torch.int32, device=x.device
        )

        if layer.use_grouped_topk:
            routing_weights, selected_experts = torch.ops._moe_C.fused_grouped_topk(
                x,
                router_logits,
                layer.top_k,
                layer.renormalize,
                n_expert_group=layer.num_expert_group,
                n_topk_group=layer.topk_group,
                scoring_func=layer.scoring_func,
                routed_scaling_factor=layer.routed_scaling_factor,
                bias=layer.e_score_correction_bias,
            )
        else:
            torch.ops._moe_C.topk_softmax(
                routing_weights,
                selected_experts,
                token_expert_indices,
                router_logits,
                layer.renormalize,
                layer.e_score_correction_bias,
            )

        return xpu_fused_moe(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_bias=layer.w13_bias if self.moe.has_bias else None,
            w13_scales=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_bias=layer.w2_bias if self.moe.has_bias else None,
            w2_scales=layer.w2_weight_scale,
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            n_experts_per_token=layer.top_k,
            activation=layer.activation,
            num_experts=layer.local_num_experts,
            is_mxfp4=True,
        )
