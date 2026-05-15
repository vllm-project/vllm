# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.modelopt_utils import (
    ModelOptKVCacheMethod,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class ModelOptMxFp8Config(ModelOptQuantConfigBase):
    """Config class for ModelOpt MXFP8."""

    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
    ) -> None:
        super().__init__(exclude_modules)
        self.is_checkpoint_mxfp8_serialized = is_checkpoint_mxfp8_serialized

        if not is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 quantization requires a serialized checkpoint. "
                "Dynamic quantization is not supported."
            )

        logger.warning(
            "Detected ModelOpt MXFP8 checkpoint. Please note that "
            "the format is experimental and could change in future."
        )

        self.kv_cache_quant_algo = kv_cache_quant_algo

    def get_name(self) -> QuantizationMethods:
        return "modelopt_mxfp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin kernel supports MXFP8 on SM80+
        return 80

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        algo = cls._extract_modelopt_quant_algo(hf_quant_cfg)
        if algo is not None and "MXFP8" in algo:
            return "modelopt_mxfp8"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        **kwargs: Any,
    ) -> "ModelOptMxFp8Config":
        is_checkpoint_mxfp8_serialized = "MXFP8" in quant_method.upper()

        # For MXFP8, validate required fields in the config
        if is_checkpoint_mxfp8_serialized and "quantization" in original_config:
            quant_config = original_config["quantization"]
            required_fields = ["kv_cache_quant_algo", "exclude_modules"]
            missing_fields = [
                field for field in required_fields if field not in quant_config
            ]
            if missing_fields:
                raise ValueError(
                    f"MXFP8 quantization requires the following fields in "
                    f"hf_quant_config.json: {missing_fields}"
                )

        return cls(
            is_checkpoint_mxfp8_serialized,
            kv_cache_quant_method,
            exclude_modules,
        )


class ModelOptMxFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt MXFP8 quantization."""

    def __init__(self, quant_config: ModelOptMxFp8Config) -> None:
        self.quant_config = quant_config

        if not self.quant_config.is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 currently only supports serialized checkpoints. "
                "Dynamic quantization is not supported."
            )

        self.kernel = init_mxfp8_linear_kernel()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size

        if not self.quant_config.is_checkpoint_mxfp8_serialized:
            raise ValueError(
                "MXFP8 quantization was selected, but checkpoint is not "
                "MXFP8 serialized. Dynamic quantization is not supported."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input dimension to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {input_size_per_partition}"
            )

        # Weight tensor: FP8 E4M3 format
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale tensor (E8M0 encoded as uint8), one scale per block of 32 along K
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Validate weight tensor
        if layer.weight.ndim != 2:
            raise ValueError(
                f"MXFP8 weight must be 2D tensor [N, K], got {layer.weight.ndim}D "
                f"with shape {tuple(layer.weight.shape)}"
            )

        if layer.weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                f"MXFP8 weight must be {MXFP8_VALUE_DTYPE} (FP8 E4M3), "
                f"got {layer.weight.dtype}. The checkpoint may not be properly "
                f"quantized with MXFP8."
            )

        # Validate weight scale tensor (should be 2D, not swizzled)
        assert layer.weight_scale.ndim == 2, (
            f"MXFP8 weight scale must be 2D, got {layer.weight_scale.ndim}D"
        )
        assert layer.weight_scale.dtype == MXFP8_SCALE_DTYPE, (
            f"MXFP8 weight scale must be {MXFP8_SCALE_DTYPE},"
            f" got {layer.weight_scale.dtype}"
        )

        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class ModelOptMxFp8FusedMoE(FusedMoEMethodBase):
    """FlashInfer TRTLLM MXFP8 block-scale MoE for ModelOpt checkpoints."""

    def __init__(
        self,
        quant_config: ModelOptMxFp8Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        self.quant_config = quant_config
        assert self.quant_config.is_checkpoint_mxfp8_serialized

        self.mxfp8_backend, self.experts_cls = select_mxfp8_moe_backend(config=self.moe)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert layer.intermediate_size_per_partition == intermediate_size_per_partition
        assert layer.hidden_size == hidden_size
        layer.orig_dtype = params_dtype

        if hidden_size % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 MoE requires hidden_size divisible by {MXFP8_BLOCK_SIZE}, "
                f"got {hidden_size}."
            )
        if intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                "MXFP8 MoE requires intermediate_size_per_partition divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {intermediate_size_per_partition}."
            )

        layer.num_experts = num_experts
        weight_loader = extra_weight_attrs.get("weight_loader")
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # GEMM 1 weights: [E, (2I or I), H]
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2 weights: [E, H, I]
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        # Per-block (K=32) E8M0 scales.
        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Ensure the generic MoE weight-loader treats these as block scales.
        set_weight_attrs(
            layer.w13_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )
        set_weight_attrs(
            layer.w2_weight_scale,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )

    @staticmethod
    def _check_weight_dtypes(layer: torch.nn.Module) -> None:
        """Validate weight and scale dtypes before processing."""
        expected = {
            "w13_weight": MXFP8_VALUE_DTYPE,
            "w2_weight": MXFP8_VALUE_DTYPE,
            "w13_weight_scale": MXFP8_SCALE_DTYPE,
            "w2_weight_scale": MXFP8_SCALE_DTYPE,
        }
        for name, expected_dtype in expected.items():
            actual = getattr(layer, name).dtype
            if actual != expected_dtype:
                raise ValueError(
                    f"Expected {name} dtype {expected_dtype}, got {actual}."
                )

    def _shuffle_weights_for_trtllm(self, layer: torch.nn.Module) -> None:
        """Shuffle weights and scales into FlashInfer TRTLLM MXFP8 layout."""
        from flashinfer import (
            reorder_rows_for_gated_act_gemm,
            shuffle_matrix_a,
            shuffle_matrix_sf_a,
        )

        epilogue_tile_m = 128
        num_experts = layer.w13_weight.shape[0]
        is_gated = self.moe.is_act_and_mul
        intermediate_size_factor = 2 if is_gated else 1

        w13_weight = layer.w13_weight.data
        w13_scale = layer.w13_weight_scale.data
        if is_gated:
            # FI TRTLLM gated kernels use W31 ordering. Model checkpoints store
            # gated projection as W13, so convert once before shuffling.
            w13_weight = swap_w13_to_w31(w13_weight)
            w13_scale = swap_w13_to_w31(w13_scale)

        w13_weight_shuffled = []
        w2_weight_shuffled = []
        w13_scale_shuffled = []
        w2_scale_shuffled = []
        for i in range(num_experts):
            w13_i = w13_weight[i].reshape(
                intermediate_size_factor * layer.intermediate_size_per_partition, -1
            )
            w13_sf_i = w13_scale[i].reshape(
                intermediate_size_factor * layer.intermediate_size_per_partition, -1
            )
            if is_gated:
                # Reorder rows for gated activation layout expected by TRTLLM.
                w13_i = reorder_rows_for_gated_act_gemm(w13_i.clone())
                w13_sf_i = reorder_rows_for_gated_act_gemm(w13_sf_i.clone())

            w13_shuffled_i = shuffle_matrix_a(w13_i.view(torch.uint8), epilogue_tile_m)
            w2_shuffled_i = shuffle_matrix_a(
                layer.w2_weight.data[i].view(torch.uint8), epilogue_tile_m
            )
            w13_weight_shuffled.append(
                w13_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE)
            )
            w2_weight_shuffled.append(
                w2_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE)
            )
            w13_sf_shuffled_i = shuffle_matrix_sf_a(
                w13_sf_i.view(torch.uint8).reshape(
                    intermediate_size_factor * layer.intermediate_size_per_partition,
                    -1,
                ),
                epilogue_tile_m,
            )
            w2_sf_shuffled_i = shuffle_matrix_sf_a(
                layer.w2_weight_scale.data[i]
                .view(torch.uint8)
                .reshape(layer.hidden_size, -1),
                epilogue_tile_m,
            )
            w13_scale_shuffled.append(
                w13_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE)
            )
            w2_scale_shuffled.append(
                w2_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE)
            )

        replace_parameter(
            layer, "w13_weight", torch.stack(w13_weight_shuffled).contiguous()
        )
        replace_parameter(
            layer, "w2_weight", torch.stack(w2_weight_shuffled).contiguous()
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            torch.stack(w13_scale_shuffled).contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            torch.stack(w2_scale_shuffled).contiguous(),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        self._check_weight_dtypes(layer)

        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.mxfp8_backend,
            layer=layer,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_input_scale=None,
            w2_input_scale=None,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.mxfp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return make_fp8_moe_quant_config(
            fp8_backend=self.mxfp8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=self.weight_block_size,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
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
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


# Register the method classes for ModelOptMxFp8Config
ModelOptMxFp8Config.LinearMethodCls = ModelOptMxFp8LinearMethod
ModelOptMxFp8Config.FusedMoEMethodCls = ModelOptMxFp8FusedMoE
ModelOptMxFp8Config.KVCacheMethodCls = ModelOptKVCacheMethod
