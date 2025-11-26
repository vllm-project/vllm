# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    build_flashinfer_fp4_cutlass_moe_prepare_finalize,
    flashinfer_trtllm_fp4_moe,
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    select_nvfp4_gemm_impl,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
    run_nvfp4_emulations,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4", "CompressedTensorsW4A4Nvfp4MoeMethod"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.backend = "none"
        if envs.VLLM_NVFP4_GEMM_BACKEND is None:
            if has_flashinfer():
                self.backend = "flashinfer-cutlass"
            elif cutlass_fp4_supported():
                self.backend = "cutlass"
        elif envs.VLLM_USE_FBGEMM:
            self.backend = "fbgemm"
            try:
                import fbgemm_gpu  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Backend fbgemm requires fbgemm.f4f4bf16 operator, "
                    "Please install with: pip install fbgemm-gpu-genai"
                ) from exc
        elif envs.VLLM_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.VLLM_NVFP4_GEMM_BACKEND
            assert has_flashinfer(), f"FlashInfer is required for {self.backend}"
        elif envs.VLLM_NVFP4_GEMM_BACKEND == "cutlass":
            self.backend = "cutlass"
            assert cutlass_fp4_supported(), f"Cutlass is required for {self.backend}"

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 GEMM backend found. "
                "Please check your platform capability."
            )

        logger.info_once(f"Using {self.backend} for NVFP4 GEMM")
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            return 80
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        if self.backend == "flashinfer-trtllm":
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            if self.backend == "fbgemm":
                swizzled_weight_scale = swizzled_weight_scale.view(-1).view(torch.uint8)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            out = run_nvfp4_emulations(
                x=x,
                input_global_scale=layer.input_global_scale,
                weight=layer.weight_packed,
                weight_scale_swizzled=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
            )
            if bias is not None:
                out = out + bias
            return out

        output_dtype = x.dtype
        output_shape = [x.shape[0], layer.weight_packed.shape[0]]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_global_scale)

        mm_args = (
            x_fp4,
            layer.weight_packed,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
        )
        if self.backend.startswith("flashinfer-"):
            backend_name = self.backend[len("flashinfer-") :]
            out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
        elif self.backend == "fbgemm":
            out = torch.ops.fbgemm.f4f4bf16(
                x_fp4,
                layer.weight_packed,
                x_blockscale.view(-1).view(torch.uint8),
                layer.weight_scale,
                layer.alpha,
                use_mx=False,
            ).to(output_dtype)
        else:
            assert self.backend == "cutlass"
            out = cutlass_scaled_fp4_mm(*mm_args)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class CompressedTensorsW4A4Nvfp4MoeMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        from vllm.model_executor.layers.quantization.utils.nvfp4_moe_support import (  # noqa: E501
            detect_nvfp4_moe_support,
        )

        super().__init__(moe)
        _nvfp4 = detect_nvfp4_moe_support(self.__class__.__name__)
        self.cutlass_nvfp4_supported = _nvfp4.cutlass_supported
        self.allow_flashinfer = _nvfp4.allow_flashinfer
        self.use_marlin = _nvfp4.use_marlin
        self.group_size = 16
        self.flashinfer_moe_backend = None
        if self.allow_flashinfer:
            self.flashinfer_moe_backend = get_flashinfer_moe_backend()
            logger.info_once(
                f"Using FlashInfer {self.flashinfer_moe_backend.value} kernels"
                " for CompressedTensorsW4A4Nvfp4MoeMethod."
            )
        elif self.use_marlin:
            logger.info_once("Using Marlin for CompressedTensorsW4A4Nvfp4MoeMethod.")
        else:
            logger.info_once("Using Cutlass for CompressedTensorsW4A4Nvfp4MoeMethod.")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # reorder GEMM1 weights and block scales for FlashInfer CUTLASS kernel.
        if self.allow_flashinfer:
            w, s = reorder_w1w3_to_w3w1(
                layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2
            )
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)

        if not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected."
            )

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False
        )

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False
        )

        if self.use_marlin:
            prepare_moe_fp4_layer_for_marlin(layer)
            return
        # w13
        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            w13_input_global_scale = (
                layer.w13_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_experts)
            )
        else:
            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(
                torch.float32
            )
        layer.g1_alphas = torch.nn.Parameter(
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False,
        )

        layer.w13_input_scale_quant = torch.nn.Parameter(
            (w13_input_global_scale), requires_grad=False
        )

        # w2
        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            w2_input_global_scale = (
                layer.w2_input_global_scale.min()
                .to(torch.float32)
                .expand(layer.num_experts)
            )
        else:
            w2_input_global_scale = layer.w2_input_global_scale

        layer.g2_alphas = torch.nn.Parameter(
            ((1 / w2_input_global_scale) * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        layer.w2_input_scale_quant = torch.nn.Parameter(
            (w2_input_global_scale), requires_grad=False
        )

        # TensorRT-LLM specific processing
        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            # Prepare static weights for TRT-LLM kernel
            # alternate: prepare_static_weight_layouts_for_trtllm_moe
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = prepare_static_weights_for_trtllm_fp4_moe(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )
            logger.debug_once("Finished shuffling weights for TRT-LLM MOE")

            layer.gemm1_weights_fp4_shuffled = Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_weights_fp4_shuffled = Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm1_scales_fp4_shuffled = Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_scales_fp4_shuffled = Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False
            )

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del layer.w2_weight
            del layer.w2_weight_scale
            del layer.w13_weight
            del layer.w13_weight_scale
        else:
            # swizzle weight scales
            layer.w13_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
            )

            layer.w2_weight_scale = torch.nn.Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
            )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        if self.use_marlin or (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            return None
        elif not self.allow_flashinfer:
            return super().maybe_make_prepare_finalize(routing_tables)

        prepare_finalize = build_flashinfer_fp4_cutlass_moe_prepare_finalize(self.moe)
        logger.debug_once("%s", prepare_finalize.__class__.__name__)
        return prepare_finalize

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        """Return the appropriate GEMM experts implementation."""
        experts = select_nvfp4_gemm_impl(
            self.moe,
            self.moe_quant_config,
            allow_flashinfer=self.allow_flashinfer,
        )
        logger.debug_once("Using %s", experts.__class__.__name__)
        return experts

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if (
            self.use_marlin
            or self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            return None

        return nvfp4_moe_quant_config(
            g1_alphas=layer.g1_alphas,
            g2_alphas=layer.g2_alphas,
            a1_gscale=layer.w13_input_scale_quant,
            a2_gscale=layer.w2_input_scale_quant,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert activation == "silu", "Only SiLU activation is supported."

        if (
            self.allow_flashinfer
            and self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            if enable_eplb:
                raise NotImplementedError(
                    "EPLB not supported for `CompressedTensorsW4A4MoeMethod` yet."
                )

            return flashinfer_trtllm_fp4_moe(
                layer=layer,
                x=x,
                router_logits=router_logits,
                top_k=top_k,
                global_num_experts=global_num_experts,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                custom_routing_function=custom_routing_function,
                e_score_correction_bias=e_score_correction_bias,
            )

        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        if self.use_marlin:
            return fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                None,
                None,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=layer.w13_weight_scale_2,
                global_scale2=layer.w2_weight_scale_2,
                quant_type_id=scalar_types.float4_e2m1f.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                workspace=layer.workspace,
            )

        # FlashInfer fused experts path
        elif self.allow_flashinfer:
            from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (  # noqa: E501
                flashinfer_cutlass_moe_fp4,
                is_valid_flashinfer_cutlass_fused_moe,
            )

            assert is_valid_flashinfer_cutlass_fused_moe(
                x, layer.w13_weight, layer.w2_weight
            ), "Flashinfer CUTLASS Fused MoE not applicable!"

            assert self.moe_quant_config is not None

            return flashinfer_cutlass_moe_fp4(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_config=self.moe_quant_config,
                inplace=False,  # TODO(shuw): fix later, now output is high prec
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4

            assert expert_map is None, (
                "Expert Parallelism / expert_map "
                "is currently not supported for "
                "CompressedTensorsW4A4Nvfp4MoeMethod."
            )
            assert self.moe_quant_config is not None

            # Cutlass moe takes in activations in BF16/Half precision
            # and fp4 quantized weights loaded from the checkpoint
            return cutlass_moe_fp4(
                a=x,
                w1_fp4=layer.w13_weight,
                w2_fp4=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                # TODO(bnell): derive these from arguments
                m=x.shape[0],
                n=layer.w2_weight.shape[2] * 2,
                k=x.shape[1],
                e=layer.w13_weight.shape[0],
            ).to(x.dtype)
