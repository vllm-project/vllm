# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 routed experts re-quantized to MXFP4 at load (ROCm).

``VLLM_FP8_MOE_REQUANT=mxfp4`` keeps the checkpoint's block-FP8 expert
weights for loading and, once loaded, dequantizes each expert to bf16 and
quantizes it to MXFP4 (e2m1 values, e8m0 block-32 scales) for the aiter MXFP4
MoE kernels: half the expert bytes of FP8. ``VLLM_FP8_MOE_REQUANT_ACT`` selects
the activation dtype: a8w4 (fp8 per-token; aiter's interleaved FlyDSL kernels,
the ``AITER_MXFP4_BF16`` backend; default), a4w4 (MXFP4 activations,
``AITER_MXFP4_MXFP4``) or a16w4 (bf16, ``AITER_TRITON_MXFP4_BF16``).
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


def dequant_block_fp8_to_bf16(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: list[int]
) -> torch.Tensor:
    """``[N, K]`` fp8 x ``[ceil(N/bn), ceil(K/bk)]`` fp32 block scales -> bf16."""
    n, k = weight.shape
    bn, bk = block_size
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


class Fp8BlockExpertsAsMxfp4FusedMoE(Mxfp4OnlineMoEMethod):
    """Block-FP8 routed experts stored as MXFP4 (VLLM_FP8_MOE_REQUANT=mxfp4).

    Weights are created and loaded by ``Fp8MoEMethod`` (fp8 values + block
    ``weight_scale_inv``), so the checkpoint loader is unchanged; the
    re-quantization happens in ``process_weights_after_loading``. Rounding the
    values to the e2m1 grid with power-of-two block-32 scales is lossy: the
    weight error is of the order of an MXFP4 quantization of the bf16 weights.
    """

    # FP8 expert parameters live on the meta device and are materialized one
    # layer at a time by the layerwise online-processing loader, so each layer
    # is re-quantized as soon as its experts are loaded: peak memory is the
    # MXFP4 model plus one layer of FP8 experts (TP4 would not fit otherwise).
    uses_meta_device = True

    def __init__(self, quant_config, layer: RoutedExperts) -> None:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        act = envs.VLLM_FP8_MOE_REQUANT_ACT.lower()
        moe_config = layer.moe_config
        # a4w4 asks for MXFP4 activations (AITER_MXFP4_MXFP4). With bf16
        # activations the oracle picks AITER_MXFP4_BF16, whose interleaved
        # weight layout makes aiter quantize activations to fp8 (a8w4); true
        # bf16 activations need the aiter Triton backend.
        self.activation_quant_key = kMxfp4Dynamic if act == "a4w4" else None  # type: ignore[assignment]
        # The FP8 method only creates/loads the block-FP8 parameters; build it
        # before the MXFP4 backend override below (it resolves moe_backend too).
        self.fp8_method = Fp8MoEMethod(quant_config, layer)
        assert self.fp8_method.block_quant, "FP8 re-quant needs block-scaled experts"
        assert self.fp8_method.weight_block_size is not None
        self.fp8_block_size = list(self.fp8_method.weight_block_size)
        if act == "a16w4" and moe_config.moe_backend == "auto":
            moe_config.moe_backend = "aiter_triton_mxfp4_bf16"
        super().__init__(moe=moe_config)
        self.w13_precision_config = None
        self.w2_precision_config = None
        self.quant_config = quant_config
        logger.info_once(
            "FP8 block experts re-quantized to MXFP4 at load (%s, %s)",
            act,
            self.mxfp4_backend.value,
        )

    def get_fused_moe_quant_config(self, layer):
        # Triton-format backends keep the scales as PrecisionConfigs (set by
        # _setup_kernel), not as layer parameters.
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=self.w13_precision_config,
                w2_scale=self.w2_precision_config,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                layer=layer,
            )
        return super().get_fused_moe_quant_config(layer)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size != layer.moe_config.hidden_dim_unpadded or (
            intermediate_size_per_partition
            != layer.moe_config.intermediate_size_per_partition_unpadded
        ):
            raise ValueError(
                "FP8 -> MXFP4 re-quant does not pad expert dims (the FP8 "
                f"loader writes the unpadded checkpoint shapes): got "
                f"{hidden_size}x{intermediate_size_per_partition}"
            )
        self.fp8_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.weight_block_size = [1, MXFP4_BLOCK_SIZE]
        for name, param in list(layer.named_parameters(recurse=False)):
            meta = torch.nn.Parameter(
                torch.empty_like(param, device="meta"), requires_grad=False
            )
            meta.__dict__.update(param.__dict__)
            layer.register_parameter(name, meta)
        initialize_online_processing(layer)

    def _requant(self, weight: torch.Tensor, scale_inv: torch.Tensor):
        num_experts, n, k = weight.shape
        out = torch.empty(
            (num_experts, n, k // 2), dtype=torch.uint8, device=weight.device
        )
        out_scale = torch.empty(
            (num_experts, n, k // MXFP4_BLOCK_SIZE),
            dtype=torch.uint8,
            device=weight.device,
        )
        for e in range(num_experts):
            bf16 = dequant_block_fp8_to_bf16(
                weight[e], scale_inv[e], self.fp8_block_size
            )
            q, s = mxfp4_quantize(bf16)
            out[e] = q.view(torch.uint8)
            out_scale[e] = s.view(torch.uint8)
        return out, out_scale

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            raise ValueError(
                "VLLM_FP8_MOE_REQUANT=mxfp4 needs an MXFP4 MoE backend, and none "
                "is available on this platform."
            )
        w13, w13_scale = self._requant(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        w2, w2_scale = self._requant(
            layer.w2_weight.data, layer.w2_weight_scale_inv.data
        )
        delattr(layer, "w13_weight_scale_inv")
        delattr(layer, "w2_weight_scale_inv")
        # Hand the MXFP4 tensors over under the names the kernel-format
        # conversion expects (it replaces or deletes them per backend).
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        torch.accelerator.empty_cache()
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, None, None)
        layer._already_called_process_weights_after_loading = True
