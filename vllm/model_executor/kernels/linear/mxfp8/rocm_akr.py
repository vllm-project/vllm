# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1's dense MXFP8 linears on ``amd_kernels.gemm.mxfp8_common`` (the
``gemm`` route): the layers the model marks, preshuffled and compiled at load."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

from .Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from .rocm_native import RocmDotScaledMxfp8LinearKernel

logger = init_logger(__name__)


def _gemm():
    from amd_kernels.gemm.mxfp8_common import gemm

    return gemm


def is_akr_layer(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, "_akr_mxfp8", False))


def akr_codes(q: torch.Tensor, s: torch.Tensor, rows: int, k: int):
    """amd_kernels' MXFP8 codes of a [rows, k] bf16 activation, in the layout its
    consumer GEMM reads (only ``AkrMxfp8LinearKernel`` reads them)."""
    return QuantizedActivation(
        q, s, torch.bfloat16, torch.Size((rows, k)), kMxfp8Dynamic
    )


def mark_akr_linear(layer: torch.nn.Module | None) -> bool:
    """Switch ``layer`` (one of the M3 model's own linears, before the weights load)
    if its kernel is this class; returns whether it switched."""
    if layer is None:
        return False
    # Quark keeps the kernel on the layer's scheme, ModelOpt on the method.
    kernel = getattr(getattr(layer, "scheme", None), "ocp_mx_linear", None)
    if kernel is None:
        kernel = getattr(getattr(layer, "quant_method", None), "kernel", None)
    if not isinstance(kernel, AkrMxfp8LinearKernel):
        return False
    layer._akr_mxfp8 = True
    return True


class AkrMxfp8LinearKernel(RocmDotScaledMxfp8LinearKernel):
    """The stock ROCm kernel, except that marked layers run amd_kernels."""

    # the weights are shuffled at load; already-processed weights must not be
    supports_pre_processed_weights = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.models.minimax_m3.amd import akr_config

        if not akr_config.on("gemm"):
            return False, "the amd_kernels gemm route is off"
        return super().is_supported(compute_capability)

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "no batched MXFP8"
        if tuple(c.weight_shape) not in _gemm().tuned_shapes(torch.bfloat16):
            return False, f"amd_kernels has no tuned rows for {tuple(c.weight_shape)}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if not is_akr_layer(layer):
            return
        G = _gemm()
        # the model plans its fused norms on the marks: a layer cannot fall back now
        if not G.preshuffle_mxfp8_weight_(layer.weight.data, layer.weight_scale.data):
            raise ValueError(
                "amd_kernels gemm: unexpected MXFP8 weight layout "
                f"{tuple(layer.weight.shape)} / {tuple(layer.weight_scale.shape)} "
                "(a block32 checkpoint?); disable the gemm route"
            )
        logger.info_once(
            "Using amd_kernels MXFP8 GEMMs for the M3 dense linears (amd_kernels gemm)"
        )
        # compile at load (amd_kernels' compile watch arms when start-up ends)
        G.warmup_mxfp8_gemm_weight(
            layer.weight, layer.weight_scale, _max_num_tokens(), torch.bfloat16
        )

    @staticmethod
    def supports_quantized_input(layer: torch.nn.Module) -> bool:
        # a switched layer reads amd_kernels' codes only (akr_codes), never the
        # stock norm's row-major ones
        if is_akr_layer(layer):
            return False
        return RocmDotScaledMxfp8LinearKernel.supports_quantized_input(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not is_akr_layer(layer):
            return super().apply_weights(layer, x, bias)
        G = _gemm()
        w, ws = layer.weight, layer.weight_scale
        n, k = w.shape
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:  # akr_codes
            lead = qa.orig_shape[:-1]
            out = G.mxfp8_gemm_prequant(
                qa.data, qa.scale, w, ws, qa.orig_dtype, lead.numel()
            )
        else:
            assert isinstance(x, torch.Tensor)
            lead = x.shape[:-1]
            out = G.mxfp8_linear(x.reshape(-1, k), w, ws, x.dtype)
        assert bias is None  # the M3 linears have none
        return out.reshape(*lead, n)


def _max_num_tokens() -> int:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return 32768 if config is None else config.scheduler_config.max_num_batched_tokens
