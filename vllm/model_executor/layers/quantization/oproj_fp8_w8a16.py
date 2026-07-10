# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time W8A16 FP8 requant for an otherwise-bf16 dense attention O-proj.

This is a *weight-layout transform*: the attention output projection
(``self_attn.o_proj``) weight is stored in the checkpoint as bf16 (it is on the
quant config's ``ignore`` list), so it normally gets
:class:`UnquantizedLinearMethod`.  When
:data:`~vllm.envs.VLLM_OPROJ_FP8_W8A16` is set, we instead requantize that
weight to FP8 (e4m3) once at load time (halving the bytes streamed from HBM per
decode step) and dispatch the matmul through the in-tree **weight-only
FP8-Marlin** GEMM: the FP8 weight is dequantized on-chip and multiplied by
**bf16 activations** (W8A16 — activations are NOT quantized).

This is the W8A16-FP8 analog of the shipped
:class:`~vllm.model_executor.layers.quantization.qkv_fp8_requant.QkvFp8RequantLinearMethod`
(which uses a native-FP8 W8A8 CUTLASS GEMM for the QKV projection).  O-proj uses
the Marlin W8A16 path instead because NVFP4-W4A16 on these shapes exceeds the
lossy relL2 gate; per-channel FP8 keeps the weight bf16-faithful.

No CUDA kernel is authored here: ``process_weights_after_loading`` calls the
in-tree :func:`prepare_fp8_layer_for_marlin` to repack the FP8 weight into the
Marlin layout, and ``apply`` calls :func:`apply_fp8_marlin_linear`, which wraps
the ``vllm._custom_ops.marlin_gemm`` custom op.  Both are torch.compile / CUDA
graph safe by construction (registered custom op, no ``data_ptr()`` on a traced
tensor, no in-place mutation of a traced tensor, persistent workspace).
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

FP8_E4M3_MAX = 448.0


class OprojFp8W8A16LinearMethod(LinearMethodBase):
    """Requant a bf16 O-proj weight to FP8 e4m3 (W8A16) at load time.

    The weight is created as bf16 so the checkpoint loads unchanged, then
    requantized to FP8 with per-output-channel scales in
    :meth:`process_weights_after_loading` and repacked into the Marlin layout.
    ``apply`` runs the weight-only FP8-Marlin GEMM against bf16 activations
    (activations are never quantized).
    """

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
        weight_loader = extra_weight_attrs.pop("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        # Attributes prepare_fp8_layer_for_marlin / apply_fp8_marlin_linear read.
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        # Create the weight as bf16 (matches the ignore-listed checkpoint tensor
        # so the standard weight_loader path is unchanged). Layout is [N, K]
        # (output x input), row-major.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Per-output-channel FP8 e4m3 weight requant (done once, off the timed
        # path). Weight is loaded as [N, K] (output x input); each output row
        # gets its own scale for maximum bf16 fidelity.
        w_f = layer.weight.data.to(torch.float32)
        scale = (w_f.abs().amax(dim=1) / FP8_E4M3_MAX).clamp(min=1e-12)  # [N]
        w_fp8 = (
            (w_f / scale.unsqueeze(1))
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )  # [N, K]
        # prepare_fp8_layer_for_marlin with size_k_first=True expects the weight
        # in (K, N) layout — same canonicalization the in-tree W8A16 FP8 scheme
        # does (compressed_tensors_w8a16_fp8.py stores weight.t()).
        replace_parameter(layer, "weight", w_fp8.t().contiguous())
        # Channel-wise scale, one per output feature (N). prepare converts this
        # to Marlin's channel-wise layout via scales.view(1, N).
        layer.weight_scale = torch.nn.Parameter(
            scale.to(torch.float32), requires_grad=False
        )

        # Repack the FP8 weight + scales into the Marlin kernel layout.
        # size_k_first=True (non-block), input_dtype=None => W8A16 (bf16 acts):
        # apply_fp8_marlin_linear leaves activations unquantized (a_scales=None).
        prepare_fp8_layer_for_marlin(layer, size_k_first=True, input_dtype=None)

        logger.info_once(
            "VLLM_OPROJ_FP8_W8A16 active: dense O-proj requantized "
            "bf16->fp8_e4m3 W8A16 (FP8-Marlin, per-output-channel weight scale, "
            "bf16 activations)."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Weight-only FP8-Marlin GEMM: bf16 activations x dequantized FP8 weight.
        # marlin_gemm is a registered custom op (fake impl present, no data_ptr
        # on traced tensors, workspace is persistent) => torch.compile / CUDA
        # graph safe. o_proj has no bias for this model, but pass it through for
        # generality (prepare already permuted layer.bias if it existed).
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            input_dtype=None,
            bias=bias,
        )


def is_oproj_fp8_w8a16_supported() -> bool:
    """FP8-Marlin needs compute capability >= 7.5 (Turing and up)."""
    return current_platform.is_cuda() and current_platform.has_device_capability(75)
