# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time FP8 requant for an otherwise-bf16 dense QKV projection.

This is a *weight-layout transform*: the QKV projection weight is stored in the
checkpoint as bf16 (it is on the quant config's ``ignore`` list), so it normally
gets :class:`UnquantizedLinearMethod`.  When
:data:`~vllm.envs.VLLM_QKV_FP8_REQUANT` is set, we instead requantize that weight
to FP8 (e4m3) once at load time (halving the bytes streamed from HBM per decode
step) and dispatch the matmul through the pre-compiled SM120 FP8 CUTLASS GEMM,
with per-token dynamic activation quantization done in-graph.

No CUDA kernel is authored here: ``apply`` reuses the two custom ops the in-tree
``Fp8LinearMethod`` already uses under torch.compile
(``vllm._custom_ops.scaled_fp8_quant`` + ``vllm._custom_ops.cutlass_scaled_mm``),
so the torch.compile contract is satisfied by construction (functional op reuse,
no in-place mutation, no ``data_ptr()`` on a traced tensor).
"""

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

FP8_E4M3_MAX = 448.0


class QkvFp8RequantLinearMethod(LinearMethodBase):
    """Requant a bf16 QKV weight to FP8 e4m3 at load time (per-tensor scale).

    The weight is created as bf16 so the checkpoint loads unchanged, then
    requantized to FP8 in :meth:`process_weights_after_loading`.  ``apply``
    does per-token dynamic activation quant and dispatches the FP8 CUTLASS GEMM.
    """

    def __init__(self) -> None:
        # Output dtype = model activation dtype (bf16 here). cutlass_scaled_mm
        # only supports bf16/fp16 out; bind explicitly instead of relying on
        # the ambient default dtype.
        out_dtype = torch.bfloat16
        try:
            model_dtype = get_current_vllm_config().model_config.dtype
            if model_dtype in (torch.bfloat16, torch.float16):
                out_dtype = model_dtype
        except Exception:
            pass
        self.out_dtype = out_dtype

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
        # Create the weight as bf16 (matches the ignore-listed checkpoint tensor
        # so the standard weight_loader path is unchanged).
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        layer.orig_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Per-tensor FP8 e4m3 weight requant (done once, off the timed path).
        # Weight is loaded as [N, K] (output x input). w_fp8 is [N, K] row-major.
        w_f = layer.weight.data.to(torch.float32)
        scale = (w_f.abs().amax() / FP8_E4M3_MAX).clamp(min=1e-12)
        w_fp8 = (w_f / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        # cutlass_scaled_mm wants B column-major [K, N] (B.stride(0) == 1).
        # w_fp8.t() gives [K, N] with stride(0) == 1 — the exact layout, same
        # idiom as the in-tree Fp8LinearMethod (fp8.py stores weight.t()).
        replace_parameter(layer, "weight", w_fp8.t())
        layer.weight_scale = Parameter(
            scale.reshape(1, 1).to(torch.float32), requires_grad=False
        )
        logger.info_once(
            "VLLM_QKV_FP8_REQUANT active: dense QKV projection requantized "
            "bf16->fp8_e4m3 (per-tensor weight, per-token dynamic activation)."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Per-token dynamic activation quant, in-graph (functional, no mutation).
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp8, x_scale = ops.scaled_fp8_quant(
            x_2d, scale=None, use_per_token_if_dynamic=True
        )
        out = ops.cutlass_scaled_mm(
            x_fp8,
            layer.weight,
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            out_dtype=self.out_dtype,
            bias=bias,
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
