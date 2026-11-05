# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online int4 weight-only quantization.

``Int4PerGroup32AsymLMHeadMethod`` quantizes an fp16/bf16 lm_head at load time
(``quantization_config.lm_head="int4_per_group_32_asym"``). Decode reads the
whole lm_head every step, so on bandwidth-bound GPUs a full-precision head is
one of the largest per-token costs even when the checkpoint quantizes every
other linear layer.
"""

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static32Asym,
    pack_quantized_values_into_int32,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

_UINT4_MAX = 15
# Zero points pack 8 output rows per int32 word.
_PACK_FACTOR = 8


def quantize_int4_per_group_asym(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uint4 quantization of the rows of ``weight`` [N, K].

    Returns the codes [N, K] uint8, the scales [N, K // group_size] in the
    weight dtype and the zero points [N, K // group_size] uint8, such that
    ``(codes - zero_point) * scale`` approximates ``weight``. Each group's
    range is widened to include 0, so the zero point always fits in uint4.
    """
    n, k = weight.shape
    groups = weight.float().view(n, k // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0)
    w_max = groups.amax(dim=-1).clamp(min=0)
    # Round against the stored scale, so the codes match what the kernel
    # dequantizes.
    scale = (
        ((w_max - w_min) / _UINT4_MAX)
        .clamp(min=torch.finfo(weight.dtype).tiny)
        .to(weight.dtype)
    )
    scale_f = scale.float()
    zero_point = (-w_min / scale_f).round().clamp(0, _UINT4_MAX)
    codes = ((groups / scale_f.unsqueeze(-1)).round() + zero_point.unsqueeze(-1)).clamp(
        0, _UINT4_MAX
    )
    return codes.to(torch.uint8).view(n, k), scale, zero_point.to(torch.uint8)


@torch.no_grad()
def quantize_int4_to_mp_layout(
    weight: torch.Tensor, group_size: int, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` [N, K] into the `MPLinearKernel` uint4 layout.

    Returns the codes packed along K (int32 [N, K // 8]), the scales
    [N, K // group_size] and the zero points packed along N
    (int32 [N // 8, K // group_size]), the layout of compressed-tensors
    asymmetric W4A16 checkpoints. Rows are processed in chunks to bound the
    fp32 working set.
    """
    n, k = weight.shape
    packed = torch.empty(n, k // _PACK_FACTOR, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    zero_points = torch.empty(
        n, k // group_size, dtype=torch.uint8, device=weight.device
    )
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        codes, scales[rows], zero_points[rows] = quantize_int4_per_group_asym(
            weight[rows], group_size
        )
        packed[rows] = pack_quantized_values_into_int32(
            codes.to(torch.int32), scalar_types.uint4, packed_dim=1
        )
    packed_zero_points = pack_quantized_values_into_int32(
        zero_points.to(torch.int32), scalar_types.uint4, packed_dim=0
    )
    return packed, scales, packed_zero_points


class Int4PerGroup32AsymLMHeadMethod(OnlineLinearBase):
    """Weight-only asymmetric uint4 lm_head, one scale and zero point per 32
    input channels, on the platform's mixed-precision linear kernel
    (`choose_mp_linear_kernel`).

    An untied head loads its fp16/bf16 weight like any online linear layer and
    replaces it with the int4 copy. A tied head shares the input embeddings'
    weight, which stays unchanged: the int4 copy is added next to it.
    """

    weight_quant_key = kInt4Static32Asym
    activation_quant_key = None
    group_size = 32

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        # Fail before loading the checkpoint, not after.
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "int4 lm_head quantization needs an fp16/bf16 lm_head, got "
                f"{params_dtype}."
            )
        if (
            input_size_per_partition % self.group_size
            or output_size_per_partition % _PACK_FACTOR
        ):
            raise ValueError(
                "int4 lm_head quantization needs a hidden size divisible by "
                f"{self.group_size} and a vocab shard divisible by {_PACK_FACTOR}, "
                f"got {output_size_per_partition} x {input_size_per_partition}."
            )
        self.mp_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=scalar_types.uint4,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=True,
        )
        # Raises if no kernel on this platform can run the layer.
        self.kernel_cls: type[MPLinearKernel] = choose_mp_linear_kernel(self.mp_config)
        self.kernel: MPLinearKernel | None = None
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def tie_weights(self, layer: Module, embed_tokens: Module) -> Module:
        # The head reads the embeddings' loaded weight. Nothing is loaded into
        # the head itself, so it leaves layerwise loading; the model loader's
        # process_weights_after_loading pass quantizes it.
        layer.weight = embed_tokens.weight
        get_layerwise_info(layer).reset()
        layer.int4_tied_weight = True
        return layer

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        tied = getattr(layer, "int4_tied_weight", False)
        weight = self.get_weight_for_quantization(layer)
        n, k = weight.shape
        packed, scales, zero_points = quantize_int4_to_mp_layout(
            weight, self.group_size
        )
        set_weight_attrs(packed, {"packed_dim": 1})
        set_weight_attrs(scales, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(
            zero_points, {"input_dim": 1, "output_dim": 0, "packed_dim": 0}
        )
        if tied:
            # `weight` is the input embeddings' parameter: keep it, and add
            # the int4 copy under its own name.
            w_q_name = "weight_packed"
            w_q = Parameter(packed, requires_grad=False)
            set_weight_attrs(w_q, {"input_dim": 1, "output_dim": 0, "packed_dim": 1})
            layer.register_parameter(w_q_name, w_q)
            logger.info_once(
                "The lm_head is tied to the input embeddings, which keep their "
                "%s weight: the int4 lm_head adds %.2f GiB per rank instead of "
                "saving memory.",
                weight.dtype,
                (packed.nbytes + scales.nbytes + zero_points.nbytes) / 2**30,
            )
        else:
            # The loaded parameter already reports input_dim=1, output_dim=0.
            w_q_name = "weight"
            replace_parameter(layer, w_q_name, packed)
        replace_parameter(layer, "weight_scale", scales)
        replace_parameter(layer, "weight_zero_point", zero_points)

        self.kernel = self.kernel_cls(
            self.mp_config,
            w_q_param_name=w_q_name,
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
        )
        self.kernel.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)
        logger.info_once(
            "Quantized the lm_head to int4_per_group_32_asym (%d x %d per rank) "
            "with %s.",
            n,
            k,
            self.kernel_cls.__name__,
        )
        layer._already_called_process_weights_after_loading = True

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.kernel is not None
        return self.kernel.apply_weights(layer, x, bias)
