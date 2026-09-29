# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load canonical ModelOpt Q8_0 weights for Humming and embeddings."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.parameter import ModelWeightParameter


def _check_q8_0_shape(blocks: torch.Tensor, shape: tuple[int, ...]) -> None:
    if (
        blocks.dtype != torch.uint8
        or blocks.shape != (*shape[:-1], shape[-1] // 32, 34)
        or shape[-1] % 32
    ):
        raise ValueError("Q8_0 weights must be uint8 [..., K/32, 34] blocks")


def _validate_q8_0_blocks(blocks: torch.Tensor, shape: tuple[int, ...]) -> None:
    _check_q8_0_shape(blocks, shape)
    scales = blocks[..., :2].contiguous().view(torch.float16)
    if not (torch.isfinite(scales) & (scales >= 0)).all():
        raise ValueError("Q8_0 weights contain invalid FP16 scales")
    if blocks[..., 2:].view(torch.int8).amin() < -127:
        raise ValueError("Q8_0 codes must be within [-127, 127]")


def dequantize_q8_0(
    blocks: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Decode canonical Q8_0 blocks without changing their stored codes."""
    if validate:
        _validate_q8_0_blocks(blocks, shape)
    else:
        _check_q8_0_shape(blocks, shape)
    scales = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).float()
    codes = blocks[..., 2:].contiguous().view(torch.int8)
    return (codes.float() * scales.unsqueeze(-1)).reshape(shape).to(dtype)


class ModelOptQ80EmbeddingMethod(UnquantizedEmbeddingMethod):
    """Gather Q8_0 token rows, then decode only the selected rows."""

    supports_pre_processed_weights = False

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("Q8_0 embeddings are limited to TP=1")
        if input_size_per_partition % 32 or params_dtype != torch.bfloat16:
            raise ValueError("Q8_0 embeddings require K divisible by 32 and BF16")
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.zeros(
                    (sum(output_partition_sizes), input_size_per_partition // 32, 34),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            ),
        )

    def process_weights_after_loading(self, layer):
        blocks = layer.weight
        _validate_q8_0_blocks(blocks, (blocks.shape[0], layer.embedding_dim))

    def embedding(self, layer, input_):
        packed_rows = layer.weight.view(layer.weight.shape[0], -1)
        selected = F.embedding(input_, packed_rows).view(
            *input_.shape, layer.embedding_dim // 32, 34
        )
        return dequantize_q8_0(
            selected,
            (*input_.shape, layer.embedding_dim),
            layer.params_dtype,
            validate=False,
        )

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("Q8_0 embedding weights are not a linear GEMM")


@register_weight_loader_v2_supported_method
class ModelOptQ80LinearMethod(LinearMethodBase):
    """Load 32-value Q8_0 blocks without requantizing their codes or scales."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("Q8_0 Humming support is limited to TP=1")
        if input_size_per_partition % 32 or params_dtype != torch.bfloat16:
            raise ValueError("Q8_0 requires K divisible by 32 and BF16 activations")
        if (
            getattr(layer, "has_bias", False)
            or getattr(layer, "bias", None) is not None
        ):
            raise ValueError("Q8_0 Humming support is limited to bias-free layers")

        layer.input_size = input_size
        layer.output_size = output_size
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        layer.has_bias = False
        weight_loader = extra_weight_attrs["weight_loader"]

        def load_q8_0(param, loaded_weight, *args, **kwargs):
            if (
                loaded_weight.dtype != torch.uint8
                or loaded_weight.ndim != 3
                or loaded_weight.shape[-1] != 34
            ):
                raise ValueError("Q8_0 weights must be uint8 [N, K/32, 34] blocks")
            return weight_loader(param, loaded_weight, *args, **kwargs)

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                # An unloaded block has an invalid FP16 scale (NaN).
                data=torch.full(
                    (
                        layer.output_size_per_partition,
                        input_size_per_partition // 32,
                        34,
                    ),
                    255,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=load_q8_0,
            ),
        )

    def process_weights_after_loading(self, layer):
        blocks = layer.weight.data
        if blocks.dtype != torch.uint8 or blocks.ndim != 3 or blocks.shape[-1] != 34:
            raise ValueError("Expected uint8 [N, K/32, 34] Q8_0 blocks")
        scales = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if not (torch.isfinite(scales) & (scales >= 0)).all():
            raise ValueError("Q8_0 weights contain missing or invalid FP16 scales")
        codes = blocks[..., 2:].contiguous().view(torch.int8).flatten(-2)
        if (codes == -128).any():
            raise ValueError("Q8_0 codes must be within [-127, 127]")

        # Humming stores offset-unsigned INT8 codes; the XOR is reversible.
        layer.weight = Parameter(codes.view(torch.uint8) ^ 128, requires_grad=False)
        layer.weight_scale = Parameter(scales, requires_grad=False)
        convert_linear_layer_to_humming_standard(
            layer, {"weight": "weight", "weight_scale": "weight_scale"}
        )
        self.layer_config = prepare_humming_linear_layer_config(
            layer,
            {
                "quant_method": "humming",
                "dtype": "uint8",
                "group_size": 32,
                "scale_dtype": "float16",
            },
        )
        if layer.weight_scale.dtype != torch.float16:
            raise RuntimeError("Humming must preserve Q8_0 FP16 group scales")
        self.compute_config = get_humming_linear_compute_config()
        self.locks = torch.zeros(1024, dtype=torch.int32, device=layer.weight.device)

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("Q8_0 Humming support is limited to bias-free layers")
        return apply_humming_linear(
            layer,
            x,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
