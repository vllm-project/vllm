from typing import Any, Dict, List, Optional

import gguf
import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.utils import set_weight_attrs


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, ) -> None:
        pass

    def __repr__(self) -> str:
        return ("GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return GGUFEmbeddingMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
                  qweight_type: int) -> torch.Tensor:
    # use dequantize mulmat for IQmatrix, mmq for k-quants
    if qweight_type >= 16:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape)
        y = x @ weight.T
    else:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    return y


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = UninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "shard_size": {},
                "shard_id": [],
            })
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(len(output_partition_sizes),
                                             dtype=torch.uint8),
                                 requires_grad=False)
        set_weight_attrs(
            qweight_type, {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True
            })
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        shard_size = getattr(layer.qweight, "shard_size", None)
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id and shard_size:
            result = []
            offset = 0
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            for id in shard_id:
                shard_weight = layer.qweight[
                    offset:offset +
                    shard_size[id][0], :shard_size[id][1]].contiguous()
                qweight_type = layer.qweight_type.shard_weight_type[id]
                result.append(_fuse_mul_mat(x, shard_weight, qweight_type))
                offset += shard_size[id][0]
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = _fuse_mul_mat(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module,
                  x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        hidden_size = qweight.shape[1] // type_size * block_size
        if qweight_type < 2:
            return torch.embedding(qweight, x)
        x_flat = x.flatten()
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(quant, qweight_type, hidden_size,
                                      x_flat.shape[0])
        return dequant.view(*x.shape, hidden_size)
