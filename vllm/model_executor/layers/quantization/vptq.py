# Supports VPTQ compression, see https://arxiv.org/abs/2409.17066

import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class MetaData:

    def __init__(self):
        self.num_codebooks = 1
        self.num_centroids = 0
        self.num_res_centroids = 0
        self.vector_len = 0
        self.group_size = 0
        self.output_size = 0


# unpack the packed tensor to get the indices and residual indices
def unpack_index_tensor(
    pack_tensor: torch.Tensor,
    index_bits: int,
    num_elements: int,
    res_bits: int = 0,
    num_res_elements: int = 0,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    total_bits = index_bits + res_bits
    wf = torch.arange(0, 32, 1, device=pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, -1), wf)
    torch.bitwise_and(out, 1, out=out)
    pad_size = (pack_tensor.shape[-1] * 32) % (index_bits * num_elements +
                                               res_bits * num_res_elements)
    out = out.reshape(*pack_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*pack_tensor.shape[:-1], -1, total_bits)
    wf1 = torch.arange(0, total_bits, 1,
                       device=pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)

    unpack_indice = out.to(torch.uint64).view(torch.int64)

    indices = (unpack_indice & ((1 << index_bits) - 1)).view(torch.uint64).to(
        torch.int64)

    # indices = indices.squeeze()

    if res_bits > 0:
        res_indices = ((unpack_indice >> index_bits) &
                       ((1 << index_bits) - 1)).view(torch.uint64).to(
                           torch.int64)
        # res_indices = res_indices.squeeze()
    else:
        res_indices = None

    return indices, res_indices


# dequantize the weight from the quantization codes
def dequantize_weight(
        indices: torch.
    IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
        codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        res_codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        metadata: MetaData) -> torch.Tensor:
    """
    Decode float weights from quantization codes. Differentiable.
    :param codes: tensor of integer quantization codes, shape 
        [*dims, num_out_groups, num_in_groups, num_codebooks]
    :param codebooks: tensor of vectors for each quantization code, 
        [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, must be 
        broadcastble with 
        [*dims, out_groups, num_in_groups, out_group_size, in_group_size]
    :return: reconstructed weight tensor of shape 
        [*dims, num_in_groups*group_size]
    """
    output_size = metadata.output_size
    num_codebooks = indices.shape[0]
    num_centroids = metadata.num_centroids
    num_res_centroids = metadata.num_res_centroids
    vector_len = metadata.vector_len
    group_size = metadata.group_size

    codebooks = codebooks.view(num_codebooks, num_centroids, vector_len)
    res_codebooks = res_codebooks.view(
        num_codebooks, num_res_centroids,
        vector_len) if num_res_centroids > 0 else None
    index_bits = math.ceil(math.log2(num_centroids))
    enable_residual = num_res_centroids > 0
    index_res_bits = math.ceil(
        math.log2(num_res_centroids)) if enable_residual else 0

    # print(f"indices shape: {indices.shape}")
    indices, res_indices = unpack_index_tensor(
        pack_tensor=indices,
        index_bits=index_bits,
        num_elements=group_size,
        res_bits=index_res_bits,
        num_res_elements=group_size,
        index_dtype=torch.uint16,
    )

    indices = indices.unsqueeze(-1).expand(-1, -1, -1, vector_len)
    indices = indices.reshape(num_codebooks, -1, vector_len)
    selected_centroids = torch.gather(codebooks, 1, indices)
    selected_centroids = selected_centroids.view(num_codebooks, -1, group_size,
                                                 vector_len)
    selected_centroids = selected_centroids.permute(0, 1, 3, 2)

    qweight = selected_centroids.reshape(num_codebooks, -1, group_size)
    qweight = qweight.permute(1, 0, 2)
    qweight = qweight.reshape(-1, num_codebooks * group_size)

    if enable_residual:
        res_codebooks = res_codebooks.view(num_codebooks, num_res_centroids,
                                           vector_len)
        res_indices = res_indices.unsqueeze(-1).expand(-1, -1, -1, vector_len)
        res_indices = res_indices.reshape(num_codebooks, -1, vector_len)
        selected_res_centroids = torch.gather(res_codebooks, 1, res_indices)
        selected_res_centroids = selected_res_centroids.reshape(
            num_codebooks, -1, group_size, vector_len)
        selected_res_centroids = selected_res_centroids.permute(0, 1, 3, 2)
        qweight = qweight + (selected_res_centroids.reshape(
            num_codebooks, -1, group_size).permute(1, 0, 2).reshape(
                -1, num_codebooks * group_size))

    padding = -output_size % vector_len
    if padding > 0:
        qweight = qweight[:-padding, :]

    enable_perm = perm is not None
    if enable_perm:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(torch.int64))
        qweight = qweight[:, invert_perm]

    enable_norm = weight_scale is not None
    if enable_norm:
        qweight = qweight * weight_scale
        qweight = qweight + weight_bias

    return qweight


# do the quantized matmul in a generic way, it's quite slow
def generic_dequantize_gemm(
        input: torch.Tensor,  #  [..., in_features]
        indices: torch.
    IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
        codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        res_codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        bias: Optional[torch.Tensor],
        metadata: MetaData) -> torch.Tensor:
    dequantized_weight = dequantize_weight(
        indices,
        codebooks,
        res_codebooks,
        weight_scale,
        weight_bias,
        perm,
        metadata,
    )
    return F.linear(input, dequantized_weight, bias)


# call the optimized version of the dequantized matmul
def optimized_dequantize_gemm(
        input: torch.Tensor,  #  [..., in_features]
        indices: torch.
    IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
        codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        res_codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        bias: Optional[torch.Tensor],
        metadata: MetaData) -> torch.Tensor:
    codebooks = codebooks.view(metadata.num_codebooks, metadata.num_centroids,
                               metadata.vector_len)
    res_codebooks = res_codebooks.view(
        metadata.num_codebooks, metadata.num_res_centroids,
        metadata.vector_len) if metadata.num_res_centroids > 0 else None
    if input.numel() // input.shape[-1] < 3:
        return ops.vptq_gemm(
            input, indices, codebooks, weight_scale, weight_bias,
            [metadata.vector_len, weight_scale.shape[0], metadata.output_size],
            None, res_codebooks, None, None, perm, bias)
    if perm is None:
        invert_perm = None
    else:
        invert_perm = torch.argsort(perm.view(torch.uint16).to(
            torch.int64)).to(torch.uint16).view(torch.int16)
    dequantized_weight = ops.vptq_dequant(
        indices, codebooks, weight_scale, weight_bias,
        [metadata.vector_len, weight_scale.shape[0], metadata.output_size],
        None, res_codebooks, None, None, invert_perm)
    return F.linear(input, dequantized_weight, bias)


# Handle QKV projection and gate-up projection
# we will do Q K V separately
def merged_dequantize_gemm(
        input: torch.Tensor,  #  [..., in_features]
        indices: torch.
    IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
        codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        res_codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
        weight_scale: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        weight_bias: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        perm: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
        output_partition_sizes: List[int],
        bias: Optional[torch.Tensor],
        metadata: MetaData) -> torch.Tensor:
    output_shape = input.shape[:-1] + (sum(output_partition_sizes), )
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    indice_sizes = getattr(indices, "shard_sizes", [])
    output_extra_offsets = getattr(indices, "output_offset", [])
    num_codebooks = indices.shape[0]

    tp_rank = get_tensor_model_parallel_rank()
    input_size = input.shape[-1]
    input_offset = 0
    indice_offset = 0
    output_offset = 0
    codebooks_offset = 0

    num_linears = len(output_partition_sizes)
    for linear_idx, output_size, indice_size in zip(range(num_linears),
                                                    output_partition_sizes,
                                                    indice_sizes):
        metadata.output_size = output_size
        if len(output_extra_offsets) > 1:
            metadata.output_size = output_size + output_extra_offsets[tp_rank][
                linear_idx]
        shard_output = optimized_dequantize_gemm(
            input, indices.narrow(1, indice_offset, indice_size),
            codebooks.narrow(0, codebooks_offset, num_codebooks),
            res_codebooks.narrow(0, codebooks_offset, num_codebooks),
            weight_scale.narrow(0, input_offset, input_size),
            weight_bias.narrow(0, input_offset, input_size),
            perm.narrow(0, input_offset, input_size) if perm is not None else
            None, bias if bias is None else bias.narrow(
                0, output_offset, output_size), metadata)

        output_slice = output.narrow(-1, output_offset, output_size)
        if tp_rank > 0 and len(output_extra_offsets) > tp_rank:
            shard_output = shard_output.narrow(
                -1, output_extra_offsets[tp_rank][linear_idx], output_size)
        assert (output_slice.shape == shard_output.shape)
        output_slice.copy_(shard_output)
        output_offset += output_size
        indice_offset += indice_size
        codebooks_offset += num_codebooks
        input_offset += input_size
    return output


class VPTQConfig(QuantizationConfig):
    """Config class for VPTQ.

    Reference: https://github.com/microsoft/VPTQ
    """

    def __init__(
        self,
        config_for_layers: Dict[str, Dict[str, Any]],
        shared_layer_config: Dict[str, Dict[str, Any]],
    ) -> None:
        self.config_for_layers = config_for_layers
        self.shared_layer_config = shared_layer_config

    def __repr__(self) -> str:
        return (f"VPTQConfig(config_for_layers={self.config_for_layers}, "
                f"shared_layer_config={self.shared_layer_config})")

    @classmethod
    def get_name(cls) -> str:
        return "vptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VPTQConfig":
        config_for_layers: Dict[str, Any] = {}
        shared_layer_config: Dict[str, Any] = {}
        if "config_for_layers" in config:
            config_for_layers = cls.get_from_keys(config,
                                                  ["config_for_layers"])
        if "shared_layer_config" in config:
            shared_layer_config = cls.get_from_keys(config,
                                                    ["shared_layer_config"])
        assert len(config_for_layers) > 0 or len(shared_layer_config) > 0, \
            "VPTQConfig must have at least one of 'config_for_layers'\
             or 'shared_layer_config'"

        return cls(config_for_layers, shared_layer_config)

    def get_config_for_key(self, prefix, key):
        merged_name = '.'.join([prefix, key])
        if merged_name in self.config_for_layers:
            return self.config_for_layers[merged_name]
        elif key in self.shared_layer_config:
            return self.shared_layer_config[key]
        else:
            raise ValueError(f"Cannot find config for ({prefix}, {key})")

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["VPTQLinearMethod"]:
        if isinstance(layer, LinearBase):
            linear_name = prefix.split(".")[-1]
            base_name = prefix[:prefix.rfind('.')]
            if linear_name == "qkv_proj":
                quant_config = {
                    "q_proj": self.get_config_for_key(base_name, "q_proj"),
                    "k_proj": self.get_config_for_key(base_name, "k_proj"),
                    "v_proj": self.get_config_for_key(base_name, "v_proj"),
                }
            elif linear_name == "gate_up_proj":
                quant_config = {
                    "gate_proj": 
                    self.get_config_for_key(base_name, "gate_proj"),
                    "up_proj": self.get_config_for_key(base_name, "up_proj"),
                }
            else:
                quant_config = self.get_config_for_key(base_name, linear_name)
            return VPTQLinearMethod(quant_config)
        return None


class VPTQLinearMethod(LinearMethodBase):
    """Linear method for VPTQ.

    Args:
        quant_config: The VPTQ quantization config.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        self.quant_config = quant_config

    @staticmethod
    def quantized_weight_loader(
            indice_sizes,
            narrow_dim=1):  # specific for layer.indices/weight_scale&bias

        def wrap_weight_loader(param: torch.nn.Parameter,
                               loaded_weight: torch.Tensor,
                               loaded_shard_id: Optional[Union[str,
                                                               int]] = None):
            if isinstance(loaded_shard_id, str):
                _loaded_shard_id = ["q", "k", "v"].index(loaded_shard_id)
            else:
                _loaded_shard_id = loaded_shard_id or 0

            shard_sizes = [i[1] - i[0] for i in indice_sizes]
            offset, end = indice_sizes[_loaded_shard_id]
            param_data = param.data
            if loaded_shard_id is not None:
                param_data = param_data.narrow(
                    narrow_dim,
                    sum(shard_sizes[:_loaded_shard_id]),
                    shard_sizes[_loaded_shard_id],
                )

            # split for TP
            loaded_weight = loaded_weight.narrow(narrow_dim, offset,
                                                 end - offset)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)

        return wrap_weight_loader

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        row_parallel_tp_size = input_size // input_size_per_partition
        col_parallel_tp_size = output_size // sum(output_partition_sizes)

        if params_dtype != torch.half and params_dtype != torch.bfloat16:
            raise ValueError(
                "Only half and bfloat16 are currently supported by vptq")
        quant_config = self.quant_config.get("q_proj", self.quant_config)
        quant_config = quant_config.get("gate_proj", quant_config)

        num_codebooks = quant_config["group_num"]
        num_centroids = quant_config["num_centroids"][1]
        group_size = quant_config["group_size"]
        vector_len = quant_config["vector_lens"][1]
        num_res_centroids = quant_config["num_res_centroids"][1]
        enable_residual = num_res_centroids > 0
        enable_norm = quant_config["enable_norm"]
        enable_perm = quant_config["enable_perm"]
        assert not enable_perm, (
            "perm is not absorbed in this model, please process it \
by `pip install vptq && python -m vptq.tools.pre_process \
--input_path xx --output_path xx`")
        assert input_size == group_size
        group_size = input_size_per_partition
        metadata = MetaData()
        metadata.num_centroids = num_centroids
        metadata.num_res_centroids = num_res_centroids
        metadata.vector_len = vector_len
        metadata.group_size = group_size
        layer.metadata = metadata

        num_linears = len(output_partition_sizes)
        orig_weight_loader = extra_weight_attrs['weight_loader']

        if enable_norm:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                [[(
                    input_size_per_partition * tp_ind,
                    input_size_per_partition * (tp_ind + 1),
                ) for num in output_partition_sizes]
                 for tp_ind in range(row_parallel_tp_size)
                 ][get_tensor_model_parallel_rank() % row_parallel_tp_size],
                0,
            )
            extra_weight_attrs["weight_loader"] = wrapped_weight_loader

            extra_weight_attrs["output_dim"] = 0
            weight_scale = Parameter(torch.empty(input_size_per_partition *
                                                 num_linears,
                                                 dtype=params_dtype),
                                     requires_grad=False)
            weight_bias = Parameter(torch.empty(input_size_per_partition *
                                                num_linears,
                                                dtype=params_dtype),
                                    requires_grad=False)
            set_weight_attrs(weight_scale, extra_weight_attrs)
            set_weight_attrs(weight_bias, extra_weight_attrs)
            layer.register_parameter("weight_scale", weight_scale)
            layer.register_parameter("weight_bias", weight_bias)
            extra_weight_attrs["weight_loader"] = orig_weight_loader

        index_bits = int(math.log2(num_centroids))
        res_index_bits = int(
            math.log2(num_res_centroids)) if enable_residual else 0
        total_index_bits = index_bits + res_index_bits
        packed_groupsize = math.ceil(group_size * total_index_bits / 32)

        indice_sizes = [[(math.floor(num * tp_ind / vector_len),
                          math.ceil(num * (tp_ind + 1) / vector_len))
                         for num in output_partition_sizes]
                        for tp_ind in range(col_parallel_tp_size)]
        tp_output_offset = [[(num * tp_ind) % vector_len
                             for num in output_partition_sizes]
                            for tp_ind in range(col_parallel_tp_size)]
        if col_parallel_tp_size > 1:
            this_rank_indice_sizes = indice_sizes[
                get_tensor_model_parallel_rank()]
        else:
            this_rank_indice_sizes = indice_sizes[0]
        shard_sizes = [i[1] - i[0] for i in this_rank_indice_sizes]
        num_indices = sum(shard_sizes)
        indices = Parameter(torch.empty(
            (num_codebooks, num_indices, packed_groupsize), dtype=torch.int32),
                            requires_grad=False)
        if row_parallel_tp_size == 1:
            wrapped_weight_loader = VPTQLinearMethod.quantized_weight_loader(
                this_rank_indice_sizes)
            extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        set_weight_attrs(
            indices,
            {
                # metadata indicates fixed size concatenated along dim 0
                "output_partition_sizes": output_partition_sizes,
                "output_offset": tp_output_offset,
                "shard_sizes": shard_sizes,
                "input_dim": -1,
            },
        )

        extra_weight_attrs["output_dim"] = 1
        set_weight_attrs(indices, extra_weight_attrs)
        layer.register_parameter("indices", indices)
        extra_weight_attrs['weight_loader'] = orig_weight_loader

        extra_weight_attrs.pop("output_dim")
        extra_weight_attrs["is_metadata"] = True
        centroids = torch.nn.Embedding(num_codebooks * num_linears,
                                       num_centroids * vector_len,
                                       dtype=params_dtype)
        set_weight_attrs(centroids.weight, extra_weight_attrs)
        set_weight_attrs(
            centroids.weight,
            {
                # metadata indicates fixed size concatenated along dim 0
                "codebook_sizes":
                [num_centroids * vector_len for _ in output_partition_sizes],
            },
        )
        layer.centroids = centroids
        # layer.register_parameter("centroids", centroids)
        if enable_residual:
            res_centroids = torch.nn.Embedding(num_codebooks * num_linears,
                                               num_res_centroids * vector_len,
                                               dtype=params_dtype)
            set_weight_attrs(res_centroids.weight, extra_weight_attrs)
            # layer.register_parameter("res_centroids", res_centroids)
            layer.res_centroids = res_centroids
            set_weight_attrs(
                res_centroids.weight,
                {
                    # metadata indicates fixed size concatenated along dim 1
                    "codebook_sizes": [
                        num_res_centroids * vector_len
                        for _ in output_partition_sizes
                    ],
                },
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale if hasattr(layer,
                                                     "weight_scale") else None
        weight_bias = layer.weight_bias if hasattr(layer,
                                                   "weight_bias") else None
        perm = layer.perm if hasattr(layer, "perm") else None
        indices = layer.indices
        output_partition_sizes = getattr(indices, "output_partition_sizes", [])
        centroids = layer.centroids.weight
        res_centroids = layer.res_centroids.weight if hasattr(
            layer, "res_centroids") else None

        # fall back all unoptimized formats
        return merged_dequantize_gemm(x, indices, centroids, res_centroids,
                                      weight_scale, weight_bias, perm,
                                      output_partition_sizes, bias,
                                      layer.metadata)
