from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

ACTIVATION_SCHEMES = ["static", "dynamic"]


class FP8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_serialized = is_serialized
        assert activation_scheme in ACTIVATION_SCHEMES
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: PyTorch 2.3.0+ is required to run FP8 on
        # SM 89 (e.g. Ada) GPUs. Specifically, this PR has to
        # be included: https://github.com/pytorch/pytorch/pull/118881
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_serialized=is_serialized,
                   activation_scheme=activation_scheme)

    def get_linear_method(self) -> "Fp8LinearMethod":
        return Fp8LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/scale activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)
       
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes
        
        # WEIGHT
        weight_dtype = torch.float8_e4m3fn if self.quant_config.is_serialized else params_dtype
        weight = Parameter(
            torch.empty(output_size_per_partition,
                        input_size_per_partition,
                        dtype=weight_dtype),
            requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1, "output_dim": 0,
        })

        # SCALES
        #   We only need to load scales if the model is serialized FP8.
        #   Otherwise, scale creation is delayed until `process_weights_after_loading`.
        if self.quant_config.is_serialized:
            # WEIGHT SCALE
            weight_scale = Parameter(
                torch.empty(len(output_partition_sizes), dtype=torch.float32),
                requires_grad=False)
            layer.register_parameter("weight_scale", weight_scale)
            set_weight_attrs(weight_scale, {
                **extra_weight_attrs,
                "shard_indexer": self.scales_shard_indexer,
            })

            # ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                act_scale = Parameter(
                    torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    requires_grad=False)
                layer.register_parameter("act_scale", act_scale)
                set_weight_attrs(act_scale, {
                    **extra_weight_attrs,
                    "shard_indexer": self.scales_shard_indexer,
                })

    def shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id
        assert isinstance(shard_id, str)
        qkv_idxs = {"q": 0, "k": 1, "v": 2}
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_indexer(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
        logical_widths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del logical_widths
        return param[self.shard_id_as_int(shard_id)], loaded_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        # Although the linear_method is propagated to all layers,
        # only linear layers invoke "create_weights". So we check
        # whether "weight_scale" is registered to determine
        # whether the layer is a linear layer that requires quantization.
        if not hasattr(layer, "process_after_load") or not layer.process_after_load:
            return

        # If the checkpoint is fp16/bf16 (not serialized fp8), quantize the weights.
        if not self.quant_config.is_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.act_scale = None
            return

        # If the checkpoint is serialized fp8, cleanup state_dict --> apply_weights.
        # TODO: this will be cleaned up once we have the cutlass kernels.
        else: 
            # WEIGHT
            #   Tranpose weight for passing to torch._scaled_mm
            weight = layer.weight
            layer.weight = Parameter(weight.t(), requires_grad=False)
            
            # WEIGHT_SCALE
            #   If we only have one logical shard, avoid the for loop in apply weights.
            #   TODO: once we have the cutlass_gemm, this will be removed.
            if len(layer.logical_widths) == 1:
                layer.weight_scale = Parameter(layer.weight_scale.max(), requires_grad=False)
                layer.logical_widths = None
        
            # ACT_SCALE
            #   Dyanmic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the act_scales (since they are equal to eachoter).
            if self.quant_config.activation_scheme == "dynamic":
                layer.act_scale = None
            elif self.quant_config.activation_scheme == "static":
                if not all_close_1d(layer.act_scale):
                    raise ValueError(
                        "All the act_scales for the logical weights of a layer "
                        f"must be equal. But got {layer.act_scale}")
                layer.act_scale = Parameter(layer.act_scale.max(), requires_grad=False)
            else:
                raise ValueError(f"Unknown activation_scheme {self.quant_config.activation_scheme}")

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.act_scale is None and x_scale computed from x.
        #   If static,  layer.act_scale is scalar and x_scale set to act_scale.
        qinput, x_scale = ops.scaled_fp8_quant(x, layer.act_scale)

        # Case 1: we have one single scale for N logical weights.
        if layer.logical_widths is None:
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )
        
        # Case 2: We have N weigth_scales for N logical weights.
        #   Current: inefficient for loop to apply each logical GEMM_DQ.
        #   TODO: replace will cutlass gemm_dq with epilogue fusion.
        else:
            output = torch.empty(x.shape[0], layer.weight.shape[1],
                                 dtype=x.dtype, device="cuda")
            start = 0
            # Loop over the N logical shards.
            for logical_width, w_scale in zip(layer.logical_widths, layer.weight_scale):
                end = start + logical_width
                out, _ = torch._scaled_mm(
                    qinput,
                    layer.weight[:, start:end],
                    out_dtype=x.dtype,
                    scale_a=x_scale,
                    scale_b=w_scale,
                )
                output[:, start:end] = out
                start = end

        if bias is not None:
            output.add_(bias)

        return output
    
def all_close_1d(x: torch.Tensor):
    assert len(x.shape) == 1
    for i in range(x.shape[0]):
        if not torch.allclose(x[0], x[i]):
            return False
    return True
