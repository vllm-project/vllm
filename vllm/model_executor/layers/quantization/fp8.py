from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_serialized = is_serialized
        if is_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
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
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_serialized=is_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(self, layer: torch.nn.Module) -> "Fp8LinearMethod":
        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

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

    def __init__(self, quant_config: Fp8Config):
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
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_serialized else params_dtype)
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_serialized:
            # WEIGHT SCALE
            weight_scale = Parameter(torch.empty(len(output_partition_sizes),
                                                 dtype=torch.float32),
                                     requires_grad=False)
            layer.register_parameter("weight_scale", weight_scale)
            set_weight_attrs(weight_scale, {
                **extra_weight_attrs,
                "shard_indexer":
                self.scales_shard_indexer,
            })

            # ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                act_scale = Parameter(torch.empty(len(output_partition_sizes),
                                                  dtype=torch.float32),
                                      requires_grad=False)
                layer.register_parameter("act_scale", act_scale)
                set_weight_attrs(
                    act_scale, {
                        **extra_weight_attrs,
                        "shard_indexer":
                        self.scales_shard_indexer,
                    })

    def scales_shard_indexer(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv_idxs = {"q": 0, "k": 1, "v": 2}

        if isinstance(shard_id, int):
            pass
        elif isinstance(shard_id, str):
            if shard_id not in qkv_idxs:
                raise ValueError(f"Unknown shard_id: {shard_id}")
            shard_id = qkv_idxs[shard_id]
        else:
            ValueError(f"Shard id must be int or str but got {type(shard_id)}")

        return param[shard_id], loaded_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # If checkpoint is fp/bf16 (not serialized fp8), quantize the weights.
        if not self.quant_config.is_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.act_scale = None
            return

        # TODO: cutlass kernels will remove the need for much of this logic.
        # If the checkpoint is serialized fp8, we already loaded quantized,
        # so, just cleanup the Parameters for easier use in apply().
        else:
            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            weight = layer.weight
            layer.weight = Parameter(weight.t(), requires_grad=False)

            # WEIGHT_SCALE
            #   If all weight_scales are equal, use a single scale to avoid naive loop.
            if all_close_1d(layer.weight_scale):
                layer.weight_scale = Parameter(layer.weight_scale.max(),
                                               requires_grad=False)
                layer.logical_widths = None

            # ACT_SCALE
            #   Dynamic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the act_scales (since they are equal).
            if self.quant_config.activation_scheme == "dynamic":
                layer.act_scale = None
            elif self.quant_config.activation_scheme == "static":
                if not all_close_1d(layer.act_scale):
                    raise ValueError(
                        "All the act_scales for the logical weights of a layer "
                        f"must be equal. But got {layer.act_scale}")
                layer.act_scale = Parameter(layer.act_scale.max(),
                                            requires_grad=False)
            else:
                raise ValueError(
                    f"Unknown scheme {self.quant_config.activation_scheme}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.act_scale is None and x_scale computed from x.
        #   If static,  layer.act_scale is scalar and x_scale set to act_scale.
        qinput, x_scale = ops.scaled_fp8_quant(x, layer.act_scale)

        # Case 1: we have 1 weight_scale for N logical weights.
        if layer.logical_widths is None:
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )

        # TODO: replace naive loop with cutlass gemm_dq w/ epilogue fusion.
        # Case 2: We have N weight_scales for N logical weights.
        else:
            output = torch.empty(x.shape[0],
                                 layer.weight.shape[1],
                                 dtype=x.dtype,
                                 device="cuda")
            start = 0
            # Loop over the N logical shards.
            for logical_width, w_scale in zip(layer.logical_widths,
                                              layer.weight_scale):
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
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))
