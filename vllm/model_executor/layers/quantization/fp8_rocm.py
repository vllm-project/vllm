import os
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm._C
from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class Fp8RocmConfig(QuantizationConfig):

    def __init__(self) -> None:
        self._tuned = {}
        gemm_type = os.getenv("FP8_GEMM", "fp8_16")
        vllm._C.ops.create_workspace()

        self.padding_size = 256 if envs.VLLM_FP8_PADDING else 0

        self.shapes = []
        if os.getenv("TUNE_FP8") == "1" and os.path.isfile(
                "/tmp/fp8_shapes.csv"):
            self.shapes = pd.read_csv("/tmp/fp8_shapes.csv").values.tolist()

        if gemm_type == "fp8_8":
            self.gemm_out_type = torch.float8_e4m3fnuz
            tuned_filename = "/tmp/tuned_fp8_8.csv"
        elif gemm_type == "fp8_16":
            self.gemm_out_type = torch.float16
            tuned_filename = "/tmp/tuned_fp8_16.csv"
        elif gemm_type == "fp8_b16":
            self.gemm_out_type = torch.bfloat16
            tuned_filename = "/tmp/tuned_fp8_b16.csv"
        else:
            raise ValueError(f"Unknown fp8 gemm type: {gemm_type}")

        try:
            df = pd.read_csv(tuned_filename)
        except pd.errors.ParserError as e:
            logger.warning(
                "An error occurred while parsing `%s`: %s"
                "FP8 tuning results will not be used!", tuned_filename, e)
        except (IOError, pd.errors.EmptyDataError):
            return

        for i in range(len(df)):
            shape = df.iloc[i]
            m = shape["M"]
            n = shape["N"]
            k = shape["K"]
            algo = shape["solidx"]
            self._tuned[(m, n, k)] = algo

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config) -> "Fp8RocmConfig":
        return cls()

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.uint8, torch.float8_e4m3fnuz]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 94

    @classmethod
    def get_name(cls) -> str:
        return "Fp8Rocm"

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["Fp8RocmLinearMethod"]:
        if isinstance(layer, LinearBase):
            return Fp8RocmLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def save_shape(self, m, n, k):
        if os.getenv("TUNE_FP8") == "1" and [m, n, k] not in self.shapes:
            self.shapes.append([m, n, k])
            df = pd.DataFrame(self.shapes, columns=["M", "N", "K"])
            df.to_csv("/tmp/fp8_shapes.csv", index=False)


class Fp8RocmLinearMethod(LinearMethodBase):

    def __init__(self, config: Fp8RocmConfig):
        self._config = config

    def _create_scale_param(
        self,
        scale_name: str,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        **extra_weight_attrs,
    ) -> None:
        scale = Parameter(torch.empty(len(output_partition_sizes),
                                      dtype=torch.float32),
                          requires_grad=False)
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(
            scale, {
                **extra_weight_attrs,
                "fp8_scales_shard_indexer":
                self.scales_shard_indexer,
            })

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
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fnuz,
            ),
            requires_grad=False,
        )
        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs, "input_dim": 1,
            "output_dim": 0
        })

        self._create_scale_param(scale_name="weights_scaling_factor",
                                 layer=layer,
                                 output_partition_sizes=output_partition_sizes,
                                 **extra_weight_attrs)

        self._create_scale_param(scale_name="activation_scaling_factor",
                                 layer=layer,
                                 output_partition_sizes=output_partition_sizes,
                                 **extra_weight_attrs)

        self._create_scale_param(scale_name="output_scaling_factor",
                                 layer=layer,
                                 output_partition_sizes=output_partition_sizes,
                                 **extra_weight_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        layer.activation_scaling_factor = Parameter(
            layer.activation_scaling_factor.max(), requires_grad=False)
        layer.output_scaling_factor = Parameter(
            layer.output_scaling_factor.reciprocal().max(),
            requires_grad=False)

        max_w_scale = layer.weights_scaling_factor.max()
        if len(layer.logical_widths) > 1:
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = _per_tensor_dequantize(
                    layer.weight[start:end, :],
                    layer.weights_scaling_factor[idx])

                layer.weight[start:end, :] = _per_tensor_quantize(
                    weight_dq, max_w_scale)
                start = end
        layer.weights_scaling_factor = Parameter(max_w_scale,
                                                 requires_grad=False)

        weight = layer.weight
        if envs.VLLM_FP8_PADDING:
            weight = F.pad(weight, (0, self._config.padding_size), "constant",
                           0)
            torch.cuda.empty_cache()
        layer.weight = Parameter(weight, requires_grad=False)

        if layer.weight.dtype == torch.float8_e4m3fnuz:
            layer.activation_scaling_factor = Parameter(
                layer.activation_scaling_factor * 2.0, requires_grad=False)
            layer.weights_scaling_factor = Parameter(
                layer.weights_scaling_factor * 2.0, requires_grad=False)
            layer.output_scaling_factor = Parameter(
                layer.output_scaling_factor / 2.0, requires_grad=False)

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

        # To handle the scalar loaded tensor
        if loaded_weight.numel() == 1 and len(loaded_weight.shape) != 0:
            loaded_weight = torch.scalar_tensor(loaded_weight[0])

        return param[shard_id], loaded_weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight: torch.Tensor = layer.weight
        out_dtype = self._config.gemm_out_type

        asf: torch.Tensor = layer.activation_scaling_factor
        wsf: torch.Tensor = layer.weights_scaling_factor
        osf: Optional[torch.Tensor] = layer.output_scaling_factor \
            if out_dtype == torch.float8_e4m3fnuz else None

        x_quant = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        ops.convert_fp8(x_quant, x, asf)
        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        solidx = self._config._tuned.get((m, n, k), 0)
        if solidx == 0:
            self._config.save_shape(m, n, k)
        res = ops.fp8_mm(x_quant, weight.t(), out_dtype, asf, wsf, osf,
                         int(solidx), self._config.padding_size)

        if osf is not None:
            res_upscaled = torch.empty_like(res, dtype=x.dtype)
            ops.convert_fp8(res_upscaled, res, 1 / osf)
            res = res_upscaled
        if bias is not None:
            res += bias
        return res


def _per_tensor_quantize(tensor: torch.Tensor,
                         inv_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fnuz)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fnuz)


def _per_tensor_dequantize(tensor: torch.Tensor,
                           inv_scale: float) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight
