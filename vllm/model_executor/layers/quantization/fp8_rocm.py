import os
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

try:  # NOQA: SIM105
    from vllm._C import ops as vllm_ops
except ImportError:
    pass

logger = init_logger(__name__)


class Fp8RocmConfig(QuantizationConfig):

    def __init__(self) -> None:
        self._tuned = {}
        gemm_type = os.getenv("FP8_GEMM", "fp8_16")
        vllm_ops.create_workspace()
        if gemm_type == "fp8_8":
            self.gemm_method = Fp8RocmLinearMethod.apply_fp8_8
            tuned_filename = "/tmp/tuned_fp8_8.csv"
        elif gemm_type == "fp8_16":
            self.gemm_method = Fp8RocmLinearMethod.apply_fp8_16
            tuned_filename = "/tmp/tuned_fp8_16.csv"
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

        # WEIGHT
        #   Transpose weight for passing to torch._scaled_mm
        weight = layer.weight
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

    def apply_fp8_16(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        asf: torch.Tensor,
        wsf: torch.Tensor,
        osf: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not bias
        x8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        vllm_ops.convert_fp8(x8, x, asf)
        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        algo = self._config._tuned.get((m, n, k))
        if algo is None:
            _save_shape(m, n, k)
            res, _ = torch._scaled_mm(x8,
                                      weight.t(),
                                      out_dtype=x.dtype,
                                      scale_a=asf,
                                      scale_b=wsf,
                                      bias=bias)
        else:
            res = vllm_ops.fp8_gemm_16(x8, weight.t(), asf, wsf, int(algo))
        return res

    def apply_fp8_8(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        asf: torch.Tensor,
        wsf: torch.Tensor,
        osf: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not bias
        x8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        vllm_ops.convert_fp8(x8, x, asf)
        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        algo = self._config._tuned.get((m, n, k))
        if algo is None:
            _save_shape(m, n, k)
            res, _ = torch._scaled_mm(x8,
                                      weight.t(),
                                      out_dtype=x8.dtype,
                                      scale_a=asf,
                                      scale_b=wsf,
                                      scale_result=osf,
                                      bias=bias)
        else:
            res = vllm_ops.fp8_gemm(x8, weight.t(), asf, wsf, osf, int(algo))
        res16 = torch.empty_like(res, dtype=torch.float16)
        vllm_ops.convert_fp8(res16, res, 1 / osf)
        return res16

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer.weight.dtype == torch.float8_e4m3fnuz:

            return self._config.gemm_method(self, x, layer.weight,
                                            layer.activation_scaling_factor,
                                            layer.weights_scaling_factor,
                                            layer.output_scaling_factor)

        return F.linear(x, layer.weight, bias)


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


def _save_shape(m, n, k):
    if os.getenv("TUNE_FP8") == "1":
        try:
            df = pd.read_csv("/tmp/fp8_shapes.csv")
        except (IOError, pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame(columns=["M", "N", "K"])
        df = pd.concat([df, pd.DataFrame({
            "M": [m],
            "N": [n],
            "K": [k]
        })]).drop_duplicates()
        df.to_csv("/tmp/fp8_shapes.csv", index=False)
