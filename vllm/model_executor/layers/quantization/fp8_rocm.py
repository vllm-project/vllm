from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from vllm._C import ops
import pandas as pd
import os

logger = init_logger(__name__)


class Fp8RocmConfig(QuantizationConfig):
    def __init__(self) -> None:
        # self.quantized_weights_path = config["quantized_weights"]
        self._tuned = {}
        self._stats = {}
        gemm_type = os.getenv("FP8_GEMM", "fp8_16")
        self.i1 = torch.tensor(1 / 16, dtype=torch.float32, device="cuda")
        self.i2 = torch.tensor(16, dtype=torch.float32, device="cuda")
        self.i = torch.tensor(1, dtype=torch.float32, device="cuda")
        self.factor = int(os.getenv("FACTOR", 240))
        #print(f"Integral Cross factor = {self.factor}")
        if gemm_type == "fp8_8":
            self.gemm_method = Fp8RocmLinearMethod.apply_fp8_8
            tuned_filename = "/projects/tuned_fp8_8.csv"
        elif gemm_type == "fp8_8_new":
            self.gemm_method = Fp8RocmLinearMethod.apply_fp8_8_new
            tuned_filename = "/projects/tuned_fp8_8_new.csv"
        elif gemm_type == "fp8_16":
            self.gemm_method = Fp8RocmLinearMethod.apply_fp8_16
            tuned_filename = "/projects/tuned_fp8_16.csv"
        else:
            raise Exception(f"Unknown fp8 gemm type: {gemm_type}")
        try:
            df = pd.read_csv(tuned_filename)
        except:
            return

        for i in range(len(df)):
            shape = df.iloc[i]
            m = shape["M"]
            n = shape["N"]
            k = shape["K"]
            algo = shape["algo"]
            self._tuned[(m, n, k)] = algo

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config) -> "Fp8RocmConfig":
        return cls(config)

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

    def get_quant_method(self,
                         layer: torch.nn.Module) -> Optional["Fp8RocmLinearMethod"]:
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

        # orig_weight = Parameter(torch.empty(output_size_per_partition,
        #                        input_size_per_partition,
        #                        dtype=params_dtype),
        #                        requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs, 
            "input_dim": 1, 
            "output_dim": 0
        })
        # set_weight_attrs(orig_weight, {"input_dim": 1, "output_dim": 0})
        
        self._create_scale_param(
                scale_name="weights_scaling_factor",
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                **extra_weight_attrs)

        self._create_scale_param(
                    scale_name="activation_scaling_factor",
                    layer=layer,
                    output_partition_sizes=output_partition_sizes,
                    **extra_weight_attrs)
        
        self._create_scale_param(
                    scale_name="output_scaling_factor",
                    layer=layer,
                    output_partition_sizes=output_partition_sizes,
                    **extra_weight_attrs)

        
    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        layer.activation_scaling_factor = Parameter(layer.activation_scaling_factor.max(),
                                                    requires_grad=False)
        layer.output_scaling_factor = Parameter(layer.output_scaling_factor.reciprocal().max(),
                                                    requires_grad=False)

        max_w_scale = layer.weights_scaling_factor.max()
        if len(layer.logical_widths) > 1:
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = _per_tensor_dequantize(layer.weight[start:end, :],
                                                  layer.weights_scaling_factor[idx])

                layer.weight[start:end, :] = _per_tensor_quantize(
                    weight_dq, max_w_scale)
                start = end
        layer.weights_scaling_factor = Parameter(max_w_scale, requires_grad=False)

        # WEIGHT
        #   Transpose weight for passing to torch._scaled_mm
        weight = layer.weight
        layer.weight = Parameter(weight, requires_grad=False)

    
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


    def test(
        self,
        weight: torch.Tensor,
        asf: torch.Tensor,
        wsf: torch.Tensor,
        osf: torch.Tensor,
        x: torch.Tensor,
        my_osf: torch.Tensor, 
        bias: Optional[torch.Tensor] = None,
    ):
        x8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        ops.convert_fp8(x, x8, asf)

        #####
        # res8 = ops.fp8_gemm_16_1(x8, weight.t(), wsf, asf, torch.tensor(1, dtype=torch.float32, device='cuda'), 0)
        i = torch.tensor(1, dtype=torch.float32, device="cuda")
        ideal = ops.fp8_gemm_16(x8, weight.t(), asf, wsf, 0)
        # ideal = ideal.to(torch.float16)
        # ideal = ops.fp8_gemm_16(x8, weight.t(), asf, wsf, 0)
        i1 = torch.tensor(10, dtype=torch.float32, device="cuda")

        path1 = ops.fp8_gemm(x8, weight.t(), asf, wsf, i / 16, 0)
        res16_1 = torch.empty_like(path1, dtype=x.dtype)
        ops.convert_fp8(path1, res16_1, i * 16)

        path3 = ops.fp8_gemm_16(x8, weight.t(), asf, wsf, 0)
        res16_3 = path3.to(torch.float8_e4m3fnuz).to(torch.float16)

        path5 = ops.fp8_gemm(x8, weight.t(), i, i, asf * wsf * i1, 0)
        res16_5 = torch.empty_like(path5, dtype=x.dtype)
        ops.convert_fp8(path5, res16_5, 1 / i1)


        #i2 = torch.amax(ideal)
        #if i2 > my_osf:
        #    my_osf.data.copy_(i2)
        i2 = 240 / my_osf
        path6 = ops.fp8_gemm(x8, weight.t(), asf, wsf, i2, 0)
        res16_6 = torch.empty_like(path6, dtype=x.dtype)
        ops.convert_fp8(path6, res16_6, 1/i2)

        path9 = ops.fp8_gemm(x8, weight.t(), asf, wsf, osf/2, 0)
        res16_9 = torch.empty_like(path9, dtype=x.dtype)
        ops.convert_fp8(path9, res16_9, 2/osf)

        path10 = ops.fp8_gemm(x8, weight.t(), asf, wsf, osf, 0)
        res16_10 = torch.empty_like(path10, dtype=x.dtype)
        ops.convert_fp8(path10, res16_10, 1/osf)

        w16 = weight.to(torch.float16)
        w16 *= wsf

        orig_res = F.linear(x, w16, bias)
        b = {
            "ideal": torch.allclose(orig_res, ideal, atol=0.1, rtol=0.05),
            "path1": torch.allclose(orig_res, res16_1, atol=0.1, rtol=0.05),
            "path3": torch.allclose(orig_res, res16_3, atol=0.1, rtol=0.05),
            "path5": torch.allclose(orig_res, res16_5, atol=0.1, rtol=0.05),
            "path6": torch.allclose(orig_res, res16_6, atol=0.1, rtol=0.05),
            "path9": torch.allclose(orig_res, res16_9, atol=0.1, rtol=0.05),
            "path10": torch.allclose(orig_res, res16_10, atol=0.1, rtol=0.05),
        }
        for iter, v in b.items():
            if not v:
                self._config._stats[iter] = self._config._stats.get(iter, 0) + 1

        # assert torch.allclose(orig_res, res16, atol=0.1, rtol=0.01)

        return res16_6

    def apply_fp8_16(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        asf: torch.Tensor,
        wsf: torch.Tensor,
        osf: torch.Tensor,
    ) -> torch.Tensor:
        x8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
        ops.convert_fp8(x, x8, asf)
        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        algo = self._config._tuned.get((m, n, k))
        if algo is None:
            import os

            # print(f"Not found: {m} {n} {k}")
            if os.getenv("TUNE_FP8") == "1":
                try:
                    df = pd.read_csv("/projects/fp8_tune.csv")
                except:
                    df = pd.DataFrame(columns=["M", "N", "K"])
                df = pd.concat(
                    [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                ).drop_duplicates()
                df.to_csv("/projects/fp8_tune.csv", index=False)
                # print(f"{m},{n},{k}")
            algo = 0
        res = ops.fp8_gemm_16(x8, weight.t(), asf, wsf, int(algo))
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
        ops.rocm_convert_fp8(x, x8, asf)
        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        algo = self._config._tuned.get((m, n, k))
        if algo is None:
            import os

            # print(f"Not found: {m} {n} {k}")
            if os.getenv("TUNE_FP8") == "1":
                try:
                    df = pd.read_csv("/projects/fp8_tune.csv")
                except:
                    df = pd.DataFrame(columns=["M", "N", "K"])
                df = pd.concat(
                    [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                ).drop_duplicates()
                df.to_csv("/projects/fp8_tune.csv", index=False)
                # print(f"{m},{n},{k}")
            algo = 0

        res = ops.fp8_gemm(x8, weight.t(), asf, wsf, osf, int(algo))
        res16 = torch.empty_like(res, dtype=torch.float16)
        ops.convert_fp8(res, res16, 1/osf)
        return res16

    def apply_fp8_8_new(
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
        ops.convert_fp8(x, x8, asf)

        m = weight.shape[0]
        n = x.shape[0]
        k = x.shape[1]

        algo = self._config._tuned.get((m, n, k))
        if algo is None:
            import os

            # print(f"Not found: {m} {n} {k}")
            if os.getenv("TUNE_FP8") == "1":
                try:
                    df = pd.read_csv("/projects/fp8_tune.csv")
                except:
                    df = pd.DataFrame(columns=["M", "N", "K"])
                df = pd.concat(
                    [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                ).drop_duplicates()
                df.to_csv("/projects/fp8_tune.csv", index=False)
                # print(f"{m},{n},{k}")
            algo = 0

        path11 = ops.fp8_gemm_new(
            x8, weight.t(), asf, wsf, self._config.i1, int(algo)
        )
        res16_11 = torch.empty_like(path11, dtype=x.dtype)
        ops.convert_fp8(path11, res16_11, self._config.i2)
        # res8 = ops.fp8_gemm(x8, weight.t(), wsf, asf, 0)
        return res16_11

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight: torch.Tensor = layer.weight
        if weight.dtype == torch.float8_e4m3fnuz:
            asf: torch.Tensor = layer.activation_scaling_factor * 2
            wsf: torch.Tensor = layer.weights_scaling_factor * 2
            osf: torch.Tensor = layer.output_scaling_factor / 2
            #my_osf: torch.Tensor = self._config.factor / weights["my_osf"]
            #with open("ratio.txt", "a") as f:
            #    f.write(f'{weights["output_scaling_factor"].item()},{weights["my_osf"].item()}\n')
            #return self.test(weight, asf, wsf, osf, x, weights["my_osf"], bias)
            return self._config.gemm_method(self, x, weight, asf, wsf, osf)

            assert not bias
            x8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
            ops.convert_fp8(x, x8, asf)
            i1 = torch.tensor(1 / 16, dtype=torch.float32, device="cuda")
            i2 = torch.tensor(16, dtype=torch.float32, device="cuda")
            # path11 = ops.fp8_gemm_16_1(x8, weight.t(), asf, wsf, i1, 30805)
            # res16_11 = torch.empty_like(path11, dtype=x.dtype)
            # ops.convert_fp8(path11, res16_11, i2)
            # return res16_11
            # return self.test(weight, weights["orig_weight"], asf, wsf, x, bias)
            #####

            m = weight.shape[0]
            n = x.shape[0]
            k = x.shape[1]
            # df = pd.concat([df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]).drop_duplicates()
            # df.to_csv("/projects/fp8_tune.csv", index=False)
            # x8 = (x/asf).to(torch.float8_e4m3fnuz)
            algo = self._config._tuned.get((m, n, k))
            if algo is None:
                import os

                # print(f"Not found: {m} {n} {k}")
                if os.getenv("TUNE_FP8") == "1":
                    try:
                        df = pd.read_csv("/projects/fp8_tune.csv")
                    except:
                        df = pd.DataFrame(columns=["M", "N", "K"])
                    df = pd.concat(
                        [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                    ).drop_duplicates()
                    df.to_csv("/projects/fp8_tune.csv", index=False)
                    # print(f"{m},{n},{k}")
                algo = 30800

            path11 = ops.fp8_gemm_new(x8, weight.t(), asf, wsf, i1, int(algo))
            res16_11 = torch.empty_like(path11, dtype=x.dtype)
            ops.convert_fp8(path11, res16_11, i2)
            # res8 = ops.fp8_gemm(x8, weight.t(), wsf, asf, 0)
            return res16_11

            # res = F.linear(x8, weight)
            # res16 = torch.empty_like(res8, dtype=x.dtype)
            # ops.convert_fp8(res8, res16, asf * wsf)
            # assert torch.allclose(res, res16, atol=0.1, rtol=0.01)
            # return res16

            # w16 = torch.empty_like(weight, dtype=torch.float16)
            # ops.convert_fp8(weight, w16, wsf)

            # orig_weight = weights["orig_weight"]
            # assert torch.allclose(w16, orig_weight, atol=0.1, rtol=0.01)
            # w16 = weight.to(torch.float16)
            # w16 *= wsf

            # x16 = torch.empty_like(x, dtype=torch.float16)
            # ops.convert_fp8(x8, x16, asf)
            # x16_1 = x8.to(torch.float16)
            # x16_1 *= asf

            # orig_res = F.linear(x, w16, bias)
            # assert torch.allclose(orig_res, res16, atol=0.1, rtol=0.01)
            # return orig_res
        return F.linear(x, weight, bias)
    

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
