from typing import Any, Dict, Iterator, List, Optional, Tuple
from safetensors import safe_open
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm._C import ops
import pandas as pd
import os


class Fp8RocmConfig(QuantizationConfig):
    def __init__(self, config) -> None:
        self.quantized_weights_path = config["quantized_weights"]
        self._tuned = {}
        self._stats = {}
        gemm_type = os.getenv("FP8_GEMM", "fp8_16")
        #print(f"Integral Cross factor = {self.factor}")
        if gemm_type == "fp8_8":
            self.gemm_method = Fp8RocmLinearLayer.apply_fp8_8
            tuned_filename = "/tuned_fp8_8.csv"
        elif gemm_type == "fp8_16":
            self.gemm_method = Fp8RocmLinearLayer.apply_fp8_16
            tuned_filename = "/tuned_fp8_16.csv"
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
            algo = shape["solidx"]
            self._tuned[(m, n, k)] = algo

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["serenity_config.json"]

    @classmethod
    def from_config(cls, config) -> "Fp8RocmConfig":
        return cls(config)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.uint8, torch.float8_e4m3fnuz]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> str:
        return "Fp8Rocm"

    def get_linear_method(self) -> "Fp8RocmLinearLayer":
        return Fp8RocmLinearLayer(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key
        ), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


class Fp8RocmLinearLayer(LinearMethodBase):
    def __init__(self, config: Fp8RocmConfig) -> None:
        self._config = config

    def get_tensor(self) -> Iterator[Tuple[str, torch.Tensor]]:
        with safe_open(
            self._config.quantized_weights_path, framework="pt"
        ) as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        # for a, b in self.get_tensor():
        #    print(f"{a}: {b.shape}")
        #    pass
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # orig_weight = Parameter(torch.empty(output_size_per_partition,
        #                        input_size_per_partition,
        #                        dtype=params_dtype),
        #                        requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        # set_weight_attrs(orig_weight, {"input_dim": 1, "output_dim": 0})
        return {
            "weight": weight,
            # "orig_weight": orig_weight,
            "activation_scaling_factor": Parameter(
                torch.empty(1, dtype=torch.float32, device="cuda")
            ),
            "weights_scaling_factor": Parameter(
                torch.empty(1, dtype=torch.float32, device="cuda")
            ),
            "output_scaling_factor": Parameter(
                torch.empty(1, dtype=torch.float32, device="cuda")
            )
        }

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
                    df = pd.read_csv("/fp8_shapes.csv")
                except:
                    df = pd.DataFrame(columns=["M", "N", "K"])
                df = pd.concat(
                    [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                ).drop_duplicates()
                df.to_csv("/fp8_shapes.csv", index=False)
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
                    df = pd.read_csv("/fp8_shapes.csv")
                except:
                    df = pd.DataFrame(columns=["M", "N", "K"])
                df = pd.concat(
                    [df, pd.DataFrame({"M": [m], "N": [n], "K": [k]})]
                ).drop_duplicates()
                df.to_csv("/fp8_shapes.csv", index=False)
                # print(f"{m},{n},{k}")
            algo = 0

        res = ops.fp8_gemm(x8, weight.t(), asf, wsf, osf, int(algo))
        res16 = torch.empty_like(res, dtype=torch.float16)
        ops.convert_fp8(res, res16, 1/osf)
        return res16

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight: torch.Tensor = weights["weight"]
        if weight.dtype == torch.float8_e4m3fnuz:
            asf: torch.Tensor = weights["activation_scaling_factor"] * 2
            wsf: torch.Tensor = weights["weights_scaling_factor"] * 2
            osf: torch.Tensor = weights["output_scaling_factor"] / 2
            #my_osf: torch.Tensor = self._config.factor / weights["my_osf"]
            #with open("ratio.txt", "a") as f:
            #    f.write(f'{weights["output_scaling_factor"].item()},{weights["my_osf"].item()}\n')
            #return self.test(weight, asf, wsf, osf, x, weights["my_osf"], bias)
            return self._config.gemm_method(self, x, weight, asf, wsf, osf)

        return F.linear(x, weight, bias)