from torch import nn

from vllm.model_executor.layers.quantized_linear.awq import (
    AWQColumnParallelLinear, AWQRowParallelLinear)
from vllm.model_executor.layers.quantized_linear.gptq import (
    GPTQColumnParallelLinear, GPTQRowParallelLinear, GPTQLinear)
from vllm.model_executor.layers.quantized_linear.squeezellm import (
    SqueezeLLMColumnParallelLinear, SqueezeLLMRowParallelLinear)
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)

_QUANTIZED_LINEAR_REGISTRY = {
    "awq": (AWQColumnParallelLinear, AWQRowParallelLinear, None),
    "gptq": (GPTQColumnParallelLinear, GPTQRowParallelLinear, GPTQLinear),
    "squeezellm":
    (SqueezeLLMColumnParallelLinear, SqueezeLLMRowParallelLinear, None),
}


class Linear:

    @classmethod
    def linear(cls, *args, **kwargs) -> nn.Module:
        quant_config = kwargs.get("quant_config", None)
        if quant_config is None:
            kwargs.pop("quant_config", None)
            return nn.Linear(*args, **kwargs)

        name = quant_config.get_name()
        if name not in _QUANTIZED_LINEAR_REGISTRY or _QUANTIZED_LINEAR_REGISTRY[
                name][2] is None:
            raise ValueError(f"No quantized linear is found for {name}")

        quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][2]
        return quant_linear_cls(*args, **kwargs)


class ParallelLinear:

    @classmethod
    def column(cls, *args, **kwargs) -> ColumnParallelLinear:
        quant_config = kwargs.get("quant_config", None)
        if quant_config is None:
            return ColumnParallelLinear(*args, **kwargs)

        name = quant_config.get_name()
        if name not in _QUANTIZED_LINEAR_REGISTRY:
            raise ValueError(f"No quantized linear is found for {name}")

        quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][0]
        return quant_linear_cls(*args, **kwargs)

    @classmethod
    def row(cls, *args, **kwargs) -> RowParallelLinear:
        quant_config = kwargs.get("quant_config", None)
        if quant_config is None:
            return RowParallelLinear(*args, **kwargs)

        name = quant_config.get_name()
        if name not in _QUANTIZED_LINEAR_REGISTRY:
            raise ValueError(f"No quantized linear is found for {name}")

        quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][1]
        return quant_linear_cls(*args, **kwargs)
