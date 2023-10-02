from vllm.model_executor.layers.quantized_linear.awq import (
    AWQColumnParallelLinear, AWQRowParallelLinear)
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)

_QUANTIZED_LINEAR_REGISTRY = {
    "awq": (AWQColumnParallelLinear, AWQRowParallelLinear),
}


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
