from vllm.model_executor.layers.quantized_linear.awq import (
    AWQColumnParallelLinear, AWQRowParallelLinear)
from vllm.model_executor.layers.quantized_linear.squeezellm import (
    SqueezeLLMColumnParallelLinear, SqueezeLLMRowParallelLinear)

_QUANTIZED_LINEAR_REGISTRY = {
    "awq": (AWQColumnParallelLinear, AWQRowParallelLinear),
    "squeezellm":
    (SqueezeLLMColumnParallelLinear, SqueezeLLMRowParallelLinear),
}


class ParallelLinear:
    pass