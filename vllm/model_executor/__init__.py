from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_random_seed
from vllm.utils import is_neuron

if is_neuron():
    from vllm.model_executor.neuron_model_loader import get_model
else:
    from vllm.model_executor.model_loader import get_model

__all__ = [
    "InputMetadata",
    "get_model",
    "SamplingMetadata",
    "set_random_seed",
]
