import os
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_USE_MODELSCOPE: bool = False

environment_variables: Dict[str, Callable[[], Any]] = {
    # used in distributed environment to determine the master address
    'VLLM_HOST_IP':
    lambda: os.environ['VLLM_HOST_IP'] or os.environ['HOST_IP'],

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",
}


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
