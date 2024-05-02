import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_INSTANCE_ID: Optional[str] = None
    VLLM_NCCL_SO_PATH: Optional[str] = None
    VLLM_USE_TRITON_FLASH_ATTN: bool = False
    LOCAL_RANK: int = 0

environment_variables: Dict[str, Callable[[], Any]] = {
    # used in distributed environment to determine the master address
    'VLLM_HOST_IP':
    lambda: os.environ['VLLM_HOST_IP'] or os.environ['HOST_IP'],

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",

    # Instance id represents an instance of the VLLM. All processes in the same
    # instance should have the same instance id.
    "VLLM_INSTANCE_ID":
    lambda: os.environ.get("VLLM_INSTANCE_ID", None),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),

    # flag to control if vllm should use triton flash attention
    "VLLM_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),
    "LOCAL_RANK":
    lambda: int(os.environ['LOCAL_RANK']),
}


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
