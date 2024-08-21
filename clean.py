import contextlib
import gc

import torch
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.entrypoints.llm import LLM
from vllm.utils import is_cpu

def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()

llm = LLM(...)
del llm
cleanup()
