"""Utils for model executor."""
import random

import numpy as np
import torch

from vllm.model_executor.parallel_utils.parallel_state import model_parallel_is_initialized
from vllm.model_executor.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed)
