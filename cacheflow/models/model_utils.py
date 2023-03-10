import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from cacheflow.models.opt import OPTForCausalLM

MODEL_CLASSES = {
    'opt': OPTForCausalLM,
}

STR_DTYPE_TO_TORCH_DTYPE = {
    'half': torch.half,
    'float': torch.float,
    'float16': torch.float16,
    'float32': torch.float32,
}


def get_model(
    model_name: str,
    dtype: Union[torch.dtype, str],
) -> nn.Module:
    if isinstance(dtype, str):
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype.lower()]
    else:
        torch_dtype = dtype
    for model_class, hf_model in MODEL_CLASSES.items():
        if model_class in model_name:
            model = hf_model.from_pretrained(model_name, torch_dtype=torch_dtype)
            return model.eval()
    raise ValueError(f'Invalid model name: {model_name}')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
