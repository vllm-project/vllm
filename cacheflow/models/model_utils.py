from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig

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
    torch.set_default_dtype(torch_dtype)
    config = AutoConfig.from_pretrained(model_name)
    for model_class_name, model_class in MODEL_CLASSES.items():
        if model_class_name in model_name:
            model = model_class(config)
            weights_dir = model_class.download_weights(model_name)
            model.load_weights(weights_dir)
            return model.eval(), torch_dtype
    raise ValueError(f'Invalid model name: {model_name}')


