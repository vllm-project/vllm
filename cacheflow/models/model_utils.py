import os
import glob
import shutil
from typing import Union
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from huggingface_hub import snapshot_download

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
    weights_dir = download_opt_weights(model_name)
    for model_class_name, model_class in MODEL_CLASSES.items():
        if model_class_name in model_name:
            model = model_class(config)
            model.load_weights(weights_dir)
            return model.eval(), torch_dtype
    raise ValueError(f'Invalid model name: {model_name}')


def download_opt_weights(model_name: str, path: str = "/tmp/transformers"):
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    test_weight_path = os.path.join(path, "model.decoder.embed_positions.weight")
    if os.path.exists(test_weight_path):
        return path

    folder = snapshot_download(model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "model.decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "model.decoder.embed_tokens.weight", "lm_head.weight"))

    return path