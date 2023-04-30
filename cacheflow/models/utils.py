import os
import glob
import json
import filelock
from typing import Union, Optional

import numpy as np
import torch
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

_STR_DTYPE_TO_TORCH_DTYPE = {
    'half': torch.half,
    'float': torch.float,
    'float16': torch.float16,
    'float32': torch.float32,
}


def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, str):
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype.lower()]
    else:
        torch_dtype = dtype
    return torch_dtype


def get_dtype_size(dtype: Union[torch.dtype, str]) -> int:
    torch_dtype = get_torch_dtype(dtype)
    return torch.tensor([], dtype=torch_dtype).element_size()


def hf_model_weights_iterator(model_name_or_path: str,
                              cache_dir: Optional[str] = None,
                              use_np_cache: bool = False):
    # Prepare file lock directory to prevent multiple processes from
    # downloading the same model weights at the same time.
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        with lock:
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns="*.bin",
                                          cache_dir=cache_dir)
    else:
        hf_folder = model_name_or_path

    hf_bin_files = glob.glob(os.path.join(hf_folder, "*.bin"))

    if use_np_cache:
        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, 'np')
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, 'weight_names.json')
        with lock:
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_bin_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, 'w') as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, 'r') as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    else:
        for bin_file in hf_bin_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
