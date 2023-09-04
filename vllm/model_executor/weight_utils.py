"""Utilities for downloading and initializing model weights."""
import filelock
import glob
import json
import os
from typing import Iterator, List, Optional, Tuple

from huggingface_hub import snapshot_download
import numpy as np
import torch
from tqdm.auto import tqdm

from vllm.config import WeightQuantizationConfig


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def is_transposed(param_name,
                  quant_config: Optional[WeightQuantizationConfig] = None):
    """Returns True if the parameter tensor given by state_dict[param_name] is
    transposed relative to torch.nn.Linear.weight. Otherwise, returns False.
    """
    if quant_config and quant_config.method == "awq":
        return any(tag in param_name
                   for tag in ["qweight", "scales", "qzeros"])
    return False


def is_packed(param_name,
              quant_config: Optional[WeightQuantizationConfig] = None):
    """Returns True if each element of state_dict[param_name] contains more than
    one parameter. For example, with AWQ quantization, each INT32 element
    corresponds to 8 INT4 weights. Otherwise, returns False.
    """
    if quant_config and quant_config.method == "awq":
        return any(tag in param_name for tag in ["qweight", "qzeros"])
    return False


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_np_cache: bool = False,
    quant_config: Optional[WeightQuantizationConfig] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
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
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path

    hf_bin_files = [
        x for x in glob.glob(os.path.join(hf_folder, "*.bin"))
        if not x.endswith("training_args.bin")
    ]

    if use_np_cache:
        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
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
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            param = torch.from_numpy(param)
            transposed = is_transposed(name, quant_config=quant_config)
            packed = is_packed(name, quant_config=quant_config)
            if transposed:
                param = param.T
            yield name, param, transposed, packed
    else:
        for bin_file in hf_bin_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                transposed = is_transposed(name, quant_config=quant_config)
                packed = is_packed(name, quant_config=quant_config)
                if transposed:
                    param = param.T
                yield name, param, transposed, packed


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    param_name: str,
    column_parallel_weight_names: List[str],
    row_parallel_weight_names: List[str],
    tensor_model_parallel_rank: int,
) -> None:
    for p in column_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[0]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[start_idx:end_idx]
            break
    for p in row_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[:, start_idx:end_idx]
            break
    assert param.shape == loaded_weight.shape, (
        f"{param_name} shape mismatch between model and checkpoint: "
        f"{param.shape} != {loaded_weight.shape}")
    param.data.copy_(loaded_weight)


def get_param(state_dict, key, transposed=False):
    param = state_dict[key]
    if transposed:
        return param.T
    return param


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)
