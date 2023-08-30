"""Utilities for downloading and initializing model weights."""
import filelock
import glob
import json
import os
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple, Any

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file, safe_open
import numpy as np
import torch
from tqdm.auto import tqdm

from vllm.logger import init_logger

logger = init_logger(__name__)


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_safetensor: bool = False,
):
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    allow_patterns = "*.safetensors" if use_safetensor else "*.bin"
    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path
    hf_weights_files = glob.glob(os.path.join(hf_folder, allow_patterns))
    if not use_safetensor:
        hf_weights_files = [
            x for x in hf_weights_files if not x.endswith("training_args.bin")
        ]

    if len(hf_weights_files) == 0 and use_safetensor:
        logger.warning("No *.safetensors files found, "
                       "fall back to *.bin files")
        return prepare_hf_model_weights(model_name_or_path,
                                        cache_dir=cache_dir,
                                        use_safetensor=False)
    return hf_folder, hf_weights_files, use_safetensor


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_np_cache: bool = False,
    use_safetensor: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    hf_folder, hf_weights_files, use_safetensor = prepare_hf_model_weights(
        model_name_or_path, cache_dir=cache_dir, use_safetensor=use_safetensor)

    if use_np_cache:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensor is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
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
            yield name, torch.from_numpy(param)
    elif use_safetensor:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_slice(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


def load_padded_tensor_parallel_vocab(
    param: torch.Tensor,
    loaded_weight: Any,  # `torch.Tensor` or `PySafeSlice`
    tensor_model_parallel_rank: int,
) -> None:
    shard_size = param.shape[0]
    start_idx = tensor_model_parallel_rank * shard_size
    end_idx = (tensor_model_parallel_rank + 1) * shard_size
    loaded_weight = loaded_weight[start_idx:end_idx]

    # convert PySafeSlice object to torch.Tensor
    if not isinstance(loaded_weight, torch.Tensor):
        loaded_weight = loaded_weight[:]

    param[:loaded_weight.shape[0]].copy_(loaded_weight)


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: Any,  # `torch.Tensor` or `PySafeSlice`
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

    # convert PySafeSlice object to torch.Tensor
    if not isinstance(loaded_weight, torch.Tensor):
        loaded_weight = loaded_weight[:]

    assert param.shape == loaded_weight.shape, (
        f"{param_name} shape mismatch between model and checkpoint: "
        f"{param.shape} != {loaded_weight.shape}")
    param.data.copy_(loaded_weight)


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
