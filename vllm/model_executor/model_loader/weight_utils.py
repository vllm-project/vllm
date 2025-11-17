# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for downloading and initializing model weights."""

import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import filelock
import huggingface_hub.constants
import numpy as np
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from safetensors.torch import load, load_file, safe_open, save_file
from tqdm.auto import tqdm

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import PlaceholderModule

try:
    from runai_model_streamer import SafetensorsStreamer
except ImportError:
    runai_model_streamer = PlaceholderModule("runai_model_streamer")  # type: ignore[assignment]
    SafetensorsStreamer = runai_model_streamer.placeholder_attr("SafetensorsStreamer")

try:
    import gguf
except ImportError:
    gguf = PlaceholderModule("gguf")

try:
    from fastsafetensors import SafeTensorsFileLoader, SingleGroup
except ImportError:
    fastsafetensors = PlaceholderModule("fastsafetensors")
    SafeTensorsFileLoader = fastsafetensors.placeholder_attr("SafeTensorsFileLoader")
    SingleGroup = fastsafetensors.placeholder_attr("SingleGroup")

from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least

logger = init_logger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def enable_hf_transfer():
    """automatically activates hf_transfer"""
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]:
    """
    Context manager that provides an atomic file writing routine.

    The context manager writes to a temporary file and, if successful,
    atomically replaces the original file.

    Args:
        filepath (str or Path): The path to the file to write.
        mode (str): The file mode for the temporary file (e.g., 'w', 'wb').
        encoding (str): The encoding for text mode.

    Yields:
        file object: A handle to the temporary file.
    """
    # Create a temporary file in the same directory as the target file
    # to ensure it's on the same filesystem for an atomic replace.
    temp_dir = os.path.dirname(filepath)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # Open the temporary file for writing
        with os.fdopen(temp_fd, mode=mode, encoding=encoding) as temp_file:
            yield temp_file

        # If the 'with' block completes successfully,
        # perform the atomic replace.
        os.replace(temp_path, filepath)

    except Exception:
        logger.exception(
            "Error during atomic write. Original file '%s' not modified", filepath
        )
        raise
    finally:
        # Clean up the temporary file if it still exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)


def maybe_download_from_modelscope(
    model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str | None:
    """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.

    Returns the path to the downloaded model, or None if the model is not
    downloaded from ModelScope."""
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model, download_dir):
            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=ignore_patterns,
                    allow_patterns=allow_patterns,
                )
            else:
                model_path = model
        return model_path
    return None


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
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
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


# TODO(woosuk): Move this to other place.
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization in ("gguf", "inc"):
        return quant_cls()

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)

    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)

    # if hf_quant_config is None, we will try to get config from
    # hf_overrides
    hf_overrides = model_config.hf_overrides
    quantization_config_file = hf_overrides.get("quantization_config_file", None)
    if quantization_config_file is not None:
        if hasattr(quant_cls, "from_config_file"):
            return quant_cls.from_config_file(quantization_config_file)
        else:
            raise NotImplementedError(
                "from_config_file is specified in hf_override config, "
                "but quant_cls.from_config_file is not implemented in "
                f"{quant_cls}"
            )
    quantization_config_json = hf_overrides.get("quantization_config_dict_json", None)
    if quantization_config_json is not None:
        if hasattr(quant_cls, "from_config_dict_json"):
            return quant_cls.from_config_dict_json(quantization_config_json)
        else:
            raise NotImplementedError(
                "from_config_dict_json is specified in hf_override config, "
                "but quant_cls.from_config_dict_json is not implemented in "
                f"{quant_cls}"
            )

    # Inflight BNB quantization
    if model_config.quantization == "bitsandbytes":
        return quant_cls.from_config({})
    model_name_or_path = (
        maybe_download_from_modelscope(
            model_config.model,
            revision=model_config.revision,
            download_dir=load_config.download_dir,
            allow_patterns=["*.json"],
        )
        or model_config.model
    )
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_config.model, load_config.download_dir):
            hf_folder = snapshot_download(
                model_config.model,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_config.model
        elif model_config.quantization == "modelopt":
            if config["producer"]["name"] == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}."
                )

    return quant_cls.from_config(config)


def get_sparse_attention_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> dict[str, Any]:
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    config_file = os.path.join(hf_folder, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    logger.info("Loaded sparse attention config from %s", config_file)

    return config


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Attempt to reduce allow_patterns to a single pattern
        # so we only have to call snapshot_download once.
        try:
            fs = HfFileSystem()
            file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

            # Use the first pattern found in the HF repo's files.
            for pattern in allow_patterns:
                matching = fnmatch.filter(file_list, pattern)
                if len(matching) > 0:
                    allow_patterns = [pattern]
                break
        except Exception as e:
            logger.warning(
                "Failed to get file list for '%s'. Trying each pattern in "
                "allow_patterns individually until weights have been "
                "downloaded. Error: %s",
                model_name_or_path,
                e,
            )

    logger.debug("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        for allow_pattern in allow_patterns:
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_pattern,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=local_only,
            )
            # If we have downloaded weights for this allow_pattern,
            # we don't need to check the rest.
            if any(Path(hf_folder).glob(allow_pattern)):
                break
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    revision: str | None = None,
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        index_file (str): The safetensors index file name
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        revision (Optional[str]): The revision of the model.
    """
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.info("No %s found in local cache.", index_file)
        except huggingface_hub.utils.EntryNotFoundError:
            logger.info("No %s found in remote.", index_file)


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def enable_tqdm(use_tqdm_on_load: bool):
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: list[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    loading_desc = "Loading safetensors checkpoint shards"
    if safetensors_load_strategy == "eager":
        loading_desc += " (eager)"

    for st_file in tqdm(
        hf_weights_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            yield from state_dict.items()
        elif safetensors_load_strategy == "torchao":
            if not torchao_version_at_least("0.14.0"):
                raise ValueError(
                    "Please use torchao version >= 0.14.0 \
                        to load torchao safetensors checkpoint"
                )
            from torchao.prototype.safetensors.safetensors_support import (
                unflatten_tensor_state_dict,
            )

            with safe_open(st_file, framework="pt") as f:
                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    state_dict[name] = f.get_tensor(name)
                metadata = f.metadata()
                updated_state_dict = unflatten_tensor_state_dict(state_dict, metadata)
            yield from updated_state_dict.items()
        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param


def multi_thread_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model safetensor files."""

    def _load_file(st_file: str):
        result = load_file(st_file, device="cpu")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_file, st_file) for st_file in hf_weights_files]
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        for future in futures_iter:
            state_dict = future.result()
            yield from state_dict.items()


def runai_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    is_distributed: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    with SafetensorsStreamer() as streamer:
        is_cuda_alike = current_platform.is_cuda_alike()
        device = (
            f"cuda:{current_platform.current_device()}"
            if is_distributed and is_cuda_alike
            else "cpu"
        )

        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        total_tensors = sum(
            len(tensors_meta)
            for tensors_meta in streamer.files_to_tensors_metadata.values()
        )

        tensor_iter = tqdm(
            streamer.get_tensors(),
            total=total_tensors,
            desc="Loading safetensors using Runai Model Streamer",
            bar_format=_BAR_FORMAT,
            disable=not enable_tqdm(use_tqdm_on_load),
            mininterval=2,
        )

        yield from tensor_iter


def _init_loader(
    pg: torch.distributed.ProcessGroup,
    device: torch.device,
    f_list: list[str],
    *,
    nogds: bool = False,
):
    loader = SafeTensorsFileLoader(pg, device, nogds=nogds)
    rank_file_map = {i: [f] for i, f in enumerate(f_list)}
    loader.add_filenames(rank_file_map)
    return loader


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files
    using fastsafetensor library."""
    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    device = torch.device(f"cuda:{pg.rank()}")
    weight_files_sub_lists = [
        hf_weights_files[i : i + pg.size()]
        for i in range(0, len(hf_weights_files), pg.size())
    ]

    nogds = False

    for f_list in tqdm(
        weight_files_sub_lists,
        desc="Loading safetensors using Fastsafetensor loader",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        loader = _init_loader(pg, device, f_list, nogds=nogds)
        try:
            try:
                fb = loader.copy_files_to_device()
            except RuntimeError as e:
                if "gds" not in str(e):
                    raise

                loader.close()
                nogds = True
                logger.warning_once(
                    "GDS not enabled, setting `nogds=True`.\n"
                    "For more information, see: https://github.com/foundation-model-stack/fastsafetensors?tab=readme-ov-file#basic-api-usages"
                )
                loader = _init_loader(pg, device, f_list, nogds=nogds)
                fb = loader.copy_files_to_device()

            try:
                keys = list(fb.key_to_rank_lidx.keys())
                for k in keys:
                    t = fb.get_tensor(k)
                    yield k, t
            finally:
                fb.close()
        finally:
            loader.close()


def pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )
        yield from state.items()
        del state


def multi_thread_pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model bin/pt files."""

    def _load_file(bin_file: str):
        return torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_load_file, bin_file) for bin_file in hf_weights_files
        ]
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading pt checkpoint shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        for future in futures_iter:
            state = future.result()
            yield from state.items()
            del state


def get_gguf_extra_tensor_names(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf.GGUFReader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = set([tensor.name for tensor in reader.tensors])
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_weight_type_map(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    """
    Return GGUF mapped weight's name and its quant type
    """
    reader = gguf.GGUFReader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str, gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors
    """

    reader = gguf.GGUFReader(gguf_file)

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]

            if weight_type.name != "F32":
                weight_type_name = name.replace("weight", "qweight_type")
                weight_type = torch.tensor(weight_type)
                yield weight_type_name, weight_type

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight = tensor.data
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]
            if weight_type.name != "F32":
                name = name.replace("weight", "qweight")
            param = torch.tensor(weight)
            yield name, param


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Load weights that are row-parallelized."""
    tp_rank = get_tensor_model_parallel_rank()
    shard_dim = 0 if param.dim() != 1 else None

    if shard_dim is not None:
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    return default_weight_loader(param, loaded_weight)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader


def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:
    """Create a weight loader that post-processes the weights after loading"""

    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader(param, loaded_weight)
        param.data.copy_(fn(param))
        return

    return composed_loader


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            if current_platform.is_tpu():
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                # Note: The param.uniform_ function cannot be used in this
                # context because it demands more TPU HBM than directly copying
                # from a CPU tensor.
                # Note: We avoid using torch.rank_like as it doesn't currently
                # support the generator argument.
                param.copy_(
                    (high - low)
                    * torch.rand(
                        param.shape,
                        generator=generator,
                        dtype=param.dtype,
                        layout=param.layout,
                        requires_grad=param.requires_grad,
                        device="cpu",
                    )
                    + low
                )
                torch._sync(param)
                continue

            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)
            if torch.finfo(param.data.dtype).bits < 16:
                # uniform_ doesn't support < 16-bit datatypes (FP8)
                dtype = param.data.dtype
                tmp_param = param.data.to(torch.float16)
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
                param.data.copy_(tmp_param)
            else:
                param.uniform_(low, high, generator=generator)


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """
    if name.endswith(".kv_scale"):
        logger.warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            logger.warning_once(
                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                name,
                remapped_name,
            )
            return None
        return remapped_name

    if any("mla_attn" in key for key in params_dict):
        attn_str = "mla_attn.mla_attn"
        logger.debug_once(
            f"Found mla_attn with k_scale and v_scale in "
            f"the checkpoint, using {attn_str} as attn_str"
        )
    else:
        attn_str = "attn"
    # Define scale name mapping patterns in order of precedence
    scale_mapping_patterns = [
        # ModelOpt format: .self_attn.{k,v}_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (
            r"\.self_attn\.([kv])_proj\.([kv])_scale$",
            rf".self_attn.{attn_str}.\2_scale",
        ),
        # QKV proj format: .self_attn.qkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Qwen3 MoE format: .self_attn.qkqkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkqkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Default format: .{k,v}_scale -> .attn.{k,v}_scale
        (r"\.([kv])_scale$", r".attn.\1_scale"),
    ]

    # Check if name ends with k_scale or v_scale
    if name.endswith((".k_scale", ".v_scale")):
        import regex as re

        for pattern, replacement in scale_mapping_patterns:
            if re.search(pattern, name):
                remapped_name = re.sub(pattern, replacement, name)
                if remapped_name not in params_dict:
                    scale_type = name.split(".")[-1]
                    logger.warning_once(
                        "Found %s in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). %s is not loaded.",  # noqa: E501
                        scale_type,
                        name,
                        remapped_name,
                        scale_type,
                    )
                    return None
                return remapped_name

    # If there were no matches, return the untouched param name
    return name
