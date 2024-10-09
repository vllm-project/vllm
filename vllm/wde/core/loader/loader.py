# ruff: noqa: SIM117
import fnmatch
import glob
import json
import math
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    get_quant_config, np_cache_weights_iterator, pt_weights_iterator,
    safetensors_weights_iterator)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import is_pin_memory_available
from vllm.wde.core.config import DeviceConfig, LoadConfig, LoadFormat
from vllm.wde.core.layers.attention.abstract import AttentionBackend
from vllm.wde.core.loader.utils import (get_model_architecture,
                                        set_default_torch_dtype)


@contextmanager
def device_loading_context(module: torch.nn.Module,
                           target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: Dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(size=p.data.size(),
                                                   stride=p.data.stride(),
                                                   dtype=p.data.dtype,
                                                   layout=p.data.layout,
                                                   device="cpu",
                                                   pin_memory=pin_memory)
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


logger = init_logger(__name__)


def _get_quantization_config(
        model_config: ModelConfig,
        load_config: LoadConfig) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        return quant_config
    return None


def initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
    attn_backend: AttentionBackend,
    cache_config: Optional[CacheConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""

    target_device = torch.device(device_config.device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model_class = get_model_architecture(model_config)[0]
            quant_config = _get_quantization_config(model_config, load_config)

            return model_class(config=model_config.hf_config,
                               cache_config=cache_config,
                               quant_config=quant_config,
                               attn_backend=attn_backend)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def load_model(self,
                   model: nn.Module,
                   *,
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   scheduler_config: Optional[SchedulerConfig] = None,
                   cache_config: Optional[CacheConfig] = None) -> nn.Module:
        """Load a model with the given configurations."""
        ...


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _maybe_download_from_modelscope(
            self, model: str, revision: Optional[str]) -> Optional[str]:
        """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or None if the model is not
        downloaded from ModelScope."""
        if VLLM_USE_MODELSCOPE:
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str],
                         fall_back_to_pt: bool) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = self._maybe_download_from_modelscope(
            model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path, index_file,
                    self.load_config.download_dir, revision)
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str],
        fall_back_to_pt: bool
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt)
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                model_name_or_path, self.load_config.download_dir, hf_folder,
                hf_weights_files)
        elif use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        return weights_iterator

    def load_model(self,
                   model: nn.Module,
                   *,
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   scheduler_config: Optional[SchedulerConfig] = None,
                   cache_config: Optional[CacheConfig] = None) -> nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            model.load_weights(
                self._get_weights_iterator(model_config.model,
                                           model_config.revision,
                                           fall_back_to_pt=getattr(
                                               model,
                                               "fall_back_to_pt_during_load",
                                               True)), )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model.eval()


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def load_model(self,
                   model: nn.Module,
                   *,
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   scheduler_config: Optional[SchedulerConfig] = None,
                   cache_config: Optional[CacheConfig] = None) -> nn.Module:
        return model.eval()


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitAndBytes quantization."""

    default_target_modules = [
        "gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj",
        "o_proj"
    ]

    possible_config_file_names = ["adapter_config.json"]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # we don't need to quantize the whole model, only the target modules
        # that are specified in the adapter config file. If the adapter config
        # file is not provided, we will quantize the default modules.
        if (not load_config.model_loader_extra_config
                or "qlora_adapter_name_or_path"
                not in load_config.model_loader_extra_config):
            self.target_modules = self.default_target_modules
            return

        qlora_adapter = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"]

        config_file_path = self._get_config_file(qlora_adapter)

        with open(config_file_path, "r") as f:
            config = json.load(f)
            self.target_modules = config["target_modules"]

    def _get_config_file(self, qlora_adapter: str) -> str:
        is_local = os.path.isdir(qlora_adapter)
        config_file_path = None
        if is_local:
            for file in self.possible_config_file_names:
                config_file_path = os.path.join(qlora_adapter, file)
                if os.path.exists(config_file_path):
                    break
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=qlora_adapter)
            for file in self.possible_config_file_names:
                if file in repo_files:
                    config_file_path = hf_hub_download(repo_id=qlora_adapter,
                                                       filename=file)
                    break

        if not config_file_path:
            raise ValueError(
                f"Cannot find adapter config file in {qlora_adapter}")

        return config_file_path

    def _get_weight_files(
            self,
            model_name_or_path: str,
            allowed_patterns: List[str],
            revision: Optional[str] = None) -> Tuple[List[str], str]:
        """Retrieve weight files. Download the files if necessary. 
        
        Return the weight files and the file pattern."""
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(
                    os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return weight_files, pattern
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )
                    return glob.glob(os.path.join(hf_folder, pattern)), pattern

        raise RuntimeError(
            f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]) -> Tuple[List[str], bool]:
        """Prepare weight files for the model."""

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision)

        if matched_pattern != "*.safetensors":
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_weights_files, matched_pattern == "*.safetensors"

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        else:
            return pt_weights_iterator(hf_weights_files)

    def _get_quantized_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str], pre_quant: bool
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str,
                                                                     Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes
            from bitsandbytes.functional import QuantState
            if bitsandbytes.__version__ < "0.42.0":
                raise ImportError("bitsandbytes version is wrong. Please "
                                  "install bitsandbytes>=0.42.0.")
            from bitsandbytes.functional import quantize_4bit
        except ImportError as err:
            raise ImportError("Please install bitsandbytes>=0.42.0 via "
                              "`pip install bitsandbytes>=0.42.0` to use "
                              "bitsandbytes quantizer.") from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision)

        quant_state_dict = {}

        def quantized_checkpoint() -> Generator:
            # First iterate over all quant state weights
            weight_iterator = self._hf_weight_iter(hf_weights_files,
                                                   use_safetensors)
            temp_state_dict = {}
            for weight_name, weight_tensor in weight_iterator:
                if weight_name.endswith(".weight"):
                    continue
                # TODO: only nf4 quantization is supported for now
                if weight_name.endswith(".quant_state.bitsandbytes__fp4"):
                    raise NotImplementedError(
                        "Only bitsandbytes_nf4 quantization"
                        f"is supported for now. {weight_name} is fp4 quantized"
                    )
                temp_state_dict[weight_name] = weight_tensor

            # Closure to parse quant_state for each prequant weight
            def _parse_quant_state(param_name: str,
                                   temp_state_dict: Dict) -> QuantState:
                quant_state = {}
                for k in temp_state_dict:
                    if param_name + "." in k:
                        quant_state[k] = temp_state_dict[k]
                # bitsandbytes library requires
                # weight.quant_state.bitsandbytes__nf4 in CPU
                quant_state[param_name +
                            ".quant_state.bitsandbytes__nf4"] = quant_state[
                                param_name +
                                ".quant_state.bitsandbytes__nf4"].cpu().data
                return QuantState.from_dict(quant_state, device="cuda")

            # Second iterate over all prequant and normal weights
            # pre quantized weights would have a quant_state
            for weight_name, weight_tensor in self._hf_weight_iter(
                    hf_weights_files, use_safetensors):
                # Filter out all weights whose suffix is not ".weight"
                if not weight_name.endswith(".weight"):
                    continue
                if weight_name + ".quant_state.bitsandbytes__nf4" \
                    in temp_state_dict:
                    quant_state = _parse_quant_state(weight_name,
                                                     temp_state_dict)
                    weight_name = weight_name.replace(".weight", ".qweight")
                    quant_state_dict[weight_name] = quant_state
                    yield weight_name.replace(".weight",
                                              ".qweight"), weight_tensor
                else:
                    yield weight_name, weight_tensor

        def generator() -> Generator:
            for weight_name, weight_tensor in self._hf_weight_iter(
                    hf_weights_files, use_safetensors):
                if any(target_module in weight_name
                       for target_module in self.target_modules):
                    weight_name = weight_name.replace(".weight", ".qweight")
                    # bitsandbytes requires data in GPU
                    loaded_weight = weight_tensor.cuda().data
                    with set_default_torch_dtype(torch.float32):
                        processed_weight, quant_state = quantize_4bit(
                            loaded_weight,
                            compress_statistics=True,
                            quant_type="nf4")

                    quant_state_dict[weight_name] = quant_state
                else:
                    processed_weight = weight_tensor

                yield weight_name, processed_weight

        if pre_quant:
            return quantized_checkpoint(), quant_state_dict
        return generator(), quant_state_dict

    def _load_weights(self, model_config: ModelConfig,
                      model: nn.Module) -> None:
        if not hasattr(model, 'load_weights'):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(self).__name__}.")

        if not hasattr(model, 'bitsandbytes_stacked_params_mapping'):
            raise AttributeError(
                f"Model {type(self).__name__} does not support BitsAndBytes "
                "quantization yet.")

        logger.info("Loading weights with BitsAndBytes quantization. "
                    " May take a while ...")

        is_quantized_checkpoint = False
        quant_config = getattr(model_config.hf_config, "quantization_config",
                               None)
        if quant_config is not None and quant_config.get(
                'quant_method') == "bitsandbytes":
            is_quantized_checkpoint = True

        qweight_iterator, quant_state_dict = \
            self._get_quantized_weights_iterator(
            model_config.model, model_config.revision, is_quantized_checkpoint)

        model.load_weights(qweight_iterator)

        torch.cuda.empty_cache()

        param_dict = dict(model.named_parameters())
        stacked_quant_state_dict: Dict[str, Dict[int, Any]] = {}
        for quant_param_name in quant_state_dict:
            non_stacked_param_name = quant_param_name

            shard_index = 0
            for shard_name, (
                    weight_name, index
            ) in model.bitsandbytes_stacked_params_mapping.items():
                if shard_name in quant_param_name:
                    shard_index = index
                    quant_param_name = quant_param_name.replace(
                        shard_name, weight_name)
                    break

            if quant_param_name not in param_dict:
                raise ValueError(
                    f"Parameter {quant_param_name} not found in the model.")

            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = (
                quant_state_dict[non_stacked_param_name])

        # save quant_states and offsets as the attributes of the parameters
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                set_weight_attrs(param, {"bnb_quant_state": quant_states})

                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(
                        f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = math.prod(
                        quant_state.shape) // pack_ratio

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

    def load_model(self,
                   model: nn.Module,
                   *,
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   scheduler_config: Optional[SchedulerConfig] = None,
                   cache_config: Optional[CacheConfig] = None) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                self._load_weights(model_config, model)

        return model.eval()


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    return DefaultModelLoader(load_config)
