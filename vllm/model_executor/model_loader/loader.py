# ruff: noqa: SIM117
import collections
import copy
import fnmatch
import glob
import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoadFormat,
                         LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, is_vllm_tensorized, load_with_tensorizer,
    serialize_vllm_model, tensorizer_weights_iterator)
from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                    set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    get_quant_config, initialize_dummy_weights, np_cache_weights_iterator,
    pt_weights_iterator, safetensors_weights_iterator)
from vllm.model_executor.models.interfaces import (supports_lora,
                                                   supports_vision)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import is_tpu

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


def _get_model_initialization_kwargs(
    model_class: Type[nn.Module],
    lora_config: Optional[LoRAConfig],
    vlm_config: Optional[VisionLanguageConfig],
) -> Dict[str, Any]:
    """Get extra kwargs for model initialization."""
    extra_kwargs: Dict[str, Any] = {}

    if supports_lora(model_class):
        # lora_config=None is used to disable LoRA
        extra_kwargs["lora_config"] = lora_config
    elif lora_config:
        raise ValueError(
            f"Model {model_class.__name__} does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    if supports_vision(model_class):
        if vlm_config is None:
            raise ValueError("Provide `image_input_type` and other vision "
                             "related configurations through LLM entrypoint "
                             "or engine arguments.")

        extra_kwargs["vlm_config"] = vlm_config

    return extra_kwargs


def _initialize_model(model_config: ModelConfig, load_config: LoadConfig,
                      lora_config: Optional[LoRAConfig],
                      vision_language_config: Optional[VisionLanguageConfig],
                      cache_config: CacheConfig) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class = get_model_architecture(model_config)[0]
    quant_config = _get_quantization_config(model_config, load_config)

    return model_class(config=model_config.hf_config,
                       cache_config=cache_config,
                       quant_config=quant_config,
                       **_get_model_initialization_kwargs(
                           model_class, lora_config, vision_language_config))


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
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
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(model_name_or_path,
                                                 self.load_config.download_dir,
                                                 allow_patterns, revision)
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
                    model_name_or_path, self.load_config.download_dir,
                    revision)
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder)
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

        if is_tpu():
            # In PyTorch XLA, we should call `xm.mark_step` frequently so that
            # not too many ops are accumulated in the XLA program.
            import torch_xla.core.xla_model as xm

            def _xla_weights_iterator(iterator: Generator):
                for weights in iterator:
                    yield weights
                    xm.mark_step()

            weights_iterator = _xla_weights_iterator(weights_iterator)
        return weights_iterator

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)
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
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        return model.eval()


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        return model.eval()


class TensorizerLoader(BaseModelLoader):
    """Model loader using CoreWeave's tensorizer library."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if isinstance(load_config.model_loader_extra_config, TensorizerConfig):
            self.tensorizer_config = load_config.model_loader_extra_config
        else:
            self.tensorizer_config = TensorizerConfig(
                **load_config.model_loader_extra_config)

    def _verify_config(self, model_config: ModelConfig,
                       parallel_config: ParallelConfig):
        self.tensorizer_config.verify_with_model_config(model_config)
        self.tensorizer_config.verify_with_parallel_config(parallel_config)

    def _get_weights_iterator(
            self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        tensorizer_args = self.tensorizer_config._construct_tensorizer_args()
        return tensorizer_weights_iterator(tensorizer_args)

    def _load_model_serialized_cpu(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        cache_config: CacheConfig,
    ) -> nn.Module:
        """Load a serialized model with tensorizer to the CPU.

        This is only necessary when the model isn't vLLM-tensorized (see
        examples/tensorize_vllm_model.py) This should still be faster than
        default HuggingFace loading, but will be slower than loading a
        vLLM-tensorized model.
        """
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)

            model.load_weights(self._get_weights_iterator())
        return model.eval()

    def _load_model_serialized(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        cache_config: CacheConfig,
    ) -> nn.Module:
        """Load a serialized model with tensorizer.

        Expects a vLLM-tensorized model. See the
        examples/tensorize_vllm_model.py example script
        for serializing vLLM models."""
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model_class = get_model_architecture(model_config)[0]
                quant_config = _get_quantization_config(
                    model_config, self.load_config)
                extra_kwargs = _get_model_initialization_kwargs(
                    model_class, lora_config, vision_language_config)
                extra_kwargs["quant_config"] = quant_config
                extra_kwargs["cache_config"] = cache_config

                tensorizer_config = copy.copy(self.tensorizer_config)
                tensorizer_config.model_class = model_class
                tensorizer_config.hf_config = model_config.hf_config
                tensorizer_config.dtype = model_config.dtype

                model = load_with_tensorizer(tensorizer_config, **extra_kwargs)
        return model.eval()

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        self._verify_config(model_config, parallel_config)

        if parallel_config.tensor_parallel_size > 1:
            from vllm.distributed import get_tensor_model_parallel_rank
            self.tensorizer_config.tensorizer_uri = \
                self.tensorizer_config.tensorizer_uri \
                    % get_tensor_model_parallel_rank()

        if is_vllm_tensorized(self.tensorizer_config):
            return self._load_model_serialized(model_config, device_config,
                                               lora_config,
                                               vision_language_config,
                                               cache_config)
        return self._load_model_serialized_cpu(model_config, device_config,
                                               lora_config,
                                               vision_language_config,
                                               cache_config)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        serialize_vllm_model(
            model=model,
            tensorizer_config=tensorizer_config,
        )


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/save_sharded_state.py` for creating a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = ({} if load_config.model_loader_extra_config is None
                        else load_config.model_loader_extra_config.copy())
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{load_config.model_loader_extra_config.keys()}")

    @staticmethod
    def _filter_subtensors(
            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: Dict[Any, List[Tuple[
            str, torch.Tensor]]] = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: Dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(model_name_or_path,
                                            self.load_config.download_dir,
                                            allow_patterns, revision)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        from safetensors.torch import safe_open

        from vllm.distributed import get_tensor_model_parallel_rank

        local_model_path = self._prepare_weights(model_config.model,
                                                 model_config.revision)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)
            rank = get_tensor_model_parallel_rank()
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!")
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s", tensor.shape,
                                key, param_shape)
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from safetensors.torch import save_file

        from vllm.distributed import get_tensor_model_parallel_rank
        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )


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
                        model_name_or_path, self.load_config.download_dir,
                        [pattern], revision)
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

    def _get_quantized_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str,
                                                                     Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes
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
        if use_safetensors:
            weight_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weight_iterator = pt_weights_iterator(hf_weights_files)

        def generator():
            for weight_name, weight_tensor in weight_iterator:
                if any(target_module in weight_name
                       for target_module in self.target_modules):
                    weight_name = weight_name.replace(".weight", ".qweight")
                    #  bitsandbytes requires data in GPU
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

        qweight_iterator, quant_state_dict = (
            self._get_quantized_weights_iterator(model_config.model,
                                                 model_config.revision))

        model.load_weights(qweight_iterator)

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
                for seq, quant_state in enumerate(quant_states.items()):
                    num_elements[seq] = math.prod(
                        quant_state[1].shape) // pack_ratio

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)

                self._load_weights(model_config, model)

        return model.eval()


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.TENSORIZER:
        return TensorizerLoader(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    return DefaultModelLoader(load_config)
