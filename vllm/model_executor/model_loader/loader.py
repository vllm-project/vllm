# ruff: noqa: SIM117
import collections
import copy
import dataclasses
import fnmatch
import glob
import json
import math
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (Any, Dict, Generator, Iterable, List, Optional, Tuple,
                    Type, cast)

import gguf
import huggingface_hub
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoadFormat,
                         LoRAConfig, ModelConfig, MultiModalConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
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
    get_gguf_extra_tensor_names, get_quant_config, gguf_quant_weights_iterator,
    initialize_dummy_weights, np_cache_weights_iterator, pt_weights_iterator,
    safetensors_weights_iterator)
from vllm.model_executor.models import (has_inner_state, supports_lora,
                                        supports_multimodal)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import is_pin_memory_available


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
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            if capability < quant_config.get_min_capability():
                raise ValueError(
                    f"The quantization method {model_config.quantization} "
                    "is not supported for the current GPU. "
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
        multimodal_config: Optional[MultiModalConfig],
        scheduler_config: Optional[SchedulerConfig] = None) -> Dict[str, Any]:
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

    if supports_multimodal(model_class):
        assert multimodal_config is not None

        extra_kwargs["multimodal_config"] = multimodal_config

    if has_inner_state(model_class) and scheduler_config:
        extra_kwargs["scheduler_config"] = scheduler_config

    return extra_kwargs


def build_model(model_class: Type[nn.Module], hf_config: PretrainedConfig,
                cache_config: Optional[CacheConfig],
                quant_config: Optional[QuantizationConfig], *,
                lora_config: Optional[LoRAConfig],
                multimodal_config: Optional[MultiModalConfig],
                scheduler_config: Optional[SchedulerConfig]) -> nn.Module:
    extra_kwargs = _get_model_initialization_kwargs(model_class, lora_config,
                                                    multimodal_config,
                                                    scheduler_config)

    return model_class(config=hf_config,
                       cache_config=cache_config,
                       quant_config=quant_config,
                       **extra_kwargs)


def _initialize_model(
        model_config: ModelConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        cache_config: CacheConfig,
        scheduler_config: Optional[SchedulerConfig] = None) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)

    return build_model(
        model_class,
        model_config.hf_config,
        cache_config=cache_config,
        quant_config=_get_quantization_config(model_config, load_config),
        lora_config=lora_config,
        multimodal_config=model_config.multimodal_config,
        scheduler_config=scheduler_config,
    )


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

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
            self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt)
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path, self.load_config.download_dir, hf_folder,
                hf_weights_files)
        elif use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        if current_platform.is_tpu():
            # In PyTorch XLA, we should call `xm.mark_step` frequently so that
            # not too many ops are accumulated in the XLA program.
            import torch_xla.core.xla_model as xm

            def _xla_weights_iterator(iterator: Generator):
                for weights in iterator:
                    yield weights
                    xm.mark_step()

            weights_iterator = _xla_weights_iterator(weights_iterator)

        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load",
                                    True))
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(Iterable[DefaultModelLoader.Source],
                                 getattr(model, "secondary_weights", ()))
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model,
                              model_config.revision,
                              fall_back_to_pt=True)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, cache_config,
                                          scheduler_config)

            model.load_weights(self._get_all_weights(model_config, model))

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

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, cache_config,
                                          scheduler_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(
                            module, torch.device(device_config.device)):
                        quant_method.process_weights_after_loading(module)
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
                                          lora_config, cache_config)

            model.load_weights(self._get_weights_iterator())
        return model.eval()

    def _load_model_serialized(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
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
                    model_class, lora_config, model_config.multimodal_config)
                extra_kwargs["quant_config"] = quant_config
                extra_kwargs["cache_config"] = cache_config

                tensorizer_config = copy.copy(self.tensorizer_config)
                tensorizer_config.model_class = model_class
                tensorizer_config.hf_config = model_config.hf_config
                tensorizer_config.dtype = model_config.dtype

                model = load_with_tensorizer(tensorizer_config, **extra_kwargs)
        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        self.tensorizer_config.verify_with_model_config(model_config)

        with self.tensorizer_config.open_stream():
            pass

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
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
                                               lora_config, cache_config)
        return self._load_model_serialized_cpu(model_config, device_config,
                                               lora_config, cache_config)

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
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
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
                                          lora_config, cache_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
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

    # TODO: these module names are for Llama only,
    # change so that it works with other models as well
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
        self,
        model_name_or_path: str,
        revision: Optional[str],
        pre_quant: bool,
        load_8bit: bool,
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str,
                                                                     Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes
            if bitsandbytes.__version__ < "0.44.0":
                raise ImportError("bitsandbytes version is wrong. Please "
                                  "install bitsandbytes>=0.44.0.")
        except ImportError as err:
            raise ImportError("Please install bitsandbytes>=0.44.0 via "
                              "`pip install bitsandbytes>=0.44.0` to use "
                              "bitsandbytes quantizer.") from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision)

        quant_state_dict: Dict[str, Any] = {}

        if pre_quant:
            if load_8bit:
                return self._quantized_8bit_generator(
                    hf_weights_files, use_safetensors,
                    quant_state_dict), quant_state_dict
            else:
                return self._quantized_4bit_generator(
                    hf_weights_files, use_safetensors,
                    quant_state_dict), quant_state_dict

        return self._unquantized_generator(hf_weights_files, use_safetensors,
                                           quant_state_dict), quant_state_dict

    def _quantized_8bit_generator(self, hf_weights_files, use_safetensors,
                                  quant_state_dict) -> Generator:
        for weight_name, weight_tensor in self._hf_weight_iter(
                hf_weights_files, use_safetensors):
            if not weight_name.lower().endswith(".scb"):
                continue

            weight_key = weight_name.lower().replace(".scb", ".qweight")
            quant_state_dict[weight_key] = weight_tensor

        for weight_name, weight_tensor in self._hf_weight_iter(
                hf_weights_files, use_safetensors):

            if not weight_name.endswith(".weight"):
                continue

            qweight_name = weight_name.replace(".weight", ".qweight")
            if qweight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})
                yield qweight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _quantized_4bit_generator(self, hf_weights_files, use_safetensors,
                                  quant_state_dict) -> Generator:
        from bitsandbytes.functional import QuantState

        # First iterate over all quant state weights
        weight_iterator = self._hf_weight_iter(hf_weights_files,
                                               use_safetensors)
        temp_state_dict = {}
        for weight_name, weight_tensor in weight_iterator:
            if weight_name.endswith(".weight"):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            if "quant_state.bitsandbytes" in weight_name:
                temp_state_dict[weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[weight_name] = weight_tensor

        # Closure to parse quant_state for each prequant weight
        def _parse_quant_state(param_name: str,
                               temp_state_dict: Dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(quant_state, device="cuda")

        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        for weight_name, weight_tensor in self._hf_weight_iter(
                hf_weights_files, use_safetensors):
            # Filter out all weights whose suffix is not ".weight"
            if not weight_name.endswith(".weight"):
                continue
            if (f"{weight_name}.quant_state.bitsandbytes__nf4" \
                    in temp_state_dict) or \
            (f"{weight_name}.quant_state.bitsandbytes__fp4" \
                    in temp_state_dict):
                quant_state = _parse_quant_state(weight_name, temp_state_dict)
                weight_name = weight_name.replace(".weight", ".qweight")
                quant_state_dict[weight_name] = quant_state
                yield weight_name.replace(".weight", ".qweight"), weight_tensor
            else:
                yield weight_name, weight_tensor

    def _unquantized_generator(self, hf_weights_files, use_safetensors,
                               quant_state_dict) -> Generator:
        from bitsandbytes.functional import quantize_4bit
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for weight_name, weight_tensor in self._hf_weight_iter(
                hf_weights_files, use_safetensors):
            if any(target_module in weight_name
                   for target_module in self.target_modules):
                weight_name = weight_name.replace(".weight", ".qweight")

                # weight partitions of different modules occur at
                # different dimensions
                # TODO: these module names are for Llama only,
                # change so that it works with other models as well
                if 'down_proj' in weight_name or 'o_proj' in weight_name:
                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[...,
                                                      start_index:end_index]

                else:
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index,
                                                      ...]

                # bitsandbytes requires data in GPU
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.cuda()

                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight,
                        compress_statistics=True,
                        quant_type="nf4")

                quant_state_dict[weight_name] = quant_state
            else:
                processed_weight = weight_tensor

            yield weight_name, processed_weight

    def _load_weights(self, model_config: ModelConfig,
                      model: nn.Module) -> None:
        if not hasattr(model, 'load_weights'):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}.")

        if not hasattr(model, 'bitsandbytes_stacked_params_mapping'):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet.")

        logger.info("Loading weights with BitsAndBytes quantization. "
                    " May take a while ...")

        quant_config = getattr(model_config.hf_config, "quantization_config",
                               None)

        pre_quant = False
        if quant_config is not None:
            quant_method = quant_config.get('quant_method')
            if quant_method == "bitsandbytes":
                pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} "
                    "quantization")

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        if pre_quant and get_tensor_model_parallel_world_size() > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with TP is not supported."
                "Please try with PP.")

        load_8bit = False
        if pre_quant:
            load_8bit = quant_config.get('load_in_8bit', False)

        qweight_iterator, quant_state_dict = \
            self._get_quantized_weights_iterator(
            model_config.model, model_config.revision, pre_quant, load_8bit)

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

                if load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)})

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, cache_config)

                self._load_weights(model_config, model)

        return model.eval()


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"{model_name_or_path} is not a file.")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        config = model_config.hf_config
        model_type = config.model_type
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(config)
        state_dict = dummy_model.state_dict()

        gguf_to_hf_name_map = {}
        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        return gguf_to_hf_name_map

    def _get_weights_iterator(
        self, model_name_or_path: str, gguf_to_hf_name_map: Dict[str, str]
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return gguf_quant_weights_iterator(model_name_or_path,
                                           gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:

        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
                local_model_path, gguf_weights_map):
            model_config.hf_config.update({"tie_word_embeddings": True})

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, cache_config)
            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map))
        return model


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

    if load_config.load_format == LoadFormat.GGUF:
        return GGUFModelLoader(load_config)

    return DefaultModelLoader(load_config)
