# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from vllm.config.cache_config import (BlockSize, CacheConfig, CacheDType,
                                      PrefixCachingHashAlgo)
from vllm.config.compilation_config import CompilationConfig, CompilationLevel
from vllm.config.decoding_config import (DecodingConfig, GuidedDecodingBackend,
                                         GuidedDecodingBackendV1)
from vllm.config.device_config import Device, DeviceConfig
from vllm.config.kvevents_config import KVEventsConfig
from vllm.config.kvtransformer_config import KVTransferConfig
from vllm.config.load_config import LoadConfig, LoadFormat
from vllm.config.lora_config import LoRAConfig
from vllm.config.model_config import (ConfigFormat, HfOverrides, ModelConfig,
                                      ModelDType, ModelImpl, TaskOption,
                                      TokenizerMode)
from vllm.config.multimodal_config import MultiModalConfig
from vllm.config.obervability_config import (DetailedTraceModules,
                                             ObservabilityConfig)
from vllm.config.parallel_config import (DistributedExecutorBackend,
                                         ParallelConfig)
from vllm.config.pass_config import PassConfig
from vllm.config.pooler_config import PoolerConfig
from vllm.config.promptadapter_config import PromptAdapterConfig
from vllm.config.scheduler_config import SchedulerConfig, SchedulerPolicy
from vllm.config.speculative_config import SpeculativeConfig
from vllm.config.tokenizerpool_config import TokenizerPoolConfig
from vllm.config.utils import get_field  # noqa
from vllm.config.utils import config, get_attr_docs, is_init_field  # noqa
from vllm.config.vllm_config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    ConfigType = type[DataclassInstance]
else:
    QuantizationConfig = Any
    ConfigType = type

logger = init_logger(__name__)

_current_vllm_config: Optional[VllmConfig] = None


@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig, check_compile=False):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    from vllm.compilation.counter import compilation_counter
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
        yield
    except Exception:
        raise
    else:
        logger.debug("enabled custom ops: %s",
                     vllm_config.compilation_config.enabled_custom_ops)
        logger.debug("disabled custom ops: %s",
                     vllm_config.compilation_config.disabled_custom_ops)
        if check_compile and \
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE \
            and compilation_counter.num_models_seen == num_models_seen:
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model)
    finally:
        _current_vllm_config = old_vllm_config


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current vLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_vllm_config(vllm_config: VllmConfig,
                                layer_type: type[T]) -> dict[str, T]:
    return {
        layer_name: layer
        for layer_name, layer in
        vllm_config.compilation_config.static_forward_context.items()
        if isinstance(layer, layer_type)
    }


__all__ = [
    # Cache config
    "CacheConfig",
    "BlockSize",
    "CacheDType",
    "PrefixCachingHashAlgo",

    # Compilation config
    "CompilationConfig",
    "CompilationLevel",

    # Decoding config
    "DecodingConfig",
    "GuidedDecodingBackend",
    "GuidedDecodingBackendV1",

    # Device config
    "DeviceConfig",
    "Device",

    # KV events config
    "KVEventsConfig",

    # KV transformer config
    "KVTransferConfig",

    # Load config
    "LoadConfig",
    "LoadFormat",

    # LoRA config
    "LoRAConfig",

    # Model config
    "ModelConfig",
    "TaskOption",
    "TokenizerMode",
    "ModelDType",
    "ModelImpl",
    "HfOverrides",

    # Multimodal config
    "MultiModalConfig",

    # Pooler config
    "PoolerConfig",

    # Observability config
    "ObservabilityConfig",
    "DetailedTraceModules",

    # Parallel config
    "ParallelConfig",
    "DistributedExecutorBackend",

    # Pass config
    "PassConfig",

    # Prompt adapter config
    "PromptAdapterConfig",

    # Scheduler config
    "SchedulerConfig",
    "SchedulerPolicy",

    # Speculative config
    "SpeculativeConfig",

    #Tokenizerpool config
    "TokenizerPoolConfig",

    # vLLM config
    "VllmConfig",
    "SupportsMetricsInfo",

    # Others
    "set_current_vllm_config",
    "get_current_vllm_config",
    "get_layers_from_vllm_config",
    "get_field",
    "config",
    "get_attr_docs"
    "is_init_field",
    "ConfigFormat"
    "ConfigType",
    "QuantizationConfig"
]
