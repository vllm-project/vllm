# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: F401
import ast
import copy
import hashlib
import inspect
import json
import os
import textwrap
from contextlib import contextmanager
from dataclasses import field, fields, is_dataclass, replace
from functools import cached_property, lru_cache
from typing import (TYPE_CHECKING, Any, Literal, Optional, Protocol, TypeVar,
                    Union, cast)

import regex as re
import torch
from pydantic import ConfigDict, SkipValidation
from pydantic.dataclasses import dataclass
from typing_extensions import runtime_checkable

import vllm.envs as envs
from vllm import version
from vllm.config.cache import (BlockSize, CacheConfig, CacheDType, MambaDType,
                               PrefixCachingHashAlgo)
from vllm.config.compilation import (CompilationConfig, CompilationLevel,
                                     CUDAGraphMode, PassConfig)
from vllm.config.device import Device, DeviceConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.config.model import (ConvertOption, HfOverrides, LogprobsMode,
                               ModelConfig, ModelDType, ModelImpl,
                               RunnerOption, TaskOption, TokenizerMode,
                               iter_architecture_defaults,
                               try_match_architecture_defaults)
from vllm.config.multimodal import (MMCacheType, MMEncoderTPMode,
                                    MultiModalConfig)
from vllm.config.observability import DetailedTraceModules, ObservabilityConfig
from vllm.config.parallel import (DistributedExecutorBackend, EPLBConfig,
                                  ParallelConfig)
from vllm.config.pooler import PoolerConfig
from vllm.config.scheduler import RunnerType, SchedulerConfig, SchedulerPolicy
from vllm.config.speculative import SpeculativeConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.config.utils import ConfigType, config, get_attr_docs, is_init_field
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.utils import random_uuid

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
__all__ = [
    # From vllm.config.cache
    "CacheConfig",
    # From vllm.config.compilation
    "CompilationConfig",
    "CompilationLevel",
    "CUDAGraphMode",
    "PassConfig",
    # From vllm.config.device
    "DeviceConfig",
    # From vllm.config.ec_transfer
    "ECTransferConfig",
    # From vllm.config.kv_events
    "KVEventsConfig",
    # From vllm.config.kv_transfer
    "KVTransferConfig",
    # From vllm.config.load
    "LoadConfig",
    # From vllm.config.lora
    "LoRAConfig",
    # From vllm.config.model
    "ModelConfig",
    "iter_architecture_defaults",
    "try_match_architecture_defaults",
    # From vllm.config.multimodal
    "MultiModalConfig",
    # From vllm.config.observability
    "ObservabilityConfig",
    # From vllm.config.parallel
    "EPLBConfig",
    "ParallelConfig",
    # From vllm.config.pooler
    "PoolerConfig",
    # From vllm.config.scheduler
    "SchedulerConfig",
    # From vllm.config.speculative
    "SpeculativeConfig",
    # From vllm.config.speech_to_text
    "SpeechToTextConfig",
    # From vllm.config.structured_outputs
    "StructuredOutputsConfig",
    # From vllm.config.utils
    "ConfigType",
    "SupportsMetricsInfo",
    "config",
    "get_attr_docs",
    "is_init_field",
    "update_config",
    # From vllm.config.vllm
    "VllmConfig",
    "get_cached_compilation_config",
    "get_current_vllm_config",
    "set_current_vllm_config",
    "get_layers_from_vllm_config",
]
