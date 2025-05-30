# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import enum
import hashlib
import json
from dataclasses import field, replace
from typing import Any, Optional, Protocol, Union

import pydantic
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from transformers import PretrainedConfig
from typing_extensions import runtime_checkable

import vllm.envs as envs
from vllm.config.cache_config import CacheConfig
from vllm.config.compilation_config import CompilationConfig, CompilationLevel
from vllm.config.decoding_config import DecodingConfig
from vllm.config.device_config import DeviceConfig
from vllm.config.kvevents_config import KVEventsConfig
from vllm.config.kvtransformer_config import KVTransferConfig
from vllm.config.load_config import LoadConfig
from vllm.config.lora_config import LoRAConfig
from vllm.config.model_config import ModelConfig
from vllm.config.obervability_config import ObservabilityConfig
from vllm.config.parallel_config import ParallelConfig
from vllm.config.promptadapter_config import PromptAdapterConfig
from vllm.config.scheduler_config import SchedulerConfig
from vllm.config.speculative_config import SpeculativeConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.utils import random_uuid

logger = init_logger(__name__)


@runtime_checkable
class SupportsHash(Protocol):

    def compute_hash(self) -> str:
        ...


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> dict[str, str]:
        ...


class ModelImpl(str, enum.Enum):
    AUTO = "auto"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = None  # type: ignore
    """Model configuration."""
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration."""
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = field(default_factory=LoadConfig)
    """Load configuration."""
    lora_config: Optional[LoRAConfig] = None
    """LoRA configuration."""
    speculative_config: Optional[SpeculativeConfig] = None
    """Speculative decoding configuration."""
    decoding_config: DecodingConfig = field(default_factory=DecodingConfig)
    """Decoding configuration."""
    observability_config: Optional[ObservabilityConfig] = None
    """Observability configuration."""
    prompt_adapter_config: Optional[PromptAdapterConfig] = None
    """Prompt adapter configuration."""
    quant_config: Optional[QuantizationConfig] = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig)
    """`torch.compile` configuration for the model.

    When it is a number (0, 1, 2, 3), it will be interpreted as the
    optimization level.

    NOTE: level 0 is the default level without any optimization. level 1 and 2
    are for internal testing only. level 3 is the recommended level for
    production.

    Following the convention of traditional compilers, using `-O` without space
    is also supported. `-O3` is equivalent to `-O 3`.

    You can specify the full compilation config like so:
    `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    kv_transfer_config: Optional[KVTransferConfig] = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: Optional[KVEventsConfig] = None
    """The configurations for event publishing."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: Union[dict, SupportsHash] = field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the vLLM instance."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        from vllm import __version__
        vllm_factors.append(__version__)
        vllm_factors.append(envs.VLLM_USE_V1)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
            # LoRA creates static buffers based on max_num_batched_tokens.
            # The tensor sizes and strides get captured in the torch.compile
            # graph explicitly.
            vllm_factors.append(
                str(self.scheduler_config.max_num_batched_tokens))
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.decoding_config:
            vllm_factors.append(self.decoding_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.observability_config:
            vllm_factors.append(self.observability_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.prompt_adapter_config:
            vllm_factors.append(self.prompt_adapter_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = hashlib.md5(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_capture_size
        return self.compilation_config.bs_to_padded_graph_size[batch_size]

    @staticmethod
    def _get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        """Get the quantization config."""
        from vllm.platforms import current_platform
        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import (
                get_quant_config)
            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}.")
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}")
            return quant_config
        return None

    @staticmethod
    def get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(copy.deepcopy(model_config),
                                                   load_config)

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: Optional[list[str]] = None,
    ) -> VllmConfig:
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config

        return replace(self, model_config=model_config)

    def __post_init__(self):
        """Verify configs are valid & consistent with each other.
        """
        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config,
                                                       self.speculative_config,
                                                       self.device_config)
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(
                self.load_config)

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_lora_support()
        if self.prompt_adapter_config is not None:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        from vllm.platforms import current_platform
        if self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.enable_async_tp:
            self.compilation_config.pass_config.enable_sequence_parallelism = \
                True
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append("+rms_norm")
        if envs.VLLM_USE_V1 and self.model_config is not None and \
            not self.model_config.enforce_eager:
            # FIXME(rob): Add function to set all of these.
            if not self.compilation_config.custom_ops:
                self.compilation_config.custom_ops = ["none"]
            self.compilation_config.use_cudagraph = True
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.pass_config.enable_fusion = False
            self.compilation_config.pass_config.enable_noop = False
            self.compilation_config.level = CompilationLevel.PIECEWISE
            self.compilation_config.set_splitting_ops_for_v1()

        self._set_cudagraph_sizes()

        if self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION \
                and not envs.VLLM_USE_V1:
            logger.warning(
                "CPU offload is not supported with `torch.compile` in v0 yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if ((not envs.VLLM_USE_V1) and self.lora_config is not None
                and self.compilation_config.level
                != CompilationLevel.NO_COMPILATION):
            logger.warning(
                "LoRA for V0 is not supported with `torch.compile` yet. "
                "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if self.compilation_config.full_cuda_graph and \
            not self.model_config.disable_cascade_attn:
            logger.warning_once(
                "full_cuda_graph is not supported with "
                "cascade attention. Disabling cascade attention.")
            self.model_config.disable_cascade_attn = True
            self.cache_config.enable_prefix_caching = False

        if (self.kv_events_config is not None
                and self.kv_events_config.enable_kv_cache_events
                and not self.cache_config.enable_prefix_caching):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled."
                "Use --enable-prefix-caching to enable.")
        if (self.kv_events_config is not None
                and self.kv_events_config.publisher != "null"
                and not self.kv_events_config.enable_kv_cache_events):
            logger.warning("KV cache events are disabled,"
                           "but the scheduler is configured to publish them."
                           "Modify KVEventsConfig.enable_kv_cache_events"
                           "to True to enable.")
        current_platform.check_and_update_config(self)

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

    def update_sizes_for_sequence_parallelism(self,
                                              possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled", removed_sizes,
                self.parallel_config.tensor_parallel_size)

        return [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_cudagraph_sizes(self):
        """
        cudagraph batchsize padding logic:

        `[1, 2, 4] + [8 * i for i in range(1, 1025)]` is a list of all possible
        batch sizes that cudagraph will capture.

        Depending on the engine's configuration of `max_num_seqs`, the
        candidate batch sizes to capture cudagraph will shrink to the subset
        which just cover the range of `[1, max_num_seqs]`. In the common case,
        `max_num_seqs` is 256, and the cudagraph batch sizes will be
        `[1, 2, 4, 8, 16, 24, 32, 40, ..., 256]`.

        However, if users specify the cudagraph capture sizes through
        compilation config, we will use the specified sizes instead.

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in descending order).

        During runtime, if batchsize is larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        no cudagraph will be used.
        If the batch size is no larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        we can quickly find the padded graph size for a given batch size by
        looking up `vllm_config.compilation_config.bs_to_padded_graph_size`.
        """

        # calculate the default `batch_size_capture_list`
        if not envs.VLLM_USE_V1:
            batch_size_capture_list = []
            max_batchsize_to_capture = 0
            if self.scheduler_config is not None and \
                self.model_config is not None and \
                    not self.model_config.enforce_eager:

                possible_sizes = [1, 2, 4] + [8 * i for i in range(1, 1025)]
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    possible_sizes = self.update_sizes_for_sequence_parallelism(
                        possible_sizes)

                # find the minimum size that is larger than max_num_seqs,
                # which then becomes the max_batchsize_to_capture
                larger_sizes = [
                    x for x in possible_sizes
                    if x >= self.scheduler_config.max_num_seqs
                ]
                if larger_sizes:
                    max_batchsize_to_capture = larger_sizes[0]
                else:
                    max_batchsize_to_capture = possible_sizes[-1]

                # filter out the sizes that are
                # larger than max_batchsize_to_capture
                batch_size_capture_list = [
                    size for size in possible_sizes
                    if size <= max_batchsize_to_capture
                ]
        else:
            batch_size_capture_list = []
            if self.model_config is not None and \
                not self.model_config.enforce_eager:
                cuda_graph_sizes = self.scheduler_config.cuda_graph_sizes
                if len(cuda_graph_sizes) == 1:
                    batch_size_capture_list = [1, 2, 4] + [
                        i for i in range(8, cuda_graph_sizes[0] + 1, 8)
                    ]
                elif len(cuda_graph_sizes) > 1:
                    batch_size_capture_list = sorted(cuda_graph_sizes)
                else:
                    raise TypeError(f"Invalid value for {cuda_graph_sizes=}.")
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    batch_size_capture_list = \
                        self.update_sizes_for_sequence_parallelism(batch_size_capture_list)
                max_num_tokens = self.scheduler_config.max_num_batched_tokens
                batch_size_capture_list = [
                    size for size in batch_size_capture_list
                    if size <= max_num_tokens
                ]

        self.compilation_config.init_with_cudagraph_sizes(
            batch_size_capture_list)

    def recalculate_max_model_len(self, max_model_len: int):
        model_config = self.model_config
        max_model_len = model_config.get_and_verify_max_len(max_model_len)
        self.model_config.max_model_len = max_model_len
        self.scheduler_config.max_model_len = max_model_len
        self.compute_hash()

    def __str__(self):
        return (
            f"model={self.model_config.model!r},"
            f" speculative_config={self.speculative_config!r},"
            f" tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init},"
            f" tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"override_neuron_config={self.model_config.override_neuron_config},"
            f" tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len},"
            f" download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size},"
            f" pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f" device_config={self.device_config.device}, "
            f"decoding_config={self.decoding_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"num_scheduler_steps={self.scheduler_config.num_scheduler_steps}, "
            f"multi_step_stream_outputs={self.scheduler_config.multi_step_stream_outputs}, "  # noqa
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"chunked_prefill_enabled={self.scheduler_config.chunked_prefill_enabled}, "  # noqa
            f"use_async_output_proc={self.model_config.use_async_output_proc}, "
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}")


# Avoid pydantic.errors.PydanticUserError: `VllmConfig` is not fully defined;
pydantic.dataclasses.rebuild_dataclass(VllmConfig)
