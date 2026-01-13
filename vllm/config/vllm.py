# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config.speculative import EagleModelTypes
from vllm.logger import enable_trace_function_call, init_logger
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.utils import random_uuid
from vllm.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .ec_transfer import ECTransferConfig
from .kv_events import KVEventsConfig
from .kv_transfer import KVTransferConfig
from .load import LoadConfig
from .lora import LoRAConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .parallel import ParallelConfig
from .profiler import ProfilerConfig
from .scheduler import SchedulerConfig
from .speculative import SpeculativeConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
else:
    PretrainedConfig = Any

    QuantizationConfig = Any

    KVCacheConfig = Any

logger = init_logger(__name__)


class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2s."""


IS_QUANTIZED = False
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# See https://github.com/vllm-project/vllm/issues/25689.


def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""

    return cfg.compilation_config.is_custom_op_enabled(
        "rms_norm"
    ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")


def enable_act_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either SiLU+Mul or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""
    return cfg.compilation_config.is_custom_op_enabled(
        "silu_and_mul"
    ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")


OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": False,
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = Field(default=None)
    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig.default_factory,
    )
    """Scheduler configuration."""
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    """Load configuration."""
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    """Attention configuration."""
    lora_config: LoRAConfig | None = None
    """LoRA configuration."""
    speculative_config: SpeculativeConfig | None = None
    """Speculative decoding configuration."""
    structured_outputs_config: StructuredOutputsConfig = Field(
        default_factory=StructuredOutputsConfig
    )
    """Structured outputs configuration."""
    observability_config: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    """Observability configuration."""
    quant_config: QuantizationConfig | None = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, one can append compilation arguments via
    -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

    You can specify the full compilation config like so:
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig)
    """Profiling configuration."""
    kv_transfer_config: KVTransferConfig | None = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: KVEventsConfig | None = None
    """The configurations for event publishing."""
    ec_transfer_config: ECTransferConfig | None = None
    """The configurations for distributed EC cache transfer."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the vLLM instance."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -02 is used by defult. See  OptimizationLevel for full
    description."""

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
        if self.attention_config:
            vllm_factors.append(self.attention_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.structured_outputs_config:
            vllm_factors.append(self.structured_outputs_config.compute_hash())
        if self.profiler_config:
            vllm_factors.append(self.profiler_config.compute_hash())
        else:
            vllm_factors.append("None")
        vllm_factors.append(self.observability_config.compute_hash())
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
        if self.ec_transfer_config:
            vllm_factors.append(self.ec_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_cudagraph_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_cudagraph_capture_size
        return self.compilation_config.bs_to_padded_graph_size[batch_size]

    @property
    def needs_dp_coordinator(self) -> bool:
        """
        Determine if the DPCoordinator process is needed.

        The DPCoordinator is needed in two cases:
        1. For MoE models with DP > 1: to handle wave coordination
           (even in external LB mode, since wave coordination runs in the coordinator)
        2. For non-MoE models in internal/hybrid LB mode: to collect and publish
           queue stats for load balancing across DP ranks

        Returns:
            True if DPCoordinator process is needed, False otherwise.
        """

        # For non-MoE models, only need coordinator in internal/hybrid LB mode
        # (for stats collection).
        return self.parallel_config.data_parallel_size > 1 and (
            self.model_config is None
            or self.model_config.is_moe
            or not self.parallel_config.data_parallel_external_lb
        )

    def enable_trace_function_call_for_thread(self) -> None:
        """
        Set up function tracing for the current thread,
        if enabled via the `VLLM_TRACE_FUNCTION` environment variable.
        """
        if envs.VLLM_TRACE_FUNCTION:
            tmp_dir = tempfile.gettempdir()
            # add username to tmp_dir to avoid permission issues
            tmp_dir = os.path.join(tmp_dir, getpass.getuser())
            filename = (
                f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                f"_thread_{threading.get_ident()}_at_{datetime.now()}.log"
            ).replace(" ", "_")
            log_path = os.path.join(
                tmp_dir,
                "vllm",
                f"vllm-instance-{self.instance_id}",
                filename,
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            enable_trace_function_call(log_path)

    @staticmethod
    def _get_quantization_config(
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        """Get the quantization config."""
        from vllm.platforms import current_platform

        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import get_quant_config

            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}"
                )
            quant_config.maybe_update_config(model_config.model)
            return quant_config
        return None

    @staticmethod
    def get_quantization_config(
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(
            copy.deepcopy(model_config), load_config
        )

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: list[str] | None = None,
    ) -> "VllmConfig":
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config
        model_config.model_arch_config = model_config.get_model_arch_config()

        return replace(self, model_config=model_config)

    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user.

        Args:
            config_obj: Configuration object to update.
            key: Attribute name.
            value: Default value (static or callable).
        """
        if getattr(config_obj, key) is None:
            # Some config values are known before initialization and are
            # hard coded.
            # Other values depend on the user given configuration, so they are
            # implemented with lambda functions and decided at run time.
            setattr(config_obj, key, value(self) if callable(value) else value)

    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten.

        If the user configuration does not specify a value for a default field
        and if the default field is still None after all user selections are
        applied, then default values will be applied to the field. User speciied
        fields will not be overridden by the default.

        Args:
            defaults: Dictionary of default values to apply.
        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            """Recursively apply defaults to config_obj, using self as root."""
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    def _post_init_kv_transfer_config(self) -> None:
        """Update KVTransferConfig based on top-level configs in VllmConfig.

        Right now, this function reads the offloading settings from
        CacheConfig and configures the KVTransferConfig accordingly.
        """
        if (kv_offloading_backend := self.cache_config.kv_offloading_backend) is None:
            return

        # If no KVTransferConfig is provided, create a default one.
        if self.kv_transfer_config is None:
            self.kv_transfer_config = KVTransferConfig()

        if (kv_offloading_size := self.cache_config.kv_offloading_size) is None:
            raise ValueError(
                "You must set kv_offloading_size when kv_offloading_backend is set."
            )
        num_kv_ranks = (
            self.parallel_config.tensor_parallel_size
            * self.parallel_config.pipeline_parallel_size
        )

        if kv_offloading_backend == "native":
            self.kv_transfer_config.kv_connector = "OffloadingConnector"
            self.kv_transfer_config.kv_connector_extra_config.update(
                {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
            )
        elif kv_offloading_backend == "lmcache":
            self.kv_transfer_config.kv_connector = "LMCacheConnectorV1"
            kv_gb_per_rank = kv_offloading_size / num_kv_ranks
            self.kv_transfer_config.kv_connector_extra_config = {
                "lmcache.local_cpu": True,
                "lmcache.max_local_cpu_size": kv_gb_per_rank,
            }

        # This is the same for all backends
        self.kv_transfer_config.kv_role = "kv_both"

    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""

        # To give each torch profile run a unique instance name.
        self.instance_id = f"{time.time_ns()}"

        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(self.load_config)

            self.parallel_config.is_moe_model = self.model_config.is_moe

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_model_config(self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config
            )

        executor_backend = self.parallel_config.distributed_executor_backend
        executor_supports_async_sched = executor_backend in (
            "mp",
            "uni",
            "external_launcher",
        )

        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "Async scheduling is not yet compatible with "
                    "pipeline_parallel_size > 1."
                )
            # Currently, async scheduling only support eagle speculative
            # decoding.
            if self.speculative_config is not None:
                if self.speculative_config.method not in get_args(EagleModelTypes):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP kind of speculative decoding."
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            if not executor_supports_async_sched:
                raise ValueError(
                    "Currently, async scheduling only supports `mp`, `uni`, or "
                    "`external_launcher` distributed executor backend, but you chose "
                    f"`{executor_backend}`."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if self.parallel_config.pipeline_parallel_size > 1:
                logger.warning_once(
                    "Async scheduling is not yet supported with "
                    "pipeline_parallel_size > 1 and will be disabled.",
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend (only `mp`, `uni`, and "
                    "`external_launcher` are supported).",
                    executor_backend,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        logger.info_once(
            "Asynchronous scheduling is %s.",
            "enabled" if self.scheduler_config.async_scheduling else "disabled",
        )

        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            if self.scheduler_config.async_scheduling:
                logger.info_once(
                    "Disabling NCCL for DP synchronization "
                    "when using async scheduling.",
                    scope="local",
                )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        from vllm.platforms import current_platform

        if (
            self.model_config is not None
            and self.scheduler_config.enable_chunked_prefill
            and self.model_config.dtype == torch.float32
            and current_platform.get_device_capability() == (7, 5)
        ):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels."
            )

        if (
            self.optimization_level > OptimizationLevel.O0
            and self.model_config is not None
            and self.model_config.enforce_eager
        ):
            logger.warning("Enforce eager set, overriding optimization level to -O0")
            self.optimization_level = OptimizationLevel.O0

        if self.compilation_config.backend == "eager" or (
            self.compilation_config.mode is not None
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        def has_blocked_weights():
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)

        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.info(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.fuse_gemm_comms:
            self.compilation_config.pass_config.enable_sp = True
        if self.compilation_config.pass_config.enable_sp:
            if "-rms_norm" in self.compilation_config.custom_ops:
                logger.warning(
                    "RMS norm force disabled, sequence parallelism might break"
                )
            else:
                self.compilation_config.custom_ops.append("+rms_norm")

        if current_platform.support_static_graph_mode():
            # if cudagraph_mode has full cudagraphs, we need to check support
            if model_config := self.model_config:
                if (
                    self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                    and model_config.pooler_config is not None
                ):
                    logger.warning_once(
                        "Pooling models do not support full cudagraphs. "
                        "Overriding cudagraph_mode to PIECEWISE."
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                elif (
                    model_config.is_encoder_decoder
                    and self.compilation_config.cudagraph_mode
                    not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
                ):
                    logger.info_once(
                        "Encoder-decoder models do not support %s. "
                        "Overriding cudagraph_mode to FULL_DECODE_ONLY.",
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = (
                        CUDAGraphMode.FULL_DECODE_ONLY
                    )

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # override related settings when enforce eager
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                self.compilation_config.cudagraph_num_of_warmups = 1

            self._set_cudagraph_sizes()
        else:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if self.cache_config.kv_sharing_fast_prefill:
            if (
                self.speculative_config is not None
                and self.speculative_config.use_eagle()
            ):
                raise ValueError(
                    "Fast prefill optimization for KV sharing is not "
                    "compatible with EAGLE as EAGLE requires correct logits "
                    "for all tokens while fast prefill gives incorrect logits "
                    "for prompt tokens."
                )

            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings."
            )
        # TODO: Move after https://github.com/vllm-project/vllm/pull/26847 lands
        self._set_compile_ranges()

        if (
            self.model_config
            and self.model_config.architecture == "WhisperForConditionalGeneration"
            and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            logger.warning(
                "Whisper is known to have issues with "
                "forked workers. If startup is hanging, "
                "try setting 'VLLM_WORKER_MULTIPROC_METHOD' "
                "to 'spawn'."
            )

        if (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
            and not self.cache_config.enable_prefix_caching
        ):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled. "
                "Use --enable-prefix-caching to enable."
            )
        if (
            self.kv_events_config is not None
            and self.kv_events_config.publisher != "null"
            and not self.kv_events_config.enable_kv_cache_events
        ):
            logger.warning(
                "KV cache events are disabled, "
                "but the scheduler is configured to publish them. "
                "Modify KVEventsConfig.enable_kv_cache_events "
                "to True to enable."
            )
        current_platform.check_and_update_config(self)

        # If DCP, ensure the block size is right.
        if self.parallel_config.decode_context_parallel_size > 1:
            if self.parallel_config.dcp_kv_cache_interleave_size > 1 and (
                self.parallel_config.cp_kv_cache_interleave_size
                != self.parallel_config.dcp_kv_cache_interleave_size
            ):
                self.parallel_config.cp_kv_cache_interleave_size = (
                    self.parallel_config.dcp_kv_cache_interleave_size
                )
                logger.warning_once(
                    "cp_kv_cache_interleave_size is overridden by dcp_kv_cache"
                    "_interleave_size. And dcp-kv-cache-interleave-size will be "
                    "deprecated when PCP is fully supported."
                )
            assert (
                self.parallel_config.cp_kv_cache_interleave_size
                <= self.cache_config.block_size
                and self.cache_config.block_size
                % self.parallel_config.cp_kv_cache_interleave_size
                == 0
            ), (
                f"Block_size({self.cache_config.block_size}) should be greater "
                "than or equal to and divisible by cp_kv_cache_interleave_size "
                f"({self.parallel_config.cp_kv_cache_interleave_size})."
            )

        # Do this after all the updates to compilation_config.mode
        effective_dp_size = (
            self.parallel_config.data_parallel_size
            if self.model_config is None or self.model_config.is_moe
            else 1
        )
        self.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=self.parallel_config.all2all_backend,
            data_parallel_size=effective_dp_size,
        )

        if self.compilation_config.pass_config.enable_sp:
            # With pipeline parallelism or dynamo partitioning,
            # native rms norm tracing errors due to incorrect residual shape.
            # Use custom rms norm to unblock. In the future,
            # the pass will operate on higher-level IR to avoid the issue.
            # TODO: https://github.com/vllm-project/vllm/issues/27894
            if self.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                logger.warning(
                    "Sequence parallelism is enabled, but running in wrong "
                    "vllm compile mode: %s.",
                    self.compilation_config.mode,
                )

            is_fullgraph = (
                self.compilation_config.use_inductor_graph_partition
                or len(self.compilation_config.splitting_ops) == 0
            )
            if self.parallel_config.pipeline_parallel_size > 1 or not is_fullgraph:
                if "-rms_norm" not in self.compilation_config.custom_ops:
                    self.compilation_config.custom_ops.append("+rms_norm")
                else:
                    regime = (
                        "Dynamo partition"
                        if not is_fullgraph
                        else "pipeline parallelism"
                    )
                    logger.warning_once(
                        "Sequence parallelism not supported with "
                        "native rms_norm when using %s, "
                        "this will likely lead to an error.",
                        regime,
                    )

        # final check of cudagraph mode after all possible updates
        if current_platform.is_cuda_alike():
            if (
                self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and self.model_config is not None
                and not self.model_config.disable_cascade_attn
                and not self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()  # noqa: E501
            ):
                logger.warning_once(
                    "No piecewise cudagraph for executing cascade attention."
                    " Will fall back to eager execution if a batch runs "
                    "into cascade attentions."
                )

            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert self.compilation_config.mode == CompilationMode.VLLM_COMPILE, (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )

        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
            ], (
                "Microbatching currently only supports the deepep_low_latency and "
                f"deepep_high_throughput all2all backend. {a2a_backend} is not "
                "supported. To fix use --all2all-backend=deepep_low_latency or "
                "--all2all-backend=deepep_high_throughput and install the DeepEP"
                " kernels."
            )

            if not self.model_config.disable_cascade_attn:
                self.model_config.disable_cascade_attn = True
                logger.warning_once("Disabling cascade attention when DBO is enabled.")

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        if not self.scheduler_config.disable_hybrid_kv_cache_manager:
            # logger should only print warning message for hybrid models. As we
            # can't know whether the model is hybrid or not now, so we don't log
            # warning message here and will log it later.
            if not current_platform.support_hybrid_kv_cache():
                # Hybrid KV cache manager is not supported on non-GPU platforms.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_transfer_config is not None:
                # NOTE(Kuntai): turn HMA off for connector for now.
                # TODO(Kuntai): have a more elegent solution to check and
                # turn off HMA for connector that does not support HMA.
                logger.warning(
                    "Turning off hybrid kv cache manager because "
                    "`--kv-transfer-config` is set. This will reduce the "
                    "performance of vLLM on LLMs with sliding window attention "
                    "or Mamba attention. If you are a developer of kv connector"
                    ", please consider supporting hybrid kv cache manager for "
                    "your connector by making sure your connector is a subclass"
                    " of `SupportsHMA` defined in kv_connector/v1/base.py."
                )
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_events_config is not None:
                # Hybrid KV cache manager is not compatible with KV events.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if (
                self.model_config is not None
                and self.model_config.attention_chunk_size is not None
            ):
                if (
                    self.speculative_config is not None
                    and self.speculative_config.use_eagle()
                ):
                    # Hybrid KV cache manager is not yet supported with chunked
                    # local attention + eagle.
                    self.scheduler_config.disable_hybrid_kv_cache_manager = True
                elif not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
                    logger.warning(
                        "There is a latency regression when using chunked local"
                        " attention with the hybrid KV cache manager. Disabling"
                        " it, by default. To enable it, set the environment "
                        "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
                    )
                    # Hybrid KV cache manager is not yet supported with chunked
                    # local attention.
                    self.scheduler_config.disable_hybrid_kv_cache_manager = True

        if self.compilation_config.debug_dump_path:
            self.compilation_config.debug_dump_path = (
                self.compilation_config.debug_dump_path.absolute().expanduser()
            )
        if envs.VLLM_DEBUG_DUMP_PATH is not None:
            env_path = Path(envs.VLLM_DEBUG_DUMP_PATH).absolute().expanduser()
            if self.compilation_config.debug_dump_path:
                logger.warning(
                    "Config-specified debug dump path is overridden"
                    " by VLLM_DEBUG_DUMP_PATH to %s",
                    env_path,
                )
            self.compilation_config.debug_dump_path = env_path

        def has_blocked_weights():
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # Handle the KV connector configs
        self._post_init_kv_transfer_config()

    def update_sizes_for_sequence_parallelism(self, possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled",
                removed_sizes,
                self.parallel_config.tensor_parallel_size,
            )

        return [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_cudagraph_sizes(self):
        """
        vLLM defines the default candidate list of batch sizes for CUDA graph
        capture as:

        ```python
        max_graph_size = min(max_num_seqs * 2, 512)
        # 1, 2, 4, then multiples of 8 up to 256 and then multiples of 16
        # up to max_graph_size
        cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_graph_size + 1, 16))

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in ascending order).

        These sizes are used to capture and reuse CUDA graphs for
        performance-critical paths (e.g., decoding). Capturing enables
        significantly faster kernel dispatch by avoiding Python overhead. The
        list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
        most GPUs), which controls the total allowed number of tokens in a
        batch. Since each sequence may have a variable number of tokens, the
        maximum usable batch size will depend on actual sequence lengths.

        Example:
            With `max_num_batched_tokens = 8192`, and typical sequences
            averaging ~32 tokens, most practical batch sizes fall below 256.
            However, the system will still allow capture sizes up to 512 if
            shape and memory permit.

        Note:
            If users explicitly specify cudagraph capture sizes in the
            compilation config, those will override this default logic.
            At runtime:

            - If batch size <= one of the `cudagraph_capture_sizes`, the closest
            padded CUDA graph will be used.
            - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
            not be used.
        """

        if (
            self.model_config is not None
            and not self.model_config.enforce_eager
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            if max_cudagraph_capture_size is None:
                decode_query_len = 1
                if (
                    self.speculative_config
                    and self.speculative_config.num_speculative_tokens
                ):
                    decode_query_len += self.speculative_config.num_speculative_tokens
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
                )
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )

            # determine the cudagraph_capture_sizes
            if self.compilation_config.cudagraph_capture_sizes is not None:
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                # de-duplicate the sizes provided by the config
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                # sort to make sure the sizes are in ascending order
                cudagraph_capture_sizes.sort()
            else:
                cudagraph_capture_sizes = [
                    i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                ]
                if max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )

            if (
                self.parallel_config.tensor_parallel_size > 1
                and self.compilation_config.pass_config.enable_sp
            ):
                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                    cudagraph_capture_sizes
                )

            # user-specific compilation_config.max_cudagraph_capture_size get
            # truncated to valid_max_size when they are inconsistent.
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            if (
                self.compilation_config.max_cudagraph_capture_size is not None
                and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # raise error only when both two flags are user-specified
                # and they are inconsistent with each other
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )

                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d",
                    valid_max_size,
                )
            # always set the final max_cudagraph_capture_size
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                # If users have specified capture sizes, we only need to
                # compare the lens before and after modification since the modified
                # list is only the subset of the original list.
                logger.warning(
                    (
                        "cudagraph_capture_sizes specified in compilation_config"
                        " %s is overridden by config %s"
                    ),
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            # always write back the final sizes
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

        else:
            # no cudagraph in use
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # complete the remaining process.
        self.compilation_config.post_init_cudagraph_sizes()

    def _set_compile_ranges(self):
        """
        Set the compile ranges for the compilation config.
        """
        compilation_config = self.compilation_config
        computed_compile_ranges_split_points = []

        # The upper bound of the compile ranges is the max_num_batched_tokens
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        if max_num_batched_tokens is not None:
            computed_compile_ranges_split_points.append(max_num_batched_tokens)

        # Add the compile ranges for flashinfer
        if compilation_config.pass_config.fuse_allreduce_rms:
            tp_size = self.parallel_config.tensor_parallel_size
            max_size = compilation_config.pass_config.flashinfer_max_size(tp_size)
            if max_size is not None:
                max_token_num = max_size // (
                    self.model_config.get_hidden_size()
                    * self.model_config.dtype.itemsize
                )
                if (
                    max_num_batched_tokens is not None
                    and max_token_num < max_num_batched_tokens
                ):
                    computed_compile_ranges_split_points.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below allreduce-rms fusion threshold, "
                        "allreduce-rms fusion will be enabled for all num_tokens."
                    )

        if compilation_config.compile_ranges_split_points is not None:
            for x in compilation_config.compile_ranges_split_points:
                assert isinstance(x, int)
                assert x > 0, f"Invalid compile range split point: {x}"
                if (
                    max_num_batched_tokens is not None
                    and x < max_num_batched_tokens
                    and x > 1
                ):
                    computed_compile_ranges_split_points.append(x)
        compilation_config.compile_ranges_split_points = sorted(
            computed_compile_ranges_split_points
        )

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import SequenceClassificationConfig

            SequenceClassificationConfig.verify_and_update_config(self)

        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
            self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                "runai_streamer",
                "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from S3, 'load_format' "
                    f"must be 'runai_streamer' or 'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

    def compile_debug_dump_path(self) -> Path | None:
        """Returns a rank-aware path for dumping
        torch.compile debug information.
        """
        if self.compilation_config.debug_dump_path is None:
            return None
        tp_rank = self.parallel_config.rank
        dp_rank = self.parallel_config.data_parallel_index
        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
        path = self.compilation_config.debug_dump_path / append_path
        return path

    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"enable_return_routed_experts={self.model_config.enable_return_routed_experts}, "  # noqa
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"structured_outputs_config={self.structured_outputs_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"enable_chunked_prefill={self.scheduler_config.enable_chunked_prefill}, "  # noqa
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}"
        )

    @model_validator(mode="after")
    def validate_mamba_block_size(self) -> "VllmConfig":
        if self.model_config is None:
            return self
        mamba_block_size_is_set = (
            self.cache_config.mamba_block_size is not None
            and self.cache_config.mamba_block_size != self.model_config.max_model_len
        )
        if mamba_block_size_is_set and not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "--mamba-block-size can only be set with --enable-prefix-caching"
            )
        return self


_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from vllm.compilation.counter import compilation_counter

    num_models_seen = compilation_counter.num_models_seen
    try:
        # Clear the compilation config cache when context changes.
        # This is needed since the old config may have been accessed
        # and cached before the new config is set.
        get_cached_compilation_config.cache_clear()

        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            vllm_config.compilation_config.custom_op_log_check()

        if (
            check_compile
            and vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and compilation_counter.num_models_seen == num_models_seen
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model,
            )
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    return get_current_vllm_config().compilation_config


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_vllm_config' pytest fixture from tests/conftest.py."
        )
    return _current_vllm_config


def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_vllm_config(
    vllm_config: VllmConfig,
    layer_type: type[T],
    layer_names: list[str] | None = None,
) -> dict[str, T]:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vLLM config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    if layer_names is None:
        layer_names = list(vllm_config.compilation_config.static_forward_context.keys())

    forward_context = vllm_config.compilation_config.static_forward_context

    return {
        layer_name: forward_context[layer_name]
        for layer_name in layer_names
        if isinstance(forward_context[layer_name], layer_type)
    }
