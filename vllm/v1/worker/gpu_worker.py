# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import (
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
)
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.parallel_state import (
    Handle,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.weight_transfer import (
    WeightTransferEngine,
    WeightTransferEngineFactory,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
)
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check, with_gpu_sync_check
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from ...model_executor.model_loader import TensorizerLoader
from .gpu.warmup import warmup_kernels
from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class Worker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Weight transfer engine is created in `load_model` once the model
        # is available, since the engine needs a reference to the model.
        self.weight_transfer_engine: WeightTransferEngine | None = None
        self._weight_update_active = False

        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        # Profiler wrapper is created lazily in profile() when start is called,
        # so we have all the information needed for proper trace naming.
        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config

        # Only validate profiler config is valid, don't instantiate yet
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list[Handle] = []

        # Resolved lazily on first sleep/wake; persists worker-process state.
        self._sleep_mode_backend: SleepModeBackend | None = None

    def _get_sleep_mode_backend(self) -> "SleepModeBackend":
        if self._sleep_mode_backend is None:
            from vllm.device_allocator.sleep_mode_backend import (
                SleepModeBackendFactory,
            )

            self._sleep_mode_backend = SleepModeBackendFactory.create_backend(
                self.vllm_config.model_config
            )
        return self._sleep_mode_backend

    def sleep(self, level: int = 1) -> None:
        torch.accelerator.synchronize()
        free_bytes_before_sleep = torch.accelerator.get_memory_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        self._get_sleep_mode_backend().suspend(level)

        torch.accelerator.synchronize()
        deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)
        while True:
            free_bytes_after_sleep, total = torch.accelerator.get_memory_info()
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            if freed_bytes >= 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        self._get_sleep_mode_backend().resume(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if (
            current_platform.is_cuda_alike()
            and not self.vllm_config.model_config.enable_cumem_allocator
        ):
            return nullcontext()

        if (
            current_platform.is_xpu()
            and not self.vllm_config.model_config.enable_sleep_mode
        ):
            return nullcontext()

        if current_platform.is_cpu():
            return nullcontext()

        allocator = get_mem_allocator_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "CuMem allocator can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)

    @contextmanager
    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        """Temporarily set max_split_size_mb to reduce allocator fragmentation at the
        cost of more cudaMalloc calls (negligible in practice). Restores the original
        value on exit."""
        if not current_platform.is_cuda():
            yield
            return

        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        match = re.search(r"max_split_size_mb:(\d+)", conf)
        original_value = match.group(1) if match else None

        torch._C._accelerator_setAllocatorSettings(
            f"max_split_size_mb:{max_split_size_mb}"
        )
        try:
            yield
        finally:
            # PyTorch defaults to SIZE_MAX (no limit).
            _SIZE_MAX_MB = (2**64 - 1) // (1024 * 1024)
            restore = original_value if original_value else str(_SIZE_MAX_MB)
            torch._C._accelerator_setAllocatorSettings(f"max_split_size_mb:{restore}")

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks.
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                # NOTE(patch pr45026): local_world_size is derived from
                # parallel_config.nnodes, which is only set for the "mp"
                # multi-node backend. With the "ray"/"external_launcher"
                # backends nnodes stays 1, so local_world_size collapses to
                # the full world_size and this check wrongly fires on
                # cross-node deployments. assigned_physical_gpu_ids is already
                # per-node and the local_rank bound above fully validates the
                # mapping for these backends, so skip the check for them.
                if parallel_config.distributed_executor_backend not in (
                    "ray",
                    "external_launcher",
                ):
                    assert self.parallel_config.local_world_size <= len(
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({self.parallel_config.local_world_size})"
                        " exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

        if self.vllm_config.weight_transfer_config is not None:
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config,
                self.device,
                self.model_runner.get_model(),
            )

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return self._reserve_mm_ipc_gpu_memory(kv_cache_memory_bytes)

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

            profile_torch_peak = torch.accelerator.memory_stats(self.device).get(
                "allocated_bytes.all.peak", 0
            )

            # Profile CUDA graph memory if graphs will be captured.
            # Skip on ROCm/HIP/XPU as graph pool handles and get_memory_info
            # behave differently and can produce incorrect/negative estimates.
            cudagraph_memory_estimate = 0
            if (
                current_platform.is_cuda()
                and self.vllm_config.compilation_config.cudagraph_mode
                != CUDAGraphMode.NONE
            ):
                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # Use the pre-cudagraph torch peak to avoid double-counting.
        profile_result.torch_peak_increase = (
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )

        # On ROCm, cudagraph_memory_estimate is always 0 so this is a no-op.
        # On CUDA, respect the opt-in flag as originally designed.
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.cudagraph_memory_estimate = cudagraph_memory_estimate
        self.cudagraph_memory_persistent_estimate = getattr(
            self.model_runner, "cudagraph_memory_persistent_estimate", 0
        )
        self.cudagraph_memory_graph_pool_estimate = getattr(
            self.model_runner, "cudagraph_memory_graph_pool_estimate", 0
        )

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )

        if cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                equiv_util = round(current_util - cg_util_delta, 4)
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(
                    "CUDA graph memory profiling is enabled (default since "
                    "v0.21.0). The current --gpu-memory-utilization=%.4f is "
                    "equivalent to --gpu-memory-utilization=%.4f without "
                    "CUDA graph memory profiling. To maintain the same "
                    "effective KV cache size as before, increase "
                    "--gpu-memory-utilization to %.4f. To disable, set "
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                    current_util,
                    equiv_util,
                    suggested_util,
                )
            else:
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.warning(
                    "CUDA graph memory profiling is disabled "
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                    "Without it, CUDA graph memory is not accounted for "
                    "during KV cache allocation, which may require lowering "
                    "--gpu-memory-utilization to avoid OOM. Consider "
                    "re-enabling it (the default as of v0.21.0) and increasing "
                    "--gpu-memory-utilization from %.4f to %.4f.",
                    current_util,
                    suggested_util,
                )

        return self._reserve_mm_ipc_gpu_memory(
            int(self.available_kv_cache_memory_bytes)
        )

    @staticmethod
    def _uses_pynvvideocodec_video_backend(mm_config) -> bool:
        video_kwargs = mm_config.media_io_kwargs.get("video", {})
        video_loader_backend = (
            video_kwargs.get("video_backend") or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        codec_backend = video_kwargs.get("backend")
        return (
            video_loader_backend == PYNVVIDEOCODEC_VIDEO_BACKEND
            or codec_backend == PYNVVIDEOCODEC_VIDEO_BACKEND
        )

    def _reserve_mm_ipc_gpu_memory(self, available_kv_cache_memory_bytes: int) -> int:
        """Carve frontend multimodal GPU memory out of the KV cache.

        The frontend (API-server) process allocates GPU memory for hardware
        multimodal decoding. Raw decoded frames are bounded by
        ``mm_ipc_gpu_memory_gb`` and acquired by the frontend semaphore. Some
        decoders also keep persistent surfaces around; reserve a fixed upper
        bound for those when the corresponding backend is configured.
        """
        mm_config = self.model_config.multimodal_config
        if mm_config is None:
            return available_kv_cache_memory_bytes

        raw_frame_reserved_bytes = int(mm_config.mm_ipc_gpu_memory_gb * GiB_bytes)
        # Each api_server_count process runs its OWN decoder surfaces + NVDEC/CUVID
        # CUDA context on the GPU, outside this (worker) memory pool. Reserve that
        # per-server footprint x api_server_count so gpu_memory_utilization bounds
        # TOTAL GPU usage across all API-server processes. Without the multiply,
        # HW decode overshoots the budget by ~(api_server_count-1) x per-server and
        # OOMs at high gmu, while SW decode (no per-server GPU allocation) does not.
        num_api_servers = max(1, getattr(self.parallel_config, "_api_process_count", 1))
        per_server_decoder_bytes = (
            PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES
            * PYNVVIDEOCODEC_MAX_RETAINED_DECODERS
            + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
        )
        decoder_reserved_bytes = (
            num_api_servers * per_server_decoder_bytes
            if self._uses_pynvvideocodec_video_backend(mm_config)
            else 0
        )
        reserved_bytes = raw_frame_reserved_bytes + decoder_reserved_bytes
        if reserved_bytes <= 0:
            return available_kv_cache_memory_bytes

        remaining = available_kv_cache_memory_bytes - reserved_bytes
        if remaining <= 0:
            raise ValueError(
                f"frontend multimodal GPU decoding reserves "
                f"{format_gib(reserved_bytes)} GiB "
                f"({format_gib(raw_frame_reserved_bytes)} GiB raw-frame budget, "
                f"{format_gib(decoder_reserved_bytes)} GiB decoder cache budget), "
                f"but only {format_gib(available_kv_cache_memory_bytes)} GiB is "
                "available for the KV cache. Reduce mm_ipc_gpu_memory_gb, use a "
                "different video backend, or increase gpu_memory_utilization."
            )
        logger.info_once(
            "Reserving %s GiB of GPU memory for frontend multimodal decoding "
            "(%s GiB raw-frame semaphore budget, %s GiB decoder+CUDA-context "
            "across %d API server(s) @ %s GiB/server); "
            "KV cache memory reduced to %s GiB.",
            format_gib(reserved_bytes),
            format_gib(raw_frame_reserved_bytes),
            format_gib(decoder_reserved_bytes),
            num_api_servers,
            format_gib(per_server_decoder_bytes),
            format_gib(remaining),
        )
        return remaining

    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available.

        Returned dict is keyed by `(pp_rank, tp_rank)`.
        """

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.
        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()

    @instrument(span_name="Warmup (GPU)")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        warmup_sizes: list[int] = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            graph_pool_estimate = self.cudagraph_memory_graph_pool_estimate
            if graph_pool_estimate == 0:
                graph_pool_estimate = self.cudagraph_memory_estimate
            diff = abs(cuda_graph_memory_bytes - graph_pool_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(graph_pool_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )
            if self.cudagraph_memory_persistent_estimate > 0:
                logger.info(
                    "CUDA graph persistent memory: %s GiB (estimated).",
                    GiB(self.cudagraph_memory_persistent_estimate),
                )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            cuda_graph_memory_for_sizing = cuda_graph_memory_bytes
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                cuda_graph_memory_for_sizing = max(
                    cuda_graph_memory_for_sizing,
                    self.cudagraph_memory_estimate,
                )

            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_for_sizing
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and "
                f"{format_gib(cuda_graph_memory_for_sizing)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            from vllm.compilation.compiler_interface import (
                trigger_inductor_lazy_init,
            )

            trigger_inductor_lazy_init(self.device)

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        from vllm.utils.jit_monitor import activate as activate_jit_monitor

        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )

        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        enable_gpu_sync_check()

        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_compilation_match_table(self) -> dict[str, int]:
        from vllm.compilation.passes.vllm_inductor_pass import get_match_table

        return get_match_table()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    @with_gpu_sync_check
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    @with_gpu_sync_check
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config

        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # currently only supported by V1 GPUModelRunner
            assert not self.use_v2_model_runner
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # Build the full trace name
            if profile_prefix:
                trace_name = f"{profile_prefix}_{rank_suffix}"
            else:
                trace_name = rank_suffix

            # Create the profiler wrapper only on the first start call
            if self.profiler is None:
                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                    logger.debug(
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
                    logger.debug("Starting CUDA profiler")
                else:
                    # Config validation should prevent this code being reached
                    raise ValueError(
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _check_weight_transfer_engine(self) -> None:
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info
        """
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        # Parse dict into backend-specific typed dataclass
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def start_weight_update(self) -> None:
        """
        Start a new weight update session.

        Delegates engine-specific preparation (e.g. layerwise reload setup) to
        the configured weight transfer engine. The worker only tracks that a
        session is active.
        """
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is already "
                "active. Call finish_weight_update first."
            )

        self.weight_transfer_engine.start_weight_update()
        self._weight_update_active = True

    def update_weights(self, update_info: dict) -> None:
        """
        Receive one weight update chunk from the trainer.

        start_weight_update must be called before update_weights and
        finish_weight_update must be called after all chunks have been sent.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before update_weights."
            )

        try:
            self.weight_transfer_engine.update_weights(update_info)
        except BaseException:
            self._weight_update_active = False
            raise

    def finish_weight_update(self) -> None:
        """Finish the current weight update session."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        if not self._weight_update_active:
            raise RuntimeError(
                "finish_weight_update called without a matching start_weight_update."
            )

        self.weight_transfer_engine.finish_weight_update()
        self._weight_update_active = False

    def shutdown(self) -> None:
        gc.unfreeze()

        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if ensure_ec_transfer_shutdown is not None:
            ensure_ec_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

        # Release GPU resources held by the model runner so that memory
        # can be reclaimed when running in-process
        if model_runner := getattr(self, "model_runner", None):
            model_runner.shutdown()

        # Release kept-alive cumem pools while the pluggable allocator wrappers
        # and callbacks are still alive, so MemPool teardown is not deferred to
        # interpreter finalization (pytorch/pytorch#145168).
        if current_platform.is_cuda_alike():
            from vllm.device_allocator.cumem import CuMemAllocator

            if CuMemAllocator.instance is not None:
                CuMemAllocator.instance.release_pools()

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()
    override_envs_for_eplb(
        parallel_config,
        moe_backend=getattr(vllm_config.kernel_config, "moe_backend", None),
    )
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
