# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import platform
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional

import psutil
import torch

from vllm.logger import init_logger
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS

from .interface import CpuArchEnum, Platform, PlatformEnum, _Backend

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def get_max_threads(pid=0):
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(pid))
    elif platform.system() == 'Darwin':
        return os.cpu_count()
    else:
        raise NotImplementedError("Unsupported OS")


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.get_cpu_architecture() == CpuArchEnum.POWERPC:
            return [torch.bfloat16, torch.float32]
        elif sys.platform.startswith(
                "darwin") and self.get_cpu_architecture() == CpuArchEnum.ARM:
            # TODO: change this condition to check if the platform support bf16
            # instead of checking the OS. For instance M2 shall supports bf16
            # already. But we need to modify `cpu_extension.cmake` to activate
            # the feature in the build.
            return [torch.float16, torch.float32]
        # x86/aarch64 CPU has supported both bf16 and fp16 natively.
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if selected_backend and selected_backend != _Backend.TORCH_SDPA:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        if use_mla:
            logger.info("Using CPU MLA backend.")
            return "vllm.attention.backends.cpu_mla.CPUMLABackend"
        logger.info("Using Torch SDPA backend.")
        if use_v1:
            return "vllm.v1.attention.backends.cpu_attn.TorchSDPABackend"
        else:
            return "vllm.attention.backends.torch_sdpa.TorchSDPABackend"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm.envs as envs
        from vllm.utils import GiB_bytes
        model_config = vllm_config.model_config
        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        if not model_config.enforce_eager:
            model_config.enforce_eager = True

        model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config

        ipex_available = find_spec("intel_extension_for_pytorch") is not None

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128 if ipex_available else 16

        if not ipex_available and cache_config.block_size != 16:
            raise RuntimeError(
                f"--block-size={cache_config.block_size} requires"
                " intel_extension_for_pytorch")

        scheduler_config = vllm_config.scheduler_config
        if ((scheduler_config.chunked_prefill_enabled
             or cache_config.enable_prefix_caching)
                and cache_config.cache_dtype != "auto"):
            raise RuntimeError("Chunked-prefill and prefix-cache on the CPU "
                               "backend is not compatible with FP8 KV cache.")

        if cache_config.cache_dtype == "fp8_e4m3":
            cache_config.cache_dtype = "fp8_e5m2"
            logger.warning(
                "CPU backend doesn't support fp8_e4m3 KV cache type, "
                "cast to fp8_e5m2.")

        if (cache_config.cache_dtype != "auto"
                and model_config.dtype == torch.half):
            logger.warning("FP8 KV cache on the CPU backend only does not"
                           " support fp16 for now, cast to bf16.")
            model_config.dtype = torch.bfloat16

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                cache_config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) "
                    "for CPU backend is not set, using 4 by default.")
            else:
                cache_config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore # noqa
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")

        parallel_config = vllm_config.parallel_config
        if (parallel_config.world_size > 1
                and parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "mp"):
            logger.warning(("%s is not supported on CPU, fallback to mp "
                            "distributed executor backend."),
                           parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "mp"
        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.cpu_worker.CPUWorker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.cpu_worker.CPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.cpu_worker.CPUWorker"

        # Note: workaround for v1 gpu_model_runner
        from vllm.config import CompilationLevel
        vllm_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = vllm_config.compilation_config
        if (envs.VLLM_USE_V1 and vllm_config.compilation_config.level
                == CompilationLevel.PIECEWISE):
            compilation_config.level = CompilationLevel.DYNAMO_ONCE
            compilation_config.backend = "eager"
            compilation_config.custom_ops += ["none"]
            compilation_config.inductor_compile_config.update({
                "dce":
                True,
                "size_asserts":
                False,
                "nan_asserts":
                False,
                "memory_planning":
                True,
                "epilogue_fusion":
                True,
            })

        if vllm_config.lora_config is not None:
            compilation_config.level = CompilationLevel.NO_COMPILATION

        assert vllm_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Note: to avoid the error 'nthreads cannot be larger than environment
        #  variable "NUMEXPR_MAX_THREADS" (64)'.
        os.environ["NUMEXPR_MAX_THREADS"] = str(get_max_threads())

        # Set default threads num for OpenMP parallel
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Share the cpusets list among ranks by spawning process instead
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Intel OpenMP setting
        ld_prealod_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_prealod_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ['KMP_BLOCKTIME'] = "1"
            # Prevents the CPU to run into low performance state
            os.environ['KMP_TPAUSE'] = "0"
            # Provides fine granularity parallelism
            os.environ['KMP_FORKJOIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_PLAIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_REDUCTION_BARRIER_PATTERN'] = "dist,dist"

        # To hint IPEX uses shared memory based AllReduce
        os.environ["LOCAL_WORLD_SIZE"] = str(
            vllm_config.parallel_config.tensor_parallel_size)

        if vllm_config.model_config and vllm_config.model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled.")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on CPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"  # noqa

    @classmethod
    def supports_structured_output(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True
