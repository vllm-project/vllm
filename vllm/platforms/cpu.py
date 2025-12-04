# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil
import regex as re
import torch

from vllm import envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.logger import init_logger

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def get_max_threads(pid=0):
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(pid))
    elif platform.system() == "Darwin":
        return os.cpu_count()
    else:
        raise NotImplementedError("Unsupported OS")


@dataclass
class LogicalCPUInfo:
    id: int = -1
    physical_core: int = -1
    numa_node: int = -1

    @classmethod
    def _int(cls, value: str) -> int:
        try:
            int_value = int(value)
        except Exception:
            int_value = -1
        return int_value

    @staticmethod
    def json_decoder(obj_dict: dict):
        id = obj_dict.get("cpu")
        physical_core = obj_dict.get("core")
        numa_node = obj_dict.get("node")

        if not (id is None or physical_core is None or numa_node is None):
            return LogicalCPUInfo(
                id=LogicalCPUInfo._int(id),
                physical_core=LogicalCPUInfo._int(physical_core),
                numa_node=LogicalCPUInfo._int(numa_node),
            )
        else:
            return obj_dict


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"
    device_control_env_var = "CPU_VISIBLE_MEMORY_NODES"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.get_cpu_architecture() == CpuArchEnum.POWERPC:
            return [torch.bfloat16, torch.float32]
        elif self.get_cpu_architecture() == CpuArchEnum.ARM and sys.platform.startswith(
            "darwin"
        ):
            if (
                subprocess.check_output(
                    ["sysctl -n hw.optional.arm.FEAT_BF16"], shell=True
                ).strip()
                == b"1"
            ):
                return [torch.bfloat16, torch.float16, torch.float32]
            return [torch.float16, torch.float32]
        elif self.get_cpu_architecture() == CpuArchEnum.RISCV:
            # Workaround for Issue #25655: RISC-V scheduler bug with float16
            #
            # Background:
            # - RISC-V currently uses scalar code path
            # - There is a latent bug in the vLLM scheduler that provides
            # invalid
            #   physical_block_idx values under certain conditions
            # - This bug causes segmentation faults when using float16
            # dtype on RISC-V
            # - Testing shows that forcing float32 successfully bypasses
            # this issue
            #
            # Technical details:
            # - The bug manifests as out-of-bounds physical_block_idx in
            # block_tables
            # - Only occurs on RISC-V hardware
            # tested on Sophgo SG2044
            # - Does not reproduce on x86 or other architectures
            # - Root cause is in Python-level scheduling logic,
            # not C++ kernels
            #
            # This is a temporary workaround until the scheduler bug is fixed.
            # See: https://github.com/vllm-project/vllm/issues/25655
            return [torch.float32]
        # x86/aarch64 CPU has supported both bf16 and fp16 natively.
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        if use_mla:
            raise NotImplementedError("MLA is not supported on CPU.")
        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on CPU.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        from vllm.utils.mem_constants import GiB_bytes

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
        node_dir = "/sys/devices/system/node"
        if kv_cache_space is None:
            nodes = (
                [d for d in os.listdir(node_dir) if d.startswith("node")]
                if os.path.exists(node_dir)
                else []
            )
            num_numa_nodes = len(nodes) or 1
            free_cpu_memory = psutil.virtual_memory().total // num_numa_nodes
            DEFAULT_CPU_MEM_UTILIZATION = 0.5
            kv_cache_space = int(free_cpu_memory * DEFAULT_CPU_MEM_UTILIZATION)
            kv_cache_space_gib = kv_cache_space / GiB_bytes
            logger.warning_once(
                "VLLM_CPU_KVCACHE_SPACE not set. Using "
                f"{kv_cache_space_gib:.2f} GiB for KV cache."
            )
        else:
            kv_cache_space *= GiB_bytes

        return kv_cache_space

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cpu.set_device(device)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config

        if cache_config.block_size is None:
            cache_config.block_size = 128

        if cache_config.block_size % 32 != 0:
            logger.warning(
                "CPU backend prefers block_size is multiples of 32, "
                "otherwise the performance is not optimized."
            )

        scheduler_config = vllm_config.scheduler_config
        if (
            scheduler_config.enable_chunked_prefill
            or cache_config.enable_prefix_caching
        ) and cache_config.cache_dtype != "auto":
            raise RuntimeError(
                "Chunked-prefill and prefix-cache on the CPU "
                "backend is not compatible with FP8 KV cache."
            )

        if cache_config.cache_dtype != "auto":
            logger.warning(
                "CPU backend doesn't support KV cache quantization fallback to auto."
            )
            cache_config.cache_dtype = "auto"

        cache_config.cpu_kvcache_space_bytes = CpuPlatform.get_device_total_memory()

        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.world_size > 1
            and parallel_config.distributed_executor_backend is not None
            and parallel_config.distributed_executor_backend != "mp"
        ):
            logger.warning(
                (
                    "%s is not supported on CPU, fallback to mp "
                    "distributed executor backend."
                ),
                parallel_config.distributed_executor_backend,
            )
            parallel_config.distributed_executor_backend = "mp"
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"
        # Disable DBO
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on CPU, disabled.")
            parallel_config.enable_dbo = False

        # Note: workaround for v1 gpu_model_runner
        from vllm.config import CompilationMode

        vllm_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = vllm_config.compilation_config
        if vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # Note: vLLM V1 is using PIECEWISE level compilation, which will
            # take time to compile kernels just-in-time with the inductor
            # backend. For CPU CI tests, most of them are executed fast and
            # compilations consume too much time, even with torch compile
            # cache. So use VLLM_CPU_CI_ENV to indicate the CI environment,
            # and just execute model with dynamo + eager mode to save time.
            # VLLM_CPU_CI_ENV is only used as an internal variable.
            if os.environ.get("VLLM_CPU_CI_ENV", "0") != "0":
                backend = "eager"
            else:
                backend = "inductor"

            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
            compilation_config.backend = backend
            compilation_config.inductor_compile_config.update(
                {
                    "dce": True,
                    "size_asserts": False,
                    "nan_asserts": False,
                    "epilogue_fusion": True,
                }
            )

        if vllm_config.lora_config is not None:
            compilation_config.mode = CompilationMode.NONE

        assert vllm_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Note: to avoid the error 'nthreads cannot be larger than environment
        # variable "NUMEXPR_MAX_THREADS" (64)'.
        os.environ["NUMEXPR_MAX_THREADS"] = str(get_max_threads())

        if envs.VLLM_CPU_OMP_THREADS_BIND != "nobind":
            # Set default threads num for OpenMP parallel
            os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
        else:
            # In this case, setting the OpenMP configuration via
            # OMP_NUM_THREADS is up to the user.
            logger.info("Disabling binding processes to CPU cores...")

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Disable multi-stream for shared experts as no Stream on CPU
        os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

        # Intel OpenMP setting
        ld_preload_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_preload_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ["KMP_BLOCKTIME"] = "1"
            # Prevents the CPU to run into low performance state
            os.environ["KMP_TPAUSE"] = "0"
            # Provides fine granularity parallelism
            os.environ["KMP_FORKJOIN_BARRIER_PATTERN"] = "dist,dist"
            os.environ["KMP_PLAIN_BARRIER_PATTERN"] = "dist,dist"
            os.environ["KMP_REDUCTION_BARRIER_PATTERN"] = "dist,dist"

        if (
            platform.system() == "Linux"
            and Platform.get_cpu_architecture()
            in (CpuArchEnum.ARM, CpuArchEnum.POWERPC)
            and not ("libomp" in ld_preload_str or "libgomp" in ld_preload_str)
        ):
            # We need to LD_PRELOAD PyTorch's libgomp, otherwise only
            # one core will be properly utilized when we thread-bind
            # See: https://github.com/vllm-project/vllm/issues/27369
            # TODO: Remove once:
            # https://github.com/pytorch/pytorch/issues/166087 is fixed

            # We need to find the location of PyTorch's libgomp
            torch_pkg = os.path.dirname(torch.__file__)
            site_root = os.path.dirname(torch_pkg)
            torch_libs = os.path.join(site_root, "torch.libs")
            pytorch_libgomp_so_candidates = glob.glob(
                os.path.join(torch_libs, "libgomp-*.so*")
            )
            if pytorch_libgomp_so_candidates:
                pytorch_libgomp_so = pytorch_libgomp_so_candidates[0]
                if ld_preload_str:
                    ld_preload_str += ":"
                ld_preload_str += pytorch_libgomp_so
                os.environ["LD_PRELOAD"] = ld_preload_str

        # To hint IPEX uses shared memory based AllReduce
        os.environ["LOCAL_WORLD_SIZE"] = str(
            vllm_config.parallel_config.tensor_parallel_size
        )

        if model_config is not None and model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

    @classmethod
    def get_allowed_cpu_core_node_list(cls) -> tuple[list[int], list[LogicalCPUInfo]]:
        assert platform.system() == "Linux"

        # Init LogicalCPUInfo from lscpu
        lscpu_output = subprocess.check_output(
            "lscpu -J -e=CPU,CORE,NODE", shell=True, text=True
        )
        lscpu_output = re.sub(r'"node":\s*-\s*(,|\n)', r'"node": 0\1', lscpu_output)
        logical_cpu_list: list[LogicalCPUInfo] = json.loads(
            lscpu_output, object_hook=LogicalCPUInfo.json_decoder
        )["cpus"]

        # Filter CPUs with invalid attributes
        logical_cpu_list = [
            x
            for x in logical_cpu_list
            if -1 not in (x.id, x.physical_core, x.numa_node)
        ]

        # Filter allowed CPUs
        if hasattr(os, "sched_getaffinity"):
            allowed_cpu_id_list = os.sched_getaffinity(0)
        else:
            raise NotImplementedError("Unsupported OS")
        logical_cpu_list = [x for x in logical_cpu_list if x.id in allowed_cpu_id_list]

        # Get allowed NUMA nodes
        allowed_numa_nodes = set()
        for x in logical_cpu_list:
            allowed_numa_nodes.add(x.numa_node)  # type: ignore
        allowed_numa_nodes_list = sorted(allowed_numa_nodes)

        env_key = CpuPlatform.device_control_env_var
        if env_key in os.environ and os.environ[env_key] != "":
            visible_nodes = [int(s) for s in os.environ[env_key].split(",")]
            allowed_numa_nodes_list = [
                x for x in visible_nodes if x in allowed_cpu_id_list
            ]

        return allowed_numa_nodes_list, logical_cpu_list

    @classmethod
    def is_pin_memory_available(cls) -> bool:
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
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True
