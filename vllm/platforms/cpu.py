# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.cpu_resource_utils import (
    DEVICE_CONTROL_ENV_VAR,
    get_memory_node_info,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None


def get_max_threads(pid=0):
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(pid))
    elif platform.system() == "Darwin":
        return os.cpu_count()
    else:
        raise NotImplementedError("Unsupported OS")


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"
    device_control_env_var = DEVICE_CONTROL_ENV_VAR

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
            return [torch.bfloat16, torch.float16, torch.float32]
        # x86/aarch64 CPU has supported both bf16 and fp16 natively.
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on CPU.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on CPU.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        meminfo = get_memory_node_info(device_id)

        return meminfo.total_memory

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cpu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config

        if not cache_config.user_specified_block_size:
            cache_config.block_size = 128

        if cache_config.block_size % 32 != 0:
            logger.warning(
                "CPU backend prefers block_size is multiples of 32, "
                "otherwise the performance is not optimized."
            )

        # Lagecy setting
        env_key = "VLLM_CPU_KVCACHE_SPACE"
        if env_key in os.environ and os.environ[env_key] != "":
            kv_cache_space = int(os.environ[env_key])
            cache_config.kv_cache_memory_bytes = kv_cache_space * GiB_bytes

        scheduler_config = vllm_config.scheduler_config
        # async scheduling is not required on CPU
        scheduler_config.async_scheduling = False
        if (
            scheduler_config.enable_chunked_prefill
            or cache_config.enable_prefix_caching
        ) and is_quantized_kv_cache(cache_config.cache_dtype):
            raise RuntimeError(
                "Chunked-prefill and prefix-cache on the CPU "
                "backend is not compatible with FP8 KV cache."
            )

        if is_quantized_kv_cache(cache_config.cache_dtype):
            logger.warning(
                "CPU backend doesn't support KV cache quantization fallback to auto."
            )
            cache_config.cache_dtype = "auto"

        parallel_config = vllm_config.parallel_config
        # OMP requires the MP executor to function correctly, UniProc is not
        # supported as it is not possible to set the OMP environment correctly
        if parallel_config.distributed_executor_backend == "uni":
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
                    "cpp.dynamic_threads": True,
                }
            )
            compilation_config.ir_enable_torch_wrap = False

        if vllm_config.lora_config is not None:
            compilation_config.mode = CompilationMode.NONE

        if (
            cls.get_cpu_architecture() == CpuArchEnum.ARM
            and "+gelu" not in compilation_config.custom_ops
            and "-gelu" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+gelu")

        vllm_config.profiler_config.torch_profiler_dump_cuda_time_total = False

        assert vllm_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Note: to avoid the error 'nthreads cannot be larger than environment
        # variable "NUMEXPR_MAX_THREADS" (64)'.
        os.environ["NUMEXPR_MAX_THREADS"] = str(get_max_threads())

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Disable multi-stream for shared experts as no Stream on CPU
        os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

        # Avoid inductor generates num_thread() and breaks the thread binding
        os.environ["TORCHINDUCTOR_CPP_DYNAMIC_THREADS"] = "1"

        ld_preload_str = os.getenv("LD_PRELOAD", "")
        cpu_architecture = Platform.get_cpu_architecture()

        if (
            platform.system() == "Linux"
            and cpu_architecture
            in (CpuArchEnum.ARM, CpuArchEnum.POWERPC, CpuArchEnum.X86)
            and not (
                "libomp" in ld_preload_str
                or "libgomp" in ld_preload_str
                or "libiomp" in ld_preload_str
            )
        ):
            # We need to LD_PRELOAD PyTorch's libgomp, otherwise only
            # one core will be properly utilized when we thread-bind
            # See: https://github.com/vllm-project/vllm/issues/27369
            # TODO: Remove once:
            # https://github.com/pytorch/pytorch/issues/166087 is fixed

            # We need to find the location of PyTorch's libgomp
            torch_pkg = os.path.dirname(torch.__file__)
            site_root = os.path.dirname(torch_pkg)
            # Search both torch.libs and torch/lib - See:
            # https://github.com/vllm-project/vllm/issues/30470
            torch_libs_paths = [
                os.path.join(site_root, "torch.libs"),
                os.path.join(torch_pkg, "lib"),
            ]
            pytorch_libgomp_so_candidates = []
            for torch_libs in torch_libs_paths:
                pytorch_libgomp_so_candidates.extend(
                    glob.glob(os.path.join(torch_libs, "libgomp*.so*"))
                )
            if pytorch_libgomp_so_candidates:
                pytorch_libgomp_so = pytorch_libgomp_so_candidates[0]
                if ld_preload_str:
                    ld_preload_str += ":"
                ld_preload_str += pytorch_libgomp_so
                os.environ["LD_PRELOAD"] = ld_preload_str

        # LD_PRELOAD libtcmalloc, bundled under vllm/libs to reduce
        # memory allocation overhead
        if (
            platform.system() == "Linux"
            and cpu_architecture in (CpuArchEnum.ARM, CpuArchEnum.X86)
            and "libtcmalloc" not in ld_preload_str
        ):
            vllm_pkg = os.path.dirname(os.path.dirname(__file__))
            tcmalloc_so = None
            for pattern in ("libtcmalloc_minimal*.so*", "libtcmalloc.so*"):
                tcmalloc_so_candidates = glob.glob(
                    os.path.join(vllm_pkg, "libs", pattern)
                )
                if tcmalloc_so_candidates:
                    tcmalloc_so = tcmalloc_so_candidates[0]
                    break

            if tcmalloc_so is not None:
                if ld_preload_str:
                    ld_preload_str = f"{tcmalloc_so}:{ld_preload_str}"
                else:
                    ld_preload_str = tcmalloc_so
                os.environ["LD_PRELOAD"] = ld_preload_str

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
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # TODO: CPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def discover_numa_topology(cls) -> list[list[int]]:
        """
        Discover NUMA topology and keep the last physical core of each numa
        into one core group list for nixl start_kv_load()
        """
        SYS_NODE = "/sys/devices/system/node"
        SYS_CPU = "/sys/devices/system/cpu"

        if not (os.path.exists(SYS_NODE) and os.path.exists(SYS_CPU)):
            return []

        core_rsv_for_kv = []
        for node in os.listdir(SYS_NODE):
            if not node.startswith("node") or not node[4:].isdigit():
                continue
            node_path = f"{SYS_NODE}/{node}"

            seen_phys = set()
            for cpu in os.listdir(node_path):
                if not cpu.startswith("cpu") or not cpu[3:].isdigit():
                    continue

                cpu_id = int(cpu[3:])
                # thread_siblings based on cpu_id
                path = f"{SYS_CPU}/cpu{cpu_id}/topology/thread_siblings_list"

                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            s = f.read()
                        cpus: list[int] = []
                        for part in s.strip().split(","):
                            if "-" in part:
                                a, b = map(int, part.split("-"))
                                cpus.extend(range(a, b + 1))
                            else:
                                cpus.append(int(part))
                        siblings = cpus if cpus else [cpu_id]
                    except (OSError, ValueError):
                        siblings = [cpu_id]
                else:
                    siblings = [cpu_id]

                phys = min(siblings)

                if phys not in seen_phys:
                    seen_phys.add(phys)

            if len(seen_phys) > 0:
                core_rsv_for_kv.append(list(seen_phys))

        return core_rsv_for_kv

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

    @classmethod
    def import_kernels(cls) -> None:
        if Platform.get_cpu_architecture() in (CpuArchEnum.X86,):
            # Note: The lib name is _C_AVX2/AVX512, but the module name is _C.
            # This will cause a exception "dynamic module does define
            # module export function". But the library is imported
            # successfully. So ignore the exception for now, until we find
            # a solution.
            ignored_msg = "dynamic module does not define module export function"
            if torch.cpu._is_avx512_supported():
                if torch.cpu._is_avx512_bf16_supported():
                    try:
                        import vllm._C  # noqa: F401
                    except ImportError as e:
                        logger.warning("Failed to import from vllm._C: %r", e)
                else:
                    try:
                        import vllm._C_AVX512  # noqa: F401
                    except ImportError as e:
                        if ignored_msg not in e.msg:
                            logger.warning(
                                "Failed to import from vllm._C_AVX512: %r", e
                            )
            else:
                try:
                    import vllm._C_AVX2  # noqa: F401
                except ImportError as e:
                    if ignored_msg not in e.msg:
                        logger.warning("Failed to import from vllm._C_AVX2: %r", e)
        else:
            try:
                import vllm._C  # noqa: F401
            except ImportError as e:
                logger.warning("Failed to import from vllm._C: %r", e)

    @classmethod
    def pack_kv_cache(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_ids: list[int],
        indices: torch.Tensor,
    ) -> None:
        """
        Rewrite the kv cache shape for the current platform.
        """
        # Import lazily: cpu_attn pulls in _custom_ops, which needs a fully
        # initialized vllm.platforms (avoid circular import while CpuPlatform loads).
        from vllm._custom_ops import cpu_attn_reshape_and_cache
        from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

        dtype = key.dtype
        # For CPU_ATTN, the shape is [N, num_kv_heads, block_size, head_size]
        _, _, block_size, head_size = key_cache.shape
        key = key.permute(0, 2, 1, 3).flatten(0, 1)
        value = value.permute(0, 2, 1, 3).flatten(0, 1)

        isa = _get_attn_isa(dtype, block_size, head_size)
        block_offsets = torch.arange(block_size, device="cpu", dtype=torch.long)
        num_blocks = len(block_ids)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + indices.reshape(num_blocks, 1) * block_size
        ).flatten()
        cpu_attn_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            isa,
        )
