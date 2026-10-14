"""
Platform hook: inject a mock CUDA platform on CPU-only hosts.

Without a driver, detection yields UnspecifiedPlatform and DeviceConfig
fails.  The hook fires while vllm.platforms.interface defines Platform;
vllm.platforms is already in sys.modules, so current_platform is set there.
"""

import sys

import torch

from vllm_simulator.hook import BaseHook
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMPlatformHook(BaseHook):
    """Hook Platform class to inject _MockCudaPlatform as current_platform."""

    HOOK_CLASS_NAME = "Platform"
    HOOK_MODULE_NAME = "vllm.platforms.interface"

    @classmethod
    def hook(cls, target_class):
        iface = sys.modules["vllm.platforms.interface"]
        PlatformEnum = iface.PlatformEnum

        class _MockCudaPlatform(target_class):
            _enum = PlatformEnum.CUDA
            device_type = "cuda"
            device_name = "cuda"
            dispatch_key = "CUDA"

            empty_cache = None  # checked as `if empty_cache is not None`

            def is_cuda(self) -> bool:
                # False skips import-time CUDA op checks that need libcuda.so.
                return False

            def is_cuda_alike(self) -> bool:
                return False

            @property
            def supported_dtypes(self):
                return [torch.bfloat16, torch.float16, torch.float32]

            @classmethod
            def get_device_capability(cls, device_id=0):
                # A100 equivalent; the meta model build swaps in the
                # simulated accelerator's capability.
                from vllm.platforms.interface import DeviceCapability
                return DeviceCapability(8, 0)

            @classmethod
            def has_device_capability(
                cls, capability, device_id: int = 0
            ) -> bool:
                if isinstance(capability, tuple):
                    capability = capability[0] * 10 + capability[1]
                return capability <= cls.get_device_capability().to_int()

            @classmethod
            def check_and_update_config(cls, vllm_config) -> None:
                parallel_config = vllm_config.parallel_config
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm.v1.worker.gpu_worker.Worker"
                    )

            @classmethod
            def pre_register_and_update(cls, parser=None) -> None:
                pass

            @classmethod
            def support_hybrid_kv_cache(cls) -> bool:
                return True

            @classmethod
            def manual_seed_all(cls, seed: int) -> None:
                pass

            @classmethod
            def set_device(cls, device) -> None:
                pass

            @classmethod
            def get_device_name(cls, device_id: int = 0) -> str:
                return "NVIDIA A100-SXM4-80GB"

            @classmethod
            def get_device_total_memory(cls, device_id: int = 0) -> int:
                return 80 * (1 << 30)

            @classmethod
            def get_attn_backend_cls(
                cls, selected_backend, attn_selector_config, num_heads=None
            ):
                # Only resolved for KV-cache-layout queries; the CPU backend
                # needs no CUDA.  Lazy import: vllm.platforms.interface is
                # still being defined.
                from vllm.v1.attention.backends.registry import (
                    AttentionBackendEnum,
                )
                if not attn_selector_config.use_mla:
                    return AttentionBackendEnum.CPU_ATTN.get_path()

                # No CPU MLA backend, and the MLA backend decides the KV
                # cache layout: select like CUDA for the simulated arch.
                from vllm.platforms.cuda import CudaPlatformBase
                valid, invalid = CudaPlatformBase.get_valid_backends(
                    cls.get_device_capability(), attn_selector_config, num_heads
                )
                if not valid:
                    raise ValueError(
                        f"No valid MLA backend for {attn_selector_config}: "
                        f"{invalid}"
                    )
                return min(valid, key=lambda c: c.priority).backend.get_path()

            @classmethod
            def verify_quantization(cls, quant: str) -> None:
                pass

            @classmethod
            def get_current_memory_usage(cls, device=None) -> float:
                return 0.0

            def __getattr__(self, name):
                """Unimplemented platform methods resolve to no-ops."""
                def _noop(*args, **kwargs):
                    return None
                return _noop

        # The parent module's attribute binding is incomplete at hook time,
        # so `import vllm.platforms` would fail.
        platforms_module = sys.modules["vllm.platforms"]
        platforms_module.current_platform = _MockCudaPlatform()
