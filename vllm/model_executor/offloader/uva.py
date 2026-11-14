# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# vllm-xpu-kernels without #605 prefetch the per-group scales of the int4
# grouped GEMM with an out-of-spec block-2D descriptor that reads past the end
# of the scale tensor (vllm-project/vllm-xpu-kernels#605). In device memory the
# overrun lands in mapped memory; in pinned host memory sized exactly to the
# tensor it loses the device. Those builds keep MoE scales on the device.
# #605 merged on 2026-11-06: every 0.1.17 build and the 0.1.18 nightlies dated
# before that (0.1.18.dev20261105) lack it; nightly 0.1.18.dev20261106 has it.
_XPU_KERNELS_HOST_SCALES_SAFE = (0, 1, 18)
_XPU_KERNELS_FIRST_FIXED_NIGHTLY = 20261106


def xpu_moe_scales_stay_on_device(kernels_version: str | None) -> bool:
    """Whether fused-MoE weight scales must not be offloaded on XPU with the
    installed vllm-xpu-kernels (None: not installed).

    Releases below 0.1.18 are affected, local (`+...`) builds included. On the
    0.1.18 line, nightly wheels carry their build date as the dev number and
    count as fixed from the first nightly with #605. Other 0.1.18 pre-releases
    (rc, alpha, beta, source builds with a commit-count dev number) are not
    known to carry #605 and keep the scales on the device; unparsable versions
    do too."""
    if kernels_version is None:
        return False
    from packaging.version import InvalidVersion, Version

    try:
        version = Version(kernels_version)
    except InvalidVersion:
        return True
    line = version.release[:3]
    if line < _XPU_KERNELS_HOST_SCALES_SAFE:
        return True
    if line > _XPU_KERNELS_HOST_SCALES_SAFE:
        return False
    if version.pre is not None:
        return True
    if version.dev is not None:
        build_date = str(version.dev)[:8]
        return not (
            len(build_date) == 8 and int(build_date) >= _XPU_KERNELS_FIRST_FIXED_NIGHTLY
        )
    return False


def is_moe_scale_param(name: str) -> bool:
    """A fused-MoE expert weight scale, e.g.
    `model.layers.0.mlp.experts.w13_weight_scale` or `...experts.w2_scales`."""
    segments = name.split(".")
    return "experts" in segments[:-1] and segments[-1].endswith(
        ("_scale", "_scales", "_scale_inv")
    )


def _installed_xpu_kernels_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm-xpu-kernels")
    except PackageNotFoundError:
        return None


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.

    """

    supports_tower_offload = True

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )
        self.keep_moe_scales_on_device = False
        if self.uva_offloading:
            from vllm.platforms import current_platform

            if current_platform.is_xpu():
                kernels_version = _installed_xpu_kernels_version()
                self.keep_moe_scales_on_device = xpu_moe_scales_stay_on_device(
                    kernels_version
                )
                if self.keep_moe_scales_on_device:
                    logger.info_once(
                        "vllm-xpu-kernels %s reads past the end of int4 MoE "
                        "scales (vllm-xpu-kernels#605): MoE weight scales stay "
                        "in device memory.",
                        kernels_version,
                    )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        if prefix:
            prefix = f"{prefix}."
        modules = [
            self._maybe_offload_to_cpu(module, prefix) for module in modules_generator
        ]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # Skip parameters an earlier wrap_modules call already offloaded.
            # The UVA path leaves p.device as the accelerator (a view of CPU
            # memory), so the marker is the only way to recognize those.
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{prefix}{name}."
                    for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            if self.keep_moe_scales_on_device and is_moe_scale_param(prefix + name):
                continue

            cpu_data = p.data.to(device="cpu")
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                with nullcontext() if self.pin_memory else gpu_sync_allowed():
                    device_state = {
                        # here we blindly call `to(device)`
                        # if the parameter is already on the device,
                        # it will be a no-op
                        k: v.to(device, non_blocking=True)
                        for k, v in module.state_dict().items()
                    }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
