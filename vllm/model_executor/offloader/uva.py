# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class _PrefetchRing:
    """Two device buffers for one (slot, shape, stride, dtype) and a copy
    stream: while the kernel of one layer reads one buffer, the next layer's
    weight is copied into the other.

    The order of the layers is learned from the calls of the steps that stage:
    each weight's successor is the weight staged after it for the same key.
    The buffer read by layer l is released by an event recorded on the compute
    stream when layer l+1 stages, by which point layer l's kernel is enqueued.
    """

    def __init__(self, weight: torch.Tensor, slot: str):
        device = weight.device
        mod = torch.get_device_module(device)
        self._mod = mod
        self.bufs = [
            torch.empty_strided(
                weight.shape, weight.stride(), dtype=weight.dtype, device=device
            )
            for _ in range(2)
        ]
        self.stream = mod.Stream(device)
        self.owner: list[int | None] = [None, None]
        self.ready: list[Any] = [None, None]
        self.free: list[Any] = [None, None]
        self.last: int | None = None
        self.next: dict[int, torch.Tensor] = {}
        self.prev: torch.Tensor | None = None
        logger.info(
            "Allocated 2 x %s GiB buffers to prefetch UVA-offloaded %s %s (%s).",
            format_gib(weight.numel() * weight.element_size()),
            slot,
            tuple(weight.shape),
            weight.dtype,
        )

    @staticmethod
    def _copy(dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src, non_blocking=True)

    def invalidate(self) -> None:
        """Forget which weights the buffers hold. The next stage() of every
        weight copies it again, after the kernels that read the buffers."""
        self.owner = [None, None]

    def _load(self, i: int, weight: torch.Tensor) -> None:
        with self._mod.stream(self.stream):
            if self.free[i] is not None:
                self.stream.wait_event(self.free[i])
            self._copy(self.bufs[i], weight)
            event = self._mod.Event()
            event.record(self.stream)
        self.ready[i] = event
        self.owner[i] = id(weight)

    def stage(self, weight: torch.Tensor) -> torch.Tensor:
        compute = self._mod.current_stream()
        if self.last is not None:
            # Everything enqueued so far, the previous layer's kernel included.
            done = self._mod.Event()
            done.record(compute)
            self.free[self.last] = done
        if self.prev is not None and self.prev is not weight:
            self.next[id(self.prev)] = weight
        self.prev = weight
        if id(weight) in self.owner:
            i = self.owner.index(id(weight))
        else:
            i = 0 if self.last is None else 1 - self.last
            self._load(i, weight)
        compute.wait_event(self.ready[i])
        self.last = i
        upcoming = self.next.get(id(weight))
        if upcoming is not None and id(upcoming) not in self.owner:
            self._load(1 - i, upcoming)
        return self.bufs[i]


class UVAExpertStager:
    """Copies UVA-offloaded expert weights into a device buffer for steps of
    at least `min_tokens` tokens (0 disables), see
    `UVAOffloadConfig.cpu_offload_stage_min_tokens`.

    One buffer per (slot, shape, stride, dtype, device) is shared by all
    layers. The copy and the kernel reading the buffer are issued on the same
    stream, so layer l+1's copy waits for layer l's kernel. Under DBO both
    micro-batches use one compute stream, so the caller must not yield between
    staging and the experts kernel.
    """

    def __init__(self, min_tokens: int, prefetch: bool = False):
        self.min_tokens = min_tokens
        self._buffers: dict[tuple, torch.Tensor] = {}
        self.prefetch = prefetch
        self._rings: dict[tuple, _PrefetchRing] = {}

    def invalidate(self) -> None:
        """The weights changed in place: drop prefetched copies of them."""
        for ring in self._rings.values():
            ring.invalidate()

    def maybe_stage(
        self, weight: torch.Tensor, num_tokens: int, slot: str
    ) -> torch.Tensor:
        if (
            self.min_tokens == 0
            or num_tokens < self.min_tokens
            or not getattr(weight, "_vllm_is_uva_offloaded", False)
            # A captured copy would replay at every size the graph serves.
            or torch.accelerator.current_stream().is_capturing()
        ):
            return weight
        key = (slot, tuple(weight.shape), weight.stride(), weight.dtype, weight.device)
        if self.prefetch:
            ring = self._rings.get(key)
            if ring is None:
                ring = self._rings[key] = _PrefetchRing(weight, slot)
            return ring.stage(weight)
        buf = self._buffers.get(key)
        if buf is None:
            # First allocated by the profiling run, so the KV cache accounts
            # for it.
            buf = torch.empty_strided(
                weight.shape, weight.stride(), dtype=weight.dtype, device=weight.device
            )
            self._buffers[key] = buf
            logger.info(
                "Allocated a %s GiB buffer to stage UVA-offloaded %s %s (%s).",
                format_gib(buf.numel() * buf.element_size()),
                slot,
                tuple(weight.shape),
                weight.dtype,
            )
        buf.copy_(weight, non_blocking=True)
        return buf


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
        stage_min_tokens: Resolved `cpu_offload_stage_min_tokens`; steps with
            at least this many tokens read fused-MoE expert weights from a
            device copy. 0 (or an unresolved "auto") disables it.

    """

    supports_tower_offload = True

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
        stage_min_tokens: int = 0,
        stage_prefetch: bool = False,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )
        # CUDA and XPU, not ROCm: ROCm kernels read per-tensor dispatch
        # attributes (e.g. `is_shuffled`) that a staging buffer does not carry.
        stage = self.uva_offloading and (
            current_platform.is_cuda() or current_platform.is_xpu()
        )
        self.expert_stager = UVAExpertStager(
            max(stage_min_tokens, 0) if stage else 0, prefetch=stage_prefetch
        )
        if self.expert_stager.min_tokens > 0:
            logger.info(
                "Staging UVA-offloaded MoE expert weights in VRAM for steps of at "
                "least %d tokens.",
                self.expert_stager.min_tokens,
            )

    def invalidate_staged(self) -> None:
        self.expert_stager.invalidate()

    def maybe_stage(
        self, weight: torch.Tensor, num_tokens: int, slot: str
    ) -> torch.Tensor:
        return self.expert_stager.maybe_stage(weight, num_tokens, slot)

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
