# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


def _is_sparse_expert_param(full_name: str) -> bool:
    """Check whether a parameter belongs to sparse MoE experts.

    Uses dot-delimited segment matching to avoid false positives on
    parameters such as 'shared_experts', router gates, or attention weights.
    """
    segments = full_name.lower().split(".")

    # Shared experts and router gates are dense (read on 100% of tokens).
    if any(
        seg in ("shared_expert", "shared_experts", "share_expert") for seg in segments
    ):
        return False
    if any(
        seg
        in (
            "gate",
            "router",
            "shared_expert_gate",
            "router_logits",
            "expert_gate",
        )
        for seg in segments
    ):
        return False

    # Sparse expert modules
    return any(
        seg
        in (
            "experts",
            "block_sparse_moe",
            "fused_moe",
            "switch_mlp",
            "moe_experts",
        )
        for seg in segments
    )


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

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        if prefix:
            prefix = f"{prefix}."
        modules = list(modules_generator)
        self._offload_modules(modules, prefix)
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        self._offload_modules([module], prefix)
        return module

    def _offload_modules(
        self,
        modules: list[nn.Module],
        prefix: str = "",
    ) -> None:
        """Offload parameters from modules to CPU using UVA if budget allows."""
        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return

        candidates: list[tuple[int, nn.Module, str, nn.Parameter, str]] = []
        for module in modules:
            params = next(module.parameters(), None)
            if params is None or params.device == torch.device("cpu"):
                continue

            for name, p in module.named_parameters():
                if p.device.type == "cpu" or getattr(
                    p, "_vllm_is_uva_offloaded", False
                ):
                    continue

                full_name = f"{prefix}{name}"
                if self.cpu_offload_params:
                    # Segment match
                    should_offload = any(
                        f".{param}." in f".{full_name}."
                        for param in self.cpu_offload_params
                    )
                    if not should_offload:
                        continue
                    priority = 0
                else:
                    # Sparse expert weights are prioritized over dense weights
                    priority = 0 if _is_sparse_expert_param(full_name) else 1

                candidates.append((priority, module, name, p, full_name))

        # Stable sort by priority tier while preserving relative declaration order
        candidates.sort(key=lambda item: item[0])

        offloaded_modules: set[nn.Module] = set()
        dense_offloaded = False

        for priority, module, name, p, full_name in candidates:
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                break

            cpu_data = p.data.to(device="cpu")
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_modules.add(module)
            if priority == 1:
                dense_offloaded = True

        if dense_offloaded and not self.cpu_offload_params:
            logger.info_once(
                "Dense parameters were offloaded to CPU because the offload "
                "budget exceeded sparse expert weights or all parameters were "
                "eligible. This may increase PCIe memory traffic."
            )

        if not self.uva_offloading:
            for module in offloaded_modules:
                self._wrap_non_uva_forward(module)

    def _wrap_non_uva_forward(self, module: nn.Module) -> None:
        """Wrap module with functional_call for non-UVA CPU offloading."""
        original_forward = module.forward
        device = next(module.parameters()).device

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }
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
