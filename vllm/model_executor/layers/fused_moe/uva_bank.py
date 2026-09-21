# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-selective host banking for MoE expert weights.

`UVAOffloader` offloads whole parameters, chosen by name and by a byte
budget. Expert-granular MoE residency needs something narrower: keeping a
*subset of the rows* of an ``[num_experts, ...]`` expert weight off the
device while the remaining rows stay resident. The two helpers here are the
storage primitives for that, and nothing else in this module decides which
experts are cold -- callers hand them an explicit list of expert ids.

`HostBank` keeps the cold rows in page-locked host memory and exposes them
as a device-readable tensor through UVA, so a kernel dereferences them
across PCIe and their bytes never occupy device memory.

`StagingBank` is the alternative for workloads where one bulk H2D copy per
step beats per-access PCIe latency: cold rows stay in pinned host memory
with no persistent device view, and each step copies only the distinct cold
rows that step actually routed to into a small reusable device buffer.

Both are dtype- and model-agnostic.
"""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

__all__ = ["HostBank", "StagingBank"]


class HostBank:
    """Hold tensors in pinned host memory, readable by device kernels."""

    @staticmethod
    def pin(tensor: torch.Tensor) -> torch.Tensor:
        """Return a page-locked, contiguous CPU copy of ``tensor``.

        Returns ``tensor`` unchanged when it is already CPU, contiguous and
        pinned. Pinning needs a CUDA context, so this requires an
        accelerator even though no device memory is allocated for the
        result.
        """
        if tensor.device.type != "cpu":
            tensor = tensor.detach().cpu()
        tensor = tensor.contiguous()
        if tensor.is_pinned():
            return tensor
        pinned = torch.empty_like(tensor, pin_memory=True)
        pinned.copy_(tensor)
        return pinned

    @staticmethod
    def as_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Pin ``tensor`` and return a device view over the same host pages.

        The returned tensor has the same shape and dtype as the input and
        allocates no device memory: reads go over PCIe to the pinned host
        pages. The view keeps the pinned buffer alive for as long as it is
        itself reachable.
        """
        pinned = HostBank.pin(tensor)
        view = get_accelerator_view_from_cpu_tensor(pinned)
        # get_accelerator_view_from_cpu_tensor does not itself keep the CPU
        # tensor alive; freeing the host pages while a kernel still holds
        # the device pointer would be a use-after-free on the device.
        view._vllm_uva_host_keepalive = pinned
        return view


class StagingBank:
    """Stage the cold rows a step actually needs into a reusable buffer.

    ``max_rows_per_step`` bounds the device buffer. Callers size it from the
    worst-case number of distinct cold experts a single step can route to.
    """

    def __init__(
        self,
        per_row_shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, torch.dtype],
        max_rows_per_step: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.max_rows_per_step = max_rows_per_step
        self._buffers: dict[str, torch.Tensor] = {
            name: torch.empty(
                (max_rows_per_step, *shape), dtype=dtypes[name], device=device
            )
            for name, shape in per_row_shapes.items()
        }

    def stage(
        self,
        cold_bank: dict[str, torch.Tensor],
        needed_row_ids: torch.Tensor,
        non_blocking: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Copy ``needed_row_ids`` out of each host tensor into the buffer.

        Ids index the cold bank's own ``0..num_cold_rows-1`` numbering.
        Returns views sized exactly ``[len(needed_row_ids), ...]``, a prefix
        of the preallocated buffer, so the hot path allocates nothing.
        """
        n = int(needed_row_ids.numel())
        if n > self.max_rows_per_step:
            raise ValueError(
                f"StagingBank: {n} distinct cold rows needed this step "
                f"exceeds max_rows_per_step={self.max_rows_per_step}"
            )
        staged: dict[str, torch.Tensor] = {}
        for name, host_tensor in cold_bank.items():
            src = host_tensor.index_select(0, needed_row_ids.to(host_tensor.device))
            dst = self._buffers[name][:n]
            dst.copy_(src, non_blocking=non_blocking)
            staged[name] = dst
        return staged

    @staticmethod
    def remap_for_staged(
        cold_expert_map: torch.Tensor,
        needed_row_ids: torch.Tensor,
        num_cold_rows: int | None = None,
    ) -> torch.Tensor:
        """Rewrite an expert_map to point at this step's staged rows.

        ``cold_expert_map`` has the usual ``[global_num_experts]`` shape and
        semantics: a cold row index, or -1 for "not resident here". The
        result points each staged expert at its row in the compacted buffer
        (``0..len(needed_row_ids)-1``) and leaves everything else at -1.

        ``num_cold_rows`` sizes the scatter table. When omitted it is
        derived from the map, which costs a device sync; pass it from the
        bank's row count on the hot path to avoid that.
        """
        resident = cold_expert_map >= 0
        if num_cold_rows is None:
            num_cold_rows = (
                int(cold_expert_map.max().item()) + 1 if bool(resident.any()) else 0
            )
        inverse = torch.full(
            (max(num_cold_rows, 1),),
            -1,
            dtype=cold_expert_map.dtype,
            device=cold_expert_map.device,
        )
        row_ids = needed_row_ids.to(cold_expert_map.device)
        inverse[row_ids] = torch.arange(
            row_ids.numel(), dtype=cold_expert_map.dtype, device=inverse.device
        )
        gathered = inverse[cold_expert_map.clamp(min=0)]
        return torch.where(resident, gathered, torch.full_like(cold_expert_map, -1))
