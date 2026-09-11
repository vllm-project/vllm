# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact multicast of the PCP rows consumed by sampling."""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None  # type: ignore[assignment]


class PCPMulticastUnavailableError(RuntimeError):
    """All PCP ranks must fall back together when multicast is unavailable."""


class PCPMulticastHiddenStateRestorer:
    def __init__(
        self,
        *,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        if torch_symm_mem is None:
            raise PCPMulticastUnavailableError("Symmetric memory is unavailable")
        if max_num_tokens <= 0 or hidden_size <= 0:
            raise ValueError("PCP output dimensions must be positive")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("PCP multicast requires BF16 or FP16")
        self._group = group
        self._group_name = group.group_name
        self._world_size = group.size()
        self._next_buffer = 0
        self._multicast_storage = self._ordered_outputs = self._packed_input = None
        error = None
        try:
            self._multicast_storage = torch_symm_mem.empty(
                (max_num_tokens * self._world_size, hidden_size),
                dtype=dtype,
                device=device,
            )
            # Alternate outputs so sampling can retain the preceding batch.
            self._ordered_outputs = torch.empty(
                (2, max_num_tokens, hidden_size),
                dtype=dtype,
                device=device,
            )
            self._packed_input = torch.empty(
                (max_num_tokens, hidden_size),
                dtype=dtype,
                device=device,
            )
        except RuntimeError as exc:
            error = exc
        self._agree(error, "allocation")
        try:
            handle = torch_symm_mem.rendezvous(self._multicast_storage, group)
            if not handle.multicast_ptr:
                raise PCPMulticastUnavailableError("CUDA multicast is unsupported")
        except RuntimeError as exc:
            error = exc
        self._agree(error, "rendezvous")

    def _agree(self, error: RuntimeError | None, phase: str) -> None:
        ready = torch.tensor([error is None], dtype=torch.int32, device="cpu")
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=self._group)
        if not ready.item():
            self._multicast_storage = self._ordered_outputs = self._packed_input = None
            raise PCPMulticastUnavailableError(
                f"PCP multicast {phase} failed on at least one rank"
            ) from error

    @property
    def local_output(self) -> torch.Tensor:
        if self._ordered_outputs is None:
            raise RuntimeError("PCP restorer is closed")
        return self._ordered_outputs[self._next_buffer ^ 1]

    def restore_selected(
        self,
        hidden_states: torch.Tensor,
        local_row_indices: torch.Tensor,
        restore_indices: torch.Tensor,
        *,
        num_selected_rows: int,
    ) -> torch.Tensor:
        if self._packed_input is None:
            raise RuntimeError("PCP restorer is closed")
        assert self._multicast_storage is not None and self._ordered_outputs is not None
        n = local_row_indices.numel()
        if (
            n > self._packed_input.shape[0]
            or num_selected_rows > self._packed_input.shape[0]
        ):
            raise ValueError("PCP sampled rows exceed the allocated capacity")
        assert restore_indices.shape == (num_selected_rows,)
        packed = self._packed_input[:n]
        gathered = self._multicast_storage[: n * self._world_size]
        out = self._ordered_outputs[self._next_buffer, :num_selected_rows]
        torch.index_select(hidden_states, 0, local_row_indices, out=packed)
        torch.ops.symm_mem.multimem_all_gather_out(packed, self._group_name, gathered)
        torch.index_select(gathered, 0, restore_indices, out=out)
        self._next_buffer ^= 1
        return out

    def close(self) -> None:
        if self._multicast_storage is not None:
            torch.accelerator.synchronize()
            dist.barrier(group=self._group)
            self._multicast_storage = self._ordered_outputs = self._packed_input = None
