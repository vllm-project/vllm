# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact restoration of final PCP hidden-state rows."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None  # type: ignore[assignment]


class PCPMulticastUnavailableError(RuntimeError):
    """Raised when CUDA symmetric-memory multicast cannot be initialized."""


class PCPMulticastHiddenStateRestorer:
    """Restore PCP hidden states with NVLink multicast symmetric memory.

    Every rank multicasts its contiguous rank-local shard into the same
    rank-major staging layout on all PCP ranks. A local index-select then
    restores global token order. The multicast operation includes the
    cross-rank publication needed before the local reorder.
    """

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
            raise PCPMulticastUnavailableError(
                "CUDA symmetric-memory support is unavailable."
            )
        if max_num_tokens <= 0 or hidden_size <= 0:
            raise ValueError("PCP hidden output dimensions must be positive.")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "PCP multicast hidden-state restore supports BF16 or FP16, "
                f"got {dtype}."
            )
        self._group = group
        self._group_name = group.group_name
        self._world_size = group.size()
        self._device = device
        self._closed = False
        self._next_buffer = 0
        self._current_buffer = 0
        self._max_num_tokens = max_num_tokens
        # Final-row ownership can be skewed: every selected row may belong to
        # one PCP rank. Reserve max_num_tokens rows per producer rank rather
        # than assuming an even split.
        self._expanded_capacity = max_num_tokens * self._world_size

        self._multicast_storage: torch.Tensor | None = None
        self._ordered_outputs: torch.Tensor | None = None
        self._packed_input: torch.Tensor | None = None
        allocation_error: RuntimeError | None = None
        try:
            self._multicast_storage = torch_symm_mem.empty(
                (self._expanded_capacity, hidden_size),
                dtype=dtype,
                device=device,
            )
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
        except RuntimeError as error:
            allocation_error = error

        allocation_ready = torch.tensor(
            [allocation_error is None],
            dtype=torch.int32,
        )
        dist.all_reduce(
            allocation_ready,
            op=dist.ReduceOp.MIN,
            group=group,
        )
        if not allocation_ready.item():
            self._multicast_storage = None
            self._ordered_outputs = None
            self._packed_input = None
            raise PCPMulticastUnavailableError(
                "CUDA symmetric-memory multicast allocation failed on at "
                "least one PCP rank."
            ) from allocation_error

        assert self._multicast_storage is not None
        rendezvous_error: RuntimeError | None = None
        try:
            handle = torch_symm_mem.rendezvous(self._multicast_storage, group)
            if handle.multicast_ptr == 0:
                rendezvous_error = PCPMulticastUnavailableError(
                    "PCP multicast hidden-state restore requires CUDA "
                    "multicast support."
                )
        except RuntimeError as error:
            rendezvous_error = error

        rendezvous_ready = torch.tensor(
            [rendezvous_error is None],
            dtype=torch.int32,
        )
        dist.all_reduce(
            rendezvous_ready,
            op=dist.ReduceOp.MIN,
            group=group,
        )
        if not rendezvous_ready.item():
            self._multicast_storage = None
            self._ordered_outputs = None
            self._packed_input = None
            raise PCPMulticastUnavailableError(
                "CUDA symmetric-memory multicast initialization failed on at "
                "least one PCP rank."
            ) from rendezvous_error

    @property
    def local_output(self) -> torch.Tensor:
        if self._closed or self._ordered_outputs is None:
            raise RuntimeError("PCP multicast hidden-state restorer is closed.")
        return self._ordered_outputs[self._current_buffer]

    def restore(
        self,
        hidden_states: torch.Tensor,
        restore_indices: torch.Tensor,
        *,
        num_global_tokens: int,
    ) -> torch.Tensor:
        if (
            self._closed
            or self._multicast_storage is None
            or self._ordered_outputs is None
        ):
            raise RuntimeError("PCP multicast hidden-state restorer is closed.")
        if hidden_states.ndim != 2:
            raise ValueError(
                "PCP hidden-state restore expects a 2D input, got "
                f"{tuple(hidden_states.shape)}."
            )
        if hidden_states.dtype != self._multicast_storage.dtype:
            raise ValueError(
                "PCP hidden-state input and multicast output dtypes must match."
            )
        if hidden_states.device != self._device:
            raise ValueError(
                "PCP hidden-state input and multicast output must share a device."
            )
        if hidden_states.shape[1] != self._multicast_storage.shape[1]:
            raise ValueError("PCP hidden sizes do not match.")
        if restore_indices.shape != (num_global_tokens,):
            raise ValueError(
                "PCP restore indices must match the global row count: "
                f"{tuple(restore_indices.shape)} != ({num_global_tokens},)."
            )
        if restore_indices.device != hidden_states.device:
            raise ValueError("PCP restore indices must be on the hidden-state device.")
        if restore_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("PCP restore indices must use int32 or int64.")
        if not 0 <= num_global_tokens <= self._max_num_tokens:
            raise ValueError(
                "PCP global token count exceeds the multicast output capacity: "
                f"{num_global_tokens} > {self._max_num_tokens}."
            )

        num_expanded_tokens = hidden_states.shape[0] * self._world_size
        if num_expanded_tokens > self._expanded_capacity:
            raise ValueError(
                "PCP padded token count exceeds the multicast staging capacity: "
                f"{num_expanded_tokens} > {self._expanded_capacity}."
            )
        gathered = self._multicast_storage[:num_expanded_tokens]
        torch.ops.symm_mem.multimem_all_gather_out(
            hidden_states,
            self._group_name,
            gathered,
        )

        buffer_index = self._next_buffer
        local_output = self._ordered_outputs[buffer_index]
        torch.index_select(
            gathered,
            0,
            restore_indices,
            out=local_output[:num_global_tokens],
        )
        self._current_buffer = buffer_index
        self._next_buffer = buffer_index ^ 1
        return local_output[:num_global_tokens]

    def restore_selected(
        self,
        hidden_states: torch.Tensor,
        local_row_indices: torch.Tensor,
        restore_indices: torch.Tensor,
        *,
        num_selected_rows: int,
    ) -> torch.Tensor:
        """Pack locally owned final rows, multicast, and restore request order."""
        if self._closed or self._packed_input is None:
            raise RuntimeError("PCP multicast hidden-state restorer is closed.")
        if local_row_indices.ndim != 1:
            raise ValueError("PCP local selected-row indices must be one-dimensional.")
        num_local_rows = local_row_indices.shape[0]
        if num_local_rows > self._packed_input.shape[0]:
            raise ValueError(
                "PCP selected-row count exceeds the multicast input capacity: "
                f"{num_local_rows} > {self._packed_input.shape[0]}."
            )
        if local_row_indices.device != hidden_states.device:
            raise ValueError(
                "PCP selected-row indices must be on the hidden-state device."
            )
        if local_row_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("PCP selected-row indices must use int32 or int64.")
        packed_input = self._packed_input[:num_local_rows]
        torch.index_select(
            hidden_states,
            0,
            local_row_indices,
            out=packed_input,
        )
        return self.restore(
            packed_input,
            restore_indices,
            num_global_tokens=num_selected_rows,
        )

    def close(self) -> None:
        if self._closed:
            return
        torch.accelerator.synchronize()
        dist.barrier(group=self._group)
        self._ordered_outputs = None
        self._packed_input = None
        self._multicast_storage = None
        self._closed = True
