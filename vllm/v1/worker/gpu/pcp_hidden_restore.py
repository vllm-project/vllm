# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct peer-memory restoration of final PCP hidden states."""

from __future__ import annotations

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)
from vllm.distributed.device_communicators.peer_memory import (
    PeerMemoryFence,
    make_rank_major_tensor_view,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _direct_hidden_restore_kernel(
    hidden_states_ptr,
    peer_output_ptr,
    global_row_indices_ptr,
    hidden_row_stride,
    hidden_col_stride,
    peer_rank_stride,
    output_row_stride,
    output_col_stride,
    output_capacity,
    HIDDEN_SIZE: tl.constexpr,
    PCP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    local_row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    global_row = tl.load(global_row_indices_ptr + local_row)
    valid = (global_row >= 0) & (global_row < output_capacity)
    mask = valid & (column < HIDDEN_SIZE)
    values = tl.load(
        hidden_states_ptr + local_row * hidden_row_stride + column * hidden_col_stride,
        mask=mask,
    )
    output_row = global_row * output_row_stride + column * output_col_stride
    for destination_rank in range(PCP_SIZE):
        tl.store(
            peer_output_ptr + destination_rank * peer_rank_stride + output_row,
            values,
            mask=mask,
        )


def direct_hidden_state_restore(
    hidden_states: torch.Tensor,
    peer_output: torch.Tensor,
    global_row_indices: torch.Tensor,
) -> None:
    """Fan canonical rank-local rows into globally ordered output replicas."""
    if hidden_states.ndim != 2:
        raise ValueError(
            "PCP hidden-state restore expects a 2D input, got "
            f"{tuple(hidden_states.shape)}."
        )
    if peer_output.ndim != 3:
        raise ValueError(
            "PCP peer output must have shape [peer, token, hidden], got "
            f"{tuple(peer_output.shape)}."
        )
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "PCP direct hidden-state restore supports BF16 or FP16, got "
            f"{hidden_states.dtype}."
        )
    if peer_output.dtype != hidden_states.dtype:
        raise ValueError(
            "PCP hidden-state input and peer output dtypes must match: "
            f"{hidden_states.dtype} != {peer_output.dtype}."
        )
    if hidden_states.device != peer_output.device:
        raise ValueError("PCP hidden-state input and peer output must share a device.")
    if hidden_states.shape[1] != peer_output.shape[2]:
        raise ValueError(
            "PCP hidden sizes do not match: "
            f"{hidden_states.shape[1]} != {peer_output.shape[2]}."
        )
    num_local_rows = hidden_states.shape[0]
    if global_row_indices.shape != (num_local_rows,):
        raise ValueError(
            "PCP global-row indices must match the local row count: "
            f"{tuple(global_row_indices.shape)} != ({num_local_rows},)."
        )
    if global_row_indices.device != hidden_states.device:
        raise ValueError("PCP global-row indices must be on the hidden-state device.")
    if global_row_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("PCP global-row indices must use int32 or int64.")
    if num_local_rows == 0:
        return

    block_size = min(256, triton.next_power_of_2(hidden_states.shape[1]))
    _direct_hidden_restore_kernel[
        (num_local_rows, triton.cdiv(hidden_states.shape[1], block_size))
    ](
        hidden_states,
        peer_output,
        global_row_indices,
        hidden_states.stride(0),
        hidden_states.stride(1),
        peer_output.stride(0),
        peer_output.stride(1),
        peer_output.stride(2),
        peer_output.shape[1],
        HIDDEN_SIZE=hidden_states.shape[1],
        PCP_SIZE=peer_output.shape[0],
        BLOCK_SIZE=block_size,
    )


class PCPHiddenStateRestorer:
    """Reusable globally ordered hidden-state output for one PCP group.

    Two output slabs alternate. Stores for step N+1 therefore cannot overwrite
    step N while sampling or prompt-logprob work still consumes it. The
    publication fence for N+1 is stream-ordered after those consumers and
    retires slab N before N+2 reuses it. This needs one fence per restore.
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
        if max_num_tokens <= 0 or hidden_size <= 0:
            raise ValueError("PCP hidden output dimensions must be positive.")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"PCP direct hidden-state restore supports BF16 or FP16, got {dtype}."
            )
        self._group = group
        self._device = device
        self._closed = False
        self._next_buffer = 0
        self._current_buffer = 0
        self._allocation: RankMajorPeerView = create_rank_major_peer_view(
            (2, max_num_tokens, hidden_size),
            dtype=dtype,
            group=group,
            require_native_atomics=False,
            device=device,
        )
        try:
            local_outputs = self._allocation.local_view[
                :2, :max_num_tokens, :hidden_size
            ]
            self._local_outputs: torch.Tensor | None = local_outputs
            self._peer_outputs: torch.Tensor | None = make_rank_major_tensor_view(
                self._allocation, local_outputs
            )
            self._fence: PeerMemoryFence | None = PeerMemoryFence(group, device)
        except BaseException:
            self._allocation.close()
            raise

    @property
    def local_output(self) -> torch.Tensor:
        if self._closed or self._local_outputs is None:
            raise RuntimeError("PCP hidden-state restorer is closed.")
        return self._local_outputs[self._current_buffer]

    @property
    def peer_output(self) -> torch.Tensor:
        if self._closed or self._peer_outputs is None:
            raise RuntimeError("PCP hidden-state restorer is closed.")
        return self._peer_outputs[:, self._current_buffer]

    def restore(
        self,
        hidden_states: torch.Tensor,
        global_row_indices: torch.Tensor,
        *,
        num_global_tokens: int,
    ) -> torch.Tensor:
        if self._closed or self._local_outputs is None or self._peer_outputs is None:
            raise RuntimeError("PCP hidden-state restorer is closed.")
        if hidden_states.shape[0] != global_row_indices.shape[0]:
            raise ValueError(
                "PCP hidden-state rows and publish-map rows must match: "
                f"{hidden_states.shape[0]} != {global_row_indices.shape[0]}."
            )
        if not 0 <= num_global_tokens <= self._local_outputs.shape[1]:
            raise ValueError(
                "PCP global token count exceeds the direct output capacity: "
                f"{num_global_tokens} > {self._local_outputs.shape[1]}."
            )
        buffer_index = self._next_buffer
        local_output = self._local_outputs[buffer_index]
        peer_output = self._peer_outputs[:, buffer_index]
        assert self._fence is not None
        direct_hidden_state_restore(
            hidden_states,
            peer_output,
            global_row_indices,
        )
        self._fence()
        self._current_buffer = buffer_index
        self._next_buffer = buffer_index ^ 1
        return local_output[:num_global_tokens]

    def close(self) -> None:
        if self._closed:
            return
        assert self._fence is not None
        self._fence.close()
        self._allocation.close()
        self._fence = None
        self._local_outputs = None
        self._peer_outputs = None
        self._closed = True
