# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs for the checkpoint-configured LongSpark proposal pass."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


@triton.jit(
    do_not_specialize=["num_reqs", "captured_num_reqs", "padding_request_slot"]
)
def _prepare_graph_inputs_kernel(
    selected_hidden_ptrs,
    query_start_loc,
    num_rejected,
    selected_hidden_out,
    num_sampled,
    last_sampled,
    next_prefill_tokens,
    anchors_out,
    current_positions,
    current_positions_out,
    position_ids_out,
    request_slots,
    request_slots_out,
    state_committed_lens,
    starts_out,
    ends_out,
    num_reqs,
    captured_num_reqs,
    padding_request_slot,
    current_position_stride: tl.constexpr,
    request_slot_stride: tl.constexpr,
    state_committed_len_stride: tl.constexpr,
    query_start_stride: tl.constexpr,
    rejected_stride: tl.constexpr,
    sampled_stride: tl.constexpr,
    last_sampled_stride: tl.constexpr,
    next_prefill_stride: tl.constexpr,
    SELECTED_LAYERS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    POSITION_BLOCK: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    hidden_offsets = program * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
    selected_row_size = SELECTED_LAYERS * HIDDEN_SIZE
    hidden_requests = hidden_offsets // selected_row_size
    hidden_within_row = hidden_offsets % selected_row_size
    hidden_layers = hidden_within_row // HIDDEN_SIZE
    hidden_columns = hidden_within_row % HIDDEN_SIZE
    captured_hidden = captured_num_reqs * selected_row_size
    valid_hidden = hidden_offsets < num_reqs * selected_row_size
    source_rows = (
        tl.load(
            query_start_loc + (hidden_requests + 1) * query_start_stride,
            mask=valid_hidden,
            other=1,
        )
        - tl.load(
            num_rejected + hidden_requests * rejected_stride,
            mask=valid_hidden,
            other=0,
        )
        - 1
    )
    source_addresses = tl.load(
        selected_hidden_ptrs + hidden_layers,
        mask=valid_hidden,
        other=0,
    )
    source = source_addresses.to(
        tl.pointer_type(selected_hidden_out.dtype.element_ty)
    )
    hidden = tl.load(
        source + source_rows * HIDDEN_SIZE + hidden_columns,
        mask=valid_hidden,
        other=0.0,
    )
    tl.store(
        selected_hidden_out + hidden_offsets,
        hidden,
        mask=hidden_offsets < captured_hidden,
    )

    if program == 0:
        position_offsets = tl.arange(0, POSITION_BLOCK)
        position_requests = position_offsets // BLOCK_SIZE
        position_steps = position_offsets % BLOCK_SIZE
        captured_positions = position_offsets < captured_num_reqs * BLOCK_SIZE
        valid_positions = position_requests < num_reqs
        position_base = tl.load(
            current_positions + position_requests * current_position_stride,
            mask=valid_positions,
            other=-position_steps,
        )
        tl.store(
            position_ids_out + position_offsets,
            position_base + position_steps,
            mask=captured_positions,
        )

        request_offsets = tl.arange(0, REQUEST_BLOCK)
        captured_requests = request_offsets < captured_num_reqs
        valid_requests = request_offsets < num_reqs
        request_slot = tl.load(
            request_slots + request_offsets * request_slot_stride,
            mask=valid_requests,
            other=padding_request_slot,
        )

        sampled = tl.load(
            num_sampled + request_offsets * sampled_stride,
            mask=valid_requests,
            other=0,
        )
        sampled_token = tl.load(
            last_sampled + request_slot * last_sampled_stride,
            mask=valid_requests,
            other=0,
        )
        prefill_token = tl.load(
            next_prefill_tokens + request_slot * next_prefill_stride,
            mask=valid_requests,
            other=0,
        )
        anchor = tl.where(sampled > 0, sampled_token, prefill_token)
        tl.store(anchors_out + request_offsets, anchor, mask=captured_requests)

        current_position = tl.load(
            current_positions + request_offsets * current_position_stride,
            mask=valid_requests,
            other=1,
        )
        tl.store(
            current_positions_out + request_offsets,
            current_position,
            mask=captured_requests,
        )

        tl.store(
            request_slots_out + request_offsets,
            request_slot,
            mask=captured_requests,
        )

        start = tl.load(
            state_committed_lens + request_slot * state_committed_len_stride,
            mask=valid_requests,
            other=0,
        )
        end = current_position
        tl.store(starts_out + request_offsets, start, mask=captured_requests)
        tl.store(ends_out + request_offsets, end, mask=captured_requests)
        tl.store(
            state_committed_lens + request_slot * state_committed_len_stride,
            end,
            mask=valid_requests,
        )


@dataclass
class _GraphInputs:
    selected_hidden: torch.Tensor
    anchors: torch.Tensor
    position_ids: torch.Tensor
    global_output: torch.Tensor
    raw_keys: torch.Tensor
    raw_values: torch.Tensor
    raw_mask: torch.Tensor
    current_positions: torch.Tensor
    global_request_slots: torch.Tensor
    global_starts: torch.Tensor
    global_ends: torch.Tensor

    def slice(self, batch_size: int) -> "_GraphInputs":
        return _GraphInputs(
            selected_hidden=self.selected_hidden[:batch_size],
            anchors=self.anchors[:batch_size],
            position_ids=self.position_ids[:batch_size],
            global_output=self.global_output[:batch_size],
            raw_keys=self.raw_keys[:batch_size],
            raw_values=self.raw_values[:batch_size],
            raw_mask=self.raw_mask[:batch_size],
            current_positions=self.current_positions[:batch_size],
            global_request_slots=self.global_request_slots[:batch_size],
            global_starts=self.global_starts[:batch_size],
            global_ends=self.global_ends[:batch_size],
        )


class LongSparkCudaGraphManager(CudaGraphManager):
    """Capture the eager-free draft backbone and greedy Markov sampler."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        dtype: torch.dtype,
        selected_layers: int,
        hidden_size: int,
        query_heads: int,
        global_slots: int,
        head_dim: int,
        block_size: int,
    ) -> None:
        super().__init__(
            vllm_config,
            device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=1,
        )
        self.dtype = dtype
        self.selected_layers = selected_layers
        self.hidden_size = hidden_size
        self.query_heads = query_heads
        self.global_slots = global_slots
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_capture_batch = max(
            (
                desc.num_reqs or 0
                for descs in self._capture_descs.values()
                for desc in descs
            ),
            default=0,
        )
        self.static_inputs: dict[BatchExecutionDescriptor, _GraphInputs] = {}
        self._shared_inputs: _GraphInputs | None = None
        self._selected_hidden_addresses: tuple[int, ...] = ()
        self._selected_hidden_ptrs_host = torch.empty(
            selected_layers,
            dtype=torch.uint64,
            device="cpu",
            pin_memory=True,
        )
        self._selected_hidden_ptrs = torch.empty(
            selected_layers,
            dtype=torch.uint64,
            device=device,
        )

    def _allocate_inputs(self, batch_size: int) -> _GraphInputs:
        return _GraphInputs(
            selected_hidden=torch.zeros(
                batch_size,
                self.selected_layers,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            ),
            anchors=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            position_ids=torch.arange(
                self.block_size, dtype=torch.long, device=self.device
            )[None]
            .expand(batch_size, -1)
            .clone(),
            global_output=torch.zeros(
                batch_size,
                self.selected_layers,
                self.query_heads,
                self.global_slots,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            raw_keys=torch.zeros(
                batch_size,
                self.selected_layers,
                1,
                1,
                1,
                dtype=self.dtype,
                device=self.device,
            ),
            raw_values=torch.zeros(
                batch_size,
                self.selected_layers,
                1,
                1,
                1,
                dtype=self.dtype,
                device=self.device,
            ),
            raw_mask=torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=self.device,
            ),
            current_positions=torch.zeros(
                batch_size, dtype=torch.long, device=self.device
            ),
            global_request_slots=torch.arange(
                batch_size, dtype=torch.long, device=self.device
            ),
            global_starts=torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            ),
            global_ends=torch.ones(batch_size, dtype=torch.int32, device=self.device),
        )

    def _make_inputs(self, batch_size: int) -> _GraphInputs:
        if batch_size > self.max_capture_batch:
            raise ValueError(
                f"LongSpark graph batch {batch_size} exceeds the largest "
                f"configured capture size {self.max_capture_batch}"
            )
        if self._shared_inputs is None:
            self._shared_inputs = self._allocate_inputs(self.max_capture_batch)
        return self._shared_inputs.slice(batch_size)

    def capture(self, forward_fn: Callable[..., None]) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor, warmup: bool
        ) -> Callable[[CUDAGraphMode], None]:
            assert desc.num_reqs is not None
            inputs = self._make_inputs(desc.num_reqs)
            if not warmup:
                self.static_inputs[desc] = inputs

            def forward(cg_mode: CUDAGraphMode) -> None:
                del cg_mode
                forward_fn(
                    inputs.selected_hidden,
                    inputs.anchors,
                    inputs.position_ids,
                    inputs.global_output,
                    inputs.raw_keys,
                    inputs.raw_values,
                    inputs.raw_mask,
                    inputs.current_positions,
                    desc.num_reqs,
                    inputs.global_request_slots,
                    inputs.global_starts,
                    inputs.global_ends,
                )

            return forward

        super().capture(create_forward_fn, "Capturing LongSpark CUDA graphs")

    def release_graphs(self) -> None:
        super().release_graphs()
        self.static_inputs.clear()
        self._shared_inputs = None
        self._selected_hidden_addresses = ()

    def select(self, num_reqs: int) -> BatchExecutionDescriptor | None:
        desc = self.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            uniform_token_count=1,
            num_active_loras=0,
        )
        return desc if desc.cg_mode == CUDAGraphMode.FULL else None

    def get_inputs(self, desc: BatchExecutionDescriptor) -> _GraphInputs:
        """Return graph-owned buffers that can be populated in-place."""
        return self.static_inputs[desc]

    def replay(
        self,
        desc: BatchExecutionDescriptor,
        *,
        num_reqs: int,
        selected_hidden_states: list[torch.Tensor],
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        global_output: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        current_positions: torch.Tensor,
        global_request_slots: torch.Tensor,
        state_committed_lens: torch.Tensor,
        padding_request_slot: int,
    ) -> None:
        inputs = self.static_inputs[desc]
        captured_num_reqs = desc.num_reqs
        assert captured_num_reqs is not None and captured_num_reqs >= num_reqs
        if len(selected_hidden_states) != self.selected_layers:
            raise ValueError("LongSpark selected hidden-state count does not match")
        if any(
            state.device != self.device
            or state.dtype != self.dtype
            or state.ndim != 2
            or state.shape[1] != self.hidden_size
            or not state.is_contiguous()
            for state in selected_hidden_states
        ):
            raise ValueError("LongSpark selected hidden states have invalid layouts")
        hidden_ptrs = tuple(state.data_ptr() for state in selected_hidden_states)
        if hidden_ptrs != self._selected_hidden_addresses:
            for index, address in enumerate(hidden_ptrs):
                self._selected_hidden_ptrs_host[index] = address
            self._selected_hidden_ptrs.copy_(
                self._selected_hidden_ptrs_host,
                non_blocking=True,
            )
            self._selected_hidden_addresses = hidden_ptrs
        selected_row_size = self.selected_layers * self.hidden_size
        copy_block = 256
        position_block = triton.next_power_of_2(captured_num_reqs * self.block_size)
        request_block = triton.next_power_of_2(captured_num_reqs)
        copy_elements = captured_num_reqs * max(selected_row_size, self.block_size)
        _prepare_graph_inputs_kernel[(triton.cdiv(copy_elements, copy_block),)](
            self._selected_hidden_ptrs,
            query_start_loc,
            num_rejected,
            inputs.selected_hidden,
            num_sampled,
            last_sampled,
            next_prefill_tokens,
            inputs.anchors,
            current_positions,
            inputs.current_positions,
            inputs.position_ids,
            global_request_slots,
            inputs.global_request_slots,
            state_committed_lens,
            inputs.global_starts,
            inputs.global_ends,
            num_reqs,
            captured_num_reqs,
            padding_request_slot,
            current_positions.stride(0),
            global_request_slots.stride(0),
            state_committed_lens.stride(0),
            query_start_loc.stride(0),
            num_rejected.stride(0),
            num_sampled.stride(0),
            last_sampled.stride(0),
            next_prefill_tokens.stride(0),
            SELECTED_LAYERS=self.selected_layers,
            HIDDEN_SIZE=self.hidden_size,
            BLOCK_SIZE=self.block_size,
            COPY_BLOCK=copy_block,
            POSITION_BLOCK=position_block,
            REQUEST_BLOCK=request_block,
            num_warps=4,
        )
        # Some inputs may already have been gathered directly into graph-owned
        # storage. Avoid redundant device copies in that case.
        if inputs.global_output is not global_output:
            inputs.global_output[:num_reqs].copy_(global_output)
        if inputs.raw_keys is not raw_keys:
            inputs.raw_keys[:num_reqs].copy_(raw_keys)
        if inputs.raw_values is not raw_values:
            inputs.raw_values[:num_reqs].copy_(raw_values)
        if inputs.raw_mask is not raw_mask:
            inputs.raw_mask[:num_reqs].copy_(raw_mask)
        self.run_fullgraph(desc)


__all__ = ["LongSparkCudaGraphManager"]
