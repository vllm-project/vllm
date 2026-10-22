# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs for the checkpoint-configured LongSpark proposal pass."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
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
        selected_hidden: torch.Tensor,
        anchors: torch.Tensor,
        position_ids: torch.Tensor,
        global_output: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        current_positions: torch.Tensor,
        global_request_slots: torch.Tensor,
        global_starts: torch.Tensor,
        global_ends: torch.Tensor,
        padding_request_slot: int,
    ) -> None:
        inputs = self.static_inputs[desc]
        captured_num_reqs = desc.num_reqs
        assert captured_num_reqs is not None and captured_num_reqs >= num_reqs
        inputs.selected_hidden[:num_reqs].copy_(selected_hidden)
        inputs.anchors[:num_reqs].copy_(anchors)
        inputs.position_ids[:num_reqs].copy_(position_ids)
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
        inputs.current_positions[:num_reqs].copy_(current_positions)
        inputs.global_request_slots[:num_reqs].copy_(global_request_slots)
        inputs.global_starts[:num_reqs].copy_(global_starts)
        inputs.global_ends[:num_reqs].copy_(global_ends)
        if captured_num_reqs > num_reqs:
            inputs.selected_hidden[num_reqs:].zero_()
            inputs.anchors[num_reqs:].zero_()
            inputs.position_ids[num_reqs:].zero_()
            inputs.current_positions[num_reqs:].fill_(1)
            inputs.global_request_slots[num_reqs:].fill_(padding_request_slot)
            inputs.global_starts[num_reqs:].zero_()
            inputs.global_ends[num_reqs:].fill_(1)
        self.run_fullgraph(desc)


__all__ = ["LongSparkCudaGraphManager"]
