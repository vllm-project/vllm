# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod


class AgGemm:
    """Overlap equal-shard all-gather with a BF16 GEMM.

    All ranks must provide the same contiguous ``[local_M, K]`` input shape.
    Outputs contain ``world_size * local_M`` rows concatenated in rank order.
    The caller owns the logical unpadded token count and must slice outputs when
    sequence-parallel inputs contain suffix padding.

    The receive workspace has no cross-rank exit barrier. Before calling again,
    callers must complete a collective or equivalent synchronization that
    guarantees every rank has finished consuming its received shards.
    """

    def __init__(self, max_global_tokens: int, hidden_size: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        tp_group = get_tp_group()
        self.rank = tp_group.rank_in_group
        self.world_size = tp_group.world_size
        self.max_local_tokens = (
            max_global_tokens + self.world_size - 1
        ) // self.world_size
        self.max_global_tokens = self.max_local_tokens * self.world_size
        self.hidden_size = hidden_size
        self.receive = symm_mem.empty(
            (self.world_size, self.max_local_tokens, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.receive_handle = symm_mem.rendezvous(self.receive, tp_group.device_group)
        self.comm_stream = torch.cuda.Stream(priority=-1)
        self.wait_stream = torch.cuda.Stream(priority=-1)
        self.ready_events = [torch.cuda.Event() for _ in range(self.world_size - 1)]

    def can_run(self, linear: LinearBase) -> bool:
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            return False
        weight = linear.weight
        return (
            weight.ndim == 2
            and weight.shape[1] == self.hidden_size
            and weight.dtype == torch.bfloat16
            and weight.device == self.receive.device
            and weight.is_contiguous()
        )

    def __call__(self, local_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        from cuda.bindings import driver

        assert local_input.ndim == 2
        assert local_input.shape[0] <= self.max_local_tokens
        assert local_input.shape[1] == self.hidden_size
        assert local_input.dtype == torch.bfloat16
        assert local_input.is_contiguous()

        parent_stream = torch.cuda.current_stream()
        self.comm_stream.wait_stream(parent_stream)
        self.wait_stream.wait_stream(parent_stream)
        shard_bytes = local_input.nbytes

        # Each source owns a max-sized slot at every destination.
        receive_rank_stride = (
            self.max_local_tokens * self.hidden_size * local_input.element_size()
        )
        next_rank = (self.rank + 1) % self.world_size
        previous_rank = (self.rank - 1) % self.world_size

        def push_to_next_rank(
            source_buffer: torch.Tensor,
            source_rank: int,
            channel: int,
        ) -> None:
            destination_ptr = (
                self.receive_handle.buffer_ptrs[next_rank]
                + source_rank * receive_rank_stride
            )
            result = driver.cuMemcpyDtoDAsync(
                destination_ptr,
                source_buffer.data_ptr(),
                shard_bytes,
                self.comm_stream.cuda_stream,
            )
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyDtoDAsync failed: {result[0]}")
            self.receive_handle.put_signal(next_rank, channel=channel)

        # Push shards clockwise around the ring and compute each one on arrival.
        with torch.cuda.stream(self.comm_stream):
            push_to_next_rank(local_input, self.rank, channel=1)

        output = torch.empty(
            (self.world_size, local_input.shape[0], weight.shape[0]),
            dtype=local_input.dtype,
            device=local_input.device,
        )

        # Compute the local shard while peer copies are in flight.
        torch.mm(local_input, weight.T, out=output[self.rank])

        for step, ready_event in enumerate(self.ready_events, start=1):
            source_rank = (self.rank - step) % self.world_size
            with torch.cuda.stream(self.wait_stream):
                self.receive_handle.wait_signal(previous_rank, channel=step)
                ready_event.record(self.wait_stream)

            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ready_event)
                # Do not send the final shard back to its owner.
                if step < self.world_size - 1:
                    push_to_next_rank(
                        self.receive[source_rank, : local_input.shape[0]],
                        source_rank,
                        channel=step + 1,
                    )

            # Keep signal waits off the GEMM stream and release it via events.
            parent_stream.wait_event(ready_event)
            remote_input = self.receive[source_rank, : local_input.shape[0]]
            torch.mm(remote_input, weight.T, out=output[source_rank])

        # This only protects local source buffers; workspace reuse across ranks
        # is governed by the class contract above.
        parent_stream.wait_stream(self.comm_stream)
        return output.flatten(0, 1)


_ag_gemm: AgGemm | None = None


def init_ag_gemm(max_global_tokens: int, hidden_size: int) -> None:
    """Collectively initialize the process-wide AG-GEMM state."""
    global _ag_gemm
    if _ag_gemm is not None:
        assert _ag_gemm.max_global_tokens >= max_global_tokens
        assert _ag_gemm.hidden_size == hidden_size
        return
    _ag_gemm = AgGemm(max_global_tokens, hidden_size)


def get_ag_gemm() -> AgGemm | None:
    return _ag_gemm
