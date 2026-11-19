# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel residual stream of large batches (TP > 1, CUDA).

Each rank owns a contiguous token shard of the hidden stream. The attention
partial sums (written straight into this rank's symmetric buffer) are
reduce-scattered to the shard owners; the MoE combine pushes its partial rows
straight into the owners' per-source slots instead. The owners run the short
conv (on the slots' rank-order sum) and the AttnRes mix + RMSNorm on their
shard only and push the normalized mix outputs into every rank's all-gather
buffer.
"""

import torch
import torch.distributed as dist

# Barrier flag rows at the start of every symmetric buffer: one per
# reduce-scatter CTA, then the all-gather's and the MoE push's.
_MAX_CTAS = 1024
_FLAG_BYTES = 64 * 1024
_MAX_QUERIES = 3
# The conv taps' halo rows before a shard.
HALO_ROWS = 3


class SeqParallel:
    """Symmetric (MNNVL) buffers of up to `max_tokens` rows: the partial
    sums (aliased by the per-source slots of pushed partial rows), the
    `(3, max_tokens, hidden)` all-gather destination and the staging rows
    of the conv-state cache.

    Later steps read the conv-state cache rows of each request's last
    `tail_rows` tokens (decode and verify rollbacks, the next chunk) and of
    the tokens before prefix-cache hit boundaries; their owners push them to
    every rank, which writes them to its cache.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        max_tokens: int,
        tail_rows: int,
    ) -> None:
        from flashinfer.comm.mnnvl import (
            SymmDeviceMemory,
            TorchDistBackend,
            create_tensor_from_cuda_memory,
        )

        self.world_size = group.size()
        self.rank = dist.get_rank(group)
        # The smallest batches that run sequence parallel: below, the
        # all-reduce of the full stream is faster end to end.
        self.min_tokens = 2048 if self.world_size <= 4 else 1536
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.tail_rows = tail_rows
        row_bytes = hidden_size * 2
        # Each source rank's slot: the halo rows, then the shard's rows. A
        # rank pushes into them (or overwrites its partial sums) only after
        # the all-gather barrier of a later conv + mix, which every rank
        # passes once done reading them.
        self.slot_rows = -(-max_tokens // self.world_size) + HALO_ROWS
        partial_rows = max(max_tokens, self.world_size * self.slot_rows)
        self.partial_offset = _FLAG_BYTES
        self.gather_offset = self.partial_offset + partial_rows * row_bytes
        self.staging_offset = self.gather_offset + _MAX_QUERIES * max_tokens * row_bytes
        size = self.staging_offset + max_tokens * row_bytes
        device = torch.accelerator.current_device_index()
        self.memory = SymmDeviceMemory(
            size,
            self.world_size,
            self.rank,
            device,
            TorchDistBackend(group=group),
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self.buffer = create_tensor_from_cuda_memory(
            self.memory.uc_ptrs[self.rank], (size,), torch.uint8, device
        )
        self.buffer[:_FLAG_BYTES].zero_()
        self.bases = self.memory.uc_ptrs_dev
        # Every rank's buffer, starting with this rank's (push order).
        ptrs = self.memory.uc_ptrs
        self.push_bases = torch.tensor(
            ptrs[self.rank :] + ptrs[: self.rank], dtype=torch.int64, device=device
        )
        # This rank's slot in every rank's buffer.
        self.slot_bases = torch.tensor(
            [
                ptr + self.partial_offset + self.rank * self.slot_rows * row_bytes
                for ptr in ptrs
            ],
            dtype=torch.int64,
            device=device,
        )
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # One reduce-scatter CTA per SM: more only add barrier traffic.
        self.epochs = torch.zeros(num_sms, dtype=torch.int32, device=device)
        self.gather_slot = _MAX_CTAS
        # The all-gather barrier's epoch, finished-CTA count and released
        # epoch.
        self.gather_epochs = torch.zeros(3, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group)

    def _view(self, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= s
        return self.buffer[offset : offset + 2 * numel].view(torch.bfloat16).view(shape)

    def shard(self, num_tokens: int) -> tuple[int, int, int]:
        """This rank's tokens `[begin, end)` and the shard capacity."""
        size = -(-num_tokens // self.world_size)
        begin = min(self.rank * size, num_tokens)
        return begin, min(begin + size, num_tokens), size

    def partial(self, num_tokens: int) -> torch.Tensor:
        """Where this rank writes its `(T, hidden)` partial sums."""
        assert num_tokens <= self.max_tokens
        return self._view(self.partial_offset, (num_tokens, self.hidden_size))

    def gathered(self, num_queries: int, num_tokens: int) -> torch.Tensor:
        return self._view(
            self.gather_offset, (_MAX_QUERIES, self.max_tokens, self.hidden_size)
        )[:num_queries, :num_tokens]

    def reduce_scatter(self, num_tokens: int) -> tuple[torch.Tensor, int]:
        """The reduced rows of this rank's shard, preceded by the conv taps'
        halo rows (and their count)."""
        begin, end, _ = self.shard(num_tokens)
        halo = min(begin, 3)
        out = torch.empty(
            (end - begin + halo, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.epochs.device,
        )
        torch.ops._C.inkling2_reduce_scatter(
            self.bases,
            self.epochs,
            self.rank,
            self.world_size,
            self.partial_offset,
            begin - halo,
            out,
        )
        return out, halo

    def pushed(self, num_tokens: int) -> tuple[torch.Tensor, int]:
        """The `(world, halo + shard, hidden)` partial rows of this rank's
        shard every rank pushed (`ops.moe.combine_push`), and the halo count."""
        begin, end, _ = self.shard(num_tokens)
        halo = min(begin, HALO_ROWS)
        slots = self._view(
            self.partial_offset, (self.world_size, self.slot_rows, self.hidden_size)
        )
        return slots[:, HALO_ROWS - halo : HALO_ROWS + end - begin], halo
