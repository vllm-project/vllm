# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill staging for the pool's partition path.

Large uncaptured partition steps read their non-resident rows from a device
buffer instead of the pinned host view: Marlin reads its B operand once per
M-block, so a host row would cross PCIe once per block; staged, it crosses
once. Two full-layer buffers alternate by layer (residency per layer moves
under the global LRU, so a layer may have any number of non-resident rows).

Every layer's full pinned source is copied by DMA on a side stream: layer
i+1's rows land while layer i computes. Copying resident rows too avoids a
host readback of the device-managed placement. The GEMM still selects only
non-resident rows from these buffers. DMA copies do not compete with the
expert kernel for SMs.
"""

from __future__ import annotations

from typing import Any

import torch


class PrefillStage:
    """Two full-layer staging buffers and their copy stream."""

    def __init__(
        self,
        device: torch.device,
        source: dict[str, torch.Tensor],
        num_experts: int,
        min_tokens: int,
    ) -> None:
        self.min_tokens = min_tokens
        self.num_experts = num_experts
        self.buffers = [
            {
                name: torch.empty(
                    (num_experts, *t.shape[1:]), dtype=t.dtype, device=device
                )
                for name, t in source.items()
            }
            for _ in range(2)
        ]
        self.bytes = sum(
            t.numel() * t.element_size() for b in self.buffers for t in b.values()
        )
        self.stream = torch.cuda.Stream(device)
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.layers: list[Any] = []  # PoolLayers in forward order (install)
        self.staged = 0
        self.copied_rows = 0

    def applies(self, x: torch.Tensor) -> bool:
        return (
            x.shape[0] >= self.min_tokens
            and not torch.cuda.is_current_stream_capturing()
        )

    def acquire(self, layer: Any) -> dict[str, torch.Tensor]:
        """The buffer holding `layer`'s non-resident rows for this step (indexed
        by expert id, like the host view); queues the next layer's copy."""
        index = layer.index
        compute = torch.cuda.current_stream()
        if index == 0:
            self._fill(0, compute)
        compute.wait_event(self.ready[index % 2])
        if index + 1 < len(self.layers):
            self._fill(index + 1, compute)
        self.staged += 1
        return self.buffers[index % 2]

    def _fill(self, index: int, compute: torch.cuda.Stream) -> None:
        # After all queued compute work, which includes the expert kernel that
        # last read this buffer (layer index - 2).
        self.stream.wait_stream(compute)
        buffer = self.buffers[index % 2]
        with torch.cuda.stream(self.stream):
            for name, src in self.layers[index].sources.items():
                buffer[name].copy_(src, non_blocking=True)
            self.ready[index % 2].record(self.stream)
        self.copied_rows += self.num_experts
