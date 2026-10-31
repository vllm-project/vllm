# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the row table: the device table of one tier and the tier choice."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.models.glm53.nvidia.ops.cute_dsl import row_table
from vllm.models.glm53.nvidia.ops.cute_dsl.row_table import (
    PACKED_SLOTS,
    RowTableLayout,
)


class RowTable:
    """The device row table of one tier."""

    def __init__(
        self, layout: RowTableLayout, device: torch.device, packed: bool = False
    ) -> None:
        capacity = PACKED_SLOTS[layout.rmax] if packed else layout.bmax
        if layout.bmax != capacity:
            raise ValueError(
                f"a packed row table has {capacity} request words (the kernels of "
                f"a packed build are compiled for them), not {layout.bmax}"
            )
        self.layout: RowTableLayout = layout
        # Zeros: the first replay may read idle words before any step wrote them.
        self.tensor: torch.Tensor = torch.zeros(
            layout.table_words if packed else layout.words,
            dtype=torch.int32,
            device=device,
        )
        self._images: dict[tuple[int, int], torch.Tensor] = {}

    def set_uniform_step(self, requests: int, rows_per_request: int) -> None:
        """Write the table of a step of ``requests`` requests of equal row counts, its
        context words 0 (the prologue writes them).
        """
        key = (requests, rows_per_request)
        image = self._images.get(key)
        if image is None:
            words = row_table.header(self.layout, [rows_per_request] * requests)
            image = torch.tensor(words, dtype=torch.int32, device=self.tensor.device)
            self._images[key] = image
        self.tensor[: self.layout.words].copy_(image)

    def fill(self, rows_per_request: Sequence[int], contexts: Sequence[int]) -> None:
        """Write the whole table from the host (tests, where no prologue writes the
        contexts).
        """
        if len(contexts) != len(rows_per_request):
            raise ValueError("one context per request")
        words = row_table.header(self.layout, rows_per_request)
        for request, context in enumerate(contexts):
            words[self.layout.context_word(request)] = context
        self.tensor[: self.layout.words].copy_(
            torch.tensor(words, dtype=torch.int32), non_blocking=False
        )
