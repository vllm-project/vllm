# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.pooling_params import PoolingParams
from vllm.utils import is_pin_memory_available
from vllm.v1.pool.metadata import PoolingCursor, build_pooling_cursor


class PoolingMetadata:
    """Metadata for pooling operations in the Pooler layer.

    This class holds the necessary information for pooling operations,
    providing context for how to perform pooling and other related operations.

    Attributes:
        seq_groups: List of (seq_ids, pooling_params).
        seq_data: A mapping of sequence ID to additional sequence data.
        prompt_lens: List of the lengths of each prompt.
    """

    def __init__(
            self,
            seq_groups: list[tuple[list[int], PoolingParams]],
            seq_data: dict[int, Any],  # Specific data related to sequences
            prompt_lens: list[int],
            pooling_cursor: Optional[PoolingCursor] = None) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.pooling_cursor: Optional[PoolingCursor] = pooling_cursor

    def __repr__(self) -> str:
        return ("PoolingMetadata("
                f"seq_groups={self.seq_groups}, "
                f"seq_data={self.seq_data}, "
                f"prompt_lens={self.prompt_lens})")

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            seq_groups=self.seq_groups[indices],
            seq_data=dict(list(self.seq_data.items())[indices]),
            prompt_lens=self.prompt_lens[indices],
            pooling_cursor=None
            if self.pooling_cursor is None else self.pooling_cursor[indices],
        )

    def build_pooling_cursor(self, num_scheduled_tokens: list[int],
                             device: torch.device):
        prompt_lens = torch.tensor(self.prompt_lens, device="cpu")
        self.pooling_cursor = build_pooling_cursor(num_scheduled_tokens,
                                                   prompt_lens,
                                                   device=device)


@dataclass
class PoolingTensors:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor

    @classmethod
    def from_pooling_metadata(
        cls,
        pooling_metadata: "PoolingMetadata",
        device: torch.device,
    ) -> "PoolingTensors":
        """
        Create PoolingTensors from PoolingMetadata.

        Args:
            pooling_metadata: PoolingMetadata instance to convert.
            device: Device to store the tensors.
        """
        # Convert prompt lengths to tensor
        pin_memory = is_pin_memory_available()

        prompt_lens_t = torch.tensor(
            pooling_metadata.prompt_lens,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        )

        return cls(prompt_lens=prompt_lens_t.to(device=device,
                                                non_blocking=True), )
