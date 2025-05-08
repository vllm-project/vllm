# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.pooling_params import PoolingParams
from vllm.utils import is_pin_memory_available


class PoolingMetadata:
    """Metadata for pooling operations in the Pooler layer.

    This class holds the necessary information for pooling operations,
    providing context for how to perform pooling and other related operations.

    Attributes:
        seq_groups: List of (seq_ids, pooling_params).
        seq_data: A mapping of sequence ID to additional sequence data.
        prompt_lens: List of the lengths of each prompt.
        prompt_offsets: List of prompt start offsets for each prompt
        when flat out with padding
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], PoolingParams]],
        seq_data: Dict[int, Any],  # Specific data related to sequences
        prompt_lens: List[int],
        prompt_offsets: Optional[List[int]] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.prompt_offsets = prompt_offsets

    def __repr__(self) -> str:
        return ("PoolingMetadata("
                f"seq_groups={self.seq_groups}, "
                f"seq_data={self.seq_data}, "
                f"prompt_lens={self.prompt_lens}, "
                f"prompt_offsets={self.prompt_offsets})")


@dataclass
class PoolingTensors:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_offsets: torch.Tensor

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
        if pooling_metadata.prompt_offsets is not None:
            prompt_offsets_t = torch.tensor(
                pooling_metadata.prompt_offsets,
                device="cpu",
                dtype=torch.long,
                pin_memory=pin_memory,
            ).to(device=device, non_blocking=True)
        else:
            prompt_offsets_t = None
        return cls(prompt_lens=prompt_lens_t.to(device=device,
                                                non_blocking=True),
                   prompt_offsets=prompt_offsets_t)