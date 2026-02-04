# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AFD metadata definitions for communication between attention and
FFN workers."""

import time
import typing
from dataclasses import dataclass, field
from typing import Any

import torch


class FFNNeedForwardData:
    def __init__(
        self,
        moe_comm_method: typing.Any,
        num_input_tokens: int,
        with_prefill: bool,
        total_num_scheduled_tokens: int | None,
        is_dummy_run: bool = False,
    ):
        self.moe_comm_method = moe_comm_method
        self.num_input_tokens = num_input_tokens
        self.with_prefill = with_prefill
        self.total_num_scheduled_tokens = total_num_scheduled_tokens
        self.is_dummy_run = is_dummy_run


@dataclass
class AFDConnectorMetadata:
    """Lightweight AFD metadata containing core information needed for
    communication."""

    layer_idx: int
    stage_idx: int
    seq_lens: list[int]  # Length of each sequence, supports variable length and
    # multiple sequences
    dtype: torch.dtype
    device: torch.device
    topk_idx: torch.Tensor | None = None  # indices token which expert to be sended
    topk_weights: torch.Tensor | None = None  # the expert weights
    moe_expert_num: int | None = None  # number of moe experts
    shared_expert_num: int | None = None  # number of share experts
    scale: torch.Tensor | None = None  #  quant scale
    expertTokenNumsOut: torch.Tensor | None = (
        None  # The number of tokens received by each expert is used as input for the subsequent GMM.
    )
    recv_handle_list: list[Any] | None = (
        None  # the communication handles (list of Work objects returned by torch.distributed.irecv)
    )

    # Optional fields for debugging and extensibility
    request_id: str | None = None
    timestamp: float | None = None
    """ffn need forward data"""
    ffn_need_forward_data: FFNNeedForwardData | None = None
    num_of_stages: int = 1
    afd_tokens_lens: list = field(default_factory=list)

    def __post_init__(self):
        """Validate data consistency."""
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("All sequence lengths must be positive")

    @property
    def total_tokens(self) -> int:
        """Total number of tokens."""
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        """Number of sequences."""
        return len(self.seq_lens)

    @property
    def is_single_sequence(self) -> bool:
        """Whether this is a single sequence (attention side characteristic)."""
        return len(self.seq_lens) == 1

    @property
    def is_multi_sequence(self) -> bool:
        """Whether this is multiple sequences (FFN side characteristic)."""
        return len(self.seq_lens) > 1

    @classmethod
    def create_attention_metadata(
        cls,
        layer_idx: int,
        stage_idx: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        request_id: str | None = None,
        ffn_need_forward_data: FFNNeedForwardData | None = None,
        num_of_stages: int = 1,
        afd_tokens_lens: list[int] = [],
    ) -> "AFDConnectorMetadata":
        """Create metadata for attention side (single sequence)."""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],
            dtype=dtype,
            device=device,
            request_id=request_id,
            ffn_need_forward_data=ffn_need_forward_data,
            timestamp=1234,
            num_of_stages=num_of_stages,
            afd_tokens_lens=afd_tokens_lens,
        )

    @classmethod
    def create_ffn_metadata(
        cls,
        layer_idx: int,
        stage_idx: int,
        seq_lens: list[int],
        dtype: torch.dtype,
        device: torch.device,
        request_id: str | None = None,
    ) -> "AFDConnectorMetadata":
        """Create metadata for FFN side (multiple sequences)."""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=seq_lens.copy(),  # Prevent external modification
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=time.time(),
        )

    def get_split_indices(self) -> list[int]:
        """Get tensor split indices for FFN side output splitting."""
        if len(self.seq_lens) <= 1:
            return []

        indices = []
        cumsum = 0
        for length in self.seq_lens[:-1]:  # Exclude the last one
            cumsum += length
            indices.append(cumsum)
        return indices

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        """Validate if tensor shape is consistent with metadata."""
        if len(tensor_shape) < 1:
            return False
        return tensor_shape[0] == self.total_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary format for serialization and debugging."""
        return {
            "layer_idx": self.layer_idx,
            "stage_idx": self.stage_idx,
            "seq_lens": self.seq_lens,
            "dtype": self.dtype,
            "device": self.device,
            "total_tokens": self.total_tokens,
            "num_sequences": self.num_sequences,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        """Friendly string representation."""
        return (
            f"AFDConnectorMetadata(layer={self.layer_idx}, "
            f"stage={self.stage_idx}, seq_lens={self.seq_lens}, "
            f"total_tokens={self.total_tokens}, dtype={self.dtype}, "
            f"device={self.device}, request_id={self.request_id})"
        )
