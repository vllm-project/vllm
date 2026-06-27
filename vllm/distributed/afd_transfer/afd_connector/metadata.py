# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AFD metadata definitions for communication between attention and
FFN workers."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from abc import ABC, abstractmethod


class AFDRecvHandle(ABC):
    """
    Abstract base class for AFD receive handles.

    This provides a handle interface for managing asynchronous AFD operations,
    allowing waiting for completion of data transfer operations.
    """
    @abstractmethod
    def __init__(self, handle: Any):
        """Initialize the AFD receive handle.

        Args:
            handle: Backend-specific handle object
        """
        raise NotImplementedError

    @abstractmethod
    def wait(self):
        """Wait for the operation associated with this handle to complete.

        Blocks until the data transfer or computation is finished.
        """
        raise NotImplementedError


class AFDConnectorData:
    """Base class for connector-specific metadata objects."""
    pass


@dataclass
class AFDRecvOutput:
    """Standardized output for recv_attn_output across all connectors."""
    hidden_states: torch.Tensor
    metadata: Optional[Any] = None  # AFDConnectorMetadata

    # Common / Shared fields
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    dynamic_scales: Optional[torch.Tensor] = None
    group_list: Optional[torch.Tensor] = None

    # M2N specific
    handle: Optional[Any] = None

    # P2P specific
    router_logits: Optional[torch.Tensor] = None
    row_idx: Optional[torch.Tensor] = None

    # CAM specific fields (mapped from raw lists)
    expand_idx: Optional[torch.Tensor] = None
    ep_recv_counts: Optional[torch.Tensor] = None
    atten_batch_size: Optional[torch.Tensor] = None
    x_active_mask: Optional[torch.Tensor] = None
    cam_p2p_ep_name: Optional[str] = None


@dataclass
class AFDConnectorMetadata:
    """Lightweight AFD metadata containing core information needed for
    communication."""
    layer_idx: int
    stage_idx: int
    seq_lens: list[
        int]  # Length of each sequence, supports variable length and
    # multiple sequences
    dtype: torch.dtype
    device: torch.device
    num_ubatches: int = 1

    # Generic field for connector-specific data
    connector_data: Optional["AFDConnectorData"] = None

    topk_idx: Optional[torch.Tensor] = None # indices token which expert to be sended
    topk_weights: Optional[torch.Tensor] = None # the expert weights
    topk_ids: Optional[torch.Tensor] = None
    row_idx: Optional[torch.Tensor] = None
    moe_expert_num: Optional[int] = None # number of moe experts
    shared_expert_num: Optional[int] = None # number of share experts
    scale: Optional[torch.Tensor] = None #  quant scale
    expertTokenNumsOut: Optional[torch.Tensor] = None # The number of tokens received by each expert is used as input for the subsequent GMM.
    send_handle_list: Optional[list[Any]] = None # the communication handles (list of Work objects returned by torch.distributed.isend)
    recv_handle_list: Optional[list[Any]] = None # the communication handles (list of Work objects returned by torch.distributed.irecv)

    # Optional fields for debugging and extensibility
    request_id: Optional[str] = None
    timestamp: Optional[float] = None

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
            num_ubatches: int = 1,
            request_id: Optional[str] = None,
            connector_data: Optional["AFDConnectorData"] = None,
            topk_weights: Optional[torch.Tensor] = None,
            topk_ids: Optional[torch.Tensor] = None,
            row_idx: Optional[torch.Tensor] = None,
            **kwargs) -> "AFDConnectorMetadata":
        """Create metadata for attention side (single sequence)."""
        return cls(layer_idx=layer_idx,
                   stage_idx=stage_idx,
                   seq_lens=[seq_len],
                   dtype=dtype,
                   device=device,
                   num_ubatches=num_ubatches,
                   request_id=request_id,
                #    timestamp=time.time(),
                   connector_data=connector_data,
                   topk_weights=topk_weights,
                   topk_ids=topk_ids,
                   row_idx=row_idx,
                #    extra_fields = extra_fields
                   )

    @classmethod
    def create_ffn_metadata(
            cls,
            layer_idx: int,
            stage_idx: int,
            seq_lens: list[int],
            dtype: torch.dtype,
            device: torch.device,
            request_id: Optional[str] = None) -> "AFDConnectorMetadata":
        """Create metadata for FFN side (multiple sequences)."""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=seq_lens.copy(),  # Prevent external modification
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=time.time())

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
