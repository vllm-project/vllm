# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AFDConnectorBase Class for Distributed AFD FFN computation

The class provides the four core AFD communication interfaces:
1. send_attn_output(): Send attention output to FFN servers (Attention Worker)
2. recv_ffn_output(): Receive FFN computation result (Attention Worker)
3. recv_attn_output(): Receive attention output from workers (FFN Server)
4. send_ffn_output(): Send FFN computation result back (FFN Server)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .metadata import AFDConnectorMetadata


class AFDConnectorBase(ABC):
    """
    Abstract base class for AFD connectors.

    This provides the four core interfaces for AFD communication between
    attention workers and FFN servers.
    """

    @abstractmethod
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        """Initialize the AFD connector.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFDConfig
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the connector and release resources."""
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        raise NotImplementedError

    def get_connector_rank(self) -> int:
        """Get the rank of this connector."""
        return getattr(self, "rank", 0)

    def get_connector_local_rank(self) -> int:
        """Get the local rank of this connector."""
        return getattr(self, "local_rank", 0)

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: "AFDConnectorMetadata",
        ubatch_idx: int | None = None,
    ) -> Any:
        """Send attention output to FFN servers.

        Args:
            hidden_states: Attention output tensor
            metadata: AFD metadata containing layer_idx, stage_idx, seq_len info
            ubatch_idx: Optional micro-batch index for ubatching

        Returns:
            Any: Handle for tracking this request (backend-specific)
        """
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(
        self,
        handle: Any = None,
        ubatch_idx: int | None = None,
    ) -> torch.Tensor:
        """Wait for and receive FFN computation result.

        Args:
            handle: Handle returned by send_attn_output()
            ubatch_idx: Optional micro-batch index for ubatching

        Returns:
            torch.Tensor: FFN computation result
        """
        raise NotImplementedError

    @abstractmethod
    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
    ) -> tuple[torch.Tensor, "AFDConnectorMetadata"]:
        """Receive attention output from attention workers.

        Args:
            timeout_ms: Optional timeout in milliseconds
            ubatch_idx: Optional micro-batch index for ubatching

        Returns:
            tuple: (hidden_states, metadata)
                - hidden_states: Concatenated attention outputs
                - metadata: Inferred AFD metadata containing
                            seq_lens and other info
        """
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: "AFDConnectorMetadata",
        ubatch_idx: int | None = None,
    ) -> None:
        """Send FFN computation result back to attention workers.

        Args:
            ffn_output: Computed FFN result
            metadata: AFD metadata containing seq_lens
                      for splitting and routing info
            ubatch_idx: Optional micro-batch index for ubatching
        """
        raise NotImplementedError
