# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy AFD Connector for testing and local development.

This connector provides a no-op AFDConnectorBase interface,
useful for testing and development scenarios where actual
distributed FFN computation is not needed.
"""

import time
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class DummyAFDConnector(AFDConnectorBase):
    """Dummy AFD connector that returns zero tensors.

    This connector is useful for:
    1. Testing AFD infrastructure without actual remote computation
    2. Development scenarios where FFN computation should be disabled
    3. Fallback behavior when remote FFN servers are unavailable
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        """Initialize the dummy AFD connector.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFDConfig
        """
        self.afd_config = config.afd_config
        self.rank = rank
        self.local_rank = local_rank
        self._is_initialized = False
        self.hidden_size = config.model_config.hf_config.hidden_size
        self.num_stages = config.afd_config.num_afd_stages

        self.events: deque = deque(maxlen=self.num_stages)

        logger.info("DummyAFDConnector initialized for rank %s", rank)

        self.init_afd_connector()

    def init_afd_connector(self) -> None:
        """Initialize the dummy connector.

        This is a no-op for the dummy connector.
        """
        if self._is_initialized:
            return

        logger.info("Initializing DummyAFDConnector (no-op)")
        self._is_initialized = True

    def close(self) -> None:
        """Close the dummy connector.

        This is a no-op for the dummy connector.
        """
        if not self._is_initialized:
            return

        logger.info("Closing DummyAFDConnector (no-op)")
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the connector is initialized.

        Returns:
            bool: True if initialized, False otherwise
        """
        return self._is_initialized

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> Any:
        """
        Send attention output to FFN servers (dummy implementation).
        """
        logger.debug(
            "DummyAFDConnector: send_attn_output layer=%s, stage=%s",
            metadata.layer_idx,
            metadata.stage_idx,
        )

        # Validate metadata consistency
        if not metadata.validate_tensor_shape(hidden_states.shape):
            raise ValueError(
                "Tensor shape %s doesn't match metadata %s",
                hidden_states.shape,
                metadata,
            )

        if not metadata.is_single_sequence:
            raise ValueError("Attention side should have single sequence")

        self.events.append((None, metadata))

        return None

    def recv_ffn_output(
        self,
        timeout_ms: float | None = None,
    ) -> torch.Tensor:
        """Receive FFN computation result (dummy implementation)."""
        logger.debug("DummyAFDConnector: recv_ffn_output timeout_ms=%s", timeout_ms)

        _, metadata = self.events.popleft()
        seq_len = metadata.seq_lens[0]  # Single sequence for attention side
        return torch.zeros(
            seq_len,
            self.hidden_size,
            dtype=metadata.dtype,
            device=metadata.device,
        )

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """
        Receive attention output from attention workers (dummy implementation).
        """
        logger.debug("DummyAFDConnector: recv_attn_output timeout_ms=%s", timeout_ms)

        # Generate dummy data that simulates multiple attention workers
        dummy_seq_lens = [
            2,
            2,
            2,
        ]  # Variable sequence lengths from different workers
        total_tokens = sum(dummy_seq_lens)

        dummy_tensor = torch.zeros(
            total_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        # Create dummy metadata
        dummy_metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=0,  # Dummy layer
            stage_idx=0,  # Dummy stage
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            seq_lens=dummy_seq_lens,
            request_id=f"dummy_ffn_batch_{time.time()}",
        )

        # Cache metadata for send_ffn_output
        self._current_metadata = dummy_metadata
        time.sleep(1)

        return dummy_tensor, dummy_metadata

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """Send FFN computation result back (dummy implementation)."""
        logger.debug(
            "DummyAFDConnector: send_ffn_output layer=%s, stage=%s",
            metadata.layer_idx,
            metadata.stage_idx,
        )

        # Validate that ffn_output shape matches metadata
        if not metadata.validate_tensor_shape(ffn_output.shape):
            logger.warning(
                "FFN output shape %s doesn't match metadata %s",
                ffn_output.shape,
                metadata,
            )

        # Log the splitting information for debugging
        logger.debug(
            "DummyAFDConnector: Split FFN output into %s parts with lengths %s",
            metadata.num_sequences,
            metadata.seq_lens,
        )

        # Simulate splitting (for logging purposes)
        if metadata.get_split_indices():
            split_outputs = torch.split(ffn_output, metadata.seq_lens, dim=0)
            logger.debug(
                "DummyAFDConnector: Split shapes: %s",
                [s.shape for s in split_outputs],
            )

        time.sleep(1)
        # No-op for dummy connector - just log the operation
