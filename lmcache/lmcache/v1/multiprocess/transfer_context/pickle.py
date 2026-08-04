# SPDX-License-Identifier: Apache-2.0
"""Pickle-based EngineDrivenContext implementation for multiprocess mode."""

# Standard
import pickle

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)


class EngineDrivenContextPickle(EngineDrivenContext):
    """Pickle-based implementation of :class:`EngineDrivenContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE`` (returns empty slots
      for pickle mode); ``commit_store`` serialises chunks and sends
      ``COMMIT_STORE``.
    - **Retrieve**: ``prepare_retrieve`` sends ``PREPARE_RETRIEVE`` and
      deserialises the returned bytes; ``commit_retrieve`` sends
      ``COMMIT_RETRIEVE`` (no-op for pickle).
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        mq_client: MessageQueueClient,
        mq_timeout: float,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)

    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Send PREPARE_STORE RPC. For pickle, returns no pre-allocated buffers."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return None

    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        serialised = pickle.dumps(chunks)
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, serialised],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE and deserialize the response data.

        Returns:
            Chunks on hit, or None on miss/timeout.
        """
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        chunks: list[torch.Tensor] = pickle.loads(response.data)
        return chunks

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Send COMMIT_RETRIEVE (no-op for pickle path)."""
        future = self.mq_client.submit_request(
            RequestType.COMMIT_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.COMMIT_RETRIEVE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return True

    def close(self) -> None:
        """No-op: the pickle path holds no persistent resources."""
