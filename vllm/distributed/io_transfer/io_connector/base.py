# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IOConnectorBase — abstract base for transferring IO data
between disaggregated vLLM roles (preprocessor / worker / postprocessor).

The interface mirrors ECConnectorBase but is oriented around
per-request IO data rather than encoder-cache tensors:

    put_data(object_id, data, metadata) -> DataRef  – called by the sender role
    get_data(ref) -> DataObject                     – called by the receiver role
"""

import asyncio
import enum
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import IOTransferConfig, VllmConfig
    from vllm.entrypoints.pooling.pooling.protocol import (
        DataObject,
        DataRef,
        MetadataRef,
    )

logger = init_logger(__name__)


class IOConnectorRole(enum.Enum):
    # Can only send IO data, e.g. the preprocessing instance
    SENDER = 0
    # Can only receive IO data, e.g. the postprocessing instance
    RECEIVER = 1
    # Can send and receive IO data, e.g. the inference instance
    BOTH = 2

    @classmethod
    def from_config(cls, io_transfer_config: "IOTransferConfig") -> "IOConnectorRole":
        if io_transfer_config.is_io_sender and io_transfer_config.is_io_receiver:
            return cls.BOTH
        if io_transfer_config.is_io_sender:
            return cls.SENDER
        return cls.RECEIVER


class IOConnectorBase(ABC):
    """Abstract base class for IO data transfer connectors.

    Subclasses implement the storage / transport backend
    (shared filesystem, object store, in-memory bus, …).
    """

    def __init__(self, vllm_config: "VllmConfig", role: IOConnectorRole) -> None:
        self._vllm_config = vllm_config
        self._role = role
        cfg = vllm_config.io_transfer_config
        if cfg is None:
            raise ValueError("io_transfer_config must be set for IOConnectorBase")
        self._is_io_sender: bool = cfg.is_io_sender
        self._is_io_receiver: bool = cfg.is_io_receiver

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def role(self) -> IOConnectorRole:
        return self._role

    @property
    def is_io_sender(self) -> bool:
        """True when this instance sends IO data."""
        return self._is_io_sender

    @property
    def is_io_receiver(self) -> bool:
        """True when this instance receives IO data."""
        return self._is_io_receiver

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up resources.  Called when the process is shutting down."""
        return

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def put_data(
        self,
        object_id: str,
        data: Any,
    ) -> "DataRef":
        """Persist *data* so that the receiver role can retrieve it.

        Args:
            object_id: Unique identifier for the stored object.
            data: The payload to persist.

        Returns:
            A ``DataRef`` that the receiver can pass to ``get_data``.

        """

    @abstractmethod
    def get_data(
        self,
        ref: "DataRef",
    ) -> "DataObject":
        """Retrieve previously stored data for *ref*.

        Args:
            ref: A ``DataRef`` returned by a prior call to ``put_data``.

        Returns:
            The ``DataObject`` associated with *ref*.

        Raises:
            KeyError: If no entry exists for *ref*.

        """

    @abstractmethod
    def cleanup_data(
        self,
        ref: "DataRef",
    ) -> None:
        """Delete stored data."""
        ...

    @abstractmethod
    def put_metadata(
        self,
        request_id: str,
        metadata: dict,
    ) -> "MetadataRef":
        """Store per-request *metadata* under *request_id*.

        Metadata is keyed by request_id (not object_id), so the same
        data object can be reused across requests with different metadata.

        Args:
            request_id: The request identifier — unique per inference request.
            metadata:   Arbitrary key/value dict (e.g. CRS, transform, output
                        path, out_data_format).

        Returns:
            A ``MetadataRef`` that the receiver can pass to ``get_metadata``.

        """

    @abstractmethod
    def get_metadata(
        self,
        ref: "MetadataRef",
    ) -> dict:
        """Retrieve per-request metadata stored by ``put_metadata``.

        Args:
            ref: A ``MetadataRef`` returned by a prior call to ``put_metadata``.

        Returns:
            The metadata dict.

        Raises:
            KeyError: If no entry exists for *ref*.

        """

    @abstractmethod
    def cleanup_metadata(
        self,
        ref: "MetadataRef",
    ) -> None:
        """Delete stored metadata referenced by *ref*."""
        ...

    # ------------------------------------------------------------------
    # Async interface (optional override)
    # ------------------------------------------------------------------

    async def put_data_async(
        self,
        object_id: str,
        data: Any,
        executor: Executor | None = None,
    ) -> "DataRef":
        """Async variant of ``put_data``.

        The default implementation offloads ``put_data`` to *executor*.
        Subclasses may override when the put path is cheap enough to run
        inline or when a native async implementation is available.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.put_data,
            object_id,
            data,
        )

    async def get_data_async(
        self,
        ref: "DataRef",
        executor: Executor | None = None,
    ) -> "DataObject":
        """Async variant of ``get_data``.

        The default implementation offloads ``get_data`` to *executor*
        (or the running loop's default executor when *executor* is None).
        Subclasses that have a native async transport (e.g. ZMQ/asyncio)
        should override this to avoid occupying an OS thread for the RTT.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.get_data,
            ref,
        )

    async def put_metadata_async(
        self,
        request_id: str,
        metadata: dict,
        executor: Executor | None = None,
    ) -> "MetadataRef":
        """Async variant of ``put_metadata``.  Default offloads to *executor*."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.put_metadata,
            request_id,
            metadata,
        )

    async def get_metadata_async(
        self,
        ref: "MetadataRef",
        executor: Executor | None = None,
    ) -> dict:
        """Async variant of ``get_metadata``.  Default offloads to *executor*."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.get_metadata,
            ref,
        )
