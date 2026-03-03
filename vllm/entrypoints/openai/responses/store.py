# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
from abc import ABC, abstractmethod

from openai.types.responses import ResponseStatus

from vllm.entrypoints.openai.responses.protocol import ResponsesResponse

logger = logging.getLogger(__name__)


class ResponseStore(ABC):
    """Abstract interface for persisting Responses API state.

    vLLM ships an in-memory default (``InMemoryResponseStore``).  Users who
    need persistence or multi-node consistency can implement this interface and
    point ``VLLM_RESPONSES_STORE_BACKEND`` to their class's fully-qualified
    name (e.g. ``mypackage.redis_store.RedisResponseStore``).
    """

    @abstractmethod
    async def get_response(self, response_id: str) -> ResponsesResponse | None:
        """Return the stored response, or ``None`` if not found."""

    @abstractmethod
    async def put_response(
        self,
        response_id: str,
        response: ResponsesResponse,
        *,
        unless_status: ResponseStatus | None = None,
    ) -> bool:
        """Store *response* under *response_id*.

        If *unless_status* is given and the **currently stored** response
        already has that status, the write is skipped and ``False`` is
        returned.  Otherwise the write succeeds and ``True`` is returned.
        """

    @abstractmethod
    async def update_response_status(
        self,
        response_id: str,
        new_status: ResponseStatus,
        *,
        allowed_current_statuses: set[ResponseStatus] | None = None,
    ) -> ResponsesResponse | None:
        """Atomically transition a response's status.

        If *allowed_current_statuses* is given the update only proceeds when
        the current status is in the set; otherwise the call is a no-op and
        returns ``None``.

        Returns the (possibly updated) ``ResponsesResponse``, or ``None``
        when the response was not found or the transition was rejected.
        """

    @abstractmethod
    async def get_messages(self, response_id: str) -> list | None:
        """Return the stored input messages for a response, or ``None``."""

    @abstractmethod
    async def put_messages(self, response_id: str, messages: list) -> None:
        """Store the input messages for a response."""

    async def close(self) -> None:
        """Release resources.  Override to clean up connections/pools."""
        return


class InMemoryResponseStore(ResponseStore):
    """Default in-memory implementation — wraps the previous dict behaviour.

    All response mutations are guarded by an internal ``asyncio.Lock`` so
    callers no longer need an external ``response_store_lock``.
    """

    def __init__(self) -> None:
        self._responses: dict[str, ResponsesResponse] = {}
        self._messages: dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def get_response(self, response_id: str) -> ResponsesResponse | None:
        async with self._lock:
            return self._responses.get(response_id)

    async def put_response(
        self,
        response_id: str,
        response: ResponsesResponse,
        *,
        unless_status: ResponseStatus | None = None,
    ) -> bool:
        async with self._lock:
            existing = self._responses.get(response_id)
            if (
                unless_status is not None
                and existing is not None
                and existing.status == unless_status
            ):
                return False
            self._responses[response_id] = response
            return True

    async def update_response_status(
        self,
        response_id: str,
        new_status: ResponseStatus,
        *,
        allowed_current_statuses: set[ResponseStatus] | None = None,
    ) -> ResponsesResponse | None:
        async with self._lock:
            response = self._responses.get(response_id)
            if response is None:
                return None
            if (
                allowed_current_statuses is not None
                and response.status not in allowed_current_statuses
            ):
                return None
            response.status = new_status
            return response

    async def get_messages(self, response_id: str) -> list | None:
        return self._messages.get(response_id)

    async def put_messages(self, response_id: str, messages: list) -> None:
        self._messages[response_id] = messages


def create_response_store() -> ResponseStore:
    """Factory that returns a ``ResponseStore`` instance.

    Checks ``VLLM_RESPONSES_STORE_BACKEND`` — if set, loads the class via
    ``resolve_obj_by_qualname``; otherwise returns ``InMemoryResponseStore``.
    """
    from vllm import envs

    backend = envs.VLLM_RESPONSES_STORE_BACKEND
    if not backend:
        return InMemoryResponseStore()

    from vllm.utils.import_utils import resolve_obj_by_qualname

    cls = resolve_obj_by_qualname(backend)
    if not (isinstance(cls, type) and issubclass(cls, ResponseStore)):
        raise TypeError(
            f"VLLM_RESPONSES_STORE_BACKEND={backend!r} resolved to "
            f"{cls!r}, which is not a ResponseStore subclass."
        )
    logger.info("Using custom response store backend: %s", backend)
    return cls()
