# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side warm-prefetch dispatch.

Forwards a client's warm-prefetch request to one named MP server and proxies
its status. The MP server's ``POST /cache/prefetches`` submits the load and returns
a ``request_id`` immediately (the load runs in the server's storage-manager
thread); the coordinator relays that id back to the client, which then polls
``GET /cache/prefetches/{instance_id}/{request_id}`` on the coordinator until the
server reports completion. There is no background polling on either side -- the
submit and status calls are quick and the client drives completion on demand.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    import httpx

    # First Party
    from lmcache.v1.mp_coordinator.registry import MPInstance

logger = init_logger(__name__)


class PrefetchManager:
    """Submit warm-prefetch requests to MP servers and proxy their status."""

    async def submit_prefetch(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
    ) -> dict[str, Any]:
        """``POST /cache/prefetches`` to ``target`` and return its JSON reply.

        Args:
            target: The MP server to warm.
            http_client: Shared async client for outbound coordinator calls.
            model_name: Model whose layout the target uses to allocate L1.
            world_size: World size selecting the layout and per-rank fan-out.
            token_ids: Prompt tokens whose complete chunks should be warmed.
            cache_salt: Per-tenant isolation salt applied to the produced keys.

        Returns:
            The server's reply, e.g. ``{"request_id", "chunks", "status"}`` or
            ``{"chunks": 0, "status": "noop"}``.

        Raises:
            httpx.HTTPError: If the target is unreachable or returns non-2xx.
        """
        url = f"http://{target.ip}:{target.http_port}/cache/prefetches"
        body = {
            "model_name": model_name,
            "world_size": world_size,
            "token_ids": token_ids,
            "cache_salt": cache_salt,
        }
        resp = await http_client.post(url, json=body)
        resp.raise_for_status()
        logger.info(
            "Prefetch submitted to %s: %d tokens", target.instance_id, len(token_ids)
        )
        return resp.json()

    async def get_status(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        request_id: str,
    ) -> tuple[int, dict[str, Any]]:
        """Proxy ``GET /cache/prefetches/{request_id}`` on ``target``.

        Args:
            target: The MP server holding the job.
            http_client: Shared async client for outbound coordinator calls.
            request_id: The id returned by :meth:`submit_prefetch`.

        Returns:
            ``(status_code, body)`` from the server (e.g. 200 with a status
            body, or 404 for an unknown id), relayed verbatim to the caller.

        Raises:
            httpx.HTTPError: If the target is unreachable (transport error).
        """
        url = f"http://{target.ip}:{target.http_port}/cache/prefetches/{request_id}"
        resp = await http_client.get(url)
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        return resp.status_code, body
