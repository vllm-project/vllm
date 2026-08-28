# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client half of EPD dynamic registration.

An instance announces itself to the proxy once it is serving, then keeps
announcing on an interval. The repeat is what makes a proxy restart
survivable: a proxy that comes back with an empty roster refills on its own
instead of needing every instance restarted.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

import aiohttp

from vllm.distributed.ec_transfer.proxy.registry import InstanceRole
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

DEFAULT_ANNOUNCE_INTERVAL = 30.0
_RETRY_INTERVAL = 5.0
_REQUEST_TIMEOUT = 5.0


class ProxyRegistrar:
    """Keeps one instance present in a proxy's roster.

    Args:
        proxy_urls: Proxies to announce to. More than one is a proxy that
            is itself replicated; each gets the same record.
        role: Which stage this instance serves.
        url: Base URL other components should reach this instance at.
        ec_zmq_addrs: Encoder-cache receive addresses, if this instance
            consumes embeddings over a point-to-point transport.
        dp_size: Data-parallel replicas behind `url`.
        interval: Seconds between announcements.
    """

    def __init__(
        self,
        proxy_urls: list[str],
        role: InstanceRole,
        url: str,
        ec_zmq_addrs: list[str] | None = None,
        dp_size: int = 1,
        interval: float = DEFAULT_ANNOUNCE_INTERVAL,
    ):
        self.proxy_urls = [proxy.rstrip("/") for proxy in proxy_urls]
        self.interval = interval
        self.payload = {
            "role": role.value,
            "url": url.rstrip("/"),
            "ec_zmq_addrs": ec_zmq_addrs or [],
            "dp_size": dp_size,
        }
        self._task: asyncio.Task | None = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        proxy_urls: list[str],
        role: InstanceRole,
        url: str,
        interval: float = DEFAULT_ANNOUNCE_INTERVAL,
    ) -> ProxyRegistrar:
        """Build a registrar, asking the EC connector for its receive addresses."""
        return cls(
            proxy_urls=proxy_urls,
            role=role,
            url=url,
            ec_zmq_addrs=_receive_addresses(vllm_config),
            dp_size=vllm_config.parallel_config.data_parallel_size,
            interval=interval,
        )

    def start(self) -> None:
        """Announce in the background; never block the instance's startup."""
        if self._task is None and self.proxy_urls:
            self._task = asyncio.create_task(self._announce_loop())

    async def stop(self) -> None:
        """Stop announcing and leave the roster, so no request is sent here."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._request("delete")

    async def _announce_loop(self) -> None:
        # The first announcement retries fast so a proxy that is still coming
        # up costs seconds, not a whole announce interval.
        while not await self._request("post"):
            await asyncio.sleep(_RETRY_INTERVAL)
        while True:
            await asyncio.sleep(self.interval)
            await self._request("post")

    async def _request(self, method: str) -> bool:
        """Send `payload` to every proxy. Returns whether all of them took it."""
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(
                *(
                    self._request_one(session, method, proxy)
                    for proxy in self.proxy_urls
                ),
                return_exceptions=True,
            )
        return all(result is True for result in results)

    async def _request_one(
        self, session: aiohttp.ClientSession, method: str, proxy: str
    ) -> bool:
        try:
            async with session.request(
                method, f"{proxy}/instances", json=self.payload
            ) as resp:
                if resp.status == 200:
                    return True
                logger.warning(
                    "EPD proxy %s rejected registration with %s: %s",
                    proxy,
                    resp.status,
                    await resp.text(),
                )
        except Exception as exc:
            logger.debug("Could not reach EPD proxy %s: %s", proxy, exc)
        return False


def _receive_addresses(vllm_config: VllmConfig) -> list[str]:
    ec_config = getattr(vllm_config, "ec_transfer_config", None)
    if ec_config is None or not ec_config.is_ec_consumer:
        return []
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

    try:
        connector_cls = ECConnectorFactory.get_connector_class(ec_config)
    except Exception:
        logger.warning(
            "Could not resolve the EC connector class; registering without "
            "receive addresses. An encoder will not be told where to push.",
            exc_info=True,
        )
        return []
    return connector_cls.receive_addresses(vllm_config)
