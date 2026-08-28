# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of the instances an EPD proxy routes to.

Instances register themselves once they are ready to serve; the proxy owns
liveness from then on. A probe failure is not proof of death -- a busy
encoder can miss one -- so an instance leaves the routable set only after
`fail_threshold` consecutive failures, and keeps being probed afterwards so
it rejoins on its own once it recovers.
"""

from __future__ import annotations

import asyncio
import enum
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_PROBE_INTERVAL = 5.0
DEFAULT_PROBE_TIMEOUT = 2.0
DEFAULT_FAIL_THRESHOLD = 3
# Stop probing an instance that has been down this long. 0 probes forever,
# which is what a cluster that restarts instances in place wants.
DEFAULT_EVICTED_TTL = 900.0


class InstanceRole(str, enum.Enum):
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceRecord:
    """One registered instance.

    Attributes:
        role: Which stage this instance serves.
        url: Base OpenAI-compatible URL, e.g. ``http://host:8000``.
        ec_zmq_addrs: Control addresses of this instance's encoder-cache
            receive channels, one per rank. Only an EC consumer reports
            these, and only it knows them: they are derived from its own
            connector config and rank layout. The proxy names one to the
            encoder so a push lands where the request will run.
        dp_size: Data-parallel replicas behind `url`, so the proxy can pick
            a replica and name the same one to both halves of a request.
        metadata: Anything else the instance chose to report.
    """

    role: InstanceRole
    url: str
    ec_zmq_addrs: list[str] = field(default_factory=list)
    dp_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.monotonic)


class InstanceRegistry:
    def __init__(
        self,
        probe_interval: float = DEFAULT_PROBE_INTERVAL,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
        evicted_ttl: float = DEFAULT_EVICTED_TTL,
    ):
        self._probe_interval = probe_interval
        self._probe_timeout = probe_timeout
        self._fail_threshold = fail_threshold
        self._evicted_ttl = evicted_ttl

        self._live: dict[str, InstanceRecord] = {}
        self._evicted: dict[str, InstanceRecord] = {}
        self._evicted_since: dict[str, float] = {}
        self._fail_counts: dict[str, int] = {}
        # One cursor per role, only ever incremented. Rebuilding it whenever
        # the roster changes -- what an `itertools.cycle` over a mutable list
        # forces -- restarts every fan-out at the first instance and hot-spots
        # it after each registration.
        self._cursors: dict[InstanceRole, int] = {role: 0 for role in InstanceRole}
        self._probe_task: asyncio.Task | None = None

    def register(self, record: InstanceRecord) -> bool:
        """Add or refresh an instance. Returns True if it was not already live."""
        url = record.url
        self._fail_counts.pop(url, None)
        self._evicted.pop(url, None)
        self._evicted_since.pop(url, None)
        was_new = url not in self._live
        self._live[url] = record
        logger.info(
            "%s instance %s: %s",
            "Registered" if was_new else "Refreshed",
            record.role.value,
            url,
        )
        return was_new

    def unregister(self, url: str) -> bool:
        """Drop an instance for good, so a probe cannot bring it back."""
        found = url in self._live or url in self._evicted
        self._live.pop(url, None)
        self._evicted.pop(url, None)
        self._evicted_since.pop(url, None)
        self._fail_counts.pop(url, None)
        if found:
            logger.info("Unregistered instance: %s", url)
        return found

    def instances(self, role: InstanceRole) -> list[InstanceRecord]:
        return [record for record in self._live.values() if record.role is role]

    def urls(self, role: InstanceRole) -> list[str]:
        return [record.url for record in self.instances(role)]

    def pick(self, role: InstanceRole) -> InstanceRecord | None:
        """Take the next instance of `role` in round-robin order."""
        picked = self.pick_many(role, 1)
        return picked[0] if picked else None

    def pick_many(self, role: InstanceRole, count: int) -> list[InstanceRecord]:
        """Take `count` instances, continuing the rotation across calls.

        A multimodal request fans out one encoder request per item, so the
        assignment has to be contiguous with the previous request's rather
        than restart at the first instance every time.
        """
        alive = self.instances(role)
        if not alive or count <= 0:
            return []
        start = self._cursors[role]
        self._cursors[role] = start + count
        return [alive[(start + offset) % len(alive)] for offset in range(count)]

    def status(self) -> dict[str, Any]:
        return {
            role.value: {
                "live": [record.url for record in self.instances(role)],
                "evicted": [
                    url for url, record in self._evicted.items() if record.role is role
                ],
            }
            for role in InstanceRole
        }

    def start_probing(self) -> None:
        if self._probe_task is None and self._probe_interval > 0:
            self._probe_task = asyncio.create_task(self._probe_loop())

    async def stop_probing(self) -> None:
        if self._probe_task is None:
            return
        self._probe_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._probe_task
        self._probe_task = None

    async def _probe_loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._probe_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                await asyncio.sleep(self._probe_interval)
                try:
                    await self._probe_once(session)
                except Exception:
                    logger.exception("EPD registry probe round failed")

    async def _probe_once(self, session: aiohttp.ClientSession) -> None:
        targets = list(self._live.values()) + list(self._evicted.values())
        if not targets:
            return
        results = await asyncio.gather(
            *(self._probe(session, record.url) for record in targets),
            return_exceptions=True,
        )
        now = time.monotonic()
        for record, healthy in zip(targets, results):
            if healthy is True:
                self._on_probe_success(record)
            else:
                self._on_probe_failure(record, now)
        self._drop_expired(now)

    async def _probe(self, session: aiohttp.ClientSession, url: str) -> bool:
        async with session.get(f"{url}/health") as resp:
            return resp.status == 200

    def _on_probe_success(self, record: InstanceRecord) -> None:
        self._fail_counts.pop(record.url, None)
        if record.url in self._evicted:
            self._evicted.pop(record.url, None)
            self._evicted_since.pop(record.url, None)
            self._live[record.url] = record
            logger.info(
                "Instance %s (%s) is healthy again; routing resumed",
                record.url,
                record.role.value,
            )

    def _on_probe_failure(self, record: InstanceRecord, now: float) -> None:
        if record.url in self._evicted:
            return
        failures = self._fail_counts.get(record.url, 0) + 1
        self._fail_counts[record.url] = failures
        if failures < self._fail_threshold:
            return
        self._live.pop(record.url, None)
        self._evicted[record.url] = record
        self._evicted_since[record.url] = now
        logger.warning(
            "Instance %s (%s) failed %d consecutive probes; stopped routing "
            "to it. It rejoins on its own once it responds again.",
            record.url,
            record.role.value,
            failures,
        )

    def _drop_expired(self, now: float) -> None:
        if self._evicted_ttl <= 0:
            return
        for url, since in list(self._evicted_since.items()):
            if now - since < self._evicted_ttl:
                continue
            self._evicted.pop(url, None)
            self._evicted_since.pop(url, None)
            self._fail_counts.pop(url, None)
            logger.warning("Instance %s stayed down; forgetting it", url)
