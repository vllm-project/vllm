# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated instance settings and scheduler-process background peer discovery."""

import hashlib
import ipaddress
import math
import os
import queue
import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import msgspec
import numpy as np

from vllm.logger import init_logger

from .data import PeerMemory
from .native import PlacementReplica, description

logger = init_logger(__name__)

RDMA_LEASE_MS = 30000
IPC_LEASE_MS = 1000
MIN_GAIN_PAGES = 4


@dataclass(frozen=True)
class Domain:
    """Hold immutable sharing settings in each scheduler and worker process.

    An empty instance ID selects same-instance IPC; explicit instances require
    compatible identity, control endpoints and a shared credential.
    """

    instance_id: str = ""
    compatibility_id: str = ""
    host: str = ""
    port: int = 0
    seeds: tuple[str, ...] = ()
    credential: bytes = field(default=b"", repr=False)
    role: str = "full"
    rdma_transport: str | None = None
    protected_cap_blocks: int | None = None
    lease_ms: int = RDMA_LEASE_MS
    timeout_s: float = 1.0
    min_gain_pages: int = MIN_GAIN_PAGES

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "Domain":
        """Validate public options and read the configured inherited secret."""
        allowed = {
            "role",
            "instance_id",
            "kv_compatibility_id",
            "control_host",
            "control_port",
            "peer_seeds",
            "shared_secret_env",
            "rdma_transport",
            "protected_cap_blocks",
            "read_timeout_s",
        }
        unknown = options.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown peer L1 options: {', '.join(sorted(unknown))}")
        secret = ""
        secret_env = str(options.get("shared_secret_env", ""))
        if secret_env:
            secret = os.environ.get(secret_env, "")
            if not secret:
                raise ValueError("Peer shared_secret_env is not set")
        if options.get("control_host") and not secret_env:
            raise ValueError("Peer L1 control_host requires shared_secret_env")
        result = cls(
            instance_id=str(options.get("instance_id", "")),
            compatibility_id=str(options.get("kv_compatibility_id", "")),
            host=str(options.get("control_host", "")),
            port=int(options.get("control_port", 0)),
            seeds=tuple(options.get("peer_seeds", ())),
            credential=hashlib.sha256(secret.encode()).digest() if secret else b"",
            role=str(options.get("role", "full")),
            rdma_transport=options.get("rdma_transport"),
            protected_cap_blocks=(
                int(options["protected_cap_blocks"])
                if "protected_cap_blocks" in options
                else None
            ),
            timeout_s=float(options.get("read_timeout_s", 1.0)),
        )
        if result.role not in ("full", "decode"):
            raise ValueError("Peer L1 role must be full or decode")
        if result.protected_cap_blocks is not None and result.protected_cap_blocks < 0:
            raise ValueError("Peer L1 protected_cap_blocks must be nonnegative")
        if not math.isfinite(result.timeout_s) or result.timeout_s <= 0:
            raise ValueError("Peer L1 read_timeout_s must be finite and positive")
        if result.rdma_transport is not None:
            from .transport import TRANSPORTS

            if result.rdma_transport not in TRANSPORTS:
                raise ValueError(
                    f"Unknown RDMA transport {result.rdma_transport!r}; "
                    f"choose from {list(TRANSPORTS)}"
                )
        configured = (
            result.instance_id
            or result.compatibility_id
            or result.host
            or result.port
            or result.seeds
        )
        if configured:
            if not (
                result.instance_id
                and result.compatibility_id
                and result.host
                and 0 < result.port < 65000
            ):
                raise ValueError(
                    "Cross-instance peers require instance_id, kv_compatibility_id, "
                    "control_host and control_port"
                )
            if ipaddress.ip_address(result.host).is_unspecified:
                raise ValueError(
                    "control_host must be an advertised, reachable numeric IP"
                )
            if len(result.seeds) > 128 or not all(
                isinstance(seed, str) and seed.startswith("tcp://")
                for seed in result.seeds
            ):
                raise ValueError("peer_seeds must contain at most 128 TCP endpoints")
        return result

    def endpoint(self, rank: int) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        port = self.port + 2 * rank
        if not 0 < port < 65535:
            raise ValueError(
                "Peer control/subscription ports exceed the TCP port range"
            )
        return f"tcp://{host}:{port}"


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    instance_id: str
    members: tuple[PeerMemory, ...]
    world_size: int


def compatible(local: PeerMemory, remote: PeerMemory) -> bool:
    try:
        endpoint = urlsplit(remote.endpoint)
        if (
            endpoint.scheme != "tcp"
            or endpoint.port is None
            or not 0 < endpoint.port < 65535
        ):
            return False
        ipaddress.ip_address(endpoint.hostname or "")
    except ValueError:
        return False
    return bool(
        local.kv_compatibility_id
        and remote.role in ("full", "decode")
        and (local.role != "decode" or remote.role == "full")
        and local.kv_compatibility_id == remote.kv_compatibility_id
        and local.compatibility == remote.compatibility
        and local.hash_settings == remote.hash_settings
        and "rdma" in remote.session_kinds
        and remote.rdma
        and local.rdma_transport == remote.rdma_transport
        and remote.num_blocks > 1
        and remote.epoch > 0
        and math.isfinite(remote.lease_s)
        and remote.lease_s > 0
        and remote.endpoint.startswith("tcp://")
        and not remote.handles
    )


@dataclass(frozen=True)
class RemoteSnapshot:
    version: int
    members: tuple[PeerMemory, ...]
    replica: PlacementReplica


class Resolver:
    """Discover compatible owners off the scheduler's request path.

    The Python discovery thread performs bounded RPCs and swaps snapshots under
    `_lock`; the scheduler claims a snapshot until it adopts its replacement.
    """

    def __init__(self, domain: Domain, local: PeerMemory, timeout_ms: int = 50):
        self.domain, self.local, self.timeout_ms = domain, local, timeout_ms
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._snapshot: RemoteSnapshot | None = None
        self._retired: list[RemoteSnapshot] = []
        self._claimed: set[int] = set()
        self._next: dict[str, tuple[float, float]] = {}
        self._invalidated: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._instances: dict[str, tuple[Manifest, float]] = {}
        self._endpoint_instances: dict[str, str] = {}
        self._rotations: dict[str, int] = {}
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="peer-resolver"
        )

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> RemoteSnapshot | None:
        """Claim the current replica under a short lock without network I/O."""
        with self._lock:
            if self._snapshot is not None:
                self._claimed.add(self._snapshot.version)
            return self._snapshot

    def acknowledge(self, version: int) -> None:
        """Keep the adopted version claimed and allow older replicas to close."""
        with self._lock:
            self._claimed = {version}

    def invalidate(self, endpoint: str) -> None:
        """Queue an early discovery retry without network I/O on the scheduler."""
        self._invalidated.put(endpoint)

    def _fetch(self, seed: str) -> Manifest:
        raw = description(
            seed,
            0,
            members=True,
            credential=self.domain.credential,
            timeout_ms=self.timeout_ms,
        )
        manifest = msgspec.json.decode(raw, type=Manifest)
        if (
            manifest.schema_version != 1
            or not manifest.instance_id
            or not 0 < len(manifest.members) <= 256
            or manifest.world_size != len(manifest.members)
        ):
            raise ValueError("Unsupported or incomplete MEMBERS snapshot")
        owners = set()
        previous = self._instances.get(manifest.instance_id)
        verified = (
            {} if previous is None else {m.endpoint: m for m in previous[0].members}
        )
        for member in manifest.members:
            if self._stop.is_set():
                raise ConnectionError("Resolver is closing")
            if (
                member.instance_id != manifest.instance_id
                or member.rank in owners
                or member.epoch <= 0
            ):
                raise ValueError("Inconsistent MEMBERS snapshot")
            owners.add(member.rank)
            if verified.get(member.endpoint) == member:
                continue
            hello = description(
                member.endpoint,
                member.epoch,
                credential=self.domain.credential,
                timeout_ms=self.timeout_ms,
            )
            if msgspec.json.decode(hello, type=PeerMemory) != member:
                raise ValueError("HELLO differs from MEMBERS")
        if owners != set(range(manifest.world_size)):
            raise ValueError("MEMBERS ranks are not a complete instance")
        return manifest

    def _refresh(self) -> None:
        now = monotonic()
        while True:
            try:
                endpoint = self._invalidated.get_nowait()
                self._next.pop(self._endpoint_instances.get(endpoint, endpoint), None)
            except queue.Empty:
                break
        groups: dict[str, list[str]] = {}
        for seed in self.domain.seeds:
            groups.setdefault(self._endpoint_instances.get(seed, seed), []).append(seed)
        for manifest, _ in self._instances.values():
            groups.setdefault(manifest.instance_id, []).extend(
                m.endpoint for m in manifest.members
            )
        attempted: set[str] = set()
        for key, endpoints in list(groups.items())[:128]:
            if self._stop.is_set():
                break
            endpoints = list(dict.fromkeys(endpoints))
            if key in attempted or any(
                self._endpoint_instances.get(e) in attempted for e in endpoints
            ):
                continue
            at, delay = self._next.get(key, (0.0, 0.05))
            if now < at:
                continue
            attempted.add(key)
            rotation = self._rotations.get(key, 0)
            seed = endpoints[rotation % len(endpoints)]
            try:
                manifest = self._fetch(seed)
                attempted.add(manifest.instance_id)
                if manifest.instance_id != self.domain.instance_id:
                    self._instances[manifest.instance_id] = manifest, now
                self._next[manifest.instance_id] = now + 1.0, 0.05
                self._rotations[manifest.instance_id] = rotation
                for endpoint in (seed, *(m.endpoint for m in manifest.members)):
                    self._endpoint_instances[endpoint] = manifest.instance_id
                if key != manifest.instance_id:
                    self._next.pop(key, None)
            except (ValueError, ConnectionError, RuntimeError) as exc:
                self._next[key] = now + delay, min(5.0, delay * 2)
                self._rotations[key] = rotation + 1
                logger.debug(
                    "Peer MEMBERS unavailable endpoint=%s reason=%s", seed, exc
                )
        self._instances = {
            k: v for k, v in self._instances.items() if now - v[1] < 10.0
        }
        active = (
            set(groups)
            | set(self._instances)
            | {self._endpoint_instances.get(seed, seed) for seed in self.domain.seeds}
        )
        self._next = {k: v for k, v in self._next.items() if k in active}
        members = tuple(
            sorted(
                (
                    member
                    for manifest, _ in self._instances.values()
                    for member in manifest.members
                    if compatible(self.local, member)
                ),
                key=lambda m: (m.instance_id, m.rank),
            )
        )
        if len(members) > 256:
            members = members[:256]
        with self._lock:
            current = self._snapshot
        if members != (() if current is None else current.members):
            replica = PlacementReplica(
                [
                    (m.endpoint, m.epoch, m.num_blocks * max(1, len(m.segments)))
                    for m in members
                ],
                credential=self.domain.credential,
            )
            with self._lock:
                if current is not None:
                    self._retired.append(current)
                self._snapshot = RemoteSnapshot(
                    1 if current is None else current.version + 1, members, replica
                )
        with self._lock:
            retired = [s for s in self._retired if s.version not in self._claimed]
            self._retired = [s for s in self._retired if s.version in self._claimed]
        for old in retired:
            old.replica.close()

    def refresh(self) -> None:
        """Serialize discovery; may block on multiple RPCs and replica shutdown."""
        with self._refresh_lock:
            self._refresh()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(0.02)

    def close(self) -> None:
        """Join discovery and close replicas after the scheduler stops querying them."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        for old in self._retired:
            old.replica.close()
        if self._snapshot is not None:
            self._snapshot.replica.close()


class CandidateReplicas:
    """Present local and remote hints as one scheduler-thread peer index space."""

    def __init__(
        self, local: PlacementReplica, remote: PlacementReplica, local_count: int
    ):
        self.local, self.remote, self.local_count = local, remote, local_count

    def generation(self, peer: int) -> int:
        return (
            self.local.generation(peer)
            if peer < self.local_count
            else self.remote.generation(peer - self.local_count)
        )

    def window(self, peer: int) -> int:
        return (
            self.local.window(peer)
            if peer < self.local_count
            else self.remote.window(peer - self.local_count)
        )

    def query(self, keys: list[bytes]) -> np.ndarray:
        return np.concatenate((self.local.query(keys), self.remote.query(keys)), axis=1)

    def capacity(self, peer: int) -> int:
        return (
            self.local.capacity(peer)
            if peer < self.local_count
            else self.remote.capacity(peer - self.local_count)
        )

    def try_query_any(self, keys: list[bytes]) -> np.ndarray | None:
        local = self.local.try_query_any(keys)
        remote = self.remote.try_query_any(keys)
        return None if local is None or remote is None else local | remote
