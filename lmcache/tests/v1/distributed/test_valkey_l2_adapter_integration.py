# SPDX-License-Identifier: Apache-2.0
"""
Integration test for the Valkey L2 adapter in MP mode.

Talks to a **real** Valkey (or Redis) deployment via the ``glide_sync``
client (from the ``valkey-glide-sync`` package). Skipped unless that
package is importable and a server is reachable at the configured target.

Target selection via pytest CLI options (see ``conftest.py``):

  * ``--valkey-cluster`` + ``--valkey-nodes=h:p,h:p`` — **cluster** mode.
  * ``--valkey-host`` / ``--valkey-port`` — **standalone** mode
    (defaults ``localhost:6379``).

Examples::

    pytest <thisfile> --valkey-host=localhost --valkey-port=6390 -v
    pytest <thisfile> --valkey-cluster \\
        --valkey-nodes=127.0.0.1:7000,127.0.0.1:7001 -v
"""

# Standard
from typing import Any
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _build_config(
    cluster_mode: bool, startup_nodes: str, **extra: Any
) -> ValkeyL2AdapterConfig:
    """Build a ValkeyL2AdapterConfig for the resolved target."""
    spec = {
        "type": "valkey",
        "startup_nodes": startup_nodes,
        "cluster_mode": cluster_mode,
    }
    spec.update(extra)
    return ValkeyL2AdapterConfig.from_dict(spec)


def _probe_reachable(cluster_mode: bool, startup_nodes: str) -> bool:
    """Whether a real Valkey responds at the target.

    Builds a throwaway adapter and runs a tiny lookup; any exception
    (no glide_sync, connection refused, auth error) means "not
    reachable".

    A ``glide_sync`` without an import spec is a hand-built stand-in (the
    unit tests install one), not the installed package. Treat it as
    unreachable so these tests skip rather than passing vacuously against
    an in-process fake.
    """
    try:
        # Third Party
        import glide_sync
    except ImportError:
        return False
    if getattr(glide_sync, "__spec__", None) is None:
        return False

    try:
        adapter = ValkeyL2Adapter(
            _build_config(
                cluster_mode,
                startup_nodes,
                num_workers=1,
                connection_timeout=3.0,
                request_timeout=3.0,
            )
        )
    except Exception:
        return False
    try:
        tid = adapter.submit_lookup_and_lock_task([create_object_key(0)], _EMPTY_LAYOUT)
        wait_for_event_fd(adapter.get_lookup_and_lock_event_fd(), timeout=3.0)
        adapter.query_lookup_and_lock_result(tid)
        return True
    except Exception:
        return False
    finally:
        adapter.close()


@pytest.fixture(scope="session")
def valkey_target(request) -> tuple[bool, str]:
    """Resolve ``(cluster_mode, startup_nodes)`` from CLI options."""
    if request.config.getoption("--valkey-cluster"):
        return True, request.config.getoption("--valkey-nodes")
    host = request.config.getoption("--valkey-host")
    port = request.config.getoption("--valkey-port")
    return False, f"{host}:{port}"


@pytest.fixture(scope="session")
def valkey_reachable(valkey_target) -> bool:
    """Probe the target once per session."""
    return _probe_reachable(*valkey_target)


class TestValkeyL2AdapterIntegration:
    """E2E tests that exercise the real glide↔Valkey wire path."""

    @pytest.fixture
    def adapter(self, valkey_target, valkey_reachable):
        if not valkey_reachable:
            cluster_mode, nodes = valkey_target
            pytest.skip(
                f"Valkey not reachable (cluster={cluster_mode}, nodes={nodes}) "
                "or valkey-glide-sync not installed"
            )
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cluster_mode, startup_nodes = valkey_target
        # A unique key_prefix isolates this run's keys in a (possibly
        # shared) Valkey and lets the fixture clean up only its own keys.
        adapter = create_l2_adapter(
            _build_config(
                cluster_mode,
                startup_nodes,
                num_workers=4,
                key_prefix="itest",
                max_capacity_gb=1.0,
            )
        )
        self._stored: list[ObjectKey] = []
        yield adapter
        if self._stored:
            adapter.delete(self._stored)
        adapter.close()

    def _store(self, adapter, keys, objs):
        self._stored.extend(keys)
        tid = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        return adapter.pop_completed_store_tasks()[tid]

    def _lookup(self, adapter, keys):
        tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        return adapter.query_lookup_and_lock_result(tid)

    def _load(self, adapter, keys, objs):
        tid = adapter.submit_load_task(keys, objs)
        assert wait_for_event_fd(adapter.get_load_event_fd())
        return adapter.query_load_result(tid)

    def test_roundtrip_is_byte_exact(self, adapter):
        """Real glide SET → GET must round-trip the payload bit-for-bit.

        This is the core thing the mock cannot prove: that the bytes
        actually survive real wire serialization and buffer GET.
        """
        key = create_object_key(1)
        src = create_memory_obj(size=512, fill_value=3.14159)
        dst = create_memory_obj(size=512, fill_value=0.0)

        assert self._store(adapter, [key], [src]).is_successful()

        assert self._lookup(adapter, [key]).test(0) is True
        adapter.submit_unlock([key])

        assert self._load(adapter, [key], [dst]).test(0) is True
        assert torch.equal(dst.tensor, src.tensor)

    def test_batch_roundtrip_spreads_across_nodes(self, adapter):
        """Many keys round-trip with byte-exact data.

        In cluster mode these keys hash to different slots / nodes, so
        this exercises real cross-node routing and concurrent connections
        — neither of which the single-process mock can represent.
        """
        n = 32
        keys = [create_object_key(i) for i in range(n)]
        src = [create_memory_obj(size=128, fill_value=float(i + 1)) for i in range(n)]
        dst = [create_memory_obj(size=128, fill_value=0.0) for _ in range(n)]

        assert self._store(adapter, keys, src).is_successful()

        bm = self._lookup(adapter, keys)
        assert all(bm.test(i) for i in range(n))
        adapter.submit_unlock(keys)

        bm = self._load(adapter, keys, dst)
        for i in range(n):
            assert bm.test(i) is True
            assert torch.equal(dst[i].tensor, src[i].tensor), f"mismatch at {i}"

    def test_delete_actually_removes_on_server(self, adapter):
        """A real DEL makes the key disappear from the live server."""
        key = create_object_key(1)
        self._store(adapter, [key], [create_memory_obj()])
        assert self._lookup(adapter, [key]).test(0) is True
        adapter.submit_unlock([key])

        adapter.delete([key])
        assert self._lookup(adapter, [key]).test(0) is False

    def test_cluster_reports_real_per_node_memory(self, adapter, valkey_target):
        """``report_status`` runs real ``INFO memory`` routed to all nodes.

        Entirely real-server-only: the mock has no notion of nodes.
        Skipped in standalone mode (per-node memory is cluster-only).
        """
        cluster_mode, _ = valkey_target
        if not cluster_mode:
            pytest.skip("per-node memory is cluster-only")
        # Store something so at least one node has non-zero usage.
        keys = [create_object_key(i) for i in range(8)]
        self._store(adapter, keys, [create_memory_obj(size=256) for _ in keys])

        node_mem = adapter.report_status().get("node_memory", {})
        assert node_mem, "cluster report_status must include node_memory"
        for addr, mem in node_mem.items():
            assert "used_memory" in mem and isinstance(mem["used_memory"], int)
            assert "maxmemory" in mem and isinstance(mem["maxmemory"], int)
