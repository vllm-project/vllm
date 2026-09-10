# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import ray
import zmq

from vllm.utils.network_utils import make_zmq_socket, split_zmq_path
from vllm.v1.engine.core import EngineCoreActorMixin
from vllm.v1.engine.core_client import BackgroundResources
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    EngineZmqAddresses,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from vllm.v1.utils import APIServerProcessManager


class _StubEngineCoreActor(EngineCoreActorMixin):
    def __init__(
        self,
        vllm_config: Any,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Any],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Exercise the production Ray actor mixin without loading a model.
        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )

    def _set_visible_devices(self, vllm_config: Any, local_dp_rank: int) -> None:
        pass

    def wait_for_init(self) -> None:
        pass

    def run(self) -> None:
        pass

    def get_nixl_side_channel_host(self) -> str | None:
        return os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST")

    def get_addresses(self) -> tuple[list[str], list[str]]:
        """Return the addresses snapshot the actor was constructed with.

        Used by the Ray-DP regression test to assert that no ``tcp://host:0``
        placeholders were pickled into the actor at ``.remote()`` time.
        """
        return list(self.addresses.inputs), list(self.addresses.outputs)


# Module-level stub worker for the Ray-DP regression test. Must be importable
# by ``multiprocessing.spawn`` (no closures, no nesting). Mirrors the worker
# in ``tests/entrypoints/test_api_server_process_manager.py``.
def _bind_and_report_worker(listen_address, sock, args, client_config):
    """Bind ROUTER/PULL with a kernel-assigned port, report the actual
    endpoints back via ``actual_address_pipe``, then exit."""
    ctx = zmq.Context()
    try:
        in_sock = make_zmq_socket(
            ctx, client_config["input_address"], zmq.ROUTER, bind=True
        )
        out_sock = make_zmq_socket(
            ctx, client_config["output_address"], zmq.PULL, bind=True
        )
        try:
            pipe = client_config["actual_address_pipe"]
            try:
                pipe.send(
                    {
                        "input_address": in_sock.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        "output_address": out_sock.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode(),
                    }
                )
            finally:
                pipe.close()
        finally:
            in_sock.close(linger=0)
            out_sock.close(linger=0)
    finally:
        ctx.term()


class _DummyExecutor:
    pass


def test_background_resources_passes_worker_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = 7
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    engine_manager = Mock()
    resources = BackgroundResources(ctx=None, engine_manager=engine_manager)
    resources()
    engine_manager.shutdown.assert_called_once_with(timeout=timeout)


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            enable_elastic_ep=False,
            world_size=1,
        ),
        model_config=SimpleNamespace(is_moe=False),
        kv_transfer_config=None,
    )


def _make_addresses() -> EngineZmqAddresses:
    return EngineZmqAddresses(
        inputs=["tcp://127.0.0.1:12345"],
        outputs=["tcp://127.0.0.1:12346"],
    )


def _make_cpu_placement_group():
    pg = ray.util.placement_group(
        [{"CPU": 0.001}, {"CPU": 1.0}],
        strategy="PACK",
    )
    ray.get(pg.ready())
    return pg


@pytest.fixture
def ray_context():
    started_ray = False
    if not ray.is_initialized():
        project_root = str(Path(__file__).resolve().parents[3])
        ray.init(
            num_cpus=2,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
            log_to_driver=False,
        )
        started_ray = True

    yield

    if started_ray:
        ray.shutdown()


@pytest.mark.usefixtures("ray_context")
def test_driver_nixl_side_channel_host_does_not_leak_to_engine_core_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver_marker = f"driver-only-nixl-host-{uuid.uuid4()}"
    created_placement_groups: list[Any] = []
    manager: CoreEngineActorManager | None = None

    def create_dp_placement_groups(vllm_config: Any):
        pg = _make_cpu_placement_group()
        created_placement_groups.append(pg)
        return [pg], [0]

    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", driver_marker)
    monkeypatch.setattr("vllm.v1.engine.core.EngineCoreActor", _StubEngineCoreActor)
    monkeypatch.setattr(
        CoreEngineActorManager,
        "create_dp_placement_groups",
        staticmethod(create_dp_placement_groups),
    )

    try:
        manager = CoreEngineActorManager(
            vllm_config=_make_vllm_config(),
            addresses=_make_addresses(),
            executor_class=_DummyExecutor,
            log_stats=False,
        )
        actor = manager.local_engine_actors[0]
        actor_host = ray.get(actor.get_nixl_side_channel_host.remote())
        node_host = ray.util.get_node_ip_address()

        assert actor_host != driver_marker
        assert actor_host == node_host
    finally:
        if manager is not None:
            manager.shutdown()
        else:
            for pg in created_placement_groups:
                ray.util.remove_placement_group(pg)


@pytest.fixture
def ray_context_dp2():
    """Ray context sized for two stub actors (each PG needs ~1 CPU)."""
    started_ray = False
    if not ray.is_initialized():
        project_root = str(Path(__file__).resolve().parents[3])
        ray.init(
            num_cpus=4,
            runtime_env={"env_vars": {"PYTHONPATH": project_root}},
            log_to_driver=False,
        )
        started_ray = True

    yield

    if started_ray:
        ray.shutdown()


def _make_vllm_config_ray_dp_multinode() -> SimpleNamespace:
    """Minimal vllm_config that drives the Ray-DP multi-API-server path:
    ``data_parallel_size != data_parallel_size_local`` forces TCP placeholders
    (multi-node fan-out), and ``data_parallel_backend="ray"`` routes
    ``launch_core_engines`` through the Ray branch.
    """
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=None,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_backend="ray",
            data_parallel_rpc_port=29550,
            local_engines_only=False,
            enable_elastic_ep=False,
            world_size=1,
        ),
        model_config=SimpleNamespace(multimodal_config=None, is_moe=False),
        cache_config=SimpleNamespace(),
        needs_dp_coordinator=False,
        kv_transfer_config=None,
        # ``_apply_dp_identity_suffix`` reads and rewrites this.
        instance_id="vllm-ray-dp-regression-test",
    )


@pytest.mark.timeout(120)
@pytest.mark.usefixtures("ray_context_dp2")
def test_ray_dp_addresses_resolved_before_actor_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the Ray-DP + multi-API-server hang from PR #42585.

    ``launch_core_engines`` Ray branch pickles ``addresses`` into each engine
    actor at ``.remote()`` time, and ``EngineCoreActorMixin._perform_handshakes``
    is a no-op, so the actor uses that pickled snapshot for the rest of its
    life. If ``run_multi_api_server`` allocates ``addresses`` as
    ``tcp://host:0`` placeholders (its default), the actors hold placeholders
    forever and DEALER-connect to port 0 — ZMQ ``connect`` is async and does
    not raise, so the failure mode is a deterministic hang.

    The Ray-DP carve-out in ``run_multi_api_server`` forces
    ``defer_api_server_ports=False`` when ``data_parallel_backend == "ray"``
    so addresses are pre-allocated in the driver and Ray pickles real ports
    into each actor. This test mirrors that call-site logic and asserts the
    actors hold real (non-placeholder) endpoints. If the carve-out is
    removed without an alternative fix, the test fails.
    """
    created_placement_groups: list[Any] = []

    def create_dp_placement_groups(vllm_config: Any):
        pg1 = _make_cpu_placement_group()
        pg2 = _make_cpu_placement_group()
        created_placement_groups.extend([pg1, pg2])
        return [pg1, pg2], [0, 0]

    monkeypatch.setattr("vllm.v1.engine.core.EngineCoreActor", _StubEngineCoreActor)
    monkeypatch.setattr(
        CoreEngineActorManager,
        "create_dp_placement_groups",
        staticmethod(create_dp_placement_groups),
    )

    vllm_config = _make_vllm_config_ray_dp_multinode()

    # Mirror run_multi_api_server's address-allocation logic. The Ray DP
    # carve-out forces pre-allocation so the addresses pickled into engine
    # actors at .remote() time are real, not ``tcp://host:0``.
    is_ray_dp = vllm_config.parallel_config.data_parallel_backend == "ray"
    addresses = get_engine_zmq_addresses(
        vllm_config,
        num_api_servers=2,
        defer_api_server_ports=not is_ray_dp,
    )

    sock = socket.socket()
    engine_manager: CoreEngineActorManager | None = None
    actor_snapshots: list[tuple[list[str], list[str]]] = []
    api_server_manager: APIServerProcessManager | None = None
    try:
        # Ray actors are spawned here, pickling ``addresses`` into each one.
        with launch_core_engines(
            vllm_config,
            executor_class=_DummyExecutor,
            log_stats=False,
            addresses=addresses,
        ) as engine_launch:
            engine_manager = engine_launch.engine_manager
            assert isinstance(engine_manager, CoreEngineActorManager)

            # API-server children bind to the pre-allocated ports.
            api_server_manager = APIServerProcessManager(
                listen_address="tcp://127.0.0.1:0",
                sock=sock,
                args="test_args",
                num_servers=2,
                input_addresses=addresses.inputs,
                output_addresses=addresses.outputs,
                target_server_fn=_bind_and_report_worker,
            )

            # run_multi_api_server skips ``gather_actual_addresses`` for
            # Ray DP (addresses are already real). Mirror that.
            if not is_ray_dp:
                actual_inputs, actual_outputs = (
                    api_server_manager.gather_actual_addresses(timeout=15.0)
                )
                addresses.inputs = actual_inputs
                addresses.outputs = actual_outputs

            # Snapshot what each Ray actor actually holds.
            actors = (
                engine_manager.local_engine_actors + engine_manager.remote_engine_actors
            )
            actor_snapshots = ray.get(
                [actor.get_addresses.remote() for actor in actors]
            )
    finally:
        if api_server_manager is not None:
            api_server_manager.shutdown()
            time.sleep(0.2)
        sock.close()
        if engine_manager is not None:
            engine_manager.shutdown()
        else:
            for pg in created_placement_groups:
                ray.util.remove_placement_group(pg)

    # Every Ray actor must hold real, non-placeholder addresses.
    assert actor_snapshots, "expected at least one Ray actor to be created"
    for actor_inputs, actor_outputs in actor_snapshots:
        for url in actor_inputs + actor_outputs:
            scheme, _host, port = split_zmq_path(url)
            assert scheme == "tcp", url
            assert port and int(port) > 0, (
                f"Ray actor was pickled with placeholder address {url!r}; "
                "``run_multi_api_server`` must pre-allocate ports for the "
                "Ray DP backend so the actors hold real endpoints by the "
                "time they DEALER-connect. See PR #42585 / Ray-DP "
                "multi-API-server regression."
            )


# --------------------------------------------------------------------------- #
# add_dp_placement_groups (elastic EP scale-up) must not need ray[default].
# --------------------------------------------------------------------------- #


def _make_vllm_config_elastic_ep(
    data_parallel_size: int, world_size: int, dp_master_ip: str
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_master_ip=dp_master_ip,
            world_size=world_size,
        )
    )


def _forbid_list_nodes(**kwargs):
    raise AssertionError(
        "ray.util.state.list_nodes() must not be called: it requires the Ray "
        "dashboard HTTP server (ray[default]). See PR #23822."
    )


@pytest.fixture
def fake_ray_cluster(monkeypatch: pytest.MonkeyPatch):
    """Stand in for a live two-node Ray cluster using only the resource maps
    that ``ray._private.state`` exposes (keyed by node id, live nodes only).

    - ``node-b`` is the DP master (``node:10.0.0.1``) and is deliberately
      listed *after* a worker node so the master-first sort is exercised.
    - ``node-b`` has 2 devices total, 1 already used by the existing engine.
    - ``node-a`` is a worker with 2 idle devices.
    - ``node-c`` is a CPU-only node and must be skipped.
    """
    master_ip = "10.0.0.1"
    available = {
        "node-a": {"GPU": 2.0, "CPU": 8.0, "node:10.0.0.2": 1.0},
        "node-b": {
            "GPU": 1.0,
            "CPU": 8.0,
            f"node:{master_ip}": 1.0,
            "node:__internal_head__": 1.0,
        },
        "node-c": {"CPU": 4.0, "node:10.0.0.3": 1.0},
    }
    total = {
        "node-a": {"GPU": 2.0, "CPU": 8.0, "node:10.0.0.2": 1.0},
        "node-b": {
            "GPU": 2.0,
            "CPU": 8.0,
            f"node:{master_ip}": 1.0,
            "node:__internal_head__": 1.0,
        },
        "node-c": {"CPU": 4.0, "node:10.0.0.3": 1.0},
    }
    created: list[SimpleNamespace] = []

    def fake_placement_group(name, strategy, bundles):
        pg = SimpleNamespace(name=name, strategy=strategy, bundles=bundles)
        created.append(pg)
        return pg

    monkeypatch.setattr(
        "ray._private.state.available_resources_per_node", lambda: available
    )
    monkeypatch.setattr("ray._private.state.total_resources_per_node", lambda: total)
    monkeypatch.setattr("ray.util.placement_group", fake_placement_group)
    # Regression guard: the dashboard-backed developer API must stay unused.
    monkeypatch.setattr("ray.util.state.list_nodes", _forbid_list_nodes)
    # CPU platform reports "" here; the production path uses "GPU".
    monkeypatch.setattr("vllm.v1.engine.utils.current_platform.ray_device_key", "GPU")
    return SimpleNamespace(master_ip=master_ip, available=available, created=created)


def test_add_dp_placement_groups_does_not_require_ray_default(fake_ray_cluster):
    """Scale DP 1 -> 3 (TP=1): master first, then the idle worker, CPU-only
    node skipped, without ever touching ``ray.util.state.list_nodes``."""
    master_ip = fake_ray_cluster.master_ip
    vllm_config = _make_vllm_config_elastic_ep(
        data_parallel_size=1, world_size=1, dp_master_ip=master_ip
    )

    placement_groups, local_dp_ranks = CoreEngineActorManager.add_dp_placement_groups(
        vllm_config, new_data_parallel_size=3
    )

    assert placement_groups == fake_ray_cluster.created
    assert [pg.name for pg in placement_groups] == ["dp_rank_1", "dp_rank_2"]
    assert all(pg.strategy == "STRICT_PACK" for pg in placement_groups)
    # Master node already runs one engine (2 total - 1 available), so the new
    # engine there gets local rank 1; the worker node starts at local rank 0.
    assert local_dp_ranks == [1, 0]
    # The master-node bundles are pinned to the master via its node resource.
    assert placement_groups[0].bundles == [
        {"GPU": 1.0, f"node:{master_ip}": 0.001},
        {"CPU": 1.0},
    ]
    assert placement_groups[1].bundles == [{"GPU": 1.0}, {"CPU": 1.0}]


def test_add_dp_placement_groups_respects_world_size(fake_ray_cluster):
    """With TP=2 the master node (1 idle device) cannot host a new engine; the
    single new rank lands on the worker node with 2 idle devices."""
    vllm_config = _make_vllm_config_elastic_ep(
        data_parallel_size=1, world_size=2, dp_master_ip=fake_ray_cluster.master_ip
    )

    placement_groups, local_dp_ranks = CoreEngineActorManager.add_dp_placement_groups(
        vllm_config, new_data_parallel_size=2
    )

    assert [pg.name for pg in placement_groups] == ["dp_rank_1"]
    assert local_dp_ranks == [0]
    assert placement_groups[0].bundles == [
        {"GPU": 1.0},
        {"GPU": 1.0},
        {"CPU": 1.0},
    ]


def test_add_dp_placement_groups_noop_without_growth(fake_ray_cluster):
    vllm_config = _make_vllm_config_elastic_ep(
        data_parallel_size=2, world_size=1, dp_master_ip=fake_ray_cluster.master_ip
    )

    assert CoreEngineActorManager.add_dp_placement_groups(
        vllm_config, new_data_parallel_size=2
    ) == ([], [])
    assert fake_ray_cluster.created == []


def test_add_dp_placement_groups_asserts_when_master_missing(fake_ray_cluster):
    vllm_config = _make_vllm_config_elastic_ep(
        data_parallel_size=1, world_size=1, dp_master_ip="192.0.2.99"
    )

    with pytest.raises(AssertionError, match="DP master node"):
        CoreEngineActorManager.add_dp_placement_groups(
            vllm_config, new_data_parallel_size=2
        )
    assert fake_ray_cluster.created == []
