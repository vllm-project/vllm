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
from torch.distributed import TCPStore

from vllm.utils.network_utils import get_open_port, make_zmq_socket, split_zmq_path
from vllm.v1.engine.core import EngineCoreActorMixin
from vllm.v1.engine.core_client import BackgroundResources
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    EngineZmqAddresses,
    _dp_nodes_master_first,
    _node_ip_from_resources,
    bind_engine_zmq_listeners,
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
# by ``multiprocessing.spawn`` (no closures, no nesting).
def _adopt_listener_worker(listen_address, sock, args, client_config):
    """Adopt the supervisor-bound ROUTER/PULL listeners, then exit."""
    ctx = zmq.Context()
    try:
        in_sock = make_zmq_socket(
            ctx,
            client_config["input_address"],
            zmq.ROUTER,
            bind=True,
            listener=client_config["input_listener"],
        )
        out_sock = make_zmq_socket(
            ctx,
            client_config["output_address"],
            zmq.PULL,
            bind=True,
            listener=client_config["output_listener"],
        )
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


@pytest.mark.parametrize(
    ("overrides", "expect_store"),
    [
        ({"data_parallel_backend": "mp"}, True),
        ({}, True),
        (
            {
                "data_parallel_backend": "mp",
                "data_parallel_rank": 1,
                "local_engines_only": True,
            },
            False,
        ),
        ({"data_parallel_backend": "mp", "data_parallel_rank_local": 0}, False),
        ({"data_parallel_backend": "mp", "enable_elastic_ep": True}, False),
        ({"data_parallel_backend": "mp", "data_parallel_size": 1}, False),
    ],
    ids=["mp", "ray", "non-master-node", "offline", "elastic-ep", "single-engine"],
)
def test_coordination_store_held_only_by_the_online_dp_master(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], expect_store: bool
) -> None:
    """The DP master node holds a coordination store for both engine backends,
    so engines there pick world-group ports at bind time. Every other
    deployment keeps the pre-allocated ports and must not hold one."""

    class FakeManager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    vllm_config = _make_vllm_config_ray_dp_multinode()
    parallel_config = vllm_config.parallel_config
    for name, value in overrides.items():
        setattr(parallel_config, name, value)
    # A fresh port per case so a lingering handshake socket cannot fail the next.
    parallel_config.data_parallel_rpc_port = get_open_port()
    parallel_config._coord_store_port = 0
    monkeypatch.setattr("vllm.v1.engine.utils.CoreEngineProcManager", FakeManager)
    monkeypatch.setattr("vllm.v1.engine.utils.CoreEngineActorManager", FakeManager)
    monkeypatch.setattr(
        "vllm.v1.engine.utils.wait_for_engine_startup", lambda *args, **kwargs: None
    )

    with launch_core_engines(
        vllm_config,
        executor_class=_DummyExecutor,
        log_stats=False,
        addresses=_make_addresses(),
    ) as engine_launch:
        assert engine_launch.engine_manager is not None
        if not expect_store:
            assert not parallel_config._coord_store_port
            return
        assert parallel_config._coord_store_port
        # Engines look the store up while they start, so it must be reachable
        # for as long as this frame is alive.
        client = TCPStore(
            parallel_config.data_parallel_master_ip,
            parallel_config._coord_store_port,
            is_master=False,
            wait_for_workers=False,
        )
        client.set("probe", b"1")
        assert client.get("probe") == b"1"


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
    life. The supervisor therefore binds the frontend listeners before actor
    construction and Ray pickles their resolved endpoints. This test asserts
    that each actor holds real (non-placeholder) endpoints.
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

    # Mirror run_multi_api_server: bind listeners in the supervisor so the
    # addresses pickled into Ray actors contain kernel-assigned ports.
    listeners = bind_engine_zmq_listeners(vllm_config, num_api_servers=2)
    addresses = listeners.addresses

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

            # API-server children adopt the already-bound listener FDs.
            api_server_manager = APIServerProcessManager(
                listen_address="tcp://127.0.0.1:0",
                sock=sock,
                args="test_args",
                num_servers=2,
                input_listeners=listeners.inputs,
                output_listeners=listeners.outputs,
                target_server_fn=_adopt_listener_worker,
            )

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
                "``run_multi_api_server`` must bind frontend listeners before "
                "constructing Ray DP actors so they hold real endpoints when "
                "they DEALER-connect. See PR #42585 / Ray-DP "
                "multi-API-server regression."
            )


def test_dp_nodes_master_first_orders_and_validates():
    master = {"GPU": 2.0, "node:10.0.0.1": 1.0, "node:__internal_head__": 1.0}
    worker = {"GPU": 2.0, "node:10.0.0.2": 1.0}

    nodes = _dp_nodes_master_first({"w": worker, "m": master}, "10.0.0.1")
    assert [node_id for node_id, _ in nodes] == ["m", "w"]

    with pytest.raises(AssertionError, match="DP master node"):
        _dp_nodes_master_first({"w": worker}, "10.0.0.1")
    with pytest.raises(AssertionError, match="one head node"):
        _dp_nodes_master_first({"a": master, "b": dict(master)}, "10.0.0.1")


@pytest.fixture
def ray_2gpu_node(monkeypatch: pytest.MonkeyPatch):
    """Real single-node Ray with 2 virtual GPUs and no dashboard.

    Resource maps and placement groups are real. Without the dashboard any
    ``ray.util.state.list_nodes()`` call fails, the production condition this
    path must survive. Only the platform device key is patched ("" on CPU).
    """
    from ray._private.state import available_resources_per_node

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False, log_to_driver=False)
    monkeypatch.setattr("vllm.v1.engine.utils.current_platform.ray_device_key", "GPU")

    (node_resources,) = available_resources_per_node().values()
    master_ip = _node_ip_from_resources(node_resources)
    assert master_ip is not None
    created = []

    def hold_gpus(num_gpus: int):
        """Occupy GPUs so the node looks like it already runs engines."""
        pg = ray.util.placement_group([{"GPU": 1.0}] * num_gpus)
        ray.get(pg.ready(), timeout=60)
        created.append(pg)

    yield SimpleNamespace(master_ip=master_ip, hold_gpus=hold_gpus, created=created)

    for pg in created:
        ray.util.remove_placement_group(pg)
    ray.shutdown()


def _elastic_ep_config(dp_size: int, world_size: int, master_ip: str):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_master_ip=master_ip,
            world_size=world_size,
        )
    )


@pytest.mark.timeout(120)
def test_add_dp_placement_groups_does_not_require_ray_default(ray_2gpu_node):
    """DP 1 -> 2 on an idle node: one schedulable group pinned to the master,
    found without the dashboard (``list_nodes`` would fail here)."""
    ip = ray_2gpu_node.master_ip

    pgs, local_ranks = CoreEngineActorManager.add_dp_placement_groups(
        _elastic_ep_config(dp_size=1, world_size=1, master_ip=ip), 2
    )
    ray_2gpu_node.created.extend(pgs)

    assert [pg.bundle_specs for pg in pgs] == [
        [{"GPU": 1.0, f"node:{ip}": 0.001}, {"CPU": 1.0}]
    ]
    assert local_ranks == [0]
    ray.get(pgs[0].ready(), timeout=60)


@pytest.mark.timeout(120)
def test_add_dp_placement_groups_counts_engines_already_on_node(ray_2gpu_node):
    """One GPU already held -> the new engine gets local rank 1."""
    ray_2gpu_node.hold_gpus(1)

    pgs, local_ranks = CoreEngineActorManager.add_dp_placement_groups(
        _elastic_ep_config(dp_size=1, world_size=1, master_ip=ray_2gpu_node.master_ip),
        2,
    )
    ray_2gpu_node.created.extend(pgs)

    assert len(pgs) == 1
    assert local_ranks == [1]
    ray.get(pgs[0].ready(), timeout=60)


@pytest.mark.timeout(120)
def test_add_dp_placement_groups_respects_world_size(ray_2gpu_node):
    """With TP=2 a node with one idle GPU cannot host a new engine."""
    ray_2gpu_node.hold_gpus(1)

    assert CoreEngineActorManager.add_dp_placement_groups(
        _elastic_ep_config(dp_size=1, world_size=2, master_ip=ray_2gpu_node.master_ip),
        2,
    ) == ([], [])


def test_add_dp_placement_groups_noop_without_growth():
    assert CoreEngineActorManager.add_dp_placement_groups(
        _elastic_ep_config(dp_size=2, world_size=1, master_ip="10.0.0.1"), 2
    ) == ([], [])


def test_shutdown_skips_ray_cleanup_after_driver_disconnect(monkeypatch):
    """Cleanup must not reconnect a driver whose Ray job already ended."""
    import threading

    manager = CoreEngineActorManager.__new__(CoreEngineActorManager)
    manager.manager_stopped = threading.Event()
    manager.local_engine_actors = [object()]
    manager.remote_engine_actors = [object()]
    manager.created_placement_groups = [object()]

    kills: list[Any] = []
    removed: list[Any] = []
    monkeypatch.setattr(ray, "kill", kills.append)
    monkeypatch.setattr(ray.util, "remove_placement_group", removed.append)

    # Driver already disconnected: no Ray calls may happen.
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    manager.shutdown()
    assert manager.manager_stopped.is_set()
    assert kills == [] and removed == []

    # Driver still connected: normal cleanup proceeds.
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    manager.shutdown()
    assert len(kills) == 2 and len(removed) == 1
