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
        address_broker: Any | None = None,
    ):
        # Exercise the production Ray actor mixin without loading a model.
        self._vllm_config = vllm_config
        EngineCoreActorMixin.__init__(
            self,
            vllm_config,
            addresses,
            dp_rank,
            local_dp_rank,
            address_broker,
        )

    def _set_visible_devices(self, vllm_config: Any, local_dp_rank: int) -> None:
        pass

    def wait_for_init(self) -> None:
        with self._perform_handshakes("", b"", False, self._vllm_config, None):
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
def test_ray_dp_addresses_published_after_frontend_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray actors receive endpoints only after the front-end binds them."""
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

    addresses = get_engine_zmq_addresses(vllm_config, num_api_servers=2)
    assert all(
        split_zmq_path(url)[2] == "0" for url in addresses.inputs + addresses.outputs
    )

    sock = socket.socket()
    engine_manager: CoreEngineActorManager | None = None
    actor_snapshots: list[tuple[list[str], list[str]]] = []
    api_server_manager: APIServerProcessManager | None = None
    try:
        # Ray actors start with placeholders and wait on the address broker.
        with launch_core_engines(
            vllm_config,
            executor_class=_DummyExecutor,
            log_stats=False,
            addresses=addresses,
            defer_engine_addresses=True,
        ) as engine_launch:
            engine_manager = engine_launch.engine_manager
            assert isinstance(engine_manager, CoreEngineActorManager)

            # API-server children bind first and report kernel-assigned ports.
            api_server_manager = APIServerProcessManager(
                listen_address="tcp://127.0.0.1:0",
                sock=sock,
                args="test_args",
                num_servers=2,
                input_addresses=addresses.inputs,
                output_addresses=addresses.outputs,
                target_server_fn=_bind_and_report_worker,
            )

            addresses.inputs, addresses.outputs = (
                api_server_manager.gather_actual_addresses(timeout=15.0)
            )
            engine_manager.publish_addresses(addresses)

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
                f"Ray actor retained placeholder address {url!r}; "
                "the address broker must publish the front-end's bound "
                "endpoint before the actor finishes initialization."
            )
