# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
from collections.abc import Sequence
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

import msgspec
import msgspec.msgpack
import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import (
    EEPNotificationType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.utils import EngineHandshakeMetadata, EngineZmqAddresses
from vllm.v1.utils import get_engine_client_zmq_addr

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import DPAsyncMPClient

logger = init_logger(__name__)


class ExternalElasticEPScaleUpHandshakeServer:
    """Temporary rank-0 handshake server for external EEP scale-up.

    During normal external startup the global handshake listener only exists
    while the rank starts. Scale-up needs the same handshake contract again for
    newly launched ranks, so rank 0 re-opens a temporary listener for the
    duration of the current scale operation.

    The handshake gives new ranks the information they cannot infer locally:
    front-end and DP coordinator ZMQ addresses, the new DP master address/ports,
    target DP size, and the coordination store port used for EEP
    reconfiguration.
    """

    def __init__(
        self,
        *,
        handshake_address: str,
        expected_new_ranks: list[int],
        addresses: EngineZmqAddresses,
        bootstrap: ReconfigureDistributedRequest,
    ) -> None:
        self.handshake_address = handshake_address
        self.expected_new_ranks = set(expected_new_ranks)
        self.addresses = addresses
        self.bootstrap = bootstrap
        self.started_event = Event()
        self._stop_event = Event()
        self._thread = Thread(
            target=self._run,
            name="ExternalElasticEPHandshakeServer",
            daemon=True,
        )
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread.start()
        started = self.started_event.wait(timeout=5)
        if self._error is not None:
            raise self._error
        if not started:
            raise TimeoutError(
                "Timed out waiting for external EEP handshake server to "
                f"start listening at {self.handshake_address}"
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        pending_ready_ranks = self.expected_new_ranks.copy()
        try:
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx,
                    self.handshake_address,
                    zmq.ROUTER,
                    bind=True,
                    linger=0,
                    router_handover=True,
                ) as handshake_socket,
            ):
                self.started_event.set()
                poller = zmq.Poller()
                poller.register(handshake_socket, zmq.POLLIN)

                while not self._stop_event.is_set():
                    if not pending_ready_ranks:
                        return

                    events = dict(poller.poll(timeout=1000))
                    if handshake_socket not in events:
                        continue

                    engine_identity, payload = handshake_socket.recv_multipart()
                    engine_rank = int.from_bytes(engine_identity, "little")
                    if engine_rank not in self.expected_new_ranks:
                        raise RuntimeError(
                            "Received scale-up handshake from unexpected "
                            f"dp rank {engine_rank}"
                        )

                    message = msgspec.msgpack.decode(payload)
                    status = message["status"]
                    if status == "HELLO":
                        b = self.bootstrap
                        parallel_config: dict[str, int | str | list[int]] = {
                            "data_parallel_master_ip": b.new_data_parallel_master_ip,
                            "data_parallel_master_port": (
                                b.new_data_parallel_master_port
                            ),
                            "_data_parallel_master_port_list": (
                                b.new_data_parallel_master_port_list
                            ),
                            "data_parallel_size": b.new_data_parallel_size,
                            "_coord_store_port": b.coord_store_port,
                        }
                        init_message = msgspec.msgpack.encode(
                            EngineHandshakeMetadata(
                                addresses=self.addresses,
                                parallel_config=parallel_config,
                            )
                        )
                        handshake_socket.send_multipart(
                            (engine_identity, init_message), copy=False
                        )
                    elif status == "READY":
                        pending_ready_ranks.discard(engine_rank)
                    else:
                        raise RuntimeError(
                            f"Unexpected handshake status {status} from dp rank "
                            f"{engine_rank}"
                        )
        except Exception as e:
            self._error = e
            self.started_event.set()


class ExternalElasticEPScaleCoordinator:
    def __init__(self, client: "DPAsyncMPClient") -> None:
        self.client = client
        self.active_reconfig_store: tuple[str, int] | None = None
        # Keep the previous rank-0 TCPStore server alive while the next scale
        # operation publishes bootstrap metadata through it. Overwriting
        # client._coord_store with the new store can otherwise close the old
        # server while other ranks are still polling it.
        self.control_store_ref: Any | None = None
        self.reconfig_store_ref: Any | None = None

    @staticmethod
    def key(*parts: str | int) -> str:
        return "/".join(["elastic_ep/external", *[str(part) for part in parts]])

    def _update_parallel_config(self, bootstrap: ReconfigureDistributedRequest) -> None:
        parallel_config = self.client.vllm_config.parallel_config
        parallel_config.data_parallel_size = bootstrap.new_data_parallel_size
        parallel_config.data_parallel_master_ip = bootstrap.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = (
            bootstrap.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            bootstrap.new_data_parallel_master_port_list.copy()
        )
        parallel_config._coord_store_port = bootstrap.coord_store_port

    def _get_reconfig_store(self):
        from vllm.distributed.utils import get_cached_tcp_store_client

        if self.active_reconfig_store is not None:
            store_addr = self.active_reconfig_store
        else:
            parallel_config = self.client.vllm_config.parallel_config
            if not parallel_config._coord_store_port:
                raise RuntimeError(
                    "External Elastic EP requires an active reconfiguration "
                    "coordination store."
                )
            store_addr = (
                parallel_config.data_parallel_master_ip,
                parallel_config._coord_store_port,
            )

        return get_cached_tcp_store_client(*store_addr)

    def _get_existing_engine_zmq_address(self) -> EngineZmqAddresses:
        coordinator = self.client.resources.coordinator
        if coordinator is None:
            raise RuntimeError(
                "External Elastic EP scale-up requires rank 0 to own a DP coordinator."
            )
        coordinator_input, coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        stats_publish_address = coordinator.get_stats_publish_address()

        input_endpoint = self.client.input_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        output_socket = self.client.resources.output_socket
        output_endpoint = (
            output_socket.getsockopt_string(zmq.LAST_ENDPOINT)
            if output_socket is not None
            else ""
        )

        return EngineZmqAddresses(
            inputs=[input_endpoint],
            outputs=[output_endpoint],
            coordinator_input=coordinator_input,
            coordinator_output=coordinator_output,
            frontend_stats_publish_address=stats_publish_address,
        )

    def _setup_reconfig_bootstrap(self) -> tuple[str, int]:
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self.client.vllm_config.parallel_config
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )

        ip = parallel_config.data_parallel_master_ip
        store = create_tcp_store(
            ip,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = store.port
        self.reconfig_store_ref = store
        self.control_store_ref = getattr(self.client, "_coord_store", None)
        self.client._coord_store = store
        return ip, store.port

    def _get_error(self, store: Any) -> str | None:
        error_key = self.key("error")
        if not store.check([error_key]):
            return None
        return store.get(error_key).decode()

    def _cleanup_reconfig_keys(
        self,
        store: Any,
        max_data_parallel_size: int,
    ) -> None:
        keys = [
            self.key("bootstrap"),
            self.key("prepared"),
            self.key("completed"),
            self.key("error"),
        ]
        for rank in range(max_data_parallel_size):
            keys.append(self.key("old_rank_finished", rank))
            keys.append(self.key("shutdown_complete", rank))
            for notification_type in (
                EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
            ):
                keys.append(self.key("notifications", notification_type.value, rank))

        for key in keys:
            with contextlib.suppress(Exception):
                store.delete_key(key)

    async def _wait_for_bootstrap(
        self,
        store: Any,
        requested_new_dp_size: int,
        timeout_s: float = 300,
    ) -> ReconfigureDistributedRequest:
        """Wait for rank 0 to publish matching scale bootstrap metadata.
        Non-zero ranks poll the old control store until prepared or errored."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        bootstrap_key = self.key("bootstrap")
        prepared_key = self.key("prepared")
        completed_key = self.key("completed")
        while True:
            error = self._get_error(store)
            if error is not None:
                raise RuntimeError(error)

            if store.check([bootstrap_key]):
                bootstrap = msgspec.msgpack.decode(
                    store.get(bootstrap_key),
                    type=ReconfigureDistributedRequest,
                )
                completed = store.check([completed_key])
                if bootstrap.new_data_parallel_size != requested_new_dp_size:
                    if store.check([prepared_key]) and not completed:
                        raise RuntimeError(
                            "A different external Elastic EP scaling operation is "
                            f"already in progress for target dp size "
                            f"{bootstrap.new_data_parallel_size}."
                        )
                elif store.check([prepared_key]):
                    return bootstrap

            now = loop.time()
            if now - start > timeout_s:
                raise TimeoutError(
                    "Timed out waiting for rank 0 to publish external Elastic EP "
                    "bootstrap metadata."
                )
            await asyncio.sleep(0.1)

    def _prepare_reconfig_bootstrap(
        self,
        store: Any,
        cur_data_parallel_size: int,
        new_data_parallel_size: int,
    ) -> ReconfigureDistributedRequest:
        current_error = self._get_error(store)
        if (
            current_error is None
            and store.check([self.key("prepared")])
            and not (store.check([self.key("completed")]))
        ):
            raise RuntimeError(
                "Another external Elastic EP scaling operation is already active."
            )

        self._cleanup_reconfig_keys(
            store, max(cur_data_parallel_size, new_data_parallel_size)
        )
        ip, coord_store_port = self._setup_reconfig_bootstrap()
        parallel_config = self.client.vllm_config.parallel_config
        bootstrap = ReconfigureDistributedRequest(
            new_data_parallel_size=new_data_parallel_size,
            new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip=ip,
            new_data_parallel_master_port=parallel_config.data_parallel_master_port,
            new_data_parallel_master_port_list=(
                parallel_config._data_parallel_master_port_list.copy()
            ),
            coord_store_port=coord_store_port,
        )

        bootstrap_key = self.key("bootstrap")
        store.set(bootstrap_key, msgspec.msgpack.encode(bootstrap))
        return bootstrap

    def _start_scale_up_handshake_server(
        self,
        bootstrap: ReconfigureDistributedRequest,
        cur_data_parallel_size: int,
    ) -> ExternalElasticEPScaleUpHandshakeServer:
        handshake_server = ExternalElasticEPScaleUpHandshakeServer(
            handshake_address=get_engine_client_zmq_addr(
                False,
                bootstrap.new_data_parallel_master_ip,
                self.client.vllm_config.parallel_config.data_parallel_rpc_port,
            ),
            expected_new_ranks=list(
                range(
                    cur_data_parallel_size,
                    bootstrap.new_data_parallel_size,
                )
            ),
            addresses=self._get_existing_engine_zmq_address(),
            bootstrap=bootstrap,
        )
        handshake_server.start()
        return handshake_server

    async def _wait_for_notification(
        self,
        control_store: Any,
        reconfig_store: Any,
        notification_type: EEPNotificationType,
        source_ranks: Sequence[int],
        timeout_s: float = 300,
    ) -> None:
        """Wait for source ranks to publish a specific scale notification.
        Once all are ready, forward the notification to existing engines."""
        loop = asyncio.get_running_loop()
        start = loop.time()

        def is_ready(rank: int) -> bool:
            key = self.key("notifications", notification_type.value, rank)
            return reconfig_store.check([key])

        while True:
            error = self._get_error(control_store)
            if error is not None:
                raise RuntimeError(
                    error
                    or "External Elastic EP scaling failed while waiting for "
                    f"{notification_type.value}."
                )

            ready_count = sum(1 for rank in source_ranks if is_ready(rank))
            if ready_count >= len(source_ranks):
                await self.client.call_utility_async(
                    "eep_handle_engine_core_notification",
                    notification_type.value,
                )
                return

            now = loop.time()
            if now - start > timeout_s:
                raise TimeoutError(
                    "Timed out waiting for external Elastic EP notification "
                    f"{notification_type.value}."
                )
            await asyncio.sleep(0.1)

    async def _wait_for_local_reconfig_finished(
        self,
        control_store: Any,
        reconfig_store: Any,
        bootstrap: ReconfigureDistributedRequest,
        dp_rank: int,
        scale_up: bool,
        timeout_s: float = 600,
    ) -> None:
        """Wait until this rank's EngineCore finishes the scale transition."""
        if not scale_up and dp_rank >= bootstrap.new_data_parallel_size:
            wait_key = self.key("shutdown_complete", dp_rank)
            timeout_msg = (
                "Timed out waiting for local external Elastic EP scale-down "
                "shutdown to finish."
            )
        else:
            wait_key = self.key("old_rank_finished", dp_rank)
            timeout_msg = (
                "Timed out waiting for local external Elastic EP "
                "reconfiguration to finish."
            )

        loop = asyncio.get_running_loop()
        start = loop.time()
        while True:
            error = self._get_error(control_store)
            if error is not None:
                raise RuntimeError(
                    error or "External Elastic EP scaling failed on another rank."
                )

            if reconfig_store.check([wait_key]):
                return

            now = loop.time()
            if now - start > timeout_s:
                raise TimeoutError(timeout_msg)
            await asyncio.sleep(0.1)

    async def scale(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        from vllm.distributed.utils import get_cached_tcp_store_client

        parallel_config = self.client.vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        scale_up = new_data_parallel_size > cur_data_parallel_size
        if not parallel_config._coord_store_port:
            raise RuntimeError(
                "External Elastic EP requires a runtime coordination store port."
            )
        control_store = get_cached_tcp_store_client(
            parallel_config.data_parallel_master_ip,
            parallel_config._coord_store_port,
        )
        handshake_server: ExternalElasticEPScaleUpHandshakeServer | None = None
        bootstrap: ReconfigureDistributedRequest | None = None

        try:
            if dp_rank == 0:
                bootstrap = self._prepare_reconfig_bootstrap(
                    control_store,
                    cur_data_parallel_size,
                    new_data_parallel_size,
                )
                if scale_up:
                    handshake_server = self._start_scale_up_handshake_server(
                        bootstrap, cur_data_parallel_size
                    )
                control_store.set(self.key("prepared"), b"1")
            else:
                bootstrap = await self._wait_for_bootstrap(
                    control_store, new_data_parallel_size
                )

            self.active_reconfig_store = (
                bootstrap.new_data_parallel_master_ip,
                bootstrap.coord_store_port,
            )
            reconfig_store = self._get_reconfig_store()

            reconfig_rank = (
                ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                if not scale_up and dp_rank >= bootstrap.new_data_parallel_size
                else ReconfigureRankType.KEEP_CURRENT_RANK
            )
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=bootstrap.new_data_parallel_size,
                new_data_parallel_rank=reconfig_rank,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=bootstrap.new_data_parallel_master_ip,
                new_data_parallel_master_port=bootstrap.new_data_parallel_master_port,
                new_data_parallel_master_port_list=(
                    bootstrap.new_data_parallel_master_port_list
                ),
                coord_store_port=bootstrap.coord_store_port,
            )
            await self.client.call_utility_async(
                "reinitialize_distributed", reconfig_request
            )

            if scale_up:
                new_ranks = list(
                    range(
                        cur_data_parallel_size,
                        bootstrap.new_data_parallel_size,
                    )
                )
                await self._wait_for_notification(
                    control_store,
                    reconfig_store,
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    new_ranks,
                )
                await self._wait_for_notification(
                    control_store,
                    reconfig_store,
                    EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
                    new_ranks,
                )

            await self._wait_for_local_reconfig_finished(
                control_store,
                reconfig_store,
                bootstrap,
                dp_rank,
                scale_up,
            )
            if dp_rank == 0:
                control_store.set(self.key("completed"), b"1")
            if scale_up or dp_rank < bootstrap.new_data_parallel_size:
                self._update_parallel_config(bootstrap)
        except Exception as e:
            if bootstrap is not None:
                control_store.set(
                    self.key("error"),
                    str(e).encode(),
                )
            raise
        finally:
            if handshake_server is not None:
                with contextlib.suppress(Exception):
                    handshake_server.stop()

    async def process_engine_core_notification(
        self, notification_data: tuple[str, int]
    ) -> None:
        """Record scale notifications emitted by EngineCore processes.
        The stored keys are later polled by the external scale coordinator."""
        parallel_config = self.client.vllm_config.parallel_config
        if not (
            parallel_config.enable_elastic_ep
            and parallel_config.data_parallel_external_lb
        ):
            return

        notification_type_str, dp_rank = notification_data
        notification_type = EEPNotificationType(notification_type_str)
        if not parallel_config._coord_store_port:
            logger.warning(
                "Ignoring external Elastic EP notification %s because coord "
                "store metadata is not available yet.",
                notification_type.value,
            )
            return

        reconfig_store = self._get_reconfig_store()

        if notification_type in (
            EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
            EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
        ):
            key = self.key("notifications", notification_type.value, dp_rank)
            reconfig_store.set(key, b"1")
        elif notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            key = self.key("old_rank_finished", dp_rank)
            reconfig_store.set(key, b"1")
        elif notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:
            key = self.key("shutdown_complete", dp_rank)
            reconfig_store.set(key, b"1")
