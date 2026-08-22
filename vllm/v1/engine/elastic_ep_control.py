# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Supervisor-side control channel for Elastic EP scaling.

The Rust frontend cannot create or remove engine cores or drive Ray's
``ReconfigureDistributedRequest`` flow, and there is no message path between it
and the Python supervisor today. This module implements the narrow ZMQ
request/reply channel over which the frontend asks the supervisor (which owns the
engine lifecycle via ``CoreEngineActorManager``) to scale.

Message contract (msgspec msgpack tagged union, see ``ControlMessageType``):
- ``ScaleUp { new_data_parallel_size }``: the supervisor creates the reconfigure
  rendezvous, replies ``Ack`` immediately (before spawning), runs
  ``scale_up_elastic_ep`` on a background thread, and replies ``ScaleUpResult``
  when it returns.
- ``ScaleDown { new_data_parallel_size }``: the supervisor creates the
  rendezvous, drops the removed ranks' run refs, and replies ``Ack``.
- ``ScaleDownComplete { new_data_parallel_size }``: sent once every removed rank
  has reported ``SHUTDOWN_COMPLETE``; the supervisor frees the removed
  placement groups and replies ``Ok``.
- Any request may be answered with ``Err``, which the frontend surfaces as 500.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.utils import CoreEngineActorManager

logger = init_logger(__name__)


class ControlMessage(msgspec.Struct, tag=True):
    """Base class for control channel messages."""


class ScaleUp(ControlMessage):
    new_data_parallel_size: int


class ScaleDown(ControlMessage):
    new_data_parallel_size: int


class ScaleDownComplete(ControlMessage):
    new_data_parallel_size: int


class Ack(ControlMessage):
    """Rendezvous ports the frontend relays into reinitialize_distributed."""

    new_data_parallel_master_port: int
    new_data_parallel_master_port_list: list[int]
    coord_store_port: int


class ScaleUpResult(ControlMessage):
    ok: bool


class Ok(ControlMessage):
    """Generic success reply (e.g. for ScaleDownComplete)."""


class Err(ControlMessage):
    message: str


ControlMessageType = (
    ScaleUp | ScaleDown | ScaleDownComplete | Ack | ScaleUpResult | Ok | Err
)


class ControlChannelServer:
    """ZMQ ROUTER end of the Elastic EP control channel.

    The supervisor binds this before spawning the Rust frontend and passes the
    address via ``--control-channel-address``. ``start`` launches a daemon
    thread that services requests until ``close``; the thread holds the engine
    manager and shared ``VllmConfig`` so scale operations can spawn/remove Ray
    actors and update the config used for the next spawn.
    """

    def __init__(
        self,
        engine_manager: CoreEngineActorManager,
        vllm_config: VllmConfig,
    ) -> None:
        self._engine_manager = engine_manager
        self._vllm_config = vllm_config
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="control-scale-up"
        )
        # Replies produced on the scale-up worker thread are drained by the
        # serve loop so the socket is only ever touched from one thread.
        self._pending_replies: queue.Queue[tuple[bytes, ControlMessage]] = queue.Queue()
        self._scale_pending = False
        self._scale_down_pending = False
        self._coord_store: Any | None = None

    def bind(self, host: str) -> str:
        """Bind the ROUTER socket and return its bound address."""
        from vllm.utils.network_utils import get_open_port, get_tcp_uri

        self._socket.bind(get_tcp_uri(host, get_open_port()))
        return self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._serve, name="control-channel", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False)
        with contextlib.suppress(zmq.error.ZMQError):
            self._socket.close()
        with contextlib.suppress(zmq.error.ZMQError):
            self._ctx.term()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                # Flush replies produced by background scale-up workers first
                # so a long quiet period doesn't delay their delivery.
                while not self._pending_replies.empty():
                    identity, message = self._pending_replies.get_nowait()
                    self._socket.send_multipart(
                        [identity, msgspec.msgpack.encode(message)]
                    )

                events = dict(poller.poll(1000))
            except zmq.error.ZMQError:
                # Socket closed by close() from another thread.
                break
            if not events:
                continue

            try:
                identity, payload = self._socket.recv_multipart()
            except zmq.error.ZMQError:
                break
            try:
                message = msgspec.msgpack.decode(payload, type=ControlMessageType)
            except msgspec.DecodeError as e:
                logger.warning("Dropping malformed control message: %s", e)
                self._pending_replies.put(
                    (identity, Err(message=f"Malformed control message: {e}"))
                )
                continue

            self._dispatch(identity, message)

    def _dispatch(self, identity: bytes, message: ControlMessageType) -> None:
        logger.debug("Control channel received %s", type(message).__name__)
        try:
            if isinstance(message, ScaleUp):
                self._handle_scale_up(identity, message)
            elif isinstance(message, ScaleDown):
                self._handle_scale_down(identity, message)
            elif isinstance(message, ScaleDownComplete):
                self._handle_scale_down_complete(identity, message)
            else:
                self._pending_replies.put(
                    (
                        identity,
                        Err(
                            message=f"Unexpected control message: "
                            f"{type(message).__name__}"
                        ),
                    )
                )
        except Exception as e:
            # Never let a bad message kill the control loop; surface the error
            # to the frontend so it can respond 500.
            logger.exception("Control channel handler failed: %s", e)
            self._pending_replies.put((identity, Err(message=str(e))))
            self._scale_pending = False
            self._scale_down_pending = False

    def _create_rendezvous(self) -> None:
        """Create the reconfigure TCP store + master ports on the shared config.

        The rendezvous must be shared between the workers the supervisor spawns
        and the ``reinitialize_distributed`` message the frontend sends to
        existing engines, so it is created here (where workers are launched) and
        its ports relayed to the frontend in ``Ack``.
        """
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self._vllm_config.parallel_config
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
        self._coord_store = store

    def _make_ack(self) -> Ack:
        parallel_config = self._vllm_config.parallel_config
        return Ack(
            new_data_parallel_master_port=parallel_config.data_parallel_master_port,
            new_data_parallel_master_port_list=(
                parallel_config._data_parallel_master_port_list
            ),
            coord_store_port=parallel_config._coord_store_port,
        )

    def _cur_data_parallel_size(self) -> int:
        manager = self._engine_manager
        return len(manager.local_engine_actors) + len(manager.remote_engine_actors)

    def _num_redundant_experts(self, new_size: int, cur_size: int) -> int:
        """Mirror DPLBAsyncMPClient.prepare_elastic_ep."""
        parallel_config = self._vllm_config.parallel_config
        num_experts = self._vllm_config.model_config.get_num_experts()
        return (
            num_experts + parallel_config.eplb_config.num_redundant_experts
        ) * new_size // cur_size - num_experts

    def _handle_scale_up(self, identity: bytes, message: ScaleUp) -> None:
        cur_size = self._cur_data_parallel_size()
        new_size = message.new_data_parallel_size
        if new_size <= cur_size:
            self._pending_replies.put(
                (
                    identity,
                    Err(
                        message=f"Invalid scale-up size {new_size}, "
                        f"current data parallel size is {cur_size}"
                    ),
                )
            )
            return
        if self._scale_pending:
            self._pending_replies.put(
                (identity, Err(message="Another scale operation is in progress"))
            )
            return

        self._scale_pending = True
        try:
            self._create_rendezvous()
            num_redundant_experts = self._num_redundant_experts(new_size, cur_size)
        except Exception as e:
            self._scale_pending = False
            logger.exception("Failed to prepare scale-up: %s", e)
            self._pending_replies.put((identity, Err(message=str(e))))
            return
        # Ack immediately, before any engine is spawned, so the frontend can
        # relay the rendezvous ports to the existing engines.
        self._pending_replies.put((identity, self._make_ack()))
        self._pool.submit(self._run_scale_up, identity, new_size, num_redundant_experts)

    def _run_scale_up(
        self, identity: bytes, new_size: int, num_redundant_experts: int
    ) -> None:
        try:
            self._engine_manager.scale_up_elastic_ep(
                self._vllm_config, new_size, num_redundant_experts
            )
            self._vllm_config.parallel_config.data_parallel_size_local = len(
                self._engine_manager.local_engine_actors
            )
            self._pending_replies.put((identity, ScaleUpResult(ok=True)))
        except Exception as e:
            logger.exception("scale_up_elastic_ep failed: %s", e)
            self._pending_replies.put((identity, Err(message=str(e))))
        finally:
            self._scale_pending = False

    def _handle_scale_down(self, identity: bytes, message: ScaleDown) -> None:
        cur_size = self._cur_data_parallel_size()
        new_size = message.new_data_parallel_size
        if new_size >= cur_size:
            self._pending_replies.put(
                (
                    identity,
                    Err(
                        message=f"Invalid scale-down size {new_size}, "
                        f"current data parallel size is {cur_size}"
                    ),
                )
            )
            return
        if self._scale_pending:
            self._pending_replies.put(
                (identity, Err(message="Another scale operation is in progress"))
            )
            return

        self._scale_pending = True
        self._scale_down_pending = True
        self._create_rendezvous()
        removed_dp_size = cur_size - new_size
        self._engine_manager.remove_run_refs_for_scale_down(removed_dp_size)
        self._pending_replies.put((identity, self._make_ack()))

    def _handle_scale_down_complete(
        self, identity: bytes, message: ScaleDownComplete
    ) -> None:
        if not (self._scale_pending and self._scale_down_pending):
            self._pending_replies.put(
                (identity, Err(message="No scale-down is in progress"))
            )
            return
        old_size = self._cur_data_parallel_size()
        new_size = message.new_data_parallel_size
        try:
            self._engine_manager.scale_down_elastic_ep(old_size, new_size)
            self._vllm_config.parallel_config.data_parallel_size_local = len(
                self._engine_manager.local_engine_actors
            )
            self._pending_replies.put((identity, Ok()))
        except Exception as e:
            logger.exception("scale_down_elastic_ep failed: %s", e)
            self._pending_replies.put((identity, Err(message=str(e))))
        finally:
            self._scale_pending = False
            self._scale_down_pending = False
