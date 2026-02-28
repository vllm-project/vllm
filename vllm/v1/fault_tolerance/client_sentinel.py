# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from collections.abc import Callable

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceZmqAddresses


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        shutdown_callback: Callable,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_index
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_local_size = vllm_config.parallel_config.data_parallel_size_local
        # Client manages local+remote EngineCores in pure internal LB case.
        # Client manages local EngineCores in hybrid and external LB case.
        num_dp_managed = (
            dp_local_size if vllm_config.parallel_config.local_engines_only else dp_size
        )
        super().__init__(
            vllm_config=vllm_config,
        )
        self.instance_shutdown_callback = shutdown_callback

        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )

        self.is_faulted = threading.Event()
        self.engine_status_dict: dict[int, dict[str, EngineStatusType]] = {
            engine_index: {"status": EngineStatusType.HEALTHY}
            for engine_index in range(dp_rank, dp_rank + num_dp_managed)
        }
        self.engine_status_lock = threading.Lock()
        # todo: use identities as the key, indexes as the value as the index may
        # change in dp scale down and up
        self.engine_core_sentinel_identities = (
            fault_tolerance_addresses.engine_core_sentinel_identities
        )

        threading.Thread(
            target=self.run, daemon=True, name="ClientSentinelFaultReceiverThread"
        ).start()

    def _pub_engine_status(self):
        with self.engine_status_lock:
            engine_status = self.engine_status_dict.copy()
        topic = self.ft_config.fault_state_pub_topic.encode()
        self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    def _alert_and_pause(self):
        """Receive fault info from engine."""
        try:
            _, _, message = self.fault_receiver_socket.recv_multipart()
            fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
            engine_status = (
                EngineStatusType.DEAD
                if "dead" in fault_info.type
                else EngineStatusType.UNHEALTHY
            )
            with self.engine_status_lock:
                self.engine_status_dict[int(fault_info.engine_id)] = {
                    "status": engine_status
                }
            self._pub_engine_status()
            if not self.is_faulted.is_set():
                self.is_faulted.set()

        except zmq.ZMQError:
            # Socket was closed during polling, exit loop.
            self.logger("Fault receiver socket closed, stopping thread.", level="info")
            raise

    def run(self):
        """Poll for fault messages and commands, dispatch to handlers."""
        poller = zmq.Poller()
        poller.register(self.fault_receiver_socket, zmq.POLLIN)
        try:
            while not self.sentinel_dead:
                events = poller.poll(timeout=5000)
                if not events:
                    continue
                events = dict(events)
                if self.fault_receiver_socket in events:
                    # If a fault message is received, alert and attempt to pause.
                    self._alert_and_pause()
                    # todo: implement error handling logic later.
                    time.sleep(self.ft_config.engine_recovery_timeout)
                    self.logger(
                        "Close the service after waiting for recovery timeout "
                        "since error handling is not implemented yet.",
                        level="warning",
                    )
                    self.instance_shutdown_callback()
                    break

        except zmq.ZMQError:
            # Context terminated, exit thread cleanly.
            pass

    def shutdown(self):
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        super().shutdown()
