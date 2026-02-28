# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import uuid

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.collection_utils import ThreadSafeDict
from vllm.utils.network_utils import close_sockets, get_open_port, make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
    FaultToleranceZmqAddresses,
)
from vllm.v1.utils import get_engine_client_zmq_addr


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_index
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_local_size = vllm_config.parallel_config.data_parallel_size_local
        # Client manages local+remote EngineCores in pure internal LB case.
        # Client manages local EngineCores in hybrid and external LB case.
        num_dp_managed = (
            dp_local_size if vllm_config.parallel_config.local_engines_only else dp_size
        )
        host = vllm_config.parallel_config.data_parallel_master_ip

        super().__init__(
            upstream_cmd_addr=None,
            downstream_cmd_addr=fault_tolerance_addresses.engine_core_sentinel_cmd_addr,
            sentinel_identity=None,
            sentinel_tag=None,
            vllm_config=vllm_config,
        )

        self.fault_tolerance_req_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.client_sentinel_request_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

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
        # Queue to serialize fault tolerance instruction execution:
        # (client_identity, FaultToleranceRequest)
        self.ft_request_queue: queue.Queue[
            tuple[bytes | None, FaultToleranceRequest]
        ] = queue.Queue(
            maxsize=2
        )  # allow one pause command and another fault tolerance command
        self.ft_result_queue: queue.Queue[tuple[bytes | None, FaultToleranceResult]] = (
            queue.Queue(maxsize=2)
        )
        inproc_comm_addr = get_engine_client_zmq_addr(local_only=True, host=host)
        self.inproc_res_recv_socket = make_zmq_socket(
            ctx=self.ctx, path=inproc_comm_addr, socket_type=zmq.PAIR, bind=True
        )
        self.inproc_res_send_socket = make_zmq_socket(
            ctx=self.ctx, path=inproc_comm_addr, socket_type=zmq.PAIR, bind=False
        )

        self.is_faulted = threading.Event()
        self.engine_status_dict: ThreadSafeDict[int, dict[str, EngineStatusType]] = (
            ThreadSafeDict()
        )
        self.engine_status_dict.update(
            {
                engine_index: {"status": EngineStatusType.HEALTHY}
                for engine_index in range(dp_rank, dp_rank + num_dp_managed)
            }.items()
        )
        # todo: use identities as the key, indexes as the value as the index may
        # change in dp scale down and up
        self.engine_core_sentinel_identities = (
            fault_tolerance_addresses.engine_core_sentinel_identities
        )

        threading.Thread(
            target=self.run, daemon=True, name="ClientSentinelCmdAndFaultReceiverThread"
        ).start()

        threading.Thread(
            target=self._process_ft_requests_loop,
            daemon=True,
            name="ClientSentinelFtRequestsLoopThread",
        ).start()

    def _process_ft_requests_loop(self) -> None:
        """
        Worker loop to process Fault Tolerance (FT) requests
        """
        try:
            while not self.sentinel_dead:
                try:
                    identity, ft_request = self.ft_request_queue.get(timeout=1)
                    ft_result = self._execute_cmd(ft_request)
                    self.ft_result_queue.put((identity, ft_result))
                    self.inproc_res_send_socket.send(msgspec.msgpack.encode(ft_result))
                except queue.Empty:
                    pass
        except zmq.ZMQError:
            # Socket is closed.
            pass

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        for engine_status in self.engine_status_dict.values():
            if engine_status["status"] == EngineStatusType.DEAD:
                self.logger(
                    "Engine core is dead; retry won't work.",
                    level="warning",
                )
                return False

        target_engines = set(self.engine_core_sentinel_identities.values())
        new_stateless_dp_group_port = get_open_port()
        success, _ = self._execute_command_on_downstreams(
            "retry",
            target_engines,
            new_stateless_dp_group_port=new_stateless_dp_group_port,
            timeout=timeout,
        )

        for engine_index, _ in self.engine_status_dict.items():
            self.engine_status_dict[engine_index] = {"status": EngineStatusType.HEALTHY}

        if success:
            self.is_faulted.clear()
            self._pub_engine_status()
        return success

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """Pause engine cores, return True if successful. Best-effort operation."""
        self.logger(
            "Pause operation is best-effort only. Due to the complexity of "
            "collective communications (e.g., timing dependencies and "
            "synchronization barriers), pausing may not always succeed. If "
            "the process remains unresponsive or collective operations "
            "cannot be interrupted, consider shutting down and restarting "
            "the instance.",
            level="warning",
        )
        exclude_engine_index = kwargs.get("exclude_engine_index")
        soft_pause = kwargs.get("soft_pause", False)
        alive_engines = {
            identity
            for index, identity in self.engine_core_sentinel_identities.items()
            if self.engine_status_dict[index]["status"] != EngineStatusType.DEAD
            and (exclude_engine_index is None or index not in exclude_engine_index)
        }
        success, responses = self._execute_command_on_downstreams(
            "pause",
            alive_engines,
            timeout=timeout,
            soft_pause=soft_pause,
        )
        identity_to_index = {
            identity: index
            for index, identity in self.engine_core_sentinel_identities.items()
        }
        for engine_identity, ft_result in responses.items():
            if ft_result.success:
                i = identity_to_index[engine_identity]
                engine_status = self.engine_status_dict[i]["status"]
                if engine_status == EngineStatusType.HEALTHY:
                    self.engine_status_dict[i] = {"status": EngineStatusType.PAUSED}
        return success

    def _pub_engine_status(self):
        engine_status = self.engine_status_dict.to_dict()
        topic = self.ft_config.fault_state_pub_topic.encode()
        self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    def _alert_and_pause(self):
        """Receive fault info from engine and pause engines if first fault."""
        try:
            identity, _, message = self.fault_receiver_socket.recv_multipart()
            fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
            engine_status = (
                EngineStatusType.DEAD
                if "dead" in fault_info.type
                else EngineStatusType.UNHEALTHY
            )
            self.engine_status_dict[int(fault_info.engine_id)] = {
                "status": engine_status
            }
            self._pub_engine_status()
            if not self.is_faulted.is_set():
                self.is_faulted.set()
                pause_request = FaultToleranceRequest(
                    request_id=str(uuid.uuid4()),
                    instruction="pause",
                    params={
                        "timeout": self.ft_config.gloo_comm_timeout + 5,
                        "soft_pause": False,
                    },
                )
                while not self._dispatch_fault_tolerance_request(pause_request, None):
                    # If the queue is full, it means another fault tolerance
                    # command is being executed.
                    # Wait and retry until we can add the pause command to
                    # the queue.
                    time.sleep(0.1)

        except zmq.ZMQError:
            # Socket was closed during polling, exit loop.
            self.logger("Fault receiver socket closed, stopping thread.", level="info")
            raise

    def _dispatch_fault_tolerance_request(
        self, ft_request: FaultToleranceRequest, identity: bytes | None
    ) -> bool:
        """Add fault tolerance request to queue, return False if busy."""
        try:
            self.ft_request_queue.put_nowait((identity, ft_request))
        except queue.Full:
            return False
        return True

    def run(self):
        """Poll for fault messages and commands, dispatch to handlers."""
        poller = zmq.Poller()
        poller.register(self.fault_receiver_socket, zmq.POLLIN)
        poller.register(self.fault_tolerance_req_socket, zmq.POLLIN)
        poller.register(self.inproc_res_recv_socket, zmq.POLLIN)
        try:
            while not self.sentinel_dead:
                events = poller.poll(timeout=5000)
                if not events:
                    continue
                events = dict(events)
                if self.fault_receiver_socket in events:
                    # If a fault message is received, alert and attempt to pause.
                    self._alert_and_pause()
                if self.fault_tolerance_req_socket in events:
                    # Received fault tolerance command from client.
                    # Add corresponding command to the queue.
                    parts = self.fault_tolerance_req_socket.recv_multipart()
                    identity = parts[0]
                    msg_bytes = parts[-1]
                    ft_request = msgspec.msgpack.decode(
                        msg_bytes, type=FaultToleranceRequest
                    )
                    success = self._dispatch_fault_tolerance_request(
                        ft_request, identity
                    )
                    if not success:
                        # If we're busy, reply with a busy message.
                        msg = (
                            "System busy, vLLM is executing another fault "
                            "tolerance instruction."
                        )
                        res = FaultToleranceResult(ft_request.request_id, False, msg)
                        resp = msgspec.msgpack.encode(res)
                        self.fault_tolerance_req_socket.send_multipart(
                            [identity, b"", resp]
                        )
                if self.inproc_res_recv_socket in events:
                    # Send FT execution result back to client.
                    msg = self.inproc_res_recv_socket.recv()
                    recv_res = msgspec.msgpack.decode(msg, type=FaultToleranceResult)
                    identity, ft_result = self.ft_result_queue.get_nowait()
                    assert recv_res.request_id == ft_result.request_id
                    if identity is not None:
                        self.fault_tolerance_req_socket.send_multipart(
                            [identity, b"", msg]
                        )
                    self._pub_engine_status()
        except zmq.ZMQError:
            # Context terminated, exit thread cleanly.
            pass

    def shutdown(self):
        close_sockets(
            [
                self.fault_receiver_socket,
                self.fault_state_pub_socket,
                self.inproc_res_send_socket,
                self.inproc_res_recv_socket,
                self.fault_tolerance_req_socket,
            ]
        )
        super().shutdown()
