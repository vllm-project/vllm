"""Core benchmark class for LMCache Controller ZMQ testing"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import random
import statistics
import time

# Third Party
import msgspec
import psutil
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    DeRegisterMsg,
    RegisterMsg,
    RegisterRetMsg,
)
from lmcache.v1.cache_controller.utils import KVChunkInfo
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_zmq_context,
    get_zmq_socket,
    get_zmq_socket_with_timeout,
)

# Local
from .config import ZMQBenchmarkConfig
from .constants import (
    DEFAULT_BATCH_SEND_SIZE,
    DEFAULT_OP_DISTRIBUTION_BASE,
    DEFAULT_RECV_TIMEOUT_MS,
    DEFAULT_SEND_HWM,
    DEFAULT_SEND_TIMEOUT_MS,
)
from .handlers import OPERATION_HANDLERS
from .handlers.base import SocketType

logger = init_logger(__name__)


@dataclass
class TestData:
    """Test data for benchmark operations"""

    instances: List[str]
    workers: List[int]
    locations: List[str]
    keys: List[int]


@dataclass
class OperationStats:
    """Statistics for a single operation type"""

    qps: float = 0.0  # messages per second
    rps: float = 0.0  # requests per second
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    p95_latency: float = 0.0
    errors: int = 0


@dataclass
class BenchmarkResults:
    """Overall benchmark results"""

    total_requests: int = 0
    total_messages: int = 0
    total_time: float = 0.0
    overall_rps: float = 0.0  # requests per second
    overall_qps: float = 0.0  # messages per second
    operations: Dict[str, OperationStats] = field(default_factory=dict)
    memory_usage: List[float] = field(default_factory=list)


class ZMQControllerBenchmark:
    """Benchmark class for LMCache Controller via ZMQ"""

    def __init__(self, config: ZMQBenchmarkConfig):
        self.config = config
        self.context: Optional[zmq.asyncio.Context] = None
        self.push_socket: Optional[Any] = None
        self.req_socket: Optional[Any] = None
        self.heartbeat_socket: Optional[Any] = None
        self.heartbeat_url: Optional[str] = None
        self.results = BenchmarkResults()
        self.running = False
        # Track sequence numbers per KVChunkInfo (instance_id, worker_id, location)
        self.sequence_numbers: Dict[KVChunkInfo, int] = {}

        # Track registered workers for cleanup
        self.registered_workers: List[Tuple[str, int, str, int]] = []

    async def setup(self):
        """Setup ZMQ sockets"""
        self.context = get_zmq_context(use_asyncio=True)
        self.push_socket = get_zmq_socket(
            self.context,
            self.config.controller_pull_url,
            protocol="tcp",
            role=zmq.PUSH,
            bind_or_connect="connect",
        )
        # Set send timeout to avoid blocking indefinitely when controller is down
        # SNDTIMEO: timeout in milliseconds, 0 means non-blocking
        self.push_socket.setsockopt(zmq.SNDTIMEO, DEFAULT_SEND_TIMEOUT_MS)
        # SNDHWM: high watermark for outbound messages
        self.push_socket.setsockopt(zmq.SNDHWM, DEFAULT_SEND_HWM)
        logger.info(
            "Connected to controller PULL socket at %s",
            self.config.controller_pull_url,
        )

        # Setup DEALER socket for request-reply operations (e.g., P2P lookup)
        if self.config.controller_reply_url:
            self.req_socket = get_zmq_socket_with_timeout(
                self.context,
                self.config.controller_reply_url,
                protocol="tcp",
                role=zmq.DEALER,
                bind_or_connect="connect",
                recv_timeout_ms=DEFAULT_RECV_TIMEOUT_MS,
                send_timeout_ms=DEFAULT_SEND_TIMEOUT_MS,
            )
            logger.info(
                "Connected to controller ROUTER socket at tcp://%s",
                self.config.controller_reply_url,
            )
            logger.info(
                "DEALER socket type: %d, last_endpoint: %s",
                self.req_socket.get(zmq.TYPE),
                self.req_socket.get_string(zmq.LAST_ENDPOINT, encoding="utf-8"),
            )
            # Give ZMQ time to establish the connection
            time.sleep(0.1)

        # Setup heartbeat DEALER socket if configured
        if self.config.controller_heartbeat_url:
            self.heartbeat_url = self.config.controller_heartbeat_url
            self._setup_heartbeat_socket()
            logger.info(
                "Connected to heartbeat ROUTER socket at tcp://%s",
                self.config.controller_heartbeat_url,
            )

    def cleanup(self):
        """Cleanup ZMQ sockets"""
        if self.push_socket:
            close_zmq_socket(self.push_socket)
        if self.req_socket:
            close_zmq_socket(self.req_socket)
        if self.heartbeat_socket:
            close_zmq_socket(self.heartbeat_socket)
        logger.info("ZMQ sockets closed")

    def generate_test_data(self) -> TestData:
        """Generate test data based on configuration

        Each process gets a unique range of instance IDs to avoid conflicts.
        Format: instance_p{process_id}_{instance_index}
        """
        process_id = self.config.process_id
        return TestData(
            instances=[
                "instance_p%d_%d" % (process_id, i)
                for i in range(self.config.num_instances)
            ],
            workers=list(range(self.config.num_workers)),
            locations=["location_%d" % i for i in range(self.config.num_locations)],
            keys=list(range(self.config.num_keys)),
        )

    def get_next_sequence_number(
        self, instance_id: str, worker_id: int, location: str
    ) -> int:
        """
        Get monotonically increasing sequence number for specific
        instance-worker-location
        """
        key = KVChunkInfo(instance_id, worker_id, location)
        if key not in self.sequence_numbers:
            self.sequence_numbers[key] = 0
        seq = self.sequence_numbers[key]
        self.sequence_numbers[key] += 1
        return seq

    async def send_messages(self, messages: List[Any]) -> float:
        """Send multiple messages via ZMQ PUSH socket

        Args:
            messages: List of messages to send

        Returns:
            Time taken to send all messages

        Raises:
            RuntimeError: If socket not initialized or send timeout
        """
        if self.push_socket is None:
            raise RuntimeError("Socket not initialized. Call setup() first.")
        start_time = time.time()
        encoded_msgs = [msgspec.msgpack.encode(msg) for msg in messages]
        try:
            await self.push_socket.send_multipart(encoded_msgs)
        except zmq.Again as e:
            raise RuntimeError(
                "Send timeout - Controller may not be running at %s"
                % self.config.controller_pull_url
            ) from e
        return time.time() - start_time

    async def send_request(self, message: Any) -> Tuple[float, Any]:
        """Send a message via ZMQ DEALER socket and wait for reply

        Args:
            message: Message to send

        Returns:
            Tuple of (time taken, response)

        Raises:
            RuntimeError: If socket not initialized or timeout
            zmq.ZMQError: If DEALER socket error occurs
        """
        if self.req_socket is None:
            raise RuntimeError(
                "DEALER socket not initialized. "
                "Ensure controller_reply_url is configured."
            )
        start_time = time.time()
        encoded_msg = msgspec.msgpack.encode(message)
        logger.debug(
            "Sending request to %s, message type: %s, size: %d bytes",
            self.config.controller_reply_url,
            type(message).__name__,
            len(encoded_msg),
        )

        try:
            # DEALER socket: send [empty_frame, payload]
            await self.req_socket.send_multipart([b"", encoded_msg])
            frames = await self.req_socket.recv_multipart()
            # DEALER receives: [empty_frame, payload]
            response = frames[-1]
            logger.debug("Response received, size: %d bytes", len(response))
            return time.time() - start_time, response
        except zmq.Again as e:
            logger.error("Request timeout after waiting for response")
            raise RuntimeError(
                "Request timeout - Controller may not be running at %s"
                % self.config.controller_reply_url
            ) from e
        except zmq.ZMQError as e:
            logger.error("ZMQ error: %s", e)
            # Re-raise other ZMQ errors
            raise

    async def register_workers(self, test_data: TestData):
        """Pre-register all workers before benchmark using DEALER-ROUTER mode"""
        if not self.config.register_first:
            return

        if self.req_socket is None:
            logger.warning(
                "DEALER socket not initialized, skipping worker registration"
            )
            return

        logger.info("Pre-registering workers via REQ-REP...")
        for instance in test_data.instances:
            for worker in test_data.workers:
                ip = "192.168.1.%d" % (worker + 1)
                port = 10000 + worker
                peer_port = 20000 + worker
                peer_init_url = "tcp://%s:%d" % (ip, peer_port)
                msg = RegisterMsg(
                    instance_id=instance,
                    worker_id=worker,
                    ip=ip,
                    port=port,
                    peer_init_url=peer_init_url,
                )
                try:
                    _, response = await self.send_request(msg)
                    self.registered_workers.append((instance, worker, ip, port))
                    # Extract heartbeat_url from first successful registration
                    # (only if not already configured)
                    if self.heartbeat_url is None and response:
                        self._process_register_response(response)
                except (RuntimeError, zmq.ZMQError) as e:
                    logger.error(
                        "Failed to register worker %s-%d: %s", instance, worker, e
                    )

        logger.info("Registered %d workers", len(self.registered_workers))
        # Setup heartbeat socket after getting heartbeat_url (if not already setup)
        if self.heartbeat_url and self.heartbeat_socket is None:
            self._setup_heartbeat_socket()
        await asyncio.sleep(0.5)

    def _process_register_response(self, response: bytes):
        """Process RegisterRetMsg to extract heartbeat_url"""
        try:
            ret_msg = msgspec.msgpack.decode(response, type=RegisterRetMsg)
            if ret_msg.extra_config and "heartbeat_url" in ret_msg.extra_config:
                raw_url = ret_msg.extra_config["heartbeat_url"]
                # Strip tcp:// prefix if present (get_zmq_socket adds it)
                if raw_url.startswith("tcp://"):
                    raw_url = raw_url[6:]
                # If benchmark connects to localhost but controller returns
                # a different IP, use localhost for heartbeat as well
                raw_url = self._normalize_heartbeat_url(raw_url)
                self.heartbeat_url = raw_url
                logger.info("Got heartbeat_url from register: %s", self.heartbeat_url)
        except msgspec.DecodeError as e:
            logger.warning("Failed to decode RegisterRetMsg: %s", e)

    def _normalize_heartbeat_url(self, heartbeat_url: str) -> str:
        """Normalize heartbeat URL based on controller connection.

        If benchmark connects to controller via localhost (127.0.0.1),
        but heartbeat_url contains a different IP (e.g., from get_ip()),
        replace it with 127.0.0.1 to ensure connectivity.

        Args:
            heartbeat_url: The heartbeat URL from controller (e.g., "10.0.0.1:7557")

        Returns:
            Normalized URL (e.g., "127.0.0.1:7557" if connecting locally)
        """
        if ":" not in heartbeat_url:
            return heartbeat_url

        hb_host, hb_port = heartbeat_url.rsplit(":", 1)

        # Check if we're connecting to controller via localhost
        controller_host = self.config.controller_pull_url.split(":")[0]
        if controller_host in ("127.0.0.1", "localhost"):
            # If controller returns a non-localhost IP, use localhost instead
            if hb_host not in ("127.0.0.1", "localhost"):
                logger.info(
                    "Controller returned heartbeat IP %s, "
                    "but we're connecting locally. Using 127.0.0.1 instead.",
                    hb_host,
                )
                return "127.0.0.1:%s" % hb_port

        return heartbeat_url

    def _setup_heartbeat_socket(self):
        """Setup heartbeat DEALER socket after getting heartbeat_url from register"""
        if not self.heartbeat_url or not self.context:
            logger.warning(
                "Cannot setup heartbeat socket: heartbeat_url=%s, context=%s",
                self.heartbeat_url,
                self.context is not None,
            )
            return
        if self.heartbeat_socket is not None:
            return  # Already setup
        logger.info(
            "Setting up heartbeat DEALER socket to %s, "
            "recv_timeout=%dms, send_timeout=%dms",
            self.heartbeat_url,
            DEFAULT_RECV_TIMEOUT_MS,
            DEFAULT_SEND_TIMEOUT_MS,
        )
        self.heartbeat_socket = get_zmq_socket_with_timeout(
            self.context,
            self.heartbeat_url,
            protocol="tcp",
            role=zmq.DEALER,
            bind_or_connect="connect",
            recv_timeout_ms=DEFAULT_RECV_TIMEOUT_MS,
            send_timeout_ms=DEFAULT_SEND_TIMEOUT_MS,
        )
        logger.info("Heartbeat socket created successfully")

    async def send_heartbeat(self, message: Any) -> Tuple[float, Any]:
        """Send heartbeat via dedicated heartbeat DEALER socket

        Args:
            message: HeartbeatMsg to send

        Returns:
            Tuple of (time taken, response)
        """
        if self.heartbeat_socket is None:
            raise RuntimeError(
                "Heartbeat socket not initialized. "
                "heartbeat_url=%s. Register first to get heartbeat_url."
                % self.heartbeat_url
            )
        start_time = time.time()
        encoded_msg = msgspec.msgpack.encode(message)
        try:
            # DEALER socket: send [empty_frame, payload]
            await self.heartbeat_socket.send_multipart([b"", encoded_msg])
            frames = await self.heartbeat_socket.recv_multipart()
            response = frames[-1]
            return time.time() - start_time, response
        except zmq.Again as e:
            raise RuntimeError(
                "Heartbeat timeout waiting for response from %s" % self.heartbeat_url
            ) from e

    async def deregister_workers(self):
        """Deregister all workers after benchmark"""
        if not self.registered_workers:
            return

        logger.info("Deregistering workers...")
        messages = []
        for instance, worker, ip, port in self.registered_workers:
            msg = DeRegisterMsg(
                instance_id=instance,
                worker_id=worker,
                ip=ip,
                port=port,
            )
            messages.append(msg)

        # Send in batches
        for i in range(0, len(messages), DEFAULT_BATCH_SEND_SIZE):
            batch = messages[i : i + DEFAULT_BATCH_SEND_SIZE]
            await self.send_messages(batch)

        logger.info("Deregistered %d workers", len(messages))
        self.registered_workers.clear()

    def _build_operation_distribution(self) -> List[str]:
        """Build operation distribution list based on percentages"""
        operations = []
        for op_name, percentage in self.config.operations.items():
            count = int(DEFAULT_OP_DISTRIBUTION_BASE * percentage / 100)
            operations.extend([op_name] * count)
        random.shuffle(operations)
        return operations

    async def _execute_operation(
        self, op_name: str, test_data: TestData
    ) -> Tuple[int, int, float, Optional[Exception]]:
        """Execute a single operation

        Returns:
            Tuple of (message_count, request_count, latency, error)
        """
        handler = OPERATION_HANDLERS.get(op_name)
        if not handler:
            logger.warning("Unknown operation: %s", op_name)
            return 0, 0, 0.0, ValueError("Unknown operation")

        try:
            msg = handler.create_message(self, test_data)
            socket_type = handler.socket_type

            if socket_type == SocketType.HEARTBEAT:
                latency, _ = await self.send_heartbeat(msg)
            elif socket_type == SocketType.DEALER:
                latency, _ = await self.send_request(msg)
            else:  # SocketType.PUSH
                msg_start = time.time()
                await self.send_messages([msg])
                latency = time.time() - msg_start
            return handler.get_message_count(self), 1, latency, None
        except Exception as e:
            logger.error("Error in %s: %s", op_name, e)
            return 0, 0, 0.0, e

    async def run_benchmark(self):
        """Run the main benchmark"""
        await self.setup()

        try:
            test_data = self.generate_test_data()

            # Pre-register workers
            await self.register_workers(test_data)

            # Build operation distribution
            operations = self._build_operation_distribution()

            # Initialize tracking
            latencies: Dict[str, List[float]] = {
                op: [] for op in self.config.operations.keys()
            }
            errors: Dict[str, int] = {op: 0 for op in self.config.operations.keys()}
            message_counts: Dict[str, int] = {
                op: 0 for op in self.config.operations.keys()
            }
            request_counts: Dict[str, int] = {
                op: 0 for op in self.config.operations.keys()
            }
            total_messages = 0
            total_requests = 0

            # Start monitoring
            self.running = True
            monitoring_task = asyncio.create_task(self.monitor_system())

            start_time = time.time()
            op_index = 0

            logger.info("Starting benchmark for %d seconds...", self.config.duration)

            while time.time() - start_time < self.config.duration:
                # Get next operation
                op_name = operations[op_index % len(operations)]
                op_index += 1

                msg_count, req_count, latency, error = await self._execute_operation(
                    op_name, test_data
                )
                total_messages += msg_count
                total_requests += req_count
                if error:
                    errors[op_name] += 1
                else:
                    latencies[op_name].append(latency)
                    message_counts[op_name] += msg_count
                    request_counts[op_name] += req_count

                # Small yield to prevent blocking
                if op_index % 100 == 0:
                    await asyncio.sleep(0)

            # Stop monitoring
            self.running = False
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

            # Calculate results
            total_time = time.time() - start_time
            overall_qps = total_messages / total_time if total_time > 0 else 0
            overall_rps = total_requests / total_time if total_time > 0 else 0

            self.results.total_messages = total_messages
            self.results.total_requests = total_requests
            self.results.total_time = total_time
            self.results.overall_qps = overall_qps
            self.results.overall_rps = overall_rps

            # Per-operation stats
            for op_name in self.config.operations.keys():
                if latencies[op_name]:
                    op_qps = (
                        message_counts[op_name] / total_time if total_time > 0 else 0
                    )
                    op_rps = (
                        request_counts[op_name] / total_time if total_time > 0 else 0
                    )
                    avg_latency = statistics.mean(latencies[op_name])

                    self.results.operations[op_name] = OperationStats(
                        qps=op_qps,
                        rps=op_rps,
                        avg_latency=avg_latency,
                        min_latency=min(latencies[op_name]),
                        max_latency=max(latencies[op_name]),
                        p95_latency=(
                            statistics.quantiles(latencies[op_name], n=20)[18]
                            if len(latencies[op_name]) >= 20
                            else max(latencies[op_name])
                        ),
                        errors=errors[op_name],
                    )

            # Deregister workers
            await self.deregister_workers()

        finally:
            self.cleanup()

    async def monitor_system(self):
        """Monitor system metrics during benchmark"""
        while self.running:
            try:
                memory_usage = psutil.virtual_memory().percent
                self.results.memory_usage.append(memory_usage)
            except Exception as e:
                logger.warning("Failed to get memory usage: %s", e)
            await asyncio.sleep(1)

    def print_results(self):
        """Print benchmark results"""
        print("\n" + "=" * 80)
        if self.config.num_processes > 1:
            print(
                "LMCache Controller ZMQ Benchmark Results (Process %d/%d)"
                % (self.config.process_id + 1, self.config.num_processes)
            )
        else:
            print("LMCache Controller ZMQ Benchmark Results")
        print("=" * 80)

        print("\nConfiguration:")
        print("  Controller URL: %s" % self.config.controller_pull_url)
        print("  Duration: %d seconds" % self.config.duration)
        print("  Batch Size: %d" % self.config.batch_size)
        print("  Operations: %s" % self.config.operations)
        print(
            "  Instances: %d, Workers: %d, Locations: %d, Keys: %d"
            % (
                self.config.num_instances,
                self.config.num_workers,
                self.config.num_locations,
                self.config.num_keys,
            )
        )

        print("\nOverall Performance:")
        print("  Total Requests: %d" % self.results.total_requests)
        print("  Total Messages: %d" % self.results.total_messages)
        print("  Total Time: %.2fs" % self.results.total_time)
        print("  Overall RPS (Requests/sec): %.2f" % self.results.overall_rps)
        print("  Overall QPS (Messages/sec): %.2f" % self.results.overall_qps)

        print("\nPer-Operation Performance:")
        for op_name in self.config.operations.keys():
            if op_name in self.results.operations:
                stats = self.results.operations[op_name]
                print("  %s:" % op_name)
                print("    RPS (Requests/sec): %.2f" % stats.rps)
                print("    QPS (Messages/sec): %.2f" % stats.qps)
                print(
                    "    Latency - Avg: %.3fms, Min: %.3fms, Max: %.3fms, P95: %.3fms"
                    % (
                        stats.avg_latency * 1000,
                        stats.min_latency * 1000,
                        stats.max_latency * 1000,
                        stats.p95_latency * 1000,
                    )
                )
                print("    Errors: %d" % stats.errors)

        print("\nSystem Metrics:")
        if self.results.memory_usage:
            avg_memory = statistics.mean(self.results.memory_usage)
            max_memory = max(self.results.memory_usage)
            print(
                "  Memory Usage - Avg: %.1f%%, Max: %.1f%%" % (avg_memory, max_memory)
            )

        print("=" * 80)

    def get_results(self) -> BenchmarkResults:
        """Return benchmark results for aggregation"""
        return self.results
