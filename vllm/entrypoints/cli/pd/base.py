# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base module for implementing vLLM Prefill-Decode disaggregated job management.

This module provides:
- Service type definitions (Prefill/Decode)
- Base service class implementations
- Proxy server management
- Job startup and health checking
"""

import logging
import os
import shlex
import socket
import subprocess
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ray
import requests
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.entrypoints.cli.pd.config import Config
from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger(__name__)

# Placeholder constants for variable substitution in command templates
HOST_PLACE_HOLDER = "{HOST}"
PORT_PLACE_HOLDER = "{PORT}"
KV_IP_PLACE_HOLDER = "{KV_IP}"
NODE_NAME_PLACE_HOLDER = "{NODE_NAME}"
RAY_HEAD_IP = "RAY_HEAD_IP"


class RankIDGenerator:
    """
    Global Rank ID generator.

    Used to assign unique rank_id to different service instances, ensuring each
    service has a unique identifier. Uses class variables to maintain a global
    counter with reset functionality.
    """

    _global_rank_id = 0

    @classmethod
    def next_rank_id(cls) -> int:
        """
        Get the next available rank_id.

        Returns:
            int: A new rank_id that increments with each call
        """
        rank_id = cls._global_rank_id
        cls._global_rank_id += 1
        return rank_id

    @classmethod
    def reset(cls) -> None:
        """
        Reset the rank_id counter to 0.

        Typically called when starting a new job to ensure rank_id starts from 0.
        """
        cls._global_rank_id = 0


class ServiceType(str, Enum):
    """
    Service type enumeration.

    Defines two service types in the Prefill-Decode disaggregated architecture:
    - prefill: Prefill service, responsible for initial computation of input sequences
    - decode: Decode service, responsible for incremental computation during generation
    """

    prefill = "PREFILL"
    decode = "DECODE"


class _BaseVllmService:
    """
    Base abstract class for vLLM services.

    Defines common interfaces and implementations for Prefill and Decode services,
    including:
    - Service initialization and parameter management
    - IP and port allocation
    - Environment preparation (GPU, LMCache, etc.)
    - Service startup logic
    """

    def __init__(
        self,
        job_id: str,
        service_type: ServiceType,
        rank_id: int,
        role_params: dict[str, Any],
        unique_role_extra_params: str,
        role_envs: dict[str, str],
    ):
        self.job_id = job_id
        self.service_type = service_type
        self.rank_id = rank_id
        self.role_params = role_params
        self.unique_role_extra_params = unique_role_extra_params
        self.role_envs = role_envs

        self.socket: socket.socket | None = None

    def get_cwd(self) -> str:
        """
        Get the current working directory.

        Returns:
            str: Current working directory path
        """
        return os.getcwd()

    def ip_and_port(self) -> tuple[str, int]:
        """
        Get the service IP address and port number.

        If socket is already created, return its address directly; otherwise create
        a new socket and bind it to an available port.

        Returns:
            Tuple[str, int]: (IP address, port number)
        """
        if self.socket is not None:
            return self.socket.getsockname()

        from vllm.entrypoints.openai.api_server import create_server_socket

        ip = ray.get_runtime_context().worker.node_ip_address
        # find free port
        self.socket = create_server_socket((ip, 0))
        return self.socket.getsockname()

    def _prepare_start_env(self):
        """
        Prepare environment variables required for service startup.

        Mainly handles:
        1. GPU device allocation: If no GPU is currently available, try to get
           one from the placement group
        2. LMCache configuration: If LMCache is enabled, set the config file path
        """
        if not ray.get_gpu_ids():
            # hack the flashMLA gpu devices check
            # When current worker has no GPU, try to get available GPU
            # from placement group

            @ray.remote
            def _get_gpu_ids():
                return ray.get_gpu_ids()

            current_placement_group = ray.util.get_current_placement_group()
            if current_placement_group:
                print("Trying get one possible GPU id from current placement group")
                gpu_ids = []
                # Iterate through all bundles in placement group
                # to find available GPUs
                for idx in range(current_placement_group.bundle_count):
                    scheduling_strategy = PlacementGroupSchedulingStrategy(
                        placement_group=current_placement_group,
                        placement_group_bundle_index=idx,
                    )
                    result = ray.get(
                        _get_gpu_ids.options(
                            num_cpus=0,
                            num_gpus=1,
                            scheduling_strategy=scheduling_strategy,
                        ).remote()
                    )

                    if result:
                        gpu_ids.extend([str(i) for i in result])

                if not gpu_ids:
                    print(
                        "No GPUs available for this placement group: "
                        f"{current_placement_group}"
                    )
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    @abstractmethod
    def prepare_start_role_params(self) -> dict[str, Any]:
        """
        Prepare role parameters required for service startup.

        Subclasses must implement this method to return a dictionary of parameters
        for starting the service.

        Returns:
            Dict[str, Any]: Service startup parameter dictionary
        """
        ...

    def start(self):
        """
        Start the vLLM service.

        Executes the following steps:
        1. Prepare role parameters
        2. Create command line parser
        3. Prepare environment variables
        4. Validate parameters and start the service
        """
        from vllm.entrypoints.cli.serve import ServeSubcommand
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        role_params = self.prepare_start_role_params()
        print(
            f"Starting VllmService(type={self.service_type}, "
            f"job_id={self.job_id}, rank_id={self.rank_id}), "
            f"params: {role_params}"
        )

        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")

        service_command = ServeSubcommand()
        service_command.subparser_init(subparsers)

        args = parser.parse_args(
            args=role_params, extra_params=self.unique_role_extra_params
        )

        self._prepare_start_env()

        service_command.validate(args)
        ServeSubcommand.cmd(args)


@dataclass
class Service:
    """
    Service wrapper class for managing a single vLLM service instance.

    Encapsulates the service's rank_id, actor handle, IP address, and URL,
    providing health check functionality. Uses property caching to avoid
    repeatedly fetching IP and URL.
    """

    rank_id: int
    actor_handler: ray.actor.ActorHandle
    _ip_address: tuple[str, int] | None = None
    _url: str | None = None

    @property
    def ip_and_port(self) -> tuple[str, int]:
        """
        Get the service IP address and port number.

        Uses caching mechanism: fetches from actor on first call, returns cached
        value on subsequent calls.

        Returns:
            Tuple[str, int]: (IP address, port number)
        """
        if not self._ip_address:
            self._ip_address = ray.get(self.actor_handler.ip_and_port.remote())

        return self._ip_address

    @property
    def url(self) -> str:
        """
        Get the complete service URL.

        Uses caching mechanism: builds URL on first call, returns cached value
        on subsequent calls.

        Returns:
            str: Service HTTP URL in format "http://ip:port"
        """
        if not self._url:
            ip, port = self.ip_and_port
            self._url = f"http://{ip}:{port}"
        return self._url

    def check_health(self) -> bool | Exception:
        """
        Check the service health status.

        Checks if the service is running normally by sending an HTTP GET request
        to the /health endpoint.

        Returns:
            Union[bool, Exception]:
                - True: Service is healthy (returns 200 status code)
                - Exception: Returns exception object when service is unhealthy
                  or request fails
        """
        try:
            response = requests.get(self.url + "/health", timeout=1)
            return response.status_code == 200
        except Exception as ex:
            return ex


@ray.remote
class ProxyServer:
    """
    Proxy server class for starting and managing scheduler/proxy processes.

    Responsibilities:
    1. Parse command templates from scheduler configuration
    2. Replace placeholders in commands (e.g., PREFILL_HOST_1, DECODE_PORT_2, etc.)
    3. Start proxy server subprocess
    4. Manage proxy server lifecycle
    """

    def __init__(
        self,
        prefill_services: list[Service],
        decode_services: list[Service],
        scheduler_config: dict[str, Any],
    ):
        self._prefill_services = prefill_services
        self._decode_services = decode_services
        self._scheduler_config = scheduler_config
        self._ip: str | None = None
        self._command: str | None = None

        # Prepare command with replaced placeholders
        self._prepare_command()

    def ip(self) -> str:
        """
        Get the IP address of the node where the proxy server is located.

        Uses caching mechanism: fetches from Ray runtime on first call, returns
        cached value on subsequent calls.

        Returns:
            str: Node IP address
        """
        if self._ip is None:
            self._ip = ray.get_runtime_context().worker.node_ip_address
        return self._ip

    def _prepare_command(self):
        """
        Prepare command string by replacing all placeholders with actual values.

        Supported placeholders include:
        - {PORT}: Port number
        - $PREFILL_HOST_{idx}, $PREFILL_PORT_{idx}, $PREFILL_URL_{idx}
        - $DECODE_HOST_{idx}, $DECODE_PORT_{idx}, $DECODE_URL_{idx}

        Note: Uses reverse order replacement to avoid $PREFILL_HOST_1 matching part
        of $PREFILL_HOST_10.
        """
        if (
            "command" not in self._scheduler_config
            or not self._scheduler_config["command"]
        ):
            raise ValueError("ProxyServer requires 'command' in scheduler_config")

        command = self._scheduler_config["command"]

        # Replace {PORT} placeholder with actual port number
        command = command.replace(
            "{PORT}", str(self._scheduler_config.get("port", 8021))
        )

        # Replace in reverse order to avoid $PREFILL_HOST_1
        # matching part of $PREFILL_HOST_10
        for idx in range(len(self._prefill_services), 0, -1):
            service = self._prefill_services[idx - 1]
            url = service.url.replace("http://", "").replace("https://", "")
            if ":" in url:
                host, port = url.rsplit(":", 1)
                command = command.replace(f"${{PREFILL_HOST_{idx}}}", host)
                command = command.replace(f"$PREFILL_HOST_{idx}", host)
                command = command.replace(f"${{PREFILL_PORT_{idx}}}", port)
                command = command.replace(f"$PREFILL_PORT_{idx}", port)
            command = command.replace(f"${{PREFILL_URL_{idx}}}", service.url)
            command = command.replace(f"$PREFILL_URL_{idx}", service.url)

        for idx in range(len(self._decode_services), 0, -1):
            service = self._decode_services[idx - 1]
            url = service.url.replace("http://", "").replace("https://", "")
            if ":" in url:
                host, port = url.rsplit(":", 1)
                command = command.replace(f"${{DECODE_HOST_{idx}}}", host)
                command = command.replace(f"$DECODE_HOST_{idx}", host)
                command = command.replace(f"${{DECODE_PORT_{idx}}}", port)
                command = command.replace(f"$DECODE_PORT_{idx}", port)
            command = command.replace(f"${{DECODE_URL_{idx}}}", service.url)
            command = command.replace(f"$DECODE_URL_{idx}", service.url)

        self._command = command
        logger.info("Prepared proxy command: %s", self._command)

    def start(self):
        """Start the proxy server using the prepared command."""
        if not self._command:
            raise ValueError("Command not prepared. Call _prepare_command first.")

        logger.info("Starting ProxyServer with command: %s", self._command)

        # Parse command string into list of arguments for safe execution
        # This prevents command injection vulnerabilities
        cmd_args = shlex.split(self._command)

        # Start the command as a subprocess with shell=False for security
        process = subprocess.Popen(
            cmd_args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the process (blocking)
        # In Ray remote actor, this blocking call keeps the actor alive
        # When the process exits, the actor will handle it accordingly
        process.wait()


class BasePDJob:
    """
    Base class for Prefill-Decode disaggregated jobs.

    Defines the core workflow for job management:
    1. Create placement groups for resource allocation
    2. Start Prefill and Decode services
    3. Check service health status
    4. Start scheduler/proxy server
    5. Maintain job running state
    """

    def __init__(self, job_id: str, config: Config):
        self.job_id = job_id
        self.config = config

    def _create_placement_group(self, role_num_gpus: int) -> PlacementGroup:
        """
        Create Ray placement group for resource allocation.

        Calculates the number of bundles needed based on required GPU count and
        GPUs per worker, then creates the corresponding placement group.

        Args:
            role_num_gpus: Total number of GPUs required for this role

        Returns:
            PlacementGroup: Created placement group

        Raises:
            AssertionError: If bundle count doesn't match
            TimeoutError: If placement group is not ready within timeout
        """
        # Create a placement group.
        # Calculate number of bundles needed: total GPUs / GPUs per worker
        num_bundles = int(role_num_gpus / self.config.gpus_per_worker)
        # Create placement group with specified number of GPUs
        pg = placement_group([{"GPU": self.config.gpus_per_worker}] * num_bundles)
        # CPU version for debugging:
        # pg = placement_group([{"CPU": config.gpus_per_worker}] * num_bundles)
        # Wait for placement group to be ready, with timeout
        ray.get(pg.ready(), timeout=self.config.timeout)

        assert pg.bundle_count == num_bundles

        return pg

    def _check_services_health(
        self, services: list[Service], service_type: ServiceType
    ) -> None:
        """
        Check health status of all services and wait until they are all ready.

        Continuously polls health status of all services until:
        - All services are healthy (return True)
        - Or timeout (raises exception)

        Args:
            services: List of services to check
            service_type: Service type (for logging)

        Raises:
            Exception: If services are not all ready within timeout period
        """
        logger.info("Waiting all %s ready ...", service_type)
        start_time = time.time()
        while True:
            # Check health status of all services in parallel
            check_result = [service.check_health() for service in services]
            # Check if all services are healthy (return True)
            all_succeed = all([result is True for result in check_result])
            if all_succeed:
                logger.info("All services(%s) healthy: %s", service_type, all_succeed)
                return

            # Check if the timeout is exceeded
            if time.time() - start_time > self.config.timeout:
                # Log last failed exception for debugging
                for result, service in zip(check_result, services):
                    if isinstance(result, Exception):
                        logger.error(
                            "The last exception occurred checking health"
                            "(type: %s, rank_id: %s).\n%s",
                            service_type,
                            service.rank_id,
                            str(result),
                        )

                raise Exception(
                    f"Timeout reached ({self.config.timeout} seconds) "
                    f"for checking all services({service_type}) are startup"
                )

            # Wait for 1 second
            time.sleep(1)

    def _start_llm_scheduler(
        self, prefill_services: list[Service], decode_services: list[Service]
    ) -> None:
        """
        Start the LLM scheduler/proxy server.

        Creates a ProxyServer actor, schedules it to the head node, then starts
        the proxy server process. The proxy server is responsible for routing
        requests to appropriate Prefill or Decode services.

        Args:
            prefill_services: List of Prefill services
            decode_services: List of Decode services

        Raises:
            ValueError: If 'command' field is missing in scheduler config
            AssertionError: If proxy server is not scheduled to head node
        """
        # start llm scheduler
        logger.info("=================Starting VLLM pd scheduler================")
        scheduler_name = f"{self.job_id}-scheduler"
        head_ip = get_head_ip()
        port = self.config.scheduler_config.get("port", 8021)

        # Check if custom command is provided in scheduler config
        if (
            "command" not in self.config.scheduler_config
            or not self.config.scheduler_config["command"]
        ):
            raise ValueError("'command' is required in scheduler_config")

        # Use ProxyServer to start scheduler with command replacement
        # Use resource constraints to ensure scheduling to head node
        scheduler_handler = ProxyServer.options(  # type: ignore[attr-defined]
            num_cpus=1,
            name=scheduler_name,
            resources={f"node:{head_ip}": 0.01},  # Constrain scheduling to head node
        ).remote(prefill_services, decode_services, self.config.scheduler_config)

        # Verify proxy server is indeed scheduled to head node
        scheduler_ip = ray.get(scheduler_handler.ip.remote())
        assert scheduler_ip == head_ip, "Should schedule to head node"
        logger.info("=========Starting Scheduler http://%s:%s==========", head_ip, port)

        # Start the proxy server
        # Start proxy server process asynchronously
        scheduler_handler.start.remote()

    @abstractmethod
    def start_vllm_services(self) -> tuple[list[Service], list[Service]]:
        """
        Start vLLM services (Prefill and Decode).

        Subclasses must implement this method to create and start all Prefill
        and Decode service instances.

        Returns:
            Tuple[List[Service], List[Service]]:
                (List of Prefill services, List of Decode services)
        """
        ...

    def start(self) -> None:
        """
        Start the complete Prefill-Decode disaggregated job.

        Execution flow:
        1. Start all Prefill and Decode services
        2. Wait for all services to be healthy and ready
        3. Start scheduler/proxy server
        4. Keep job running until interrupted

        If an exception occurs, the Ray cluster will be shut down.
        """
        logger.info(
            "=================Starting VLLM pd disaggregated job: %s================",
            self.job_id,
        )

        prefill_services, decode_services = self.start_vllm_services()

        self._check_services_health(prefill_services, ServiceType.prefill)
        self._check_services_health(decode_services, ServiceType.decode)

        self._start_llm_scheduler(prefill_services, decode_services)

        try:
            # Keep job running, periodically check status
            while True:
                # Wait for 1 second
                # Check every second to avoid high CPU usage
                time.sleep(1)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            logger.info("Received KeyboardInterrupt, shutting down...")
            ray.shutdown()
        except Exception as e:
            # Shutdown Ray cluster on exception to clean up resources
            logger.error("Exception occurred: %s, shutting down...", e)
            ray.shutdown()


def get_head_ip() -> str:
    """
    Get the IP address of the Ray cluster head node.

    Tries multiple methods to get head IP:
    1. First try to get from Ray state API
    2. If that fails, get from environment variable RAY_HEAD_IP
    3. If environment variable doesn't exist, parse from GCS address

    Returns:
        str: IP address of the head node

    Raises:
        AssertionError: If state API finds number of head nodes is not 1
    """
    try:
        from ray.util import state

        # Query head node from Ray state API
        nodes = state.list_nodes(filters=[("is_head_node", "=", True)])
        assert len(nodes) == 1, "There should be exactly one head node"

        return nodes[0].node_ip
    except Exception:
        # If state API fails, try to get from environment variable
        if RAY_HEAD_IP in os.environ:
            logger.info("Failed to get head ip from ray state, getting it from environ")
            return os.environ[RAY_HEAD_IP]
        else:
            # Finally try to parse from GCS address (format is usually "ip:port")
            logger.info(
                "Failed to get head ip from ray state, getting it from gcs address"
            )
            return ray.get_runtime_context().gcs_address.split(":")[0]


def set_logger(target_logger):
    """
    Add file handler to logger to output logs to Ray session directory.

    Log file path: {session_dir}/logs/driver.log
    Log level is set to DEBUG, including timestamp, level, name, and message.

    Args:
        target_logger: Logger object to configure
    """
    # Get Ray session directory path
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    # Build log file path
    ray_log_path = os.path.join(session_dir, "logs", "driver.log")

    # Create file handler
    # Create file handler to write logs to file
    file_handler = logging.FileHandler(ray_log_path)
    file_handler.setLevel(logging.DEBUG)
    # Set log format: timestamp level name message
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to existing logger
    # Add file handler to logger so it outputs to both console and file
    target_logger.addHandler(file_handler)
