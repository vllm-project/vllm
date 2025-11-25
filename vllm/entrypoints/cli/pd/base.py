# SPDX-License-Identifier: Apache-2.0

import logging
import os
import socket
import subprocess
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union, Tuple, Dict, Any

import ray
import requests
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.entrypoints.cli.pd.config import Config
from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger(__name__)

HOST_PLACE_HOLDER = "{HOST}"
PORT_PLACE_HOLDER = "{PORT}"
KV_IP_PLACE_HOLDER = "{KV_IP}"
NODE_NAME_PLACE_HOLDER = "{NODE_NAME}"
RAY_HEAD_IP = "RAY_HEAD_IP"

class RankIDGenerator:
    _global_rank_id = 0

    @classmethod
    def next_rank_id(cls) -> int:
        rank_id = cls._global_rank_id
        cls._global_rank_id += 1
        return rank_id

    @classmethod
    def reset(cls) -> None:
        cls._global_rank_id = 0


class ServiceType(str, Enum):
    prefill = "PREFILL"
    decode = "DECODE"


class _BaseVllmService:
    def __init__(self,
                 job_id: str,
                 service_type: ServiceType,
                 rank_id: int,
                 role_params: Dict[str, Any],
                 unique_role_extra_params: str,
                 role_envs: Dict[str, str]):
        self.job_id = job_id
        self.service_type = service_type
        self.rank_id = rank_id
        self.role_params = role_params
        self.unique_role_extra_params = unique_role_extra_params
        self.role_envs = role_envs

        self.socket: Optional[socket.socket] = None

    def get_cwd(self) -> str:
        return os.getcwd()
    
    def ip_and_port(self) -> Tuple[str, int]:
        if self.socket is not None:
            return self.socket.getsockname()

        from vllm.entrypoints.openai.api_server import create_server_socket

        ip = ray.get_runtime_context().worker.node_ip_address
        # find free port
        self.socket = create_server_socket((ip, 0))
        return self.socket.getsockname()

    def _prepare_start_env(self):
        if not ray.get_gpu_ids():
            # hack the flashMLA gpu devices check

            @ray.remote
            def _get_gpu_ids():
                return ray.get_gpu_ids()

            current_placement_group = ray.util.get_current_placement_group()
            if current_placement_group:
                print("Trying get one possible GPU id from current placement group")
                gpu_ids = []
                for idx in range(current_placement_group.bundle_count):
                    scheduling_strategy = PlacementGroupSchedulingStrategy(
                        placement_group=current_placement_group,
                        placement_group_bundle_index=idx,
                    )
                    result = ray.get(_get_gpu_ids.options(
                        num_cpus=0,
                        num_gpus=1,
                        scheduling_strategy=scheduling_strategy).remote())

                    if result:
                        gpu_ids.extend([str(i) for i in result])

                if not gpu_ids:
                    print(f"No GPUs available for this placement group: {current_placement_group}")
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

        # set `LMCACHE_CONFIG_FILE` if use lmcache
        if "VLLM_USE_LMCACHE" in os.environ:
            lmcache_path = os.path.join(self.get_cwd(), "lmcache.yaml")
            if not os.path.exists(lmcache_path):
                lmcache_path = "/vllm-workspace/examples/lmcache/lmcache.yaml"

            os.environ["LMCACHE_CONFIG_FILE"] = lmcache_path

    @abstractmethod
    def prepare_start_role_params(self) -> Dict[str, Any]:
        ...

    def start(self):
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.entrypoints.cli.serve import ServeSubcommand

        role_params = self.prepare_start_role_params()
        print(
            f"Starting VllmService(type={self.service_type}, job_id={self.job_id}, rank_id={self.rank_id}), params: {role_params}")

        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")

        service_command = ServeSubcommand()
        service_command.subparser_init(subparsers)

        args = parser.parse_args(args=role_params, extra_params=self.unique_role_extra_params)

        self._prepare_start_env()

        service_command.validate(args)
        ServeSubcommand.cmd(args)


@dataclass
class Service:
    rank_id: int
    actor_handler: ray.actor.ActorHandle
    _ip_address: Optional[Tuple[str, int]] = None
    _url: Optional[str] = None

    @property
    def ip_and_port(self) -> Tuple[str, int]:
        if not self._ip_address:
            self._ip_address = ray.get(self.actor_handler.ip_and_port.remote())

        return self._ip_address

    @property
    def url(self) -> str:
        if not self._url:
            ip, port = self.ip_and_port
            self._url = f"http://{ip}:{port}"
        return self._url

    def check_health(self) -> Union[bool, Exception]:
        try:
            response = requests.get(self.url + "/health", timeout=1)
            return response.status_code == 200
        except Exception as ex:
            return ex


@ray.remote
class ProxyServer:
    def __init__(self,
                 prefill_services: List[Service],
                 decode_services: List[Service],
                 scheduler_config: Dict[str, Any]):
        self._prefill_services = prefill_services
        self._decode_services = decode_services
        self._scheduler_config = scheduler_config
        self._ip: Optional[str] = None
        self._command: Optional[str] = None
        
        # Prepare command with replaced placeholders
        self._prepare_command()

    def ip(self) -> str:
        if self._ip is None:
            self._ip = ray.get_runtime_context().worker.node_ip_address
        return self._ip
    
    def _prepare_command(self):
        """Prepare command by replacing placeholders with actual values."""
        if "command" not in self._scheduler_config or not self._scheduler_config["command"]:
            raise ValueError("ProxyServer requires 'command' in scheduler_config")
        
        command = self._scheduler_config["command"]
        
        # Replace {PORT} placeholder
        command = command.replace("{PORT}", str(self._scheduler_config.get("port", 8021)))
        
        # Replace in reverse order to avoid $PREFILL_HOST_1 matching part of $PREFILL_HOST_10
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
        logger.info(f"Prepared proxy command: {self._command}")
    
    def start(self):
        """Start the proxy server using the prepared command."""
        if not self._command:
            raise ValueError("Command not prepared. Call _prepare_command first.")
        
        logger.info(f"Starting ProxyServer with command: {self._command}")
        
        # Start the command as a subprocess
        process = subprocess.Popen(
            self._command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process (blocking)
        # In Ray remote actor, this will keep the actor alive
        process.wait()


class BasePDJob:
    def __init__(self,
                 job_id: str,
                 config: Config):
        self.job_id = job_id
        self.config = config

    def _create_placement_group(self, role_num_gpus: int) -> PlacementGroup:
        # Create a placement group.
        num_bundles = int(role_num_gpus / self.config.gpus_per_worker)
        pg = placement_group([{"GPU": self.config.gpus_per_worker}] * num_bundles)
        # pg = placement_group([{"CPU": config.gpus_per_worker}] * num_bundles)
        ray.get(pg.ready(), timeout=self.config.timeout)

        assert pg.bundle_count == num_bundles

        return pg

    def _check_services_health(self, services: List[Service], service_type: ServiceType) -> None:
        logger.info(f"Waiting all {service_type} ready ...")
        start_time = time.time()
        while True:
            check_result = [service.check_health() for service in services]
            all_succeed = all([result is True for result in check_result])
            if all_succeed:
                logger.info(f"All services({service_type}) healthy: {all_succeed}")
                return

            # Check if the timeout is exceeded
            if time.time() - start_time > self.config.timeout:
                # last failed exception
                for result, service in zip(check_result, services):
                    if isinstance(result, Exception):
                        logger.error(
                            f"The last exception occurred checking health(type: {service_type}, rank_id: {service.rank_id}).\n{str(result)}")

                raise Exception(
                    f"Timeout reached ({self.config.timeout} seconds) for checking all services({service_type}) are startup")

            # Wait for 1 second
            time.sleep(1)

    def _start_llm_scheduler(self, prefill_services: List[Service], decode_services: List[Service]) -> None:
        # start llm scheduler
        logger.info("=================Starting VLLM pd scheduler================")
        scheduler_name = f"{self.job_id}-scheduler"
        head_ip = get_head_ip()
        port = self.config.scheduler_config.get("port", 8021)
        
        # Check if custom command is provided in scheduler config
        if "command" not in self.config.scheduler_config or not self.config.scheduler_config["command"]:
            raise ValueError("'command' is required in scheduler_config")
        
        # Use ProxyServer to start scheduler with command replacement
        scheduler_handler = ProxyServer.options(
            num_cpus=1,
            name=scheduler_name,
            resources={f"node:{head_ip}": 0.01}
        ).remote(
            prefill_services,
            decode_services,
            self.config.scheduler_config
        )
        
        scheduler_ip = ray.get(scheduler_handler.ip.remote())
        assert scheduler_ip == head_ip, "Should schedule to head node"
        logger.info(f"=========Starting Scheduler http://{head_ip}:{port}==========")
        
        # Start the proxy server
        scheduler_handler.start.remote()

    @abstractmethod
    def start_vllm_services(self) -> Tuple[List[Service], List[Service]]:
        ...

    def start(self) -> None:
        logger.info(f"=================Starting VLLM pd disaggregated job: {self.job_id}================")

        prefill_services, decode_services = self.start_vllm_services()

        self._check_services_health(prefill_services, ServiceType.prefill)
        self._check_services_health(decode_services, ServiceType.decode)

        self._start_llm_scheduler(prefill_services, decode_services)

        try:
            while True:
                # Wait for 1 second
                time.sleep(1)
        except Exception:
            ray.shutdown()


def get_head_ip() -> str:
    try:
        from ray.util import state

        nodes = state.list_nodes(filters=[("is_head_node", "=", True)])
        assert len(nodes) == 1, "There should be exactly one head node"

        return nodes[0].node_ip
    except Exception:
        if RAY_HEAD_IP in os.environ:
            logger.info("Failed to get head ip from ray state, getting it from environ")
            return os.environ[RAY_HEAD_IP]
        else:
            logger.info("Failed to get head ip from ray state, getting it from gcs address")
            return ray.get_runtime_context().gcs_address.split(":")[0]


def set_logger(logger):
    session_dir = ray._private.worker._global_node.get_session_dir_path()
    ray_log_path = os.path.join(session_dir, "logs", "driver.log")

    # Create file handler
    file_handler = logging.FileHandler(ray_log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to existing logger
    logger.addHandler(file_handler)
