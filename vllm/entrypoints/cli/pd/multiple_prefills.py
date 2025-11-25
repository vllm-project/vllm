# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List, Optional, Dict, Any

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.entrypoints.cli.pd.base import BasePDJob, _BaseVllmService, Service, PORT_PLACE_HOLDER, \
    KV_IP_PLACE_HOLDER, NODE_NAME_PLACE_HOLDER, ServiceType, RankIDGenerator
from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger(__name__)


@ray.remote
class _VllmService(_BaseVllmService):
    def __init__(self,
                 job_id: str,
                 service_type: ServiceType,
                 rank_id: int,
                 role_params: Dict[str, Any],
                 unique_role_extra_params: str,
                 role_envs: Dict[str, str],
                 kv_ip: Optional[str]):
        super().__init__(job_id, service_type, rank_id, role_params, unique_role_extra_params, role_envs)

        if kv_ip is None:
            assert self.rank_id == 0, "kv_ip should not be None"

        self.kv_ip = kv_ip

    def prepare_start_role_params(self) -> Dict[str, Any]:
        host, port = self.socket.getsockname()
        # replace host
        # Replace host and port in role_params
        if self.role_params.get("host"):
            self.role_params["host"] = str(host)
        if self.role_params.get("port"):
            self.role_params["port"] = str(port)

        if "kv_transfer_config" in self.role_params:
            # Replace kv_ip in kv_transfer_config if present
            kv_ip = self.kv_ip or str(host)
            self.role_params["kv_transfer_config"] = self.role_params["kv_transfer_config"].replace(KV_IP_PLACE_HOLDER, kv_ip)

            # Replace port in kv_transfer_config if present
            self.role_params["kv_transfer_config"] = self.role_params["kv_transfer_config"].replace(PORT_PLACE_HOLDER, str(port))
            
            # Replace node_name in kv_transfer_config if present
            self.role_params["kv_transfer_config"] = self.role_params["kv_transfer_config"].replace(NODE_NAME_PLACE_HOLDER, ray.get_runtime_context().get_actor_name())
            
        return self.role_params


class MultiplePrefillsPDJob(BasePDJob):
    def __init__(self,
                 job_id: str,
                 config: Config):
        super().__init__(job_id, config)

    def _start_single_service(self,
                              rank_id: int,
                              service_type: ServiceType,
                              runtime_env: RuntimeEnv,
                              role_params: Dict[str, Any],
                              role_num_gpus: int,
                              unique_role_extra_params: str,
                              role_envs: Dict[str, str],
                              kv_ip: Optional[str]) -> Service:
        pg = self._create_placement_group(role_num_gpus)
        name = f"{self.job_id}-{service_type}-{rank_id}"

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )

        # world_size is 1
        num_gpus = self.config.gpus_per_worker
        if pg.bundle_count > 1:
            # world_size is greater than 1
            num_gpus = 0

        handler = _VllmService.options(
            num_cpus=0,
            num_gpus=num_gpus,
            name=name,
            runtime_env=runtime_env,
            scheduling_strategy=scheduling_strategy,
        ).remote(
            self.job_id,
            service_type,
            rank_id,
            role_params,
            unique_role_extra_params,
            role_envs,
            kv_ip
        )

        return Service(rank_id, handler)

    def _start_prefill_services(self) -> Tuple[str, List[Service]]:
        logger.info("Starting prefill services")

        prefill_config = self.config.prefill_config

        runtime_env = None
        if prefill_config.envs:
            runtime_env = RuntimeEnv(env_vars=prefill_config.envs)

        services: List[Service] = []
        kv_ip = None
        for idx in range(prefill_config.replicas):
            rank_id = RankIDGenerator.next_rank_id()
            service = self._start_single_service(rank_id, ServiceType.prefill, runtime_env,
                                                 prefill_config.role_params, prefill_config.role_num_gpus, prefill_config.unique_role_extra_params, prefill_config.envs, kv_ip)
            if idx == 0:
                host, _ = service.ip_and_port
                kv_ip = str(host)
            
            logger.info(f"Starting prefill service(rank_id={service.rank_id}), url: {service.url}")
            service.actor_handler.start.remote()

            services.append(service)

        return kv_ip, services

    def _start_decode_services(self, kv_ip: str) -> List[Service]:
        logger.info("Starting decode services")

        decode_config = self.config.decode_config

        runtime_env = None
        if decode_config.envs:
            runtime_env = RuntimeEnv(env_vars=decode_config.envs)

        services: List[Service] = []
        for _ in range(decode_config.replicas):
            rank_id = RankIDGenerator.next_rank_id()
            service = self._start_single_service(rank_id, ServiceType.decode, runtime_env,
                                                 decode_config.role_params, decode_config.role_num_gpus, decode_config.unique_role_extra_params, decode_config.envs, kv_ip)

            logger.info(f"Starting decode service(rank_id={service.rank_id}), url: {service.url}")
            service.actor_handler.start.remote()

            services.append(service)

        return services

    def start_vllm_services(self) -> Tuple[List[Service], List[Service]]:
        kv_ip, prefill_services = self._start_prefill_services()
        decode_services = self._start_decode_services(kv_ip)

        return prefill_services, decode_services
