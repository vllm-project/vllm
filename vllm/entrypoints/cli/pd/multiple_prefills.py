# SPDX-License-Identifier: Apache-2.0
"""
Multiple Prefills Prefill-Decode job implementation.

This module provides:
- VllmService actor implementation for individual services
- MultiplePrefillsPDJob class for managing jobs with multiple Prefill services
- Service startup and configuration logic
"""

from typing import Tuple, List, Optional, Dict, Any

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.entrypoints.cli.pd.base import BasePDJob, _BaseVllmService, Service, PORT_PLACE_HOLDER, \
    KV_IP_PLACE_HOLDER, NODE_NAME_PLACE_HOLDER, ServiceType, RankIDGenerator
from vllm.entrypoints.cli.pd.config import Config
from vllm.logger import init_logger

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger(__name__)


@ray.remote
class _VllmService(_BaseVllmService):
    """
    Ray remote actor for vLLM service instances.
    
    Extends _BaseVllmService with KV connector IP handling and parameter
    replacement logic for Prefill-Decode disaggregated architecture.
    """
    def __init__(self,
                 job_id: str,
                 service_type: ServiceType,
                 rank_id: int,
                 role_params: Dict[str, Any],
                 unique_role_extra_params: str,
                 role_envs: Dict[str, str],
                 kv_ip: Optional[str]):
        super().__init__(job_id, service_type, rank_id, role_params, unique_role_extra_params, role_envs)

        # KV IP should only be None for rank 0 (first Prefill service)
        # Other services should receive the KV IP from the first Prefill service
        if kv_ip is None:
            assert self.rank_id == 0, "kv_ip should not be None"

        self.kv_ip = kv_ip

    def prepare_start_role_params(self) -> Dict[str, Any]:
        """
        Prepare role parameters for service startup.
        
        Replaces placeholders in role_params with actual values:
        - Host and port from socket
        - KV IP address (from first Prefill service or current host)
        - Port and node name in kv_transfer_config
        
        Returns:
            Dict[str, Any]: Updated role parameters with actual values
        """
        # Use ip_and_port() to ensure socket is created
        host, port = self.ip_and_port()
        
        # Replace host and port in role_params if specified
        if self.role_params.get("host"):
            self.role_params["host"] = str(host)
        if self.role_params.get("port"):
            self.role_params["port"] = str(port)

        # Process kv_transfer_config if present
        if "kv_transfer_config" in self.role_params:
            # Replace kv_ip placeholder with actual KV IP
            # Use provided kv_ip or fall back to current host (for rank 0)
            kv_ip = self.kv_ip or str(host)
            kv_config = self.role_params["kv_transfer_config"]
            kv_config = kv_config.replace(KV_IP_PLACE_HOLDER, kv_ip)

            # Replace port placeholder in kv_transfer_config
            kv_config = kv_config.replace(PORT_PLACE_HOLDER, str(port))
            
            # Replace node_name placeholder with Ray actor name
            actor_name = ray.get_runtime_context().get_actor_name()
            kv_config = kv_config.replace(NODE_NAME_PLACE_HOLDER, actor_name)
            
            self.role_params["kv_transfer_config"] = kv_config
            
        return self.role_params


class MultiplePrefillsPDJob(BasePDJob):
    """
    Prefill-Decode job implementation supporting multiple Prefill services.
    
    This class manages the lifecycle of multiple Prefill services and Decode
    services, handling their startup, configuration, and coordination.
    The first Prefill service acts as the KV store coordinator.
    """
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
        """
        Start a single vLLM service instance.
        
        Creates a placement group, configures scheduling strategy, and launches
        a _VllmService actor with the specified parameters.
        
        Args:
            rank_id: Unique rank ID for this service
            service_type: Type of service (Prefill or Decode)
            runtime_env: Runtime environment with environment variables
            role_params: Service-specific parameters
            role_num_gpus: Number of GPUs required for this role
            unique_role_extra_params: Additional command-line parameters
            role_envs: Environment variables for the service
            kv_ip: KV connector IP address (None for first Prefill service)
            
        Returns:
            Service: Service wrapper object with rank_id and actor handler
        """
        # Create placement group for resource allocation
        pg = self._create_placement_group(role_num_gpus)
        name = f"{self.job_id}-{service_type}-{rank_id}"

        # Configure scheduling strategy to use placement group
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )

        # Determine GPU allocation based on bundle count
        # If world_size is 1 (single bundle), allocate GPUs to the actor
        num_gpus = self.config.gpus_per_worker
        if pg.bundle_count > 1:
            # If world_size > 1 (multiple bundles), don't allocate GPUs here
            # GPUs will be managed by the placement group
            num_gpus = 0

        # Create and launch the service actor
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
        """
        Start all Prefill services.
        
        Launches multiple Prefill service instances. The first service's IP
        address is used as the KV connector IP for all subsequent services.
        
        Returns:
            Tuple[str, List[Service]]: 
                (KV IP address from first service, List of Prefill services)
        """
        logger.info("Starting prefill services")

        prefill_config = self.config.prefill_config

        # Create runtime environment if environment variables are specified
        runtime_env = None
        if prefill_config.envs:
            runtime_env = RuntimeEnv(env_vars=prefill_config.envs)

        services: List[Service] = []
        kv_ip = None
        
        # Start each Prefill service replica
        for idx in range(prefill_config.replicas):
            rank_id = RankIDGenerator.next_rank_id()
            service = self._start_single_service(
                rank_id, ServiceType.prefill, runtime_env,
                prefill_config.role_params, prefill_config.role_num_gpus,
                prefill_config.unique_role_extra_params, prefill_config.envs, kv_ip)
            
            # Extract KV IP from first Prefill service
            if idx == 0:
                host, _ = service.ip_and_port
                kv_ip = str(host)
            
            logger.info(f"Starting prefill service(rank_id={service.rank_id}), url: {service.url}")
            # Start the service asynchronously
            service.actor_handler.start.remote()

            services.append(service)

        return kv_ip, services

    def _start_decode_services(self, kv_ip: str) -> List[Service]:
        """
        Start all Decode services.
        
        Launches multiple Decode service instances. All Decode services use
        the KV IP address from the first Prefill service.
        
        Args:
            kv_ip: KV connector IP address from first Prefill service
            
        Returns:
            List[Service]: List of Decode services
        """
        logger.info("Starting decode services")

        decode_config = self.config.decode_config

        # Create runtime environment if environment variables are specified
        runtime_env = None
        if decode_config.envs:
            runtime_env = RuntimeEnv(env_vars=decode_config.envs)

        services: List[Service] = []
        
        # Start each Decode service replica
        for _ in range(decode_config.replicas):
            rank_id = RankIDGenerator.next_rank_id()
            service = self._start_single_service(
                rank_id, ServiceType.decode, runtime_env,
                decode_config.role_params, decode_config.role_num_gpus,
                decode_config.unique_role_extra_params, decode_config.envs, kv_ip)

            logger.info(f"Starting decode service(rank_id={service.rank_id}), url: {service.url}")
            # Start the service asynchronously
            service.actor_handler.start.remote()

            services.append(service)

        return services

    def start_vllm_services(self) -> Tuple[List[Service], List[Service]]:
        """
        Start all vLLM services (Prefill and Decode).
        
        First starts all Prefill services and extracts the KV IP from the first
        one. Then starts all Decode services using that KV IP.
        
        Returns:
            Tuple[List[Service], List[Service]]: 
                (List of Prefill services, List of Decode services)
        """
        # Start Prefill services first to get KV IP
        kv_ip, prefill_services = self._start_prefill_services()
        # Start Decode services using the KV IP from first Prefill service
        decode_services = self._start_decode_services(kv_ip)

        return prefill_services, decode_services
