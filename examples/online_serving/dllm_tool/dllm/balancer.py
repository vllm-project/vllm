import asyncio
import logging
import time
from typing import Dict, List, Optional

import aiohttp
import ray
from prometheus_client.parser import text_string_to_metric_families
from ray import actor

from dllm import constants
from dllm.constants import CONTROLLER_ACTOR_NAME, DLLM_NAMESPACE
from dllm.entities import (DispatchResult, MetricsInfo, Role, SchedulerPolicy,
                           VllmInstanceInfo, VllmInstanceStatus)

logger = logging.getLogger(__name__)


class Balancer:

    def __init__(
        self,
        policy: SchedulerPolicy = SchedulerPolicy.ROUND_ROBIN,
    ):
        self.policy = policy
        self.role_2_instances: Dict[Role, List[VllmInstanceInfo]] = {
        }  # prefill/decode/mixed => VllmInstanceInfo
        self.instance_infos: Dict[str, VllmInstanceInfo] = {
        }  # id -> VllmInstanceInfo
        self.instance_metrics: Dict[str, MetricsInfo] = {}  # id -> MetricsInfo
        self._round_robin_index_p = 0
        self._round_robin_index_d = 0
        self._round_robin_index_m = 0
        self.last_heartbeat: Dict[str, float] = {}
        self._controller_handle: Optional[actor.ActorHandle] = None
        self.all_instances_ready = False
        # start update metrics loop
        loop = asyncio.get_event_loop()
        loop.create_task(self.update_vllm_instance_metrics())

    async def update_vllm_instance_metrics(self):
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    await asyncio.gather(
                        *[
                            self._query_instance_metrics(
                                session, instance_info)
                            for instance_info in self.instance_infos.values()
                            if instance_info.uri is not None
                        ],
                        return_exceptions=True,
                    )
                await asyncio.sleep(constants.METRICS_UPDATE_CYCLE)
            except Exception as e:
                logger.error("create request session error: %s", e)

    def dispatch_request(self) -> DispatchResult:
        if self.policy == SchedulerPolicy.ROUND_ROBIN:
            return self._round_robin_pair()
        else:
            raise ValueError(f"Unsupported policy: {self.policy}")

    def get_all_instance(self) -> Dict[str, VllmInstanceInfo]:
        '''Return all vllm instance.'''
        return self.instance_infos

    async def _query_instance_metrics(self, session, instance_info):
        ins_uri = instance_info.uri
        ins_id = instance_info.id
        async with session.post(f"{ins_uri}/metrics", timeout=3) as resp:
            resp_code = resp.status
            if resp_code != constants.HTTP_OK:
                logger.error(
                    f"get metrics failed, uri:{ins_uri}, code:{resp_code}")
                return
            resp_body = await resp.text()
            # {metric_name: metric_value}
            metrics_dict = {
                metric_family.name: metric_family.samples[0].value
                for metric_family in text_string_to_metric_families(resp_body)
                if metric_family.name in MetricsInfo.METRIC_NAME_MAPPING.
                values() and metric_family.samples
            }
            if not metrics_dict:
                return
            if ins_id not in self.instance_metrics:
                self.instance_metrics[ins_id] = MetricsInfo()
            metric_info = self.instance_metrics[ins_id]
            for param_name, metric_name in MetricsInfo.METRIC_NAME_MAPPING.items(
            ):
                if metric_name not in metrics_dict:
                    continue
                # data type conversion
                target_type = metric_info.__annotations__[param_name]
                setattr(metric_info, param_name,
                        target_type(metrics_dict[metric_name]))
            logger.debug("instance metrics info: %s", self.instance_metrics)

    def _round_robin_pair(self) -> DispatchResult:
        # current policy: if has mixed, use mixed
        is_pd_disagged = Role.MIXED not in self.role_2_instances or len(
            self.role_2_instances[Role.MIXED]) == 0
        if not is_pd_disagged:
            mixed_uri = self._round_robin_selection(Role.MIXED)
            return DispatchResult(prefill_uri=None, decode_uri=mixed_uri)

        prefill_uri = self._round_robin_selection(Role.PREFILL)
        decode_uri = self._round_robin_selection(Role.DECODE)
        return DispatchResult(prefill_uri=prefill_uri, decode_uri=decode_uri)

    def _round_robin_selection(self, role: Role) -> str:
        instances = [
            item.uri for i, item in self.instance_infos.items()
            if item.role == role and item.uri is not None
        ]
        if role == Role.PREFILL:
            instance = instances[self._round_robin_index_p]
            self._round_robin_index_p = (self._round_robin_index_p +
                                         1) % len(instances)
        if role == Role.DECODE:
            instance = instances[self._round_robin_index_d]
            self._round_robin_index_d = (self._round_robin_index_d +
                                         1) % len(instances)
        if role == Role.MIXED:
            instance = instances[self._round_robin_index_m]
            self._round_robin_index_m = (self._round_robin_index_m +
                                         1) % len(instances)
        return instance

    def update_vllm_instance_info(self, infos: List[VllmInstanceInfo]):
        for item in infos:
            self.instance_infos[item.id] = item
            self.instance_metrics[item.id] = MetricsInfo()

        # reconstruct the role map
        self.role_2_instances.clear()
        for _, instance_info in self.instance_infos.items():
            if instance_info.role not in self.role_2_instances:
                self.role_2_instances[instance_info.role] = []
            self.role_2_instances[instance_info.role].append(instance_info)

    async def update_vllm_instance_health(
            self, vllm_instance_info: List[VllmInstanceInfo]) -> bool:
        """
        Update health status of VLLM instances.

        Args:
            vllm_instance_info: List of VllmInstanceInfo objects containing information

        Returns:
            bool: True if update was successful
        """

        current_time = time.time()
        for item in vllm_instance_info:
            self.instance_infos[item.id] = item
            self.last_heartbeat[item.id] = current_time
        return True

    async def is_all_instances_ready(self):
        """
        Wait until all VLLM actor running status

        Returns:
            No return value. End of function when all actor ready,.
        """
        if not self._controller_handle:
            try:
                self._controller_handle = ray.get_actor(
                    name=CONTROLLER_ACTOR_NAME, namespace=DLLM_NAMESPACE)
            except BaseException:
                logger.error('get _controller_handle fail')
        _get_expected_vllm_actors_num = await self._controller_handle._get_expected_vllm_actors_num.remote(  # type: ignore # ray remote call
        )
        while self._get_ready_vllm_actors_num(
        ) < _get_expected_vllm_actors_num:
            try:
                logger.debug(
                    f"expect {self._get_ready_vllm_actors_num()} waiting vllm actor, "
                    f"{self.instance_infos}")
                for s in self.instance_infos.values():
                    if s.status == VllmInstanceStatus.SUBPROCESS_EXITED:
                        raise RuntimeError(
                            f"vllm instance: {s} exited unexpectedly")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(
                    f"An error when waiting vllm instances ready: {e}")
                return
        logger.info("All actors are already")
        self.all_instances_ready = True
        asyncio.create_task(self._monitor_instance_health())

    def _get_ready_vllm_actors_num(self):
        """
        Get the number of ready VLLM instances.

        Returns:
            Number of ready VLLM instances.
        """
        return sum(info.status == VllmInstanceStatus.RUNNING
                   for info in self.instance_infos.values())

    def _get_unready_vllm_actors_num(self):
        """
        Get the number of unready VLLM instances.

        Returns:
            Number of unready VLLM instances.
        """
        return sum(info.status != VllmInstanceStatus.RUNNING
                   for info in self.instance_infos.values())

    async def _monitor_instance_health(self):
        """
        Monitor instance health, report to controller if >20s no response / failed status
        """
        while True:
            if self.all_instances_ready:
                current_time = time.time()
                for info in self.instance_infos.values():
                    logger.info(
                        f"Monitoring ID: {info.id}, Status: {info.status}")
                    if info.status == VllmInstanceStatus.HEALTHCHECK_FAILED:
                        logger.error(
                            f"Instance {info.id} has failed health check.")
                        self._controller_handle.report_failure_from_balancer.remote(  # type: ignore # ray remote call
                            info.id)
                        self.all_instances_ready = False
                    # Consider instance unhealthy if no heartbeat
                    elif current_time - self.last_heartbeat.get(info.id,
                                                                0) > 20:
                        logger.error(
                            f"Instance {info.id} is unhealthy (no heartbeat).")
                        self._controller_handle.report_failure_from_balancer.remote(  # type: ignore # ray remote call
                            info.id)
                        self.all_instances_ready = False
            else:
                logger.info(
                    "Waiting for all instances ready and restart the health monitoring."
                )
                await asyncio.sleep(5)
            await asyncio.sleep(1)
