import asyncio
import itertools
import logging
from typing import Dict, List, Optional

import ray
from ray import actor

from dllm.balancer import Balancer
from dllm.config import (ControllerConfig, DPConfig, EPConfig,
                         InferenceInstanceConfig, PDDistConfig,
                         VllmInstanceConfig)
from dllm.constants import BALANCER_ACTOR_NAME
from dllm.entities import Role, VllmInstanceInfo
from dllm.vllm_instance import start_vllm_instance

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


def flatten_list(multi_level_list):
    return list(itertools.chain(*multi_level_list))


def _get_accelerator_num_per_ray_node():
    accelerator_nums = []
    for e in ray.nodes():
        num = e.get("Resources", {}).get(current_platform.device_name, None)
        if num:
            accelerator_nums.append(int(num))
    return max(accelerator_nums)


def split_dp_resources(tp_size: int,
                       dp_size: int,
                       accelerators_pack_max_size: int = 8) -> List[int]:
    """
    pack DP instances into nodes, prevent cross-node DP instance at best effort
    | DP     | TP     | total  | 910C   | 910B   |
    | ------ | ------ | ------ | ------ | ------ |
    | 4      | 2      | 8      | 8      | 8      |
    | 3      | 3      | 9      | 9      | 6+3    |
    | 4      | 4      | 16     | 16     | 8+8    |
    | 32     | 1      | 32     | 16+16  | 8x4    |
    | 64     | 1      | 64     | 16x4   | 8x8    |

    TODO: optimize resource fragments
    """
    assert tp_size <= accelerators_pack_max_size, (
        f"do not allow TP size to exceed the number of accelerators on a single node {accelerators_pack_max_size}"
    )
    total_accelerators = dp_size * tp_size
    group_size = (accelerators_pack_max_size - (accelerators_pack_max_size % tp_size)
                  if accelerators_pack_max_size % tp_size != 0 else accelerators_pack_max_size)
    num_groups = total_accelerators // group_size
    remainder = total_accelerators % group_size
    packs = [group_size * num_groups]
    if remainder > 0:
        packs.append(remainder)
    return packs


async def make_dp_group(pd_role: Role,
                        pd_idx: int,
                        tp_size: int,
                        dp_size: int,
                        ep_size: int,
                        start_params: List[str],
                        env: Optional[str] = None) -> List[actor.ActorHandle]:
    """
    prepare one DP group
    1. start DP master vllm instance
        1.1. find DP master ip and a free port as DP master port
        1.2. init DP master vllm instance's DP config
    2. start all other DP instances and pass through DP master ip and port
    """
    packs = split_dp_resources(tp_size=tp_size,
                               dp_size=dp_size,
                               accelerator_pack_max_size=_get_accelerator_num_per_ray_node())
    pg = ray.util.placement_group(bundles=[{
        current_platform.device_name: p
    } for p in packs],
                                  strategy="PACK",
                                  name=f"DP-{pd_role}-{pd_idx}")
    await pg.ready()

    actors = []
    dp_master_vllm_instance_config = VllmInstanceConfig(
        exec_cmd=start_params,
        env=env,
        tp=tp_size,
        pd_config=PDDistConfig(role=pd_role, pd_rank=pd_idx),
        dp_config=DPConfig(dp_rank=0,
                           dp_size=dp_size,
                           dp_local_size=packs[0] // tp_size,
                           dp_master_ip="",
                           dp_master_port=0),
        ep_config=EPConfig(ep_size=ep_size),
    )
    dp_master_actor = start_vllm_instance(
        vllm_instance_config=dp_master_vllm_instance_config, pg=pg)
    actors.append(dp_master_actor)

    dp_master_ip, dp_master_port = await dp_master_actor.init_dp_master_ip_port.remote(
    )  # type: ignore # ray remote call
    dp_master_vllm_instance_config.dp_config.dp_master_ip = dp_master_ip  # type: ignore
    dp_master_vllm_instance_config.dp_config.dp_master_port = dp_master_port  # type: ignore
    await dp_master_actor.init_dp_config.remote(
        dp_master_vllm_instance_config.dp_config
    )  # type: ignore # ray remote call

    dp_rank = packs[0] // tp_size
    for idx in range(1, len(packs)):
        dp_vllm_instance_config = VllmInstanceConfig(
            exec_cmd=start_params,
            env=env,
            tp=tp_size,
            pd_config=PDDistConfig(role=pd_role, pd_rank=pd_idx),
            dp_config=DPConfig(
                dp_rank=dp_rank,
                dp_size=dp_size,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                dp_local_size=packs[idx] // tp_size,
            ),
            ep_config=EPConfig(ep_size=ep_size),
        )
        dp_rank += packs[idx] // tp_size
        actor = start_vllm_instance(
            vllm_instance_config=dp_vllm_instance_config, pg=pg)
        await actor.init_dp_config.remote(dp_vllm_instance_config.dp_config
                                          )  # type: ignore # ray remote call
        actors.append(actor)
    return actors


class Controller:

    def __init__(self, controller_config: ControllerConfig):
        """
        Initialize the global controller.

        Args:
            controller_config: ControllerConfig
        """
        self.config = controller_config

        self.p_instances_actors: List[actor.ActorHandle] = []
        self.d_instances_actors: List[actor.ActorHandle] = []
        self.vllm_instances_info: Dict[str, VllmInstanceInfo] = {}
        self.balancer = None

    def _get_expected_vllm_actors_num(self):
        return len(self.p_instances_actors) + len(self.d_instances_actors)

    async def make_inference_instance(
        self, pd_role: Role, pd_rank: int,
        inference_instance_config: InferenceInstanceConfig
    ) -> List[actor.ActorHandle]:
        """make inference instance (PREFILL instance, or DECODE instance)
        1. if dp enabled,     ==> start dp group
        2. if dp not enabled, ==> just start vllm instance

        Returns:
            all vllm instances actors in this inference instance
        """
        if inference_instance_config.dp and inference_instance_config.dp > 1:
            if not inference_instance_config.tp:
                inference_instance_config.tp = 1
            # enable dp
            return await make_dp_group(
                pd_role=pd_role,
                pd_idx=pd_rank,
                tp_size=inference_instance_config.tp,
                dp_size=inference_instance_config.dp,
                ep_size=inference_instance_config.ep,
                start_params=inference_instance_config.startup_params,
                env=inference_instance_config.startup_env,
            )

        # no dp
        return [
            start_vllm_instance(
                VllmInstanceConfig(
                    exec_cmd=inference_instance_config.startup_params,
                    env=inference_instance_config.startup_env,
                    tp=inference_instance_config.tp,
                    pd_config=PDDistConfig(role=pd_role, pd_rank=pd_rank),
                    dp_config=DPConfig(),
                    ep_config=EPConfig(inference_instance_config.ep),
                ))
        ]

    async def make_balancer(self) -> List[actor.ActorHandle]:
        """make balancer, and send all vllm instance info to the balancer

        Returns:
            balancer handle
        """
        balancer = ray.remote(Balancer).options(
            name=BALANCER_ACTOR_NAME).remote(
                self.config.scheduler_policy)  # type: ignore # ray remote call
        return balancer

    async def initialize(self):
        """initialize all vllm instances, construct pd/dp groups"""
        # TODO: Need to implement the resource checking logic with Ray
        logger.info(f"initialize with config: {self.config}")
        # Dictionary to track VLLM instances health status
        self.vllm_instances_info: Dict[str, VllmInstanceInfo] = {}  #

        # start VllmInstance
        # start Prefill Instances
        is_disaggregated_pd = self.config.num_prefill_instances > 0 and self.config.num_decode_instances > 0
        for p_pd_rank in range(self.config.num_prefill_instances):
            p_actors = self.make_inference_instance(
                pd_rank=p_pd_rank,
                pd_role=Role.PREFILL if is_disaggregated_pd else Role.MIXED,
                inference_instance_config=self.config.prefill_instance_config,
            )
            self.p_instances_actors.extend(await p_actors)

        # start Decode Instances
        for d_pd_rank in range(self.config.num_decode_instances):
            d_actors = self.make_inference_instance(
                pd_rank=d_pd_rank,
                pd_role=Role.DECODE if is_disaggregated_pd else Role.MIXED,
                inference_instance_config=self.config.decode_instance_config,
            )
            self.d_instances_actors.extend(await d_actors)

        logger.info("Create Balancer")
        self.balancer = await self.make_balancer()

        # init all vllm instances
        # TODO how to handle restart for reliability issues
        for vllm_instance_actor in [
                *self.p_instances_actors, *self.d_instances_actors
        ]:
            vllm_instance_actor.initialize.remote()

        # wait for all instances ready
        await self.balancer.is_all_instances_ready.remote(  # type: ignore # ray remote call
        )

        logger.info(
            f"All instances ready, VllmInstance num: {len(self.vllm_instances_info)}, updating Balancer"
        )

        # update Balancer
        self.balancer.update_vllm_instance_info.remote(  # type: ignore # ray remote call
            list(self.vllm_instances_info.values()))

        # TODO start Endpoint actors on each node:: deploy_endpoint_to_cluster()
        logger.info(
            f"Controller initialized with {self.config.num_prefill_instances} P instances and "
            f"{self.config.num_decode_instances} D instances")

    async def terminate(self, timeout_s=5):
        """
        TODO: clean all dllm actors started by controller
        """
        if self.balancer:
            ray.kill(self.balancer)

        terminate_futures = []
        for instance_actor in [
                *self.p_instances_actors, *self.d_instances_actors
        ]:
            terminate_futures.append(
                instance_actor.terminate.remote(timeout_s=timeout_s))
        await asyncio.gather(*terminate_futures)

        for instance_actor in [
                *self.p_instances_actors, *self.d_instances_actors
        ]:
            ray.kill(instance_actor)

    # TODO: Need to implement method to monitor and restart failed instances

    def report_failure_from_balancer(self, instance_id):
        """
        Report fail instance from balancer

        Returns:
            No Return required
        """
        logger.info(
            f"Received report from balancer, instance_id is {instance_id} ")
        return True
