# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from torch.distributed import ProcessGroup

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.tpu import USE_TPU_INFERENCE

from .base_device_communicator import DeviceCommunicatorBase

USE_RAY = parallel_config = (
    get_current_vllm_config().parallel_config.distributed_executor_backend == "ray"
)

logger = init_logger(__name__)

if not USE_TPU_INFERENCE:
    logger.info("tpu_inference not found, using vLLM's TpuCommunicator")
    if current_platform.is_tpu():
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from torch_xla._internal import pjrt
        from torch_xla.distributed.xla_multiprocessing import (
            create_optimized_replica_groups,
        )

        if USE_RAY:
            from vllm.v1.executor import ray_utils


class TpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

        # NOTE(woosuk): When using TP > 1 on TPUs, every TPU on the same node
        # must be used together. Therefore, the local rank and world size can
        # be simply calculated as follows.
        global_rank = self.global_rank
        global_world_size = self.global_world_size

        if USE_RAY:
            logger.info("TpuCommunicator initialized with RAY")
            # Calculate how many TPU nodes are in the current deployment. This
            # is the Ray placement group if it is deployed with Ray. Default
            # to the number of TPU nodes in the Ray cluster. The number of TPU
            # nodes is computed by the total number of TPUs divided by the
            # number of TPU accelerators per node, to account for clusters
            # with both CPUs and TPUs.
            num_nodes = ray_utils.get_num_tpu_nodes()
            num_nodes_in_pg = ray_utils.get_num_nodes_in_placement_group()
            if num_nodes_in_pg > 0:
                num_nodes = num_nodes_in_pg

            local_world_size = global_world_size // num_nodes
            local_rank = global_rank % local_world_size
        else:
            logger.info("TpuCommunicator initialized with MP")
            # Sanity: Verify we run on a single host
            num_hosts = torch_xla.tpu.num_tpu_workers()
            assert num_hosts == 1

            # Get the current number of TPUs (we have locally)
            local_world_size = torch_xla.tpu.num_available_chips()

            # Get current rank
            local_rank = global_rank % local_world_size

        # Ensure environment variables are set for multihost deployments.
        # On GKE, this is needed for libtpu and TPU driver to know which TPU
        # chip is actually visible. Otherwise the TPU driver will fail to
        # initialize because the number of devices would be different from
        # the number of visible worker addresses.
        os.environ["CLOUD_TPU_TASK_ID"] = str(global_rank)
        os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)

        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()
        self.groups = create_optimized_replica_groups()

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # TODO: Remove the groups specification after XLA compiler can support
        # auto-reordering the ring order for all-reduce.
        return xm.all_reduce(xm.REDUCE_SUM, input_, groups=self.groups)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(input_, dim=dim)
