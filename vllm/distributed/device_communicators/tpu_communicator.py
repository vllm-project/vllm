import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform

if current_platform.is_tpu():
    import ray
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from ray._private.accelerators import TPUAcceleratorManager
    from torch_xla._internal import pjrt


class TpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_tpu():
            self.disabled = True
            return
        self.disabled = False

        # NOTE(woosuk): When using TP > 1 on TPUs, every TPU on the same node
        # must be used together. Therefore, the local rank and world size can
        # be simply calculated as follows.
        global_rank = dist.get_rank(group)
        global_world_size = dist.get_world_size(group)

        # Calculate how many TPU nodes are in the current deployment. This
        # is the Ray placement group if it is deployed with Ray. Default
        # to the number of TPU nodes in the Ray cluster. The number of TPU
        # nodes is computed by the total number of TPUs divided by the
        # number of TPU accelerators per node, to account for clusters
        # with both CPUs and TPUs.
        cluster_resources = ray.cluster_resources()
        total_tpus = int(cluster_resources["TPU"])
        tpus_per_node = (
            TPUAcceleratorManager.get_current_node_num_accelerators())
        num_nodes = total_tpus // tpus_per_node

        pg_table = ray.util.placement_group_table()
        current_pg = ray.util.get_current_placement_group()

        if current_pg:
            nodes_in_pg = set()
            for pg_key, pg in pg_table.items():
                if pg_key == current_pg.id.hex():
                    for _, node in pg["bundles_to_node_id"].items():
                        nodes_in_pg.add(node)
            num_nodes = len(nodes_in_pg)

        local_world_size = global_world_size // num_nodes
        local_rank = global_rank % local_world_size

        # Ensure environment variables are set for multihost deployments.
        os.environ["CLOUD_TPU_TASK_ID"] = str(global_rank)
        os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)

        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, x)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "TPUs only support dim=-1 for all-gather."
        return xm.all_gather(x, dim=dim)
