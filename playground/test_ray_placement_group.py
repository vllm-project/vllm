import time
import os

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Initialize Ray.
import ray

class NormalActor:
    def __init__(self, index):
        self.index = index
        pass

    def log_message(self):
        import torch
        print("NormalActor", self.index, os.getpid(), torch.cuda.is_available(), ray.get_gpu_ids())

class AllocationActor:
    def __init__(self, pg):
        self.placement_group = pg
        self.a2 = ray.remote(num_cpus=1)(NormalActor).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=1,
            )
        ).remote(1)
        self.a3 = ray.remote(num_gpus=1, num_cpus=0)(NormalActor).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=2,
            )
        ).remote(2)

    def log_message(self):
        print("AllocationActor", os.getpid())
        ray.get([self.a2.log_message.remote(), self.a3.log_message.remote()])


def main():
    # Create a single node Ray cluster with 2 CPUs and 2 GPUs.
    ray.init(num_cpus=2, num_gpus=1)

    print(ray.cluster_resources())

    # Reserve a placement group of 1 bundle that reserves 1 CPU and 1 GPU.
    pg = placement_group([{"CPU": 1}, {"CPU": 1}, {"GPU": 1, "CPU": 0, "__internal_head__": 1e-5}])

    ray.get(pg.ready())
    a1 = ray.remote(num_cpus=1)(AllocationActor).options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        )
    ).remote(pg)

    ray.get(a1.log_message.remote())

main()
