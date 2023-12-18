import time

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Initialize Ray.
import ray

# Create a single node Ray cluster with 2 CPUs and 2 GPUs.
ray.init(num_cpus=2, num_gpus=1)

# Reserve a placement group of 1 bundle that reserves 1 CPU and 1 GPU.
pg = placement_group([{"CPU": 1}, {"CPU": 1}, {"GPU": 1}])

ray.get(pg.ready())

# You can look at placement group states using this API.
print(placement_group_table(pg))
