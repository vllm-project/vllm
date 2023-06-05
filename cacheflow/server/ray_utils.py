import random
from typing import List, Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

from cacheflow.config import ParallelConfig

DeviceID = Tuple[int, Optional[str], int]  # rank, node resource (node IP), device id


def initialize_cluster(
    parallel_config: ParallelConfig,
    server_use_ray: bool = False,
    address: Optional[str] = None,
) -> Tuple[str, List[List[DeviceID]]]:
    if parallel_config.worker_use_ray or server_use_ray:
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install Ray to use distributed "
                "serving.")
        # Connect to a ray cluster.
        ray.init(address=address)

    if not parallel_config.worker_use_ray:
        # Initialize cluster locally.
        port = random.randint(10000, 20000)
        # We need to setup the distributed init method to make sure
        # the distributed megatron code (e.g., get world size) works correctly.
        distributed_init_method = f"tcp://localhost:{port}"
        all_stage_devices = [[(0, None, 0)]]
        return distributed_init_method, all_stage_devices

    # Assume we have a uniform cluster that each node has the same number of
    # GPUs for now.
    valid_node_resources = []
    num_devices_per_node = None
    for node in ray.nodes():
        if (not node['Alive']) or node['Resources']['GPU'] <= 0:
            continue
        if num_devices_per_node is None:
            num_devices_per_node = node['Resources']['GPU']
        else:
            assert num_devices_per_node == node['Resources']['GPU'], (
                "The number of GPUs per node is not uniform.")
        for key in node['Resources']:
            if key.startswith('node:'):
                valid_node_resources.append(key)

    # Verify the parallel config.
    num_nodes = len(valid_node_resources)
    if parallel_config.world_size > num_nodes * num_devices_per_node:
        raise ValueError(
            "The number of required GPUs exceeds the total number of "
            "available GPUs.")
    if parallel_config.tensor_parallel_size >= num_devices_per_node:
        if parallel_config.tensor_parallel_size % num_devices_per_node != 0:
            raise ValueError(
                "The number of tensor parallelism is not divisible by the "
                "number of GPUs per node.")
    else:
        if num_devices_per_node % parallel_config.tensor_parallel_size != 0:
            raise ValueError(
                "The number of GPUs per node is not divisible by the number "
                "of tensor parallelism.")

    # Assign GPUs to pipeline stages.
    rank = 0
    current_node_id = 0
    current_device_id = 0
    distributed_init_method = None
    all_stage_devices = []

    for _ in range(parallel_config.pipeline_parallel_size):
        stage_devices = []
        for _ in range(parallel_config.tensor_parallel_size):
            node_resource = valid_node_resources[current_node_id]
            stage_devices.append((rank, node_resource, current_device_id))
            if distributed_init_method is None:
                ip = node_resource.split("node:")[-1]
                port = random.randint(10000, 20000)
                distributed_init_method = f"tcp://{ip}:{port}"
            rank += 1
            current_device_id += 1
            if current_device_id >= num_devices_per_node:
                current_node_id += 1
                current_device_id = 0
        all_stage_devices.append(stage_devices)

    return distributed_init_method, all_stage_devices
