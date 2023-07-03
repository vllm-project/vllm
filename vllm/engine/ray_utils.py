import random
from typing import List, Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

from vllm.config import ParallelConfig

# rank, node resource (node IP), device id
DeviceID = Tuple[int, Optional[str], int]


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, List[List[DeviceID]]]:
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `all_stage_devices`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `all_stage_devices` includes device IDs for
        each worker in each pipeline stage. Each device ID is a tuple of
        (rank, node resource, device id).
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install Ray to use distributed "
                "serving.")
        # Connect to a ray cluster.
        ray.init(address=ray_address)

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
        if (not node["Alive"]) or node["Resources"]["GPU"] <= 0:
            continue
        if num_devices_per_node is None:
            num_devices_per_node = node["Resources"]["GPU"]
        else:
            assert num_devices_per_node == node["Resources"]["GPU"], (
                "The number of GPUs per node is not uniform.")
        for key in node["Resources"]:
            if key.startswith("node:"):
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
