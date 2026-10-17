# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections import defaultdict

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

# Env vars that are worker-specific and must NOT be copied from the
# driver to Ray workers — they are set per-worker after GPU discovery.
WORKER_SPECIFIC_ENV_VARS: set[str] = {
    "VLLM_HOST_IP",
    "VLLM_HOST_PORT",
    "VLLM_NIXL_SIDE_CHANNEL_HOST",
    "LOCAL_RANK",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup

    ray_import_err = None

except ImportError as e:
    ray = None  # type: ignore
    # only capture string to avoid variable references in the traceback that can
    # prevent garbage collection in some cases
    ray_import_err = str(e)


def ray_is_available() -> bool:
    """Returns True if Ray is available."""
    return ray is not None


def assert_ray_available():
    """Raise an exception if Ray is not available."""
    if ray is None:
        raise ValueError(
            f"Failed to import Ray: {ray_import_err}."
            "Please install Ray with `pip install ray`."
        )


def _verify_bundles(
    placement_group: "PlacementGroup",
    parallel_config: ParallelConfig,
    device_str: str,
):
    """Warn if all tensor parallel workers cannot fit in a single node."""
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray."
    )
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> List of bundle (e.g., {"GPU": 1})
    node_id_to_bundle: dict[str, list[dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])

    for node_id, bundles in node_id_to_bundle.items():
        if len(bundles) < parallel_config.tensor_parallel_size:
            logger.warning(
                "tensor_parallel_size=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Tensor parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.",
                parallel_config.tensor_parallel_size,
                device_str,
                len(bundles),
                device_str,
                node_id,
                parallel_config.tensor_parallel_size,
            )


def build_actor_name(
    instance_id: str,
    rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
) -> str:
    """Build a descriptive Ray actor name for dashboard visibility."""
    name = f"vllm_Worker_{instance_id}"
    if tp_size > 1:
        name += f"_TP{rank % tp_size}"
    if pp_size > 1:
        name += f"_PP{(rank // tp_size) % pp_size}"
    if pcp_size > 1:
        name += f"_PCP{rank // (tp_size * pp_size)}"
    return name


def get_bundles_for_indices(
    placement_group: "PlacementGroup",
    bundle_indices: list[int],
    world_size: int,
) -> list[tuple[int, str, str]]:
    """Return GPU bundle indices paired with node IDs and node IPs for
    explicit bundle indices specified via VLLM_RAY_BUNDLE_INDICES.
    """
    assert len(bundle_indices) == world_size, (
        "VLLM_RAY_BUNDLE_INDICES must have the same size"
        f" as the world size, but got {bundle_indices=} "
        f"and {world_size=}"
    )
    assert len(set(bundle_indices)) == len(bundle_indices), (
        "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
        f" but got {bundle_indices=}"
    )

    pg_data = placement_group_table(placement_group)
    pg_bundle_to_node = pg_data["bundles_to_node_id"]
    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }
    return [
        (bid, pg_bundle_to_node[bid], node_id_to_ip[pg_bundle_to_node[bid]])
        for bid in bundle_indices
    ]


def get_bundles_sorted_by_node(
    placement_group: "PlacementGroup",
) -> list[tuple[int, str, str]]:
    """Return GPU bundle indices paired with node IDs and node IPs,
    sorted driver-first.

    This utility has to be invoked from the driver node.

    Example: 3-node cluster, driver on node-A, PG bundles spread
    across nodes:

      Input: [
          (0, node-C),
          (1, node-A),
          (2, node-B),
          (3, node-C),
          (4, node-A),
          (5, node-B),
      ]
      Output: [
          (1, node-A),
          (4, node-A),
          (2, node-B),
          (5, node-B),
          (0, node-C),
          (3, node-C),
      ]
    """
    pg_data = placement_group_table(placement_group)
    bundle_to_node = pg_data["bundles_to_node_id"]

    ray_device_key = current_platform.ray_device_key
    if not ray_device_key:
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )

    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }

    bundle_specs = placement_group.bundle_specs
    assert bundle_specs is not None
    bundle_to_node_id: list[tuple[int, str, str]] = []
    for bundle_idx, bundle in enumerate(bundle_specs):
        if bundle.get(ray_device_key):
            node_id = bundle_to_node.get(bundle_idx)
            bundle_to_node_id.append((bundle_idx, node_id, node_id_to_ip[node_id]))

    driver_node = ray.get_runtime_context().get_node_id()

    def _sort_key(item):
        _, node_id, _ = item
        return (0 if node_id == driver_node else 1, node_id)

    bundle_to_node_id.sort(key=_sort_key)

    return bundle_to_node_id


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will time out
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check `ray status` and "
            "`ray list nodes` to see if you have enough resources,"
            " and make sure the IP addresses used by ray cluster"
            " are the same as VLLM_HOST_IP environment variable"
            " specified in each node if you are running on a multi-node.",
            int(time.time() - s),
            placement_group_specs,
        )

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        # Provide more helpful error message when GPU count is exceeded
        total_gpu_required = sum(spec.get("GPU", 0) for spec in placement_group_specs)
        # If more than one GPU is required for the placement group, provide a
        # more specific error message.
        # We use >1 here because multi-GPU (tensor parallel) jobs are more
        # likely to fail due to insufficient cluster resources, and users may
        # need to adjust tensor_parallel_size to fit available GPUs.
        if total_gpu_required > 1:
            raise ValueError(
                f"Cannot provide a placement group requiring "
                f"{total_gpu_required} GPUs "
                f"(placement_group_specs={placement_group_specs}) within "
                f"{PG_WAIT_TIMEOUT} seconds.\n"
                f"Tensor parallel size may exceed available GPUs in your "
                f"cluster. Check resources with `ray status` and "
                f"`ray list nodes`.\n"
                f"If running on K8s with limited GPUs, consider reducing "
                f"--tensor-parallel-size to match available GPU resources."
            ) from None
        else:
            raise ValueError(
                "Cannot provide a placement group of "
                f"{placement_group_specs=} within "
                f"{PG_WAIT_TIMEOUT} seconds. See "
                "`ray status` and `ray list nodes` to make sure the cluster "
                "has enough resources."
            ) from None


def _wait_until_pg_removed(current_placement_group: "PlacementGroup"):
    ray.util.remove_placement_group(current_placement_group)
    s = time.time()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        pg = ray.util.get_current_placement_group()
        if pg is None:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for removing a placement group of specs for %d seconds.",
            int(time.time() - s),
        )
        time.sleep(wait_interval)


def _check_dp_backend_supports_cluster(
    parallel_config: ParallelConfig,
    driver_node_device_count: int,
    device_str: str,
) -> None:
    """Fail fast for the unsupported mp DP backend + Ray executor case.

    The mp data-parallel backend launches every local DP engine on the
    driver node and slices CUDA_VISIBLE_DEVICES assuming all DP groups'
    devices are local and contiguously indexed on that node. The cross-DP
    rendezvous address and rank-0 placement rely on the same assumption.
    When the Ray executor must spread workers beyond the driver node these
    assumptions break and startup hangs or crashes. Multi-node data
    parallelism must use the Ray data-parallel backend instead.
    """
    if (
        parallel_config.data_parallel_size <= 1
        or parallel_config.data_parallel_backend != "mp"
    ):
        return
    total_devices = parallel_config.world_size * parallel_config.data_parallel_size
    if total_devices > driver_node_device_count:
        raise ValueError(
            "Multi-node data parallelism (data_parallel_size="
            f"{parallel_config.data_parallel_size}) with the Ray distributed "
            "executor requires --data-parallel-backend=ray. The 'mp' "
            "data-parallel backend assumes all DP engines and their devices "
            f"are on the driver node, which has only "
            f"{driver_node_device_count} {device_str}(s) but "
            f"{total_devices} are required; startup would hang or crash. "
            "Re-run with --data-parallel-backend=ray."
        )


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: str | None = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    """
    assert_ray_available()
    from vllm.platforms import current_platform

    # Disable Ray usage stats collection
    if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

    # Prevalidate GPU requirements before Ray processing
    if current_platform.is_cuda() and parallel_config.world_size > 1:
        available_gpus = current_platform.device_count()
        if parallel_config.world_size > available_gpus:
            logger.warning(
                "Tensor parallel size (%d) exceeds available GPUs (%d). "
                "This may result in Ray placement group allocation failures. "
                "Consider reducing tensor_parallel_size to %d or less, "
                "or ensure your Ray cluster has %d GPUs available.",
                parallel_config.world_size,
                available_gpus,
                available_gpus,
                parallel_config.world_size,
            )

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # Try to connect existing ray instance and create a new one if not found
        try:
            ray.init("auto")
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources."
            )
            ray.init(
                address=ray_address,
                num_gpus=current_platform.device_count(),
                runtime_env=parallel_config.ray_runtime_env,
            )
    else:
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)

    device_str = current_platform.ray_device_key
    if not device_str:
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )

    # Create or get the placement group for worker processes
    if parallel_config.placement_group:
        current_placement_group = parallel_config.placement_group
    else:
        current_placement_group = ray.util.get_current_placement_group()

    if current_placement_group:
        logger.info("Using the existing placement group")

        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    f"Placement group bundle cannot have more than 1 {device_str}."
                )
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}."
            )
    else:
        logger.info("No current placement group found. Creating a new placement group.")
        current_node_id = ray.get_runtime_context().get_node_id()
        driver_node_device_count = next(
            (
                int(node.get("Resources", {}).get(device_str, 0))
                for node in ray.nodes()
                if node.get("Alive") and node.get("NodeID") == current_node_id
            ),
            0,
        )
        _check_dp_backend_supports_cluster(
            parallel_config, driver_node_device_count, device_str
        )
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        if parallel_config.world_size > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.",
                device_str,
                device_str,
            )
        # Create a new placement group
        placement_group_specs: list[dict[str, float]] = [
            {device_str: 1.0} for _ in range(parallel_config.world_size)
        ]

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK"
        )
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
