import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import msgspec

from vllm.config import ParallelConfig
from vllm.executor.msgspec_utils import decode_hook, encode_hook
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import get_ip, is_hip, is_xpu
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

try:
    import ray
    from ray._private.state import available_resources_per_node
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup

    class RayWorkerWrapper(WorkerWrapperBase):
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Since the compiled DAG runs a main execution
            # in a different thread that calls cuda.set_device.
            # The flag indicates is set_device is called on
            # that thread.
            self.compiled_dag_cuda_device_set = False

            self.input_decoder = msgspec.msgpack.Decoder(ExecuteModelRequest,
                                                         dec_hook=decode_hook)
            self.output_encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)

        def get_node_ip(self) -> str:
            return get_ip()

        def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
            node_id = ray.get_runtime_context().get_node_id()
            gpu_ids = ray.get_gpu_ids()
            return node_id, gpu_ids

        def execute_model_spmd(
            self, req_or_tuple: Union[bytes,
                                      Tuple[bytes,
                                            Optional[IntermediateTensors]]]
        ) -> bytes:
            """Execute model in SPMD fashion: used only when SPMD worker and
            compiled DAG are both enabled.

            Args:
                req_or_tuple: A request or a tuple containing the
                    request and intermediate tensors. Intermediate tensors are
                    None unless if it is provided because it is > 0 pipeline
                    stage. The request is serialized by msgspec.
            """
            if isinstance(req_or_tuple, bytes):
                serialized_req, intermediate_tensors = req_or_tuple, None
            else:
                serialized_req, intermediate_tensors = req_or_tuple

            execute_model_req = self.input_decoder.decode(serialized_req)

            # TODO(swang): This is needed right now because Ray aDAG executes
            # on a background thread, so we need to reset torch's current
            # device.
            import torch
            if not self.compiled_dag_cuda_device_set:
                torch.cuda.set_device(self.worker.device)
                self.compiled_dag_cuda_device_set = True

            output = self.worker._execute_model_spmd(execute_model_req,
                                                     intermediate_tensors)
            # Pipeline model request and output to the next pipeline stage.
            if isinstance(output, IntermediateTensors):
                output = serialized_req, output
            else:
                output = self.output_encoder.encode(output)

            return output

    ray_import_err = None

except ImportError as e:
    ray = None  # type: ignore
    ray_import_err = e
    RayWorkerWrapper = None  # type: ignore


def ray_is_available() -> bool:
    """Returns True if Ray is available."""
    return ray is not None


def assert_ray_available():
    """Raise an exception if Ray is not available."""
    if ray is None:
        raise ValueError("Failed to import Ray, please install Ray with "
                         "`pip install ray`.") from ray_import_err


def _verify_bundles(placement_group: "PlacementGroup",
                    parallel_config: ParallelConfig, device_str: str):
    """Verify a given placement group has bundles located in the right place.

    There are 2 rules.
    - Warn if all tensor parallel workers cannot fit in a single node.
    - Fail if driver node is not included in a placement group.
    """
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray.")
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> List of bundle (e.g., {"GPU": 1})
    node_id_to_bundle: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    if driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` to see if you have available GPUs in a node "
            f"{driver_node_id} before starting an vLLM engine.")

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
                parallel_config.tensor_parallel_size, device_str, len(bundles),
                device_str, node_id, parallel_config.tensor_parallel_size)


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
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
            "%d seconds. specs=%s. Check "
            "`ray status` to see if you have enough resources.",
            int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError(
            "Cannot provide a placement group of "
            f"{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See "
            "`ray status` to make sure the cluster has enough resources."
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
            "Waiting for removing a placement group of specs for "
            "%d seconds.", int(time.time() - s))
        time.sleep(wait_interval)


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
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

    # Connect to a ray cluster.
    if is_hip() or is_xpu():
        ray.init(address=ray_address,
                 ignore_reinit_error=True,
                 num_gpus=parallel_config.world_size)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True)

    if parallel_config.placement_group:
        # Placement group is already set.
        return

    device_str = "GPU" if not current_platform.is_tpu() else "TPU"
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 "
                    f"{device_str}.")
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group."
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}.")
    else:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if parallel_config.world_size > num_devices_in_cluster:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group.")
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(parallel_config.world_size)])

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
