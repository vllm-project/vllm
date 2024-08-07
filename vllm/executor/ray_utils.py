from typing import List, Optional, Tuple, Union

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import get_ip, is_hip, is_tpu, is_xpu
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

try:
    import ray

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

        def get_node_ip(self) -> str:
            return get_ip()

        def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
            node_id = ray.get_runtime_context().get_node_id()
            gpu_ids = ray.get_gpu_ids()
            return node_id, gpu_ids

        def execute_model_spmd(
            self, req_or_tuple: Union[ExecuteModelRequest,
                                      Tuple[ExecuteModelRequest,
                                            IntermediateTensors]]):
            """Execute model in SPMD fashion: used only when SPMD worker and
            compiled DAG are both enabled.

            Args:
                req_or_tuple: The request to execute the model, or a tuple
                    containing the request and intermediate tensors.
            """
            # TODO(swang): This is needed right now because Ray aDAG executes
            # on a background thread, so we need to reset torch's current
            # device.
            import torch
            if not self.compiled_dag_cuda_device_set:
                torch.cuda.set_device(self.worker.device)
                self.compiled_dag_cuda_device_set = True

            if isinstance(req_or_tuple, tuple):
                execute_model_req, intermediate_tensors = req_or_tuple
            else:
                execute_model_req = req_or_tuple
                intermediate_tensors = None

            output = self.worker._execute_model_spmd(execute_model_req,
                                                     intermediate_tensors)
            if isinstance(output, IntermediateTensors):
                return execute_model_req, output
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

    device_str = "GPU" if not is_tpu() else "TPU"
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
        placement_group_specs = ([{
            device_str: 1
        }] * parallel_config.world_size)
        current_placement_group = ray.util.placement_group(
            placement_group_specs)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)

    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
