"""Compare the outputs of HF and distributed vLLM when using greedy sampling.

Run:
```sh
cd $VLLM_PATH/tests

pytest distributed/test_multi_node.py
```
"""
import os

import pytest

import ray
from vllm.utils import cuda_device_count_stateless
from ray.cluster_utils import Cluster

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model, distributed_executor_backend, test_suite", [
    ("facebook/opt-125m", "ray", "L4"),
])
def test_multi_node_bad_topology(
    vllm_runner,
    model: str,
    distributed_executor_backend: str,
    test_suite: str,
) -> None:
    """Verify ray + multi node's bad topology raises an exception.

    This test simulates multi node ray cluster, so we don't have to start
    real 2 multi nodes.

    There are 2 potential bad issues.
    - the engine's node doesn't have enough GPUs.
    - the tensor parallel size exceeds the available GPUs in a current node.
    """
    dtype = "half"
    assert test_suite == TARGET_TEST_SUITE

    # Simulate 2 node clusters, 1 GPU each.
    cluster = Cluster()
    head_node = cluster.add_node(num_cpus=8, num_gpus=1, resources={"head": 1})
    ray.init(address=head_node.address)
    cluster.add_node(num_cpus=8, num_gpus=1)

    # Creating tp == 2. Since TP workers are supposed to spread to 2 workers
    # it should log warning.
    with vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend=distributed_executor_backend) as _:
        pass

    # Simulate there's no GPU in a current node.
    @ray.remote(num_gpus=1, resources={"head": 1})
    class Actor:
        pass

    # a is created on a head node.
    a = Actor.remote()  # type: ignore
    ray.get(a.__ray_ready__.remote())

    # Now vLLM is created on a head node, but there's no GPU. It should raise
    # an exception.
    with pytest.raises(RuntimeError), vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=1,
            distributed_executor_backend=distributed_executor_backend) as _:
        pass
