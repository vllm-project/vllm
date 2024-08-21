"""Compare the outputs of HF and distributed vLLM when using greedy sampling.

Run:
```sh
cd $VLLM_PATH/tests

pytest distributed/test_multi_node.py
```
"""

import pytest
import ray

from vllm.utils import cuda_device_count_stateless


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model, distributed_executor_backend", [
    ("facebook/opt-125m", "ray"),
])
def test_multi_node_bad_topology(
    vllm_runner,
    model: str,
    distributed_executor_backend: str,
) -> None:
    """Verify ray + multi node's bad topology raises an exception.

    This test simulates multi node ray cluster, so we don't have to start
    real 2 multi nodes.

    There are 2 potential bad issues.
    - the engine's node doesn't have enough GPUs.
    - the tensor parallel size exceeds the available GPUs in a current node.
    """
    dtype = "half"
    ray.init()
    assert ray.cluster_resources()["GPU"] == 4.0, (
        "At leasts 4 gpus are required to run a test.")
    print(ray.cluster_resources())
    print("===Test tp 4 on 2 nodes===")
    # Creating tp == 4. Since TP workers are supposed to spread to 2 workers
    # it should log warning.
    with pytest.warns() as record:
        with vllm_runner(
                model,
                dtype=dtype,
                tensor_parallel_size=4,
                distributed_executor_backend=distributed_executor_backend
        ) as _:
            pass
        print(record)

    # Simulate there's no GPU in a current node.
    @ray.remote(num_gpus=1)
    class Actor:
        pass

    # a is created on a head node.
    actors = [Actor.remote() for _ in range(2)]  # type: ignore
    ray.get([a.__ray_ready__.remote() for a in actors])

    print("===Test no GPU on a current node===")
    # Now vLLM is created on a head node, but there's no GPU. It should raise
    # an exception.
    with pytest.raises(RuntimeError), vllm_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=1,
            distributed_executor_backend=distributed_executor_backend) as _:
        pass
