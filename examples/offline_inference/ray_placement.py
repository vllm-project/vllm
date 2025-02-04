# SPDX-License-Identifier: Apache-2.0
"""
a simple demonstration to show how to control
the placement of the vLLM workers with Ray.
The key is to set VLLM_RAY_PER_WORKER_GPUS and
VLLM_RAY_BUNDLE_INDICES properly.
"""
import os

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM
from vllm.worker.worker import Worker


class MyWorker(Worker):

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        return current_platform.get_device_uuid(self.device.index)


class MyLLM(LLM):

    def __init__(self, *args, bundle_indices: list, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        # every worker will use 0.4 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            map(str, bundle_indices))
        super().__init__(*args, **kwargs)


# ray manages 4 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["RAY_DEDUP_LOGS"] = "0"
ray.init()

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg_inference.ready())
print(f"placement group has bundles {pg_inference.bundle_specs=}")

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
)

llms = []

# here we create 4 LLM instances, 2 of them will be scheduled
# on the same GPUs.
# GPUs: 0, 1, 2, 3
# instance 0: GPU 0, 1
# instance 1: GPU 0, 1
# instance 2: GPU 2, 3
# instance 3: GPU 2, 3
for bundle_indices in [[0, 1], [0, 1], [2, 3], [2, 3]]:
    print(f"creating LLM with bundle_indices={bundle_indices}")
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyLLM).remote(
        model="facebook/opt-125m",
        enforce_eager=True,
        worker_cls=MyWorker,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        bundle_indices=bundle_indices,
    )
    llms.append(llm)

# check if the device IDs are the same for two instances
device_ids = []
for llm in llms:
    device_ids.append(
        ray.get(llm.collective_rpc.remote("report_device_id", args=tuple())))
print(f"{device_ids=}")

assert device_ids[0] == device_ids[1]
assert device_ids[2] == device_ids[3]
