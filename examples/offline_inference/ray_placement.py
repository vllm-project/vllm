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
        print(f"creating LLM with bundle_indices={bundle_indices}")
        super().__init__(*args, **kwargs)


class RayTrainingActor:

    def report_device_id(self) -> str:
        # the argument for get_device_uuid is the index
        # of the GPU in the visible devices.
        # ray will set CUDA_VISIBLE_DEVICES to the assigned GPUs
        from vllm.platforms import current_platform
        return current_platform.get_device_uuid(0)


# ray manages 4 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ray.init()

# we want to co-locate vLLM instance and the training actor
# on the same set of GPUs.
# the placement plan is as follows:
# GPU 0 and 1: training actor 0, 1, and vLLM instance 0 (with TP=2)
# GPU 2 and 3: training actor 2, 3, and vLLM instance 1 (with TP=2)

pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())
print(f"placement group has bundles {pg.bundle_specs=}")

training_actors = []
training_actor_device_ids = []
inference_engines = []
inference_engine_device_ids = []

for bundle_index in [0, 1, 2, 3]:
    training_actor = ray.remote(
        num_cpus=0,
        num_gpus=0.4,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_index,
        ),
    )(RayTrainingActor).remote()
    training_actors.append(training_actor)
    device_id = ray.get(training_actor.report_device_id.remote())
    print(f"training actor {bundle_index} is on {device_id}")
    training_actor_device_ids.append(device_id)

for (i, bundle_indices) in enumerate([[0, 1], [2, 3]]):
    # IMPORTANT: when creating vLLM instances, we need to
    # make sure there are no GPU activities on the target GPUs,
    # otherwise, they will interfere with the vLLM memory profiling,
    # and cause unexpected behaviors.
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        ),
    )(MyLLM).remote(
        model="facebook/opt-125m",
        enforce_eager=True,
        worker_cls=MyWorker,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        bundle_indices=bundle_indices,
    )
    inference_engines.append(llm)
    # don't call any method on the inference engine here,
    # otherwise it will block until the vLLM instance is created.

for i, llm in enumerate(inference_engines):
    inference_engine_device_ids.append(
        ray.get(llm.collective_rpc.remote("report_device_id", args=tuple())))
    print(f"inference engine {i} is on {inference_engine_device_ids[-1]}")

# check the placement
# the first two training actors should be
# on the same GPUs as the first inference engine
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]
# the last two training actors should be
# on the same GPUs as the second inference engine
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]
