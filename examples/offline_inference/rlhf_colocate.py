# SPDX-License-Identifier: Apache-2.0
"""
a simple demonstration to show how to co-locate
vLLM worker with training actors on the same GPUs,
for RLHF-like applications.
The key points:
- Control the placement of the vLLM workers with Ray, by setting
    VLLM_RAY_PER_WORKER_GPUS and VLLM_RAY_BUNDLE_INDICES properly.
- Use cuda-ipc to pass tensors, since NCCL does not work when we have
    multiple processes on the same GPU.
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM
from vllm.worker.worker import Worker


class MyWorker(Worker):

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class MyLLM(LLM):

    def __init__(self, *args, bundle_indices: list, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # every worker will use 0.4 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            map(str, bundle_indices))
        print(f"creating LLM with bundle_indices={bundle_indices}")
        super().__init__(*args, **kwargs)


class RayTrainingActor:

    def __init__(self):
        # ray will set CUDA_VISIBLE_DEVICES to the assigned GPUs
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()
        # the argument for get_device_uuid is the index
        # of the GPU in the visible devices.
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(0)

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_weight_ipc_handles(self):
        from torch.multiprocessing.reductions import reduce_tensor
        data = {}
        for name, p in self.model.named_parameters():
            # the training actor might only have a subset of the weights
            # and need to all-gather the weights from all the actors.
            # for demonstration, here we assume all training actors have
            # the full weights.
            data[name] = reduce_tensor(p.detach())
        return {self.device_uuid: data}


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

for bundle_index, training_actor in enumerate(training_actors):
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

print("gather all the IPC handles from the training actors")
ipc_handles = {}
for actor in training_actors:
    ipc_handles.update(ray.get(actor.get_weight_ipc_handles.remote()))

print("update the weights of the inference engines")
for llm in inference_engines:
    ray.get(
        llm.collective_rpc.remote("update_weights_from_ipc_handles",
                                  args=(ipc_handles, )))
print("check if the weights are updated")
for llm in inference_engines:
    assert ray.get(
        llm.collective_rpc.remote("check_weights_changed", args=tuple()))
