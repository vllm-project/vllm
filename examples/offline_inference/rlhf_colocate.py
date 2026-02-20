# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates how to co-locate a vLLM inference worker and training
actors on the same set of GPUs for reinforcement learning from human feedback
(RLHF) workloads.

Ray serves as the distributed execution framework in this example. Ray
placement groups allocate both training actors and vLLM workers to the
same GPU bundles, enabling fast, in-GPU communication between the two
components.

The script shows how to do the following:

* Configure environment variables (`VLLM_RAY_PER_WORKER_GPUS` and
  `VLLM_RAY_BUNDLE_INDICES`) so that vLLM workers land on the desired
  devices.
* Exchange tensors between processes by means of CUDA inter-process
  communication (IPC). CUDA IPC sidesteps NCCL limitations that occur
  when multiple processes share a single GPU.

Note that this example assumes a single-node cluster with four GPUs, but Ray
supports multi-node clusters. vLLM expects exclusive use of the GPUs during
its initialization for memory profiling. Residual GPU activity interferes
with vLLM memory profiling and causes unexpected behavior.

Learn more about Ray placement groups:
https://docs.ray.io/en/latest/placement-groups.html
"""

import gc
import os
import sys

import ray
import torch
import zmq
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.multiprocessing.reductions import reduce_tensor

from vllm import LLM

if torch.version.hip is not None:
    print("Skipping test for ROCm. Ray is unsupported on vLLM ROCm.")
    sys.exit(0)


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution.

    The constructor sets environment variables that allow multiple vLLM
    workers to share a single physical GPU and that encode the bundle
    indices assigned by the placement group.

    Args:
        *args: Positional arguments forwarded to `vllm.LLM`.
        bundle_indices (list[int]): Placement-group bundle indices
            assigned to this worker.
        **kwargs: Keyword arguments forwarded to `vllm.LLM`.
    """

    def __init__(self, *args, bundle_indices: list[int], **kwargs):
        # Prevent Ray from manipulating the top-level CUDA_VISIBLE_DEVICES variable
        # so that vLLM can its own device placement inside the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Each worker uses 0.4 GPU so that two instances fit on the same GPUs.
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        print(f"creating LLM with bundle_indices={bundle_indices}")
        super().__init__(*args, **kwargs)


class RayTrainingActor:
    """Training actor that hosts a Facebook OPT-125M model from Hugging Face.

    The model is loaded onto the first GPU assigned to this actor, and expose
    the CUDA IPC handles so that colocated vLLM workers can map tensors
    directly.
    """

    def __init__(self):
        # Ray sets CUDA_VISIBLE_DEVICES to the GPUs assigned to this actor.
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")
        # Zero out all the parameters.
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()
        # The argument for `get_device_uuid` is the index of the GPU in the
        # list of visible devices.
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(0)
        self.zmq_context = zmq.Context()
        self.zmq_address_counter = 0
        self.zmq_handle = None

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_zmq_handles(self) -> dict[str, str]:
        suffix = f"{self.device_uuid}-{self.zmq_address_counter}"
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{suffix}.sock"
        self.zmq_address_counter += 1
        return {self.device_uuid: self.zmq_handle}

    def update_weights(self):
        # align size to avoid misaligned address
        align_size = 256

        def get_size(p: torch.Tensor) -> int:
            return (p.nbytes + align_size - 1) // align_size * align_size

        named_parameters: dict[str, torch.nn.Parameter] = dict(
            self.model.named_parameters()
        )
        max_tensor_size = max(get_size(p) for p in named_parameters.values())
        # use max_tensor_size * 2 as buffer size
        buffer = torch.empty(max_tensor_size * 2, dtype=torch.uint8, device="cuda:0")
        s = self.zmq_context.socket(zmq.REQ)
        s.bind(self.zmq_handle)
        handle = reduce_tensor(buffer)

        offset = 0
        buckets: list[tuple[list[dict], list[torch.Tensor]]] = []
        named_tensors: list[dict] = []
        real_tensors: list[torch.Tensor] = []
        for name, p in named_parameters.items():
            size = get_size(p)
            if offset + size > buffer.numel():
                buckets.append((named_tensors, real_tensors))
                named_tensors, real_tensors = [], []
                offset = 0
            # assume tensors are contiguous
            named_tensors.append(
                {"name": name, "dtype": p.dtype, "shape": p.shape, "offset": offset}
            )
            real_tensors.append(p)
            offset += size
        if named_tensors:
            buckets.append((named_tensors, real_tensors))
        s.send_pyobj(handle)
        s.recv()
        for named_tensors, real_tensors in buckets:
            offset = 0
            for p in real_tensors:
                buffer[offset : offset + p.nbytes].data.copy_(
                    p.data.view(-1).view(dtype=torch.uint8), non_blocking=True
                )
                offset += get_size(p)
            torch.cuda.synchronize()
            s.send_pyobj(named_tensors)
            s.recv()
        s.send_pyobj(None)
        s.recv()
        s.close()
        del buffer
        gc.collect()
        torch.cuda.empty_cache()


# Ray manages four GPUs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ray.init()

# Co-locate vLLM instances and training actors on the same set of GPUs:
#   * GPU 0 and 1: training actor 0, training actor 1, and vLLM instance 0
#     (tensor parallelism = 2).
#   * GPU 2 and 3: training actor 2, training actor 3, and vLLM instance 1
#     (tensor parallelism = 2).

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

for i, bundle_indices in enumerate([[0, 1], [2, 3]]):
    # Use the following syntax instead of the @ray.remote decorator so that
    # the placement group is customized for each bundle.
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
        worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        bundle_indices=bundle_indices,
    )
    inference_engines.append(llm)
    # Do not call any method on the inference engine at this point; the call
    # blocks until the vLLM instance finishes initialization.

for i, llm in enumerate(inference_engines):
    inference_engine_device_ids.append(
        ray.get(llm.collective_rpc.remote("report_device_id", args=tuple()))
    )
    print(f"inference engine {i} is on {inference_engine_device_ids[-1]}")

# Verify placement: the first two training actors share the same GPUs as
# the first inference engine.
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]
# Verify placement: the last two training actors share the same GPUs as
# the second inference engine.
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]

print("Gather all the ZMQ handles from the training actors.")
zmq_handles = {}
for actor in training_actors:
    zmq_handles.update(ray.get(actor.get_zmq_handles.remote()))

print(f"ZMQ handles: {zmq_handles}")

print("Update the weights of the inference engines.")
ray.get(
    [actor.update_weights.remote() for actor in training_actors]
    + [
        llm.collective_rpc.remote("update_weights_from_ipc", args=(zmq_handles,))
        for llm in inference_engines
    ]
)

print("Check if the weights are updated.")
for llm in inference_engines:
    assert ray.get(llm.collective_rpc.remote("check_weights_changed", args=tuple()))
