# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray,
with IPC-based weight syncing APIs

The script colocates the training and inference workloads onto the same GPU using Ray.

The example performs the following steps:

* Request a placement group of 1 GPU.
* Place the inference model on the above GPU using the placement group.
* Place and load the training model on the same GPU using the placement group.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using CUDA IPC handles. Note that
  for demonstration purposes we simply zero out the weights.

This example assumes a single-node cluster with a single GPU,
but can be extended to multiple GPUs.
"""

import os
from dataclasses import asdict

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferUpdateInfo


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Each worker uses 0.4 GPU so that two instances fit on the same GPU.
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        # needed for ipc handle serialization
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        super().__init__(*args, **kwargs)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


# Load the OPT-125M model onto GPU 0 for the training workload.

MODEL_NAME = "facebook/opt-125m"


@ray.remote
class TrainModel:
    def __init__(self, llm_handle: ray.ObjectRef):
        self.train_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
        )
        self.train_model.to("cuda:0")
        self.llm_handle = llm_handle

    def init_weight_transfer(self):
        # IPC backend doesn't need initialization info
        self.llm_handle.init_weight_transfer_engine.remote(dict(init_info=dict()))

    def broadcast_weights(self, llm_handle: ray.ObjectRef):
        self.llm_handle = llm_handle
        names, dtypes, shapes, ipc_handles = [], [], [], []

        for name, p in self.train_model.named_parameters():
            names.append(name)
            dtypes.append(str(p.dtype).split(".")[-1])
            shapes.append(p.shape)

            from torch.multiprocessing.reductions import reduce_tensor

            weight = p.detach().contiguous()
            ipc_handle = reduce_tensor(weight)
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handles.append(ipc_handle)

        ray.get(
            self.llm_handle.update_weights.remote(
                dict(
                    update_info=asdict(
                        IPCWeightTransferUpdateInfo(
                            names=names,
                            dtype_names=dtypes,
                            shapes=shapes,
                            ipc_handles=ipc_handles,
                        )
                    )
                )
            )
        )


ray.init(runtime_env={"excludes": [".git/objects/pack/"]})

pg_colocate = placement_group([{"GPU": 1, "CPU": 0}])
ray.get(pg_colocate.ready())


llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg_colocate,
        placement_group_capture_child_tasks=True,
    ),
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.7,
    weight_transfer_config=WeightTransferConfig(backend="ipc"),
    load_format="dummy",
)

train_model = TrainModel.options(
    num_gpus=0.1,
    num_cpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg_colocate, placement_group_capture_child_tasks=True
    ),
).remote(llm)


# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

ray.get(train_model.init_weight_transfer.remote())
# Synchronize the updated weights to the inference engine using batched API.
ray.get(train_model.broadcast_weights.remote(llm))

# Generate text with the updated model. The output is expected to be nonsense
# because the weights are zero.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
