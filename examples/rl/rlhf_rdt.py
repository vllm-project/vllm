# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning using vLLM and Ray with the
RDT (Ray Direct Transport) weight transfer backend.

Compared to ``rlhf_nccl.py``, this example uses RDT's NIXL transport for the
trainer -> inference weight sync. Rather than setting up an out-of-band NCCL
group with master_address / master_port / rank_offset, the inference workers
look up the trainer Ray actor by name and pull each weight via an
``@ray.method(tensor_transport="nixl")`` accessor on the trainer.

Prerequisites:
    pip install nixl

The script:
* Loads the training model on GPU 0 inside a Ray actor.
* Initializes vLLM across GPUs 1-2 (TP=2) with ``distributed_executor_backend="ray"``
  (mandatory for the RDT backend -- workers must be Ray actors).
* Generates from a list of prompts with dummy weights (expected: gibberish).
* Syncs the trainer's weights into vLLM via the RDT engine.
* Generates again -- output should now be coherent.

This example assumes a single-node cluster with three GPUs.
"""

import os
import sys

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

MODEL_NAME = "facebook/opt-125m"
TRAINER_ACTOR_NAME = "rdt_trainer"
# Explicit namespace so the vLLM workers -- which run inside an EngineCore
# subprocess that does its own ray.init() -- can resolve the named trainer
# actor. With ray.init() in an anonymous namespace, the worker-side init
# would land in a *different* anonymous namespace and ray.get_actor would
# fail.
RAY_NAMESPACE = "rdt_example"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, enable_tensor_transport=True)
class TrainModel:
    """Ray actor that wraps the training model and serves weights over RDT/NIXL."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
        # Cache name -> Parameter mapping for O(1) lookups. PyTorch parameters
        # mutate in place during training, so cached references stay valid.
        self._param_lookup = dict(self.model.named_parameters())

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weight(self, name: str):
        """Return the weight tensor for ``name``. NIXL handles the transport
        when called from another Ray actor (i.e. a vLLM inference worker)."""
        return self._param_lookup[name]

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for the RDT update_info."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes


# Pin Ray-actor processes to the same Python interpreter as the driver. This
# matters on managed clusters where the default worker Python may differ from
# the venv that has vLLM + nixl installed. Also propagate selected env vars
# (NCCL config, LD_PRELOAD) so vLLM's TP=2 workers see the same setup as the
# driver -- needed on hosts where the default NCCL libs SEGV under cu13.
_RUNTIME_ENV: dict[str, object] = {"py_executable": sys.executable}
_FORWARDED_ENV_VARS = {
    k: os.environ[k]
    for k in ("NCCL_CUMEM_ENABLE", "VLLM_NCCL_SO_PATH", "LD_PRELOAD")
    if k in os.environ
}
if _FORWARDED_ENV_VARS:
    _RUNTIME_ENV["env_vars"] = _FORWARDED_ENV_VARS
ray.init(runtime_env=_RUNTIME_ENV, namespace=RAY_NAMESPACE)

# Trainer actor on GPU 0; named so inference workers can resolve it via
# ray.get_actor() during init_weight_transfer_engine.
train_model = TrainModel.options(name=TRAINER_ACTOR_NAME).remote(MODEL_NAME)

# Reserve GPUs 1-2 for the vLLM inference engine via a placement group.
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# distributed_executor_backend="ray" is REQUIRED for the RDT backend: each
# vLLM worker must be a Ray actor so it can call ray.get_actor() and submit
# .remote() tasks against the trainer.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    tensor_parallel_size=2,
    data_parallel_size=1,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="rdt"),
    load_format="dummy",
    quantization="fp8",
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

# First generation: dummy weights, output is expected to be nonsense.
outputs = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
print("Before weight sync (dummy weights):")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

ray.get(llm.sleep.remote(level=0))

# Initialize the RDT engine on each worker. The only init_info needed is the
# trainer's named-actor handle -- no master_address/port, no world_size,
# no rank_offset.
ray.get(
    llm.init_weight_transfer_engine.remote(
        dict(
            init_info=dict(
                trainer_actor_name=TRAINER_ACTOR_NAME,
                trainer_actor_namespace=RAY_NAMESPACE,
            )
        )
    )
)

# Gather per-weight metadata from the trainer to feed update_info. The actual
# tensors are NOT sent here -- they're pulled by each worker during
# update_weights below.
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())

ray.get(llm.start_weight_update.remote(is_checkpoint_format=True))

# update_weights triggers the per-worker pull. Each worker iterates through
# `names`, calls `train_model.rdt_produce_weight.remote(name)`, ray.gets the
# resulting NIXL-transferred tensor, hands it to load_weights, and drops the
# reference before pulling the next one. There is no separate trainer-side
# broadcast call -- the trainer's @ray.method serves on demand.
ray.get(
    llm.update_weights.remote(
        dict(
            update_info=dict(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
            )
        )
    )
)

ray.get(llm.finish_weight_update.remote())
ray.get(llm.wake_up.remote(tags=["scheduling"]))

# Second generation: output should now be coherent.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
print("After weight sync (trainer weights pulled via RDT/NIXL):")
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
