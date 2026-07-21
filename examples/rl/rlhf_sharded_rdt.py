# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal RLHF weight sync with the **sharded** RDT (Ray Direct Transport)
weight-transfer backend.

The sharded backend pulls only the *slice* of each weight that the local
worker actually consumes, rather than the full HF-shaped tensor: at
``init_weight_transfer_engine`` the engine dry-runs vLLM's own
``model.load_weights`` against lazy placeholders and BAKES, per checkpoint
name, the op chain the loader applied (the slice) and the destination
param/offsets. Every ``update_weights`` replays that plan: the worker sends
the specs to the trainer in packed chunk pulls (one contiguous NIXL blob per
pull, received into a pre-registered ring of arenas) and scatters/quantizes
on background threads.

This is the minimal, single-node (3 GPU: 1 trainer + TP-2 inference)
variant: the trainer holds the full model resident. The trainer-side
complexity -- the NIXL serve actor, gather cache, serve rings, free
ref-counting -- is entirely hidden behind
:class:`ShardedRDTTrainerWeightTransferEngine`: the trainer actor is a plain
Ray actor that builds the engine with a :class:`ModuleSource` over its model
and calls ``send_weights()``. See rlhf_sharded_rdt_fsdp_ep.py (FSDP-sharded
trainer) and rlhf_sharded_rdt_kimi.py (1T FP8 MoE) for the full-scale
variants.

Prerequisites:
    pip install nixl
"""

import os
import sys

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    ModuleSource,
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
# tensor_parallel_size * data_parallel_size inference workers.
NUM_INFERENCE_CONSUMERS = 2
RAY_NAMESPACE = "sharded_rdt_example"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Trainer actor: the full HF model resident on one GPU. All RDT serving
    lives inside the trainer engine (which spawns its own NIXL serve actor), so
    this actor needs no producer mixin and no tensor-transport / concurrency
    actor options."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
        self.engine = None

    def setup_engine(self, llm_handle):
        """Build the trainer engine. Its trainer_init spawns the per-rank NIXL
        serve actor and (as the single sender, rank 0) drives the inference
        side's init_weight_transfer_engine, which bakes the replay plan."""
        self.engine = WeightTransferTrainerFactory.trainer_init(
            ShardedRDTTrainerInitInfo(
                rank=0,
                num_consumers=NUM_INFERENCE_CONSUMERS,
                num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                trainer_actor_namespace=RAY_NAMESPACE,
            ),
            client=RayVLLMWeightSyncClient(llm_handle),
            source=ModuleSource(self.model),
        )

    def sync_weights(self):
        """One full weight-sync round: gather + serve + drive the inference
        handshake (start/update/finish)."""
        self.engine.send_weights()


# Pin Ray-actor processes to the same Python interpreter as the driver.
_RUNTIME_ENV: dict[str, object] = {
    "py_executable": sys.executable,
    "working_dir": os.path.dirname(os.path.abspath(__file__)),
}
_FORWARDED_ENV_VARS = {
    k: os.environ[k]
    for k in (
        "NCCL_CUMEM_ENABLE",
        "VLLM_NCCL_SO_PATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "NUM_RDT_BUFFERS",
    )
    if k in os.environ
}
if _FORWARDED_ENV_VARS:
    _RUNTIME_ENV["env_vars"] = _FORWARDED_ENV_VARS
ray.init(runtime_env=_RUNTIME_ENV, namespace=RAY_NAMESPACE)

train_model = TrainModel.remote(MODEL_NAME)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# distributed_executor_backend="ray" is REQUIRED: each vLLM worker
# must be a Ray actor so it can call ray.get_actor() and submit
# .remote() tasks against the trainer's serve actor.
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
    weight_transfer_config=WeightTransferConfig(backend="sharded_rdt"),
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
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print("-" * 50)

ray.get(llm.sleep.remote(level=0))

# Build the trainer engine: spawns the NIXL serve actor and drives the
# inference side's init_weight_transfer_engine (bakes the replay plan over all
# names). The engine derives names/dtypes/shapes/group_lens from the
# ModuleSource, so the driver no longer marshals metadata by hand.
ray.get(train_model.setup_engine.remote(llm))

# One call does the whole round: start_weight_update, the concurrent
# update_weights (workers pull their slices over NIXL while the trainer serves),
# and finish_weight_update.
ray.get(train_model.sync_weights.remote())

ray.get(llm.wake_up.remote(tags=["scheduling"]))

# Second generation: output should now be coherent.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
print("After weight sync (trainer slices pulled via sharded RDT/NIXL):")
for output in outputs_updated:
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print("-" * 50)
