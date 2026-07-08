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
on background threads. The trainer side is the shared
:class:`rdt_producer.RDTShardedProducer`.

This is the minimal, single-node (3 GPU: 1 trainer + TP-2 inference),
gather-free variant: the trainer holds the full model resident and serves
slices of the LIVE parameters, so there is no gather plan and the engine's
per-group ``free_gather`` calls are no-ops. See rlhf_sharded_rdt_fsdp_ep.py
(FSDP-sharded trainer, per-group gathers) and rlhf_sharded_rdt_kimi.py
(1T FP8 MoE) for the full-scale variants.

Prerequisites:
    pip install nixl
"""

import os
import sys

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

from rdt_producer import RDTShardedProducer, layerwise_groups

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TRAINER_ACTOR_NAME = "sharded_rdt_trainer"
# Explicit namespace so vLLM workers -- which run in an EngineCore
# subprocess that does its own ray.init() -- can resolve the named
# trainer actor.
RAY_NAMESPACE = "sharded_rdt_example"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class TrainModel(RDTShardedProducer):
    """Trainer actor: the full HF model resident on one GPU, serving packed
    slice pulls of its LIVE parameters (no gather plan needed — the shared
    producer's cache is pre-populated once and never freed)."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
        self.init_rdt_producer()
        # Live parameters ARE the serve cache: produce replays each spec's op
        # chain on them and packs the slices into the registered serve ring.
        # PyTorch parameters mutate in place during training, so the cached
        # references stay valid across syncs.
        with self._cache_cond:
            self._cache.update(dict(self.model.named_parameters()))
            self._cache_cond.notify_all()

    def get_weight_metadata(self):
        """Weight names/dtypes/shapes for the engine's bake at init."""
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes


# Pin Ray-actor processes to the same Python interpreter as the driver, and
# ship this example directory so actors can import rdt_producer.
_RUNTIME_ENV: dict[str, object] = {
    "py_executable": sys.executable,
    "working_dir": os.path.dirname(os.path.abspath(__file__)),
}
_FORWARDED_ENV_VARS = {
    k: os.environ[k]
    for k in ("NCCL_CUMEM_ENABLE", "VLLM_NCCL_SO_PATH", "LD_PRELOAD")
    if k in os.environ
}
if _FORWARDED_ENV_VARS:
    _RUNTIME_ENV["env_vars"] = _FORWARDED_ENV_VARS
ray.init(runtime_env=_RUNTIME_ENV, namespace=RAY_NAMESPACE)

train_model = TrainModel.options(name=TRAINER_ACTOR_NAME).remote(MODEL_NAME)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# distributed_executor_backend="ray" is REQUIRED: each vLLM worker
# must be a Ray actor so it can call ray.get_actor() and submit
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

# The engine bakes its replay plan over ALL names at init (a meta dry run of
# model.load_weights), so the metadata goes in the INIT info.
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())
ray.get(
    llm.init_weight_transfer_engine.remote(
        dict(
            init_info=dict(
                trainer_actor_name=TRAINER_ACTOR_NAME,
                trainer_actor_namespace=RAY_NAMESPACE,
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
            )
        )
    )
)

# is_checkpoint_format=True is MANDATORY for the sharded backend --
# it requires the layerwise reload path.
ray.get(llm.start_weight_update.remote(is_checkpoint_format=True))

# ONE update_weights for the whole sync. group_lens partitions the names into
# per-layer groups: the group is the packed pull's chunk budget, so the
# receive/serve arenas stay layer-sized instead of ballooning to the whole
# model. (The trainer serves live params, so there is no gather plan and the
# engine's per-group free_gather calls are no-ops.)
groups = layerwise_groups(names)
ray.get(llm.update_weights.remote(dict(update_info=dict(
    names=[n for g in groups for n in g],
    group_lens=[len(g) for g in groups],
)))) 

ray.get(llm.finish_weight_update.remote())
ray.get(llm.wake_up.remote(tags=["scheduling"]))

# Second generation: output should now be coherent.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
print("After weight sync (trainer slices pulled via sharded RDT/NIXL):")
for output in outputs_updated:
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print("-" * 50)
