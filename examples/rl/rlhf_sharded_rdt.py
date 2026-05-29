# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning using vLLM and Ray with the
**sharded** RDT (Ray Direct Transport) weight transfer backend.

Compared to ``rlhf_rdt.py``, this example uses ``backend="sharded_rdt"``,
which pulls only the *slice* of each weight that the local TP worker
actually consumes, rather than the full HF-shaped tensor. The savings are
roughly 1/TP_size in trainer -> worker bandwidth, and the slicing is
discovered by handing vLLM's existing ``model.load_weights`` lazy
placeholders that record narrow chains while it does its usual
fused-QKV / merged-MLP routing. See ``sharded_weight_loader_rdt.md``
for the full design.

The sharded backend requires the layerwise reload path
(``is_checkpoint_format=True``); it raises otherwise. The trainer must
expose a single batched accessor that takes ``(name, [(dim, start,
size), ...])`` specs and returns one slice tensor per spec, all
decorated with ``@ray.method(tensor_transport="nixl")`` so NIXL
view-aware transport ships only the requested bytes.

Prerequisites:
    pip install nixl

This example assumes a single-node cluster with three GPUs.
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

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TRAINER_ACTOR_NAME = "sharded_rdt_trainer"
# Explicit namespace so vLLM workers -- which run in an EngineCore
# subprocess that does its own ray.init() -- can resolve the named
# trainer actor. With an anonymous namespace, the worker-side init
# would land in a *different* anonymous namespace and ray.get_actor
# would fail.
RAY_NAMESPACE = "sharded_rdt_example"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


# Mirror of the supported op set the worker-side LazyRDTTensor allows. The
# trainer refuses any op name outside this set so a misbehaving / spoofed
# spec can't reach into arbitrary tensor methods via getattr.
_ALLOWED_OPS = frozenset(
    {
        "narrow",
        "view",
        "reshape",
        "__getitem__",
        "unsqueeze",
        "squeeze",
        "transpose",
        "t",
        "permute",
        "flatten",
        "contiguous",
        "chunk",
    }
)


@ray.remote(num_gpus=1, enable_tensor_transport=True)
class TrainModel:
    """Ray actor wrapping the training model and serving slice tensors
    over RDT/NIXL.

    The producer method takes a *list of specs* in one call -- one entry
    per ``LazyRDTTensor.copy_`` the layer's loaders are about to issue --
    and returns one slice per spec. Batching matters: each worker would
    otherwise pay one RPC per fused-QKV/MergedColumn shard call, of
    which there are several per layer.

    Each spec is ``(name, [(op_name, args, kwargs_items), ...])`` and
    the chain is replayed in order on the trainer's live parameter via
    ``getattr(tensor, op_name)(*args, **dict(kwargs_items))``.
    """

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
        # Cache name -> Parameter for O(1) lookups. PyTorch parameters
        # mutate in place during training, so cached references stay valid.
        self._param_lookup: dict[str, torch.Tensor] = dict(
            self.model.named_parameters()
        )

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(
        self,
        specs,
    ) -> list[torch.Tensor]:
        """Replay each op chain on the named parameter and return one
        slice per spec.

        Each slice is materialized via ``.clone(memory_format=
        torch.contiguous_format)`` for two reasons: (1) NIXL's
        list-of-tensors transport requires contiguous tensors ("Please
        use a list of contiguous Tensors"); (2) returning views that
        share storage with the trainer's live parameters causes RDT to
        reject the second call with "still in scope as part of another
        RDT object". ``.contiguous()`` alone is not enough -- for
        dim-0 narrows on a row-major tensor the result is *already*
        contiguous, so ``.contiguous()`` returns the view as-is with
        aliased storage. ``.clone(...)`` forces a fresh allocation
        with contiguous layout. Bandwidth savings are preserved --
        the clone is only the slice size, not the full HF tensor.
        """
        out: list[torch.Tensor] = []
        for name, chain in specs:
            tensor = self._param_lookup[name]
            for op_name, args, kwargs_items in chain:
                if op_name not in _ALLOWED_OPS:
                    raise ValueError(
                        f"Spec for {name!r} requested disallowed op "
                        f"{op_name!r}; allowed: {sorted(_ALLOWED_OPS)}"
                    )
                kwargs = dict(kwargs_items)
                tensor = getattr(tensor, op_name)(*args, **kwargs)
            out.append(tensor.clone(memory_format=torch.contiguous_format))
        return out

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for the update_info."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes


# Pin Ray-actor processes to the same Python interpreter as the driver.
# Matters on managed clusters where the default worker Python may
# differ from the venv that has vLLM + nixl installed. Also propagate
# selected env vars (NCCL config, LD_PRELOAD) so vLLM's TP=2 workers
# see the same setup as the driver.
_RUNTIME_ENV: dict[str, object] = {"py_executable": sys.executable}
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
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

ray.get(llm.sleep.remote(level=0))

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

names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())

# is_checkpoint_format=True is MANDATORY for the sharded backend --
# it requires the layerwise reload path so the pre_replay_hook can
# fire between buffering and replay. The engine raises if it doesn't
# find layerwise infos.
ray.get(llm.start_weight_update.remote(is_checkpoint_format=True))

# update_weights drives model.load_weights with lazy placeholders.
# vLLM's existing loaders call narrow/view/reshape/transpose/...
# /.copy_() on them; we record the full op chain during pass 1, batch
# every layer's chains into a single rdt_produce_weights_batched RPC
# in the pre-replay hook, then copy the prefetched slices during pass
# 2. Loaders that need ops outside the supported set (e.g. .to(),
# .float(), .item(), .data, bool-mask indexing) raise from
# __torch_dispatch__ so the failure is loud rather than silent.
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
print("After weight sync (trainer slices pulled via sharded RDT/NIXL):")
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
