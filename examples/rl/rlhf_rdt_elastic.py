# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates *elastic* inference scaling on top of the RDT weight transfer
backend.

The headline benefit of RDT/NIXL over NCCL for weight sync is that the trainer
holds no process-group state about the inference fleet. Each inference replica
independently resolves the trainer Ray actor by name and pulls its weights, so
adding (or removing) a replica at runtime is a local operation -- no group
re-init, no world_size handshake, no trainer-side touch.

This example shows that by adding a second inference replica AFTER the first
one has already synced and generated:

    1. Start the trainer + Replica 1 (TP=1).
    2. Replica 1 generates with dummy weights      -> gibberish.
    3. Replica 1 syncs from the trainer            -> coherent text.
    4. Spawn Replica 2 (TP=1) with dummy weights   *while Replica 1 is live*.
    5. Replica 2 generates                         -> gibberish (its weights
                                                     are still dummy; the
                                                     trainer was not touched).
    6. Confirm Replica 1 is unaffected             -> still coherent.
    7. Replica 2 syncs from the *same* trainer     -> coherent text.

With the NCCL backend, step 4 would require destroying and rebuilding the
trainer's process group to a new world_size, which is the exact friction RDT
removes.

Prerequisites:
    pip install nixl

GPU layout (single node, 3 GPUs):
    GPU 0  -> trainer
    GPU 1  -> Replica 1 (TP=1)
    GPU 2  -> Replica 2 (TP=1)
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
# actor across an otherwise anonymous namespace boundary.
RAY_NAMESPACE = "rdt_elastic_example"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution (TP=1)."""

    def __init__(self, *args, **kwargs):
        # One bundle per TP=1 replica; bundle 0 is the only one in its PG.
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1, enable_tensor_transport=True)
class TrainModel:
    """Ray actor that wraps the training model and serves weights over RDT/NIXL.

    Identical to ``rlhf_rdt.py``; reused here unchanged to underline that the
    trainer is oblivious to how many inference replicas exist.
    """

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
        self._param_lookup = dict(self.model.named_parameters())

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weight(self, name: str):
        return self._param_lookup[name]

    def get_weight_metadata(self):
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes


def spawn_replica(label: str):
    """Create a fresh vLLM inference replica with its own placement group.

    Each replica is fully independent: own placement group, own Ray actor, own
    EngineCore subprocess. The only shared piece of state is the *name* of the
    trainer actor (resolved later via ``ray.get_actor``).
    """
    pg = placement_group([{"GPU": 1, "CPU": 0}])
    ray.get(pg.ready())
    scheduling = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling,
    )(MyLLM).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="rdt"),
        load_format="dummy",
        quantization="fp8",
    )
    print(f"[{label}] spawned (pg + Ray actor + EngineCore)")
    return llm, pg


def sync_replica(llm, label: str):
    """Run a full RDT weight sync round on ``llm`` from the named trainer."""
    print(f"[{label}] starting weight sync ...")
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
    ray.get(llm.start_weight_update.remote(is_checkpoint_format=True))
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
    print(f"[{label}] weight sync complete")


def generate_and_print(llm, label: str):
    outputs = ray.get(llm.generate.remote(prompts, sampling_params))
    header = f" {label} ".center(50, "-")
    print(header)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print("-" * 50)


# Pin Ray-actor processes to the same Python interpreter as the driver and
# forward the NCCL config / preload env vars (needed on this cluster for the
# cu13 driver). Same boilerplate as ``rlhf_rdt.py``.
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

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0)


# ---------------------------------------------------------------------------
# Phase 1: Replica 1 only. Dummy -> sync -> coherent.
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("PHASE 1: Replica 1 spawned and synced (baseline)")
print("=" * 50)
llm1, pg1 = spawn_replica("Replica 1")
generate_and_print(llm1, "Replica 1: BEFORE sync (dummy weights)")
sync_replica(llm1, "Replica 1")
generate_and_print(llm1, "Replica 1: AFTER sync (trainer weights)")


# ---------------------------------------------------------------------------
# Phase 2: Add Replica 2 dynamically. No trainer touch, no group rebuild.
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("PHASE 2: Replica 2 spawned with dummy weights (no trainer changes)")
print("=" * 50)
llm2, pg2 = spawn_replica("Replica 2")

# Pre-sync: dummy weights -> gibberish. The trainer was not touched when
# Replica 2 joined; only Replica 2's own model holds randomly initialized
# parameters at this point.
generate_and_print(llm2, "Replica 2: BEFORE sync (dummy weights)")

# Replica 1 should still produce coherent text -- adding a sibling did not
# regress it. This proves the new replica's spawn was a local operation.
generate_and_print(llm1, "Replica 1: still healthy after Replica 2 joined")


# ---------------------------------------------------------------------------
# Phase 3: Sync Replica 2 from the same trainer. Both replicas now coherent.
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("PHASE 3: Sync Replica 2 from the same trainer")
print("=" * 50)
sync_replica(llm2, "Replica 2")
generate_and_print(llm2, "Replica 2: AFTER sync (trainer weights)")
generate_and_print(llm1, "Replica 1: final state (unchanged since Phase 1)")

# Ray.shutdown() handles teardown of both replicas + trainer on exit. The
# same elastic property runs in reverse: ray.kill(llmN) + remove_placement_
# group(pgN) would tear down any single replica without affecting the others
# or the trainer, since none of them share a process group.
