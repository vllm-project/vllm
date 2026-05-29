# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training (4 GPUs) and vLLM expert-parallel inference (4 GPUs)
using the **sharded RDT** weight-transfer backend.

Mirrors ``rlhf_nccl_fsdp_ep.py`` (same model, same 4+4 GPU layout) but
swaps the NCCL broadcast for RDT/NIXL pulls. The two interesting
differences vs the NCCL version:

  1. The trainer-side producer is **lazy and layer-aligned**: rather
     than calling ``full_tensor()`` for every parameter up-front
     (which materializes the whole model on rank 0), the transfer is
     driven layer-by-layer. The parameter-name list is partitioned
     into per-layer groups (plus a "pre" group for embeddings and a
     "post" group for the final norm + lm_head). For each group the
     driver issues a collective ``gather_layer`` to every FSDP rank,
     then fires ``engine.update_weights`` for just that group's
     names. Peak resident memory on rank 0 is bounded by a small
     multiple of one layer, regardless of how many MoE experts that
     layer contains.

  2. Overlap: ``update_weights`` is fired as an ``asyncio.Task`` and
     awaited only at the *start* of the next iteration, so while
     NIXL is transporting layer K to the vLLM workers, the driver
     has already kicked off the FSDP all-gather for layer K+1.
     Backpressure is implicit — the next iteration always awaits the
     previous ``update_weights`` before firing its own, so the
     trainer cache holds at most two layers at once.

This file is intended for benchmarking; not part of the final commit.

8-GPU layout:
  Training  — 4 GPUs, PyTorch FSDP2 (fully_shard)
  Inference — 4 GPUs, vLLM AsyncLLMEngine with expert parallelism +
              data parallelism (TP=1, DP=4, enable_expert_parallel
              → EP_SIZE = TP*DP = 4)

Note: FSDP rank 0 is the named RDT trainer actor. Inference workers
resolve it via ``ray.get_actor(TRAINER_ACTOR_NAME, ...)``. Ranks 1-3
participate in collectives only.
"""

import asyncio
import os
import sys
import threading
import uuid
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferInitInfo,
    ShardedRDTWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TRAINER_ACTOR_NAME = "sharded_rdt_fsdp_trainer"
RAY_NAMESPACE = "sharded_rdt_fsdp_example"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4
# vLLM workers in the inference EP group; each one will call
# produce_method once per layer. Used by the rank-0 cache refcount so
# a non-expert param is popped only after every worker has consumed it.
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE


# Mirror of the supported op set the worker-side LazyRDTTensor allows.
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


def _layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition a flat parameter-name list into layer-aligned groups.

    Names with prefix ``model.layers.<N>.`` are grouped by ``<N>``;
    everything before the first such name becomes a single "pre" group
    (embeddings etc.) and everything after becomes a single "post" group
    (final norm, lm_head, etc.). Within each group, order matches the
    input. Groups are returned in iteration order: pre, layer 0, layer 1,
    ..., post.

    Each group is the unit of gather/cache backpressure: the gather loop
    produces one group at a time and blocks before starting the next one
    if the rank-0 cache already holds ``max_layers_in_cache`` groups.
    """
    pre: list[str] = []
    layers: dict[int, list[str]] = {}
    post: list[str] = []
    seen_layer = False
    for n in names:
        if n.startswith("model.layers."):
            seen_layer = True
            idx = int(n[len("model.layers.") :].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen_layer:
            pre.append(n)
        else:
            post.append(n)

    groups: list[list[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
    if post:
        groups.append(post)
    return groups


# max_concurrency=8 lets each rank service inbound ``gather_layer`` calls
# AND -- on rank 0 -- the concurrent ``rdt_produce_weights_batched`` calls
# from the 4 vLLM workers on separate threads in the actor's threadpool.
# (We'd need 1 + NUM_INFERENCE_CONSUMERS = 5 slots; 8 gives headroom.)
# Concurrent produce_method calls are safe because the cache uses per-name
# reference counting — a non-expert param stays resident until every
# consumer has popped it.
@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class FSDPTrainWorker:
    """One FSDP2 training worker per GPU.

    Four of these form the FSDP group. Rank 0 additionally serves
    RDT-tagged slice requests to the vLLM inference workers; ranks 1-3
    exist solely to participate in the FSDP all-gather collectives that
    ``full_tensor()`` triggers.
    """

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
        num_consumers: int = 1,
    ):
        self.rank = rank
        self.world_size = fsdp_world_size
        # Number of vLLM workers that will call rdt_produce_weights_batched.
        # Determines the initial refcount for non-expert params (they're
        # pulled by every consumer); expert params start at refcount 1
        # because only the EP-owning rank pulls them.
        self._num_consumers = num_consumers

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        torch.accelerator.set_device_index(0)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        # Capture metadata BEFORE fully_shard so we have stable names/dtypes
        # /shapes to hand to vLLM's update_info. After sharding, params
        # become DTensors but keep the same names.
        self.weight_names = [n for n, _ in model.named_parameters()]
        self.weight_dtype_names = [
            str(p.dtype).split(".")[-1] for _, p in model.named_parameters()
        ]
        self.weight_shapes = [list(p.shape) for _, p in model.named_parameters()]

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model
        # Post-sharding lookup. Each entry is a DTensor with full_tensor()
        # available as a collective.
        self._param_lookup = dict(model.named_parameters())

        # Cache of gathered full tensors. Only meaningful on rank 0.
        # Filled one layer at a time by ``gather_layer``; produce reads
        # from here. Guarded by _cache_cond so the produce thread can
        # block on "key not yet gathered."
        self._cache: dict[str, torch.Tensor] = {}
        # Per-name reference count. Initialized at gather time to the
        # number of consumers that will pull this name (NUM_CONSUMERS for
        # non-expert params, 1 for expert params). Decremented on each
        # produce_method call; the cache entry is popped when it hits 0.
        # Necessary because non-expert params are pulled by EVERY DP/EP
        # worker — popping after the first consumer would starve the
        # others and deadlock.
        self._refcount: dict[str, int] = {}
        self._cache_cond = threading.Condition()
        # Set if any gather_layer call errors; produce_method consults
        # this so workers don't hang waiting on a layer that will never
        # arrive.
        self._gather_error: BaseException | None = None

    @staticmethod
    def _expected_consumers_for_name(name: str, num_consumers: int) -> int:
        """How many produce_method calls will reference this name.

        MoE expert params are sharded across EP ranks — only one inference
        worker pulls each expert's full tensor. Everything else
        (embeddings, attention, layer norms, lm_head) is pulled by every
        inference worker.
        """
        if ".mlp.experts." in name:
            return 1
        return num_consumers

    def get_rank(self):
        return self.rank

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---------- gather (called concurrently on all ranks, once per layer) -----

    def gather_layer(self, names: list[str]) -> None:
        """Collectively all-gather one layer-aligned group of params.

        Every FSDP rank must call this with the SAME ``names`` in the SAME
        ORDER — ``full_tensor()`` is a collective and per-rank divergence
        deadlocks the group. Rank 0 stores each gathered tensor in
        ``self._cache`` with an appropriate refcount; ranks 1-3 release
        the gathered tensor immediately (they participated only so the
        collective could complete).

        Backpressure between layers is implicit in the driver loop: the
        driver awaits the previous ``update_weights`` future before
        firing the next layer's ``gather_layer`` and ``update_weights``,
        so at most two layers are ever resident in ``self._cache`` at
        once.
        """
        try:
            for name in names:
                param = self._param_lookup[name]
                full = param.full_tensor()
                if self.rank == 0:
                    rc = self._expected_consumers_for_name(name, self._num_consumers)
                    with self._cache_cond:
                        self._cache[name] = full
                        self._refcount[name] = rc
                        self._cache_cond.notify_all()
                else:
                    del full
        except BaseException as e:
            with self._cache_cond:
                self._gather_error = e
                self._cache_cond.notify_all()
            raise

    # ---------- RDT serve (rank 0 only) ----------

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs):
        """Serve a batched slice request from vLLM.

        Waits until every unique name in ``specs`` is in the cache (the
        driver will have called ``gather_layer`` on the owning group
        before firing the ``update_weights`` that triggers this RPC).
        Applies each chain to the cached full tensor, clones to a slice-
        sized contiguous buffer for NIXL, returns the list. Once the
        response is built, the named entries are refcount-decremented
        and popped from the cache when their count hits zero.
        """
        assert self.rank == 0
        needed = sorted({name for name, _ in specs})

        with self._cache_cond:
            while not all(n in self._cache for n in needed):
                if self._gather_error is not None:
                    raise RuntimeError(
                        f"gather loop errored before producing {needed}: "
                        f"{self._gather_error!r}"
                    )
                self._cache_cond.wait()

        out: list[torch.Tensor] = []
        for name, chain in specs:
            tensor = self._cache[name]
            for op_name, args, kwargs_items in chain:
                if op_name not in _ALLOWED_OPS:
                    raise ValueError(
                        f"Spec for {name!r} requested disallowed op "
                        f"{op_name!r}; allowed: {sorted(_ALLOWED_OPS)}"
                    )
                kwargs = dict(kwargs_items)
                tensor = getattr(tensor, op_name)(*args, **kwargs)
            out.append(tensor.clone(memory_format=torch.contiguous_format))

        # Decrement refcount for each consumed name; pop from cache only
        # when refcount hits 0 (i.e. every consumer has pulled this name).
        # Concurrent produce_method calls from other consumers are safe
        # because the cache_cond is held during refcount mutation, and
        # each call only decrements names it actually consumed.
        with self._cache_cond:
            for name in needed:
                self._refcount[name] -= 1
                if self._refcount[name] <= 0:
                    self._cache.pop(name, None)
                    self._refcount.pop(name, None)
            self._cache_cond.notify_all()

        return out


def create_async_engine(**kwargs):
    """Create an AsyncLLMEngine directly (no subclass needed)."""
    engine_args = vllm.AsyncEngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return vllm.AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


async def generate_batch(engine, prompts, sampling_params):
    """Generate completions for a batch of prompts."""

    async def gen_one(prompt):
        output = None
        async for request_output in engine.generate(
            {"prompt": prompt},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        return output

    return await asyncio.gather(*[gen_one(p) for p in prompts])


async def main():
    # Pin Ray workers to the driver's Python so they pick up the venv
    # (mirrors the boilerplate from the other RDT examples).
    runtime_env: dict[str, object] = {"py_executable": sys.executable}
    forwarded = {
        k: os.environ[k]
        for k in ("NCCL_CUMEM_ENABLE", "VLLM_NCCL_SO_PATH", "LD_PRELOAD")
        if k in os.environ
    }
    if forwarded:
        runtime_env["env_vars"] = forwarded
    ray.init(runtime_env=runtime_env, namespace=RAY_NAMESPACE)

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    # Rank 0 is the named RDT trainer actor; vLLM workers resolve it by name.
    # All ranks need num_consumers so they apply identical refcount math
    # (only rank 0 actually uses it, but passing it uniformly keeps the
    # constructor signature consistent).
    fsdp_workers = []
    for rank in range(FSDP_WORLD_SIZE):
        common_args = (
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
            NUM_INFERENCE_CONSUMERS,
        )
        if rank == 0:
            handle = FSDPTrainWorker.options(name=TRAINER_ACTOR_NAME).remote(
                *common_args
            )
        else:
            handle = FSDPTrainWorker.remote(*common_args)
        fsdp_workers.append(handle)
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP training workers ready.")

    print("[engine] Creating AsyncLLMEngine...")
    engine = create_async_engine(
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="sharded_rdt"),
        load_format="dummy",
        gpu_memory_utilization=0.7,
    )
    print("[engine] AsyncLLMEngine created.")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    print("[generate] Generating with dummy weights...")
    outputs = await generate_batch(engine, prompts, sampling_params)
    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    # ---- Weight transfer ----
    print("[sync] Initializing sharded RDT engine on vLLM workers...")
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                ShardedRDTWeightTransferInitInfo(
                    trainer_actor_name=TRAINER_ACTOR_NAME,
                    trainer_actor_namespace=RAY_NAMESPACE,
                )
            )
        )
    )

    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    print(f"[sync] {len(names)} parameters to transfer.")

    print("[sync] Pausing generation...")
    await engine.pause_generation(mode="abort")

    await engine.start_weight_update(is_checkpoint_format=True)

    # Group the flat name list into layer-aligned groups so we transfer
    # one layer at a time. The driver does this once and reuses the
    # groups for both the per-rank gather and the per-call update_info.
    layer_groups = _layerwise_groups(names)
    print(
        f"[sync] Partitioned {len(names)} params into {len(layer_groups)} "
        f"layer groups (max group size = "
        f"{max(len(g) for g in layer_groups)} params)."
    )
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Per-layer transfer loop. The two interleaved operations:
    #
    #   * ``gather_layer`` on every FSDP rank: a collective full_tensor
    #     for each name in the group; rank 0 caches, ranks 1-3 discard.
    #   * ``engine.update_weights`` on the inference side: triggers the
    #     vLLM workers' load_weights with lazy placeholders, which call
    #     back to rank 0's ``rdt_produce_weights_batched`` for slices.
    #
    # We fire the previous layer's ``update_weights`` as an
    # ``asyncio.Task`` and only await it at the *start* of the next
    # iteration, so the gather for layer K+1 overlaps with the worker
    # applying layer K. Backpressure is implicit — the next
    # ``update_weights`` does not fire until the previous one has
    # drained, so the rank-0 cache holds at most two layers worth of
    # full tensors at any given moment.
    print("[sync] Driving per-layer gather + update_weights...")
    prev_task: asyncio.Task | None = None
    for group_names in layer_groups:
        # Build this group's update_info from the captured metadata.
        group_info = ShardedRDTWeightTransferUpdateInfo(
            names=group_names,
            dtype_names=[dtype_names[name_to_idx[n]] for n in group_names],
            shapes=[shapes[name_to_idx[n]] for n in group_names],
        )

        # Fire the collective gather on every FSDP rank. These are Ray
        # task submissions; they return immediately so we can overlap
        # the awaiting of the previous update_weights below.
        gather_futs = [w.gather_layer.remote(group_names) for w in fsdp_workers]

        # Drain the previous layer's update_weights. While we await
        # here, the gather_futs above are already executing on the FSDP
        # actors -- that's the gather/transport overlap.
        if prev_task is not None:
            await prev_task

        # Now make sure this layer's gather has actually completed
        # before we fire update_weights for it (the producer would
        # block on the cache otherwise, but failing here surfaces gather
        # errors more cleanly).
        ray.get(gather_futs)

        # Fire this layer's update_weights as a Task; we await it on
        # the next iteration (or after the loop, for the final layer).
        prev_task = asyncio.create_task(
            engine.update_weights(
                WeightTransferUpdateRequest(update_info=asdict(group_info))
            )
        )

    if prev_task is not None:
        await prev_task

    await engine.finish_weight_update()
    print("[sync] Resuming generation...")
    await engine.resume_generation()

    print("[generate] Generating with synced weights...")
    outputs_updated = await generate_batch(engine, prompts, sampling_params)
    print("-" * 60)
    print("AFTER weight sync (real weights):")
    print("-" * 60)
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
