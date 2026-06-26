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

Note: every FSDP rank is a named RDT producer actor. ``full_tensor()``
all-gathers each layer to all ranks, so each rank can serve NIXL pulls.
Inference workers are mapped 1:1 onto the trainer ranks (static load
balancing) so the trainer-side clone + NIC egress is spread across all
ranks instead of funneling through rank 0; workers resolve their assigned
rank via ``ray.get_actor(...)``.

================================================================================
PROFILING OUTPUT — HOW TO READ IT
================================================================================
This script is instrumented with hardcoded profiling (no env vars). The goal is
to attribute every weight-sync second: clone, NIXL transfer, memory
registration, all-gather, fixed per-RPC overhead, and the worker-side copies —
and to check whether transfer is spread across NICs. If you are porting this to
bigger models/hardware, this section explains every line so you can interpret a
fresh run.

THREE PROCESSES (who measures what)
  - driver        : this script's main(). Orchestrates gather + update_weights
                    and PRINTS every ``[profile]`` line. Reads trainer counters
                    via Ray methods and worker counters via a JSONL file.
  - trainer ranks : ``FSDPTrainWorker`` actors = NIXL *producers* (4, one/FSDP
                    rank). Clone slices, serve pulls. Counters exposed via
                    ``get_produce_timing`` / ``get_nixl_timing`` Ray methods.
  - vLLM workers  : EngineCore subprocesses = NIXL *consumers* (4, DP=4). Run
                    ``receive_weights``/``_replay``/``_pull``. Their logs are NOT
                    streamed to the driver, so each appends one JSON line per
                    pull to ``/tmp/rdt_profile/consumer.jsonl``; the driver reads
                    it at the end. They also log ``[RDT-TIMING]`` lines visible in
                    the run log prefixed by ``(RayWorkerProc pid=...)``.

THE MEASUREMENT SHIM
  ``vllm/distributed/weight_transfer/_nixl_profile.py`` monkeypatches Ray's
  ``NixlTensorTransport`` (``register_memory`` / ``deregister_memory`` /
  ``transfer`` / ``check_xfer_state`` / ``extract_tensor_transport_metadata`` /
  desc calls) to accumulate per-process timers. Installed in BOTH the trainer
  actors (``FSDPTrainWorker.__init__``) and the vLLM workers (engine
  ``init_transfer_engine``). NIXL is one-sided, so the producer process only
  *registers* (never ``transfer``s) and the consumer process *transfers* — the
  same patch, read per-process, cleanly separates registration from transfer.
  It also reads per-NIC EFA/RDMA hardware counters from ``/sys`` (name-agnostic).

UNIT MODEL (critical for reading the numbers)
  one run        = SYNC_ITERS weight syncs (env RDT_SYNC_ITERS, default 3).
  one sync/iter  = N layer GROUPS (pre + one-per-decoder-layer + post).
  one group      = one ``engine.update_weights`` = one batched "produce" RPC
                   PER vLLM worker. So  #produce RPCs/iter = #groups x #workers.
  iter 0 is COLD (first registration + CUDA warmup); iters 1+ are WARM (the
  steady state RL cares about). NOTE: the RPC-floor baseline probe at init calls
  rdt_warmup, which PRE-WARMS the same NIXL connection produce uses — so the
  one-time connection-setup cost is absorbed into init, not iter 0 (this is why
  iter 0 looks less cold than it would without profiling). Read the WARM iters.

--------------------------------------------------------------------------------
PER-ITER ``[profile]`` BLOCK (printed by the driver, once per sync iter)
--------------------------------------------------------------------------------
  total weight-sync wall time   : end-to-end wall for the whole iter (the number
                                  to minimize). Everything below is a component.
  trainer all-gather (overlap'd): time the driver blocked on ``full_tensor``
                                  all-gather. NOTE: this is the OVERLAPPED TAIL —
                                  gather(K+1) is launched before awaiting
                                  update_weights(K), so warm iters show ~0. It is
                                  the critical-path contribution, not raw cost.
  trainer produce calls (all)   : total produce RPCs this iter (= #groups x
                                  #workers). The fixed-cost multiplier.
  trainer specs (slices) total  : total slices shipped (sum of batch sizes).
  trainer gather-cache wait     : time a produce call blocked waiting for its
                                  layer to be gathered. Should be ~0 (driver
                                  gathers before firing update_weights).
  trainer slice+clone (slowest) : per-rank clone (contiguous copy for NIXL) time,
                                  MAX across ranks (ranks clone in parallel, so
                                  wall ~ slowest). This is data touch #1.
  producer NIXL register (slow) : time the trainer spent in NIXL register_memory
                                  (fresh clone buffers => one reg/slice/iter).
                                  ``producer xfer`` should be ~0 (producers never
                                  call transfer; sanity check the one-sided split).
  producer method total (slow)  : wall spent INSIDE rdt_produce_weights_batched
                                  (= gather-wait + clone). The worker blocks on
                                  this during its pull.
  producer extract (slow)       : time in Ray's extract_tensor_transport_metadata
                                  (runs AFTER the method returns): cuda.sync +
                                  register + build descriptors. ``cuda.sync`` is
                                  isolated (= extract - register - descs); it has
                                  been small here, but watch it on bigger models.
  bytes produced (all ranks)    : total GiB cloned+served this iter.
  per-rank GiB                  : GiB per trainer rank — checks the 1:1 load
                                  balance is even (all ranks ~equal = balanced).
  agg clone throughput          : total bytes / slowest-rank clone time
                                  (AGGREGATE across ranks, not per-rank).

--------------------------------------------------------------------------------
END-OF-RUN BLOCKS (printed once, after all iters)
--------------------------------------------------------------------------------
RPC-FLOOR BASELINE (per worker, measured at init)
  bare Ray RTT     : round-trip of a no-op Ray actor call (``ping``). Pure Ray
                     dispatch floor, no NIXL. (~0.8 ms here.)
  nixl-ping RTT    : round-trip of a 1-element NIXL pull (``rdt_warmup``). Full
                     nixl control-plane for a trivial payload.
  nixl control-plane = nixl-ping - bare. The per-RPC FIXED overhead that does NOT
                     scale with payload (Ray object resolution + agent handshake).

CONSUMER-SIDE NIXL (WARM iters only — each worker's first ``n_groups`` replay
  records, i.e. the cold iter 0, are skipped; falls back to all records if
  SYNC_ITERS=1. An ``UNBAKED fallback fired`` line appears above it if the rare
  per-slice path ran — those records are EXCLUDED from these averages.)
  pull           : sum of the worker's ``ray.get(produce.remote(...))`` walls.
  transfer       : actual NIXL/RDMA read wall (data touch #2). Small.
  register (Nregs): consumer-side dest-buffer registration (fresh buffers/pull).
  dereg          : consumer-side deregistration.
  descs          : consumer descriptor build/serialize.
  recv-residual  : pull - transfer - register - dereg - descs. This is NOT pure
                   overhead: it still contains the clone-wait + fixed RPC floor +
                   producer extract, which all happen on the TRAINER during the
                   worker's ray.get. Split it using the producer + baseline lines.
  avg pull/RPC   : pull / (#groups x #warm iters). Compare directly to the
                   RPC-FLOOR nixl-ping: (avg pull/RPC - nixl-ping) is the
                   payload-dependent part of each pull.
  process        : printed on the next line — the post-pull scatter copy (touch
                   #3) + kernel copy (touch #4), summed. SEPARATE from pull;
                   ~two device copies of the worker's slice. Not split today.

PER-NIC RDMA byte deltas
  Per-EFA-device byte counters diffed across the run, + "max device share".
  CAVEAT: SINGLE-NODE transfers go over cuda_ipc/NVLink, NOT the NICs, so this
  reads ~0 on one node. It is only meaningful MULTI-NODE *and* where ``/sys``
  exposes EFA ``hw_counters``. Even share across devices = good balance; one
  device at ~100% = affinity/``UCX_NET_DEVICES`` problem. App-level balance is
  separately confirmed by the per-iter ``per-rank GiB`` line.

Per-pull worker log lines (in the run log, ``(RayWorkerProc pid=...)``):
  ``[RDT-TIMING] receive_weights mode=replay total=.. nixl_pull=.. (1 pull)
    process=.. | nixl_transfer=.. register=.. (N) dereg=..``
  ``process`` = the post-pull scatter work (see DATA TOUCHES). ``mode=unbaked``
  is the rare per-slice fallback path (should not appear in steady state).

--------------------------------------------------------------------------------
THE PER-RPC DECOMPOSITION (the punchline — how to split one ``pull``)
--------------------------------------------------------------------------------
  pull (worker ray.get of one produce RPC) =
      fixed Ray+nixl control-plane     <- RPC-FLOOR: bare + nixl-ping
    + clone-wait (trainer)             <- producer method total / #groups
    + transfer                         <- CONSUMER transfer
    + registration (producer+consumer) <- producer register + CONSUMER register
    + descriptors (producer+consumer)  <- CONSUMER descs (+ producer, in extract)
    + producer cuda.sync               <- producer extract - register - descs
  ``process`` (scatter/quant/kernel copy) is SEPARATE from pull (happens after).

DATA TOUCHES — the same bytes are moved 4x; only some are isolated:
  #1 trainer clone        -> per-iter "slice+clone"            (isolated)
  #2 NIXL transfer        -> CONSUMER "transfer"               (isolated)
  #3 scatter copy         -> recv buffer -> materialized param  ) both lumped into
  #4 kernel copy          -> param -> persistent kernel storage ) worker "process"
  (#3 and #4 are NOT split today; "process" ~= two device copies of the layer.
   ~half is the copy-into-dest. Splitting them is the next instrumentation TODO.)

HOW TO INTERPRET / SCALING TO BIGGER MODELS
  - FIXED cost (Ray dispatch + nixl control-plane, ~6 ms/RPC here) scales only
    with #RPCs (~#layers). Levers: fewer/larger gather groups; pipeline the pulls
    to hide latency.
  - PAYLOAD cost (clone + transfer + registration + descs + the two process
    copies) scales with model BYTES/SLICES. On bigger models this dominates; the
    clone is the single biggest payload term -> zero-copy views is the top lever.
  - So: if ``avg pull/RPC`` >> ``nixl-ping``, you are payload-bound (chase clone/
    copies). If they are close, you are RPC-bound (chase RPC count / pipelining).
  - REFERENCE (Qwen3-30B-A3B, TP1/DP4/EP4, 8xH100, nixl 1.3.0, WARM iter):
    total ~1.3 s; per-RPC pull ~18 ms ~= 33% fixed control-plane (~6 ms) +
    29% clone (~5 ms) + 15% transfer (~3 ms) + 8% registration + 8% descs +
    3% cuda.sync. process ~3 ms/RPC = two ~16 GiB/worker copies at ~250 GiB/s.
    These shift with model/hardware — recompute, don't assume.

FILES
  /tmp/rdt_profile/consumer.jsonl — truncated before init each run. Lines are
    either ``{"mode":"baseline", bare_ray_ms, nixl_ping_ms}`` (one per worker) or
    per-pull records ``{pull, transfer_seconds, register_seconds, descs_seconds,
    deregister_seconds, process, ...}``. The driver parses both at the end.
"""

import asyncio
import os
import sys
import threading
import time
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


def trainer_actor_name(rank: int) -> str:
    """Ray actor name for an FSDP rank's RDT producer.

    Rank 0 keeps the canonical name (back-compat with external tooling); ranks
    1+ get a ``_rank{N}`` suffix. All ranks are named so inference workers can
    resolve and pull from any of them for load balancing.
    """
    return TRAINER_ACTOR_NAME if rank == 0 else f"{TRAINER_ACTOR_NAME}_rank{rank}"

# RDT_WARMUP=1 -> prime the per-worker NIXL connection during
# init_weight_transfer_engine (via the trainer's rdt_warmup method) so the
# first real update_weights doesn't pay the one-time connection-setup cost.
WARMUP = os.environ.get("RDT_WARMUP", "0") == "1"

# RDT_SYNC_ITERS -> how many back-to-back weight syncs to run. The sharded RDT
# backend bakes a replay plan on the first sync for a given name set and
# replays it on subsequent syncs, so use >=2 to observe the replay speedup.
SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "3"))

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4
# vLLM workers in the inference EP group; each one calls
# rdt_produce_weights_batched once per layer. Used only to size the actor
# threadpool (one concurrent produce call per worker, plus gather).
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


# max_concurrency=8 lets each rank service inbound ``gather_layer`` calls AND
# the concurrent ``rdt_produce_weights_batched`` calls from the vLLM workers it
# serves, on separate threads in the actor's threadpool. With static 1:1 load
# balancing each rank serves a single inference worker, but 8 gives ample
# headroom (and tolerates other routing policies).
# Concurrent produce calls are read-only against the cache, so they need no
# locking beyond the gather/free synchronization: the driver frees a layer
# group only after its update_weights has fully drained.
@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class FSDPTrainWorker:
    """One FSDP2 training worker per GPU.

    Four of these form the FSDP group. Every rank serves RDT-tagged slice
    requests to the vLLM inference workers: ``full_tensor()`` all-gathers each
    layer to all ranks, so each rank can serve NIXL pulls. Inference workers are
    statically mapped 1:1 onto the ranks to spread the trainer-side clone + NIC
    egress instead of funneling everything through rank 0.
    """

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank
        self.world_size = fsdp_world_size

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

        # Cache of gathered full tensors. Held on EVERY rank (each rank serves
        # NIXL pulls for its assigned inference worker under static 1:1 routing).
        # Filled one layer group at a time by ``gather_layer``; read by
        # ``rdt_produce_weights_batched``; dropped a whole group at a time by
        # ``free_group``. Guarded by _cache_cond so a produce thread can block
        # on "key not yet gathered." No per-name refcounting: the driver frees
        # a group only after its ``update_weights`` has fully drained, which
        # guarantees every inference worker has already pulled every slice it
        # needs from that group (including its EP-local slice of each fused
        # expert tensor, which all ranks pull from the same name).
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        # Set if any gather_layer call errors; produce_method consults
        # this so workers don't hang waiting on a layer that will never
        # arrive.
        self._gather_error: BaseException | None = None

        # ---- Trainer-side profiling counters (rank 0 only) ----
        # Guarded by its own lock because rdt_produce_weights_batched runs
        # concurrently on the actor threadpool (up to NUM_INFERENCE_CONSUMERS
        # threads). Each produce call records: time waiting on the gather
        # cache (should be ~0 since the driver gathers before firing
        # update_weights), time replaying op chains + cloning slices (the
        # "slicing" cost we want to isolate from transport), the byte volume
        # of the produced slices, and call/spec counts.
        self._timing_lock = threading.Lock()
        self._produce_calls = 0
        self._produce_specs = 0
        self._produce_wait_seconds = 0.0
        self._produce_slice_seconds = 0.0
        self._produce_bytes = 0
        # Total wall time spent *inside* the produce method (entry->exit). The
        # gap between the worker's pull and this is Ray dispatch + the post-return
        # extract (cuda.sync + register) + control plane + recv.
        self._produce_method_seconds = 0.0

        # Hardcoded profiling: patch Ray's NIXL transport in this producer
        # process so register_memory (the producer-side memory registration that
        # fires for every fresh clone) is timed. This process never calls
        # transfer(), so its transfer_seconds stays ~0 -- the split is implicit.
        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing

        install_nixl_timing()

    def get_rank(self):
        return self.rank

    def ping(self):
        """No-op Ray method: measures the bare Ray actor-call RTT (no nixl)."""
        return None

    # ---------- profiling helpers ----------

    def get_nixl_timing(self) -> dict:
        """Per-process NIXL counters (producer-side registration etc.)."""
        from vllm.distributed.weight_transfer import _nixl_profile

        return _nixl_profile.snapshot()

    def reset_nixl_timing(self) -> None:
        from vllm.distributed.weight_transfer import _nixl_profile

        _nixl_profile.reset()

    def read_nic_counters(self) -> dict:
        """Read this node's per-EFA-device RDMA hardware counters."""
        from vllm.distributed.weight_transfer import _nixl_profile

        return _nixl_profile.read_efa_counters()

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---------- gather (called concurrently on all ranks, once per layer) -----

    def gather_layer(self, names: list[str]) -> None:
        """Collectively all-gather one layer-aligned group of params.

        Every FSDP rank must call this with the SAME ``names`` in the SAME
        ORDER — ``full_tensor()`` is a collective and per-rank divergence
        deadlocks the group. ``full_tensor()`` is an all-gather, so EVERY rank
        ends up holding the full tensor; each rank caches it in ``self._cache``
        so that every rank can serve NIXL pulls (load-balancing across ranks),
        not just rank 0.

        Backpressure between layers is implicit in the driver loop: the
        driver awaits (and frees) the previous ``update_weights`` future
        before firing the next layer's ``update_weights``, so at most two
        layer groups are ever resident in ``self._cache`` at once.
        """
        try:
            for name in names:
                param = self._param_lookup[name]
                full = param.full_tensor()
                with self._cache_cond:
                    self._cache[name] = full
                    self._cache_cond.notify_all()
        except BaseException as e:
            with self._cache_cond:
                self._gather_error = e
                self._cache_cond.notify_all()
            raise

    # ---------- RDT serve (rank 0 only) ----------

    @ray.method(tensor_transport="nixl")
    def rdt_warmup(self):
        """Return a tiny tensor over NIXL to prime the connection.

        Called once per inference worker during init_weight_transfer_engine.
        Independent of the gather cache (which is empty at init time), so it
        only exercises the NIXL agent/connection setup between this trainer
        and the calling worker -- absorbing the one-time first-transfer
        latency before the timed weight sync. A 1-element tensor is enough to
        force the connection handshake."""
        return torch.zeros(1, device="cuda:0")

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs):
        """Serve a batched slice request from vLLM.

        Waits until every unique name in ``specs`` is in the cache (the
        driver will have called ``gather_layer`` on the owning group
        before firing the ``update_weights`` that triggers this RPC).
        Applies each chain to the cached full tensor, clones to a slice-
        sized contiguous buffer for NIXL, and returns the list.

        This is a pure read of the cache — entries are freed separately by
        ``free_group`` once the whole group's ``update_weights`` has drained,
        so concurrent produce calls from different EP workers never race each
        other (each clones a different slice out of the same cached tensor).

        Served by every FSDP rank (each holds the full all-gathered layer), so
        inference workers spread their pulls across ranks rather than hammering
        rank 0.
        """
        _t_method0 = time.perf_counter()
        needed = sorted({name for name, _ in specs})

        _t_wait0 = time.perf_counter()
        with self._cache_cond:
            while not all(n in self._cache for n in needed):
                if self._gather_error is not None:
                    raise RuntimeError(
                        f"gather loop errored before producing {needed}: "
                        f"{self._gather_error!r}"
                    )
                self._cache_cond.wait()
        wait_seconds = time.perf_counter() - _t_wait0

        # Time the slice production: replaying each op chain (pure views,
        # cheap) plus the contiguous clone (the real GPU work — a fresh
        # allocation + copy that NIXL then ships). cuda.synchronize() before
        # stopping the clock so the clones' async GPU time is actually
        # captured rather than just the enqueue cost; this is the "slicing"
        # the trainer does, isolated from the NIXL transport that happens
        # after this method returns (during the worker's ray.get).
        _t_slice0 = time.perf_counter()
        out: list[torch.Tensor] = []
        nbytes = 0
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
            sl = tensor.clone(memory_format=torch.contiguous_format)
            out.append(sl)
            nbytes += sl.element_size() * sl.nelement()
        torch.accelerator.synchronize()
        slice_seconds = time.perf_counter() - _t_slice0

        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += len(specs)
            self._produce_wait_seconds += wait_seconds
            self._produce_slice_seconds += slice_seconds
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - _t_method0

        return out

    def get_produce_timing(self) -> dict:
        """Return accumulated rank-0 produce-side timing for profiling."""
        with self._timing_lock:
            return {
                "calls": self._produce_calls,
                "specs": self._produce_specs,
                "wait_seconds": self._produce_wait_seconds,
                "slice_seconds": self._produce_slice_seconds,
                "bytes": self._produce_bytes,
                "method_seconds": self._produce_method_seconds,
            }

    def reset_produce_timing(self) -> None:
        """Zero the rank-0 produce counters so each sync iteration can be timed
        independently (bake vs replay)."""
        with self._timing_lock:
            self._produce_calls = 0
            self._produce_specs = 0
            self._produce_wait_seconds = 0.0
            self._produce_slice_seconds = 0.0
            self._produce_bytes = 0
            self._produce_method_seconds = 0.0

    def free_group(self, names: list[str]) -> None:
        """Drop a layer group's gathered full tensors from the rank-0 cache.

        Called by the driver only AFTER ``engine.update_weights`` for this
        group has completed. That await is the synchronization point: it
        guarantees every inference worker has finished pulling all of its
        slices for the group (``update_weights`` does not return until each
        worker's ``load_weights`` — and thus every ``copy_`` that issues an
        RDT pull — has run), so the full tensors are safe to release. This
        replaces per-name refcounting with a single per-layer free.

        Runs on every rank now that every rank caches the all-gathered layer.
        """
        with self._cache_cond:
            for name in names:
                self._cache.pop(name, None)


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
    # On an Anyscale workspace a Ray head node is already running, and
    # ``RAY_OVERRIDE_RESOURCES`` pins object_store_memory to the full /dev/shm
    # size. Attach to that managed cluster rather than starting a fresh node
    # (a fresh start trips Ray's "object store exceeds /dev/shm" guard, and
    # object_store_memory must not be passed when connecting to an existing
    # cluster).
    ray.init(
        address="auto",
        runtime_env=runtime_env,
        namespace=RAY_NAMESPACE,
    )

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    # Every rank is a named RDT producer so inference workers can spread their
    # pulls across all ranks (static 1:1). Rank 0 keeps the canonical name for
    # back-compat; ranks 1+ get a ``_rank{N}`` suffix. ``producer_names`` is
    # ordered by rank and handed to the engine's ``trainer_actor_names``.
    fsdp_workers = []
    for rank in range(FSDP_WORLD_SIZE):
        common_args = (
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        handle = FSDPTrainWorker.options(name=trainer_actor_name(rank)).remote(
            *common_args
        )
        fsdp_workers.append(handle)
    producer_names = [trainer_actor_name(r) for r in range(FSDP_WORLD_SIZE)]
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
    # Fetch the trainer's full parameter metadata *before* init: the sharded RDT
    # engine bakes its replay plan over all of these during
    # init_weight_transfer_engine. The driver also partitions the flat name list
    # into layer-aligned groups for its own per-layer gather/free schedule;
    # update_weights then passes each group's gathered names.
    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    layer_groups = _layerwise_groups(names)
    print(
        f"[sync] {len(names)} params -> {len(layer_groups)} gather groups "
        f"(max group size = {max(len(g) for g in layer_groups)} params)."
    )

    # Truncate the worker-side consumer timing file BEFORE init: the engine
    # writes its per-worker RPC-baseline record (bare Ray RTT + tiny-nixl RTT)
    # during init_weight_transfer_engine, and per-pull records during the sync
    # loop. Both must survive into the driver's end-of-run read.
    import json

    consumer_file = "/tmp/rdt_profile/consumer.jsonl"
    os.makedirs(os.path.dirname(consumer_file), exist_ok=True)
    open(consumer_file, "w").close()

    print(
        f"[sync] Initializing sharded RDT engine on vLLM workers "
        f"(warmup={'on' if WARMUP else 'off'}); dry-run baking replay plans..."
    )
    _init_t0 = time.perf_counter()
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                ShardedRDTWeightTransferInitInfo(
                    trainer_actor_names=producer_names,
                    trainer_actor_namespace=RAY_NAMESPACE,
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    warmup_method_name="rdt_warmup" if WARMUP else None,
                )
            )
        )
    )
    _init_seconds = time.perf_counter() - _init_t0
    print(f"[sync] init_weight_transfer_engine (incl. bake) took {_init_seconds:.3f} s")

    print("[sync] Pausing generation...")
    await engine.pause_generation(mode="abort")

    # ---- Hardcoded profiling setup ----
    # (consumer_file already truncated above, before init, so the baseline
    # records the engine wrote during init are preserved.)
    # Snapshot per-NIC RDMA counters before any transfer. Read from rank 0's
    # actor (co-located with the GPUs/NICs that egress, unlike the driver which
    # may sit on the head node). Single node -> these are node-global counters.
    nic_before = ray.get(fsdp_workers[0].read_nic_counters.remote())

    # Run SYNC_ITERS back-to-back syncs. The plans were baked at init, so every
    # sync is a replay. Each iter brackets the per-group loop in its own
    # start/finish_weight_update, since initialize/finalize_layerwise_reload run
    # per sync.
    for sync_iter in range(SYNC_ITERS):
        # Zero produce counters on every rank so each iter is timed
        # independently (every rank now produces a share of the slices).
        ray.get([w.reset_produce_timing.remote() for w in fsdp_workers])
        # Zero per-process NIXL counters too, so producer-side registration time
        # is attributed per iter (bake-iter vs replay-iters).
        ray.get([w.reset_nixl_timing.remote() for w in fsdp_workers])

        await engine.start_weight_update(is_checkpoint_format=True)

        # Accumulated wall time spent in the collective gather (full_tensor
        # all-gather) for this iter, summed over groups. Measured as the slowest
        # rank per group since gather_futs are awaited together.
        allgather_seconds = 0.0

        # Per-layer transfer loop. The two interleaved operations:
        #
        #   * ``gather_layer`` on every FSDP rank: a collective full_tensor
        #     for each name in the group; every rank caches the result.
        #   * ``engine.update_weights`` on the inference side: triggers the
        #     vLLM workers' load_weights with lazy placeholders (bake) or a
        #     direct replay scatter (replay). Each worker pulls its slices
        #     from its statically-assigned trainer rank (1:1 load balance).
        #
        # We fire the previous layer's ``update_weights`` as an
        # ``asyncio.Task`` and only await it at the *start* of the next
        # iteration, so the gather for layer K+1 overlaps with the worker
        # applying layer K. Backpressure is implicit — the next
        # ``update_weights`` does not fire until the previous one has
        # drained, so each rank's cache holds at most two layers at once.
        print(f"[sync] iter {sync_iter} [REPLAY]: gather + update_weights...")
        _sync_t0 = time.perf_counter()
        prev_task: asyncio.Task | None = None
        prev_names: list[str] | None = None
        for group_names in layer_groups:
            # The trainer gathers/frees by name; the inference side replays the
            # baked leaf modules these gathered names cover.
            group_info = ShardedRDTWeightTransferUpdateInfo(names=group_names)
            gather_futs = [w.gather_layer.remote(group_names) for w in fsdp_workers]
            if prev_task is not None:
                await prev_task
                ray.get([w.free_group.remote(prev_names) for w in fsdp_workers])
            _t_gather = time.perf_counter()
            ray.get(gather_futs)
            allgather_seconds += time.perf_counter() - _t_gather
            prev_task = asyncio.create_task(
                engine.update_weights(
                    WeightTransferUpdateRequest(update_info=asdict(group_info))
                )
            )
            prev_names = group_names

        if prev_task is not None:
            await prev_task
            ray.get([w.free_group.remote(prev_names) for w in fsdp_workers])

        await engine.finish_weight_update()
        _sync_seconds = time.perf_counter() - _sync_t0

        # ---- Per-iter profiling summary ----
        # Trainer-side: produce calls (RPC count) and slice+clone time, summed
        # across ALL ranks (each rank now produces a share of the slices).
        # Because the ranks clone in PARALLEL, the wall-clock-relevant term is
        # the SLOWEST rank's slice+clone time (``slice_s_max``), not the sum;
        # aggregate throughput = total bytes / slowest-rank time. We also print
        # per-rank GiB to show how evenly the 1:1 routing balanced the load.
        ptimings = ray.get([w.get_produce_timing.remote() for w in fsdp_workers])
        gib = sum(p["bytes"] for p in ptimings) / (1024**3)
        per_rank_gib = [p["bytes"] / (1024**3) for p in ptimings]
        slice_s_max = max(p["slice_seconds"] for p in ptimings)
        calls = sum(p["calls"] for p in ptimings)
        specs = sum(p["specs"] for p in ptimings)
        wait_max = max(p["wait_seconds"] for p in ptimings)
        # Producer-side NIXL counters (this iter): registration is the cost that
        # fires for every fresh clone buffer. transfer_seconds should be ~0 here
        # (producers are passive RDMA responders, never call transfer()).
        ntimings = ray.get([w.get_nixl_timing.remote() for w in fsdp_workers])
        reg_s_max = max(n["register_seconds"] for n in ntimings)
        reg_calls = sum(n["register_calls"] for n in ntimings)
        prod_xfer = sum(n["transfer_seconds"] for n in ntimings)
        # Producer-side post-return extract = cuda.sync + register + descs.
        # Isolate the per-RPC cuda.synchronize() by subtracting register+descs.
        extract_s_max = max(n["extract_seconds"] for n in ntimings)
        sync_per_rank = [
            n["extract_seconds"] - n["register_seconds"] - n["descs_seconds"]
            for n in ntimings
        ]
        cuda_sync_max = max(sync_per_rank)
        method_s_max = max(p["method_seconds"] for p in ptimings)
        print("=" * 60)
        print(
            f"[profile] iter {sync_iter} [REPLAY]  warmup={'ON' if WARMUP else 'OFF'}"
        )
        if sync_iter == 0:
            print(f"[profile] init_weight_transfer_engine : {_init_seconds:.3f} s")
        print(f"[profile] total weight-sync wall time : {_sync_seconds:.3f} s")
        print(f"[profile] trainer all-gather (overlap'd): {allgather_seconds:.3f} s")
        print(f"[profile] trainer produce calls (all) : {calls}")
        print(f"[profile] trainer specs (slices) total: {specs}")
        print(f"[profile] trainer gather-cache wait    : {wait_max:.3f} s (max)")
        print(f"[profile] trainer slice+clone (slowest): {slice_s_max:.3f} s")
        print(
            f"[profile] producer NIXL register (slow): {reg_s_max:.3f} s "
            f"({reg_calls} regs; producer xfer={prod_xfer:.3f}s should be ~0)"
        )
        print(
            f"[profile] producer method total (slow) : {method_s_max:.3f} s "
            f"(time inside rdt_produce_weights_batched: wait+clone)"
        )
        print(
            f"[profile] producer extract (slow)      : {extract_s_max:.3f} s "
            f"of which cuda.sync ~= {cuda_sync_max:.3f} s  <-- scales w/ GPU work?"
        )
        print(f"[profile] bytes produced (all ranks)   : {gib:.3f} GiB")
        print(
            "[profile] per-rank GiB                 : "
            + ", ".join(f"r{r}={g:.2f}" for r, g in enumerate(per_rank_gib))
        )
        if slice_s_max > 0:
            print(
                f"[profile] agg clone throughput         : "
                f"{gib / slice_s_max:.1f} GiB/s"
            )
        print("=" * 60)

    # ---- Consumer-side NIXL breakdown (read from the worker timing file) ----
    # Lines are mode=baseline (one per worker, the RPC-floor probe), mode=replay
    # (one per baked pull), or mode=unbaked (the rare per-slice fallback path).
    # Separate all three. For replay, split each worker's records into cold
    # (iter 0) vs warm (iters 1+) using the known group count, so the
    # steady-state numbers aren't diluted by the cold iter, and so unbaked
    # records (pull=0) never sneak into the per-pull average.
    from collections import defaultdict

    n_groups = len(layer_groups)
    baselines: list[dict] = []
    replay_by_pid: dict[int, list[dict]] = defaultdict(list)
    unbaked: list[dict] = []
    with open(consumer_file) as f:
        for line in f:
            rec = json.loads(line)
            mode = rec.get("mode")
            if mode == "baseline":
                baselines.append(rec)
            elif mode == "unbaked":
                unbaked.append(rec)
            else:
                replay_by_pid[rec["pid"]].append(rec)

    def _agg(rows: list[dict]) -> dict:
        d = dict.fromkeys(
            ("pull", "transfer", "register", "dereg", "descs", "process"), 0.0
        )
        d["n"] = d["count"] = 0
        for r in rows:
            d["pull"] += r.get("pull", 0.0)
            d["transfer"] += r.get("transfer_seconds", 0.0)
            d["register"] += r.get("register_seconds", 0.0)
            d["dereg"] += r.get("deregister_seconds", 0.0)
            d["descs"] += r.get("descs_seconds", 0.0)
            d["process"] += r.get("process", 0.0)
            d["n"] += r.get("register_calls", 0)
            d["count"] += 1
        return d

    # ---- The decisive numbers: is the residual a fixed RPC cost or scaling? ----
    print("=" * 60)
    print("[profile] RPC-FLOOR BASELINE (per worker, measured at init)")
    for b in sorted(baselines, key=lambda r: r["pid"]):
        bare = b.get("bare_ray_ms") or 0.0
        nixl = b.get("nixl_ping_ms") or 0.0
        print(
            f"[profile]   pid={b['pid']}: bare Ray RTT={bare:.3f} ms  "
            f"nixl-ping RTT={nixl:.3f} ms  (nixl control-plane ~= {nixl - bare:.3f} ms)"
        )
    if unbaked:
        u = _agg(unbaked)
        print(
            f"[profile] UNBAKED fallback fired: {u['count']} records, "
            f"total={u['pull'] + u['process']:.3f}s -- per-slice path; investigate "
            f"if non-zero (these are EXCLUDED from the per-pull averages below)."
        )
    print("=" * 60)
    # Warm-only (skip each worker's first n_groups replay records = cold iter 0).
    # Falls back to all records if only the cold iter ran (SYNC_ITERS=1).
    warm_label = "WARM iters only" if SYNC_ITERS > 1 else "COLD iter only"
    print(f"[profile] CONSUMER-SIDE NIXL ({warm_label}, per worker pid)")
    for pid, rows in sorted(replay_by_pid.items()):
        warm_rows = rows[n_groups:] if len(rows) > n_groups else rows
        a = _agg(warm_rows)
        cnt = max(a["count"], 1)
        residual = a["pull"] - a["transfer"] - a["register"] - a["dereg"] - a["descs"]
        per_pull_ms = a["pull"] / cnt * 1e3
        print(
            f"[profile]   pid={pid}: pull={a['pull']:.3f}s "
            f"transfer={a['transfer']:.3f}s register={a['register']:.3f}s "
            f"({a['n']} regs) dereg={a['dereg']:.3f}s descs={a['descs']:.3f}s "
            f"recv-residual={residual:.3f}s | avg pull/RPC={per_pull_ms:.2f} ms"
        )
        print(
            f"[profile]            process (scatter+kernel copies, touches #3+#4): "
            f"{a['process']:.3f}s  (separate from pull; ~2 device copies of the data)"
        )
    print("=" * 60)

    # ---- Per-NIC RDMA counter deltas (is transfer spread across EFA devices?) --
    nic_after = ray.get(fsdp_workers[0].read_nic_counters.remote())
    print("[profile] PER-NIC RDMA byte deltas over the whole sync run")
    byte_deltas: list[tuple[str, int]] = []
    for key, after_val in nic_after.items():
        d = after_val - nic_before.get(key, 0)
        # Focus on byte-volume counters that actually moved.
        if d > 0 and "bytes" in key:
            byte_deltas.append((key, d))
    if not byte_deltas:
        print("[profile]   (no *_bytes counters moved -- check counter names below)")
        # Fall back to any counter that moved, to surface what EFA exposes.
        for key, after_val in sorted(nic_after.items()):
            d = after_val - nic_before.get(key, 0)
            if d > 0:
                print(f"[profile]   {key}  += {d}")
    else:
        for key, d in sorted(byte_deltas, key=lambda kv: -kv[1]):
            print(f"[profile]   {key}  += {d / (1024**3):.3f} GiB")
        # How evenly did the egress spread across distinct devices?
        per_dev: dict[str, int] = {}
        for key, d in byte_deltas:
            dev = key.split(":", 1)[0]
            per_dev[dev] = per_dev.get(dev, 0) + d
        n_dev = len(per_dev)
        tot = sum(per_dev.values())
        print(
            f"[profile]   -> {n_dev} EFA device(s) moved bytes; "
            f"max/total share = {max(per_dev.values()) / tot:.1%}"
            if tot
            else "[profile]   -> no bytes moved"
        )

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
