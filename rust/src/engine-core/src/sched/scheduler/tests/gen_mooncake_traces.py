# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record scheduler traces of the Python `MooncakeStoreConnector` (scheduler
role) for the native Rust port (`connectors::mooncake_store_scheduler`).

Drives `vllm.v1.core.sched.scheduler.Scheduler` (or `AsyncScheduler`) with the
real `MooncakeStoreConnector`, a fake `LookupKeyClient` (deterministic store
hits, async lookups that need several polls) and a fake worker half (store
jobs completing out of order across ranks, async loads finishing or failing a
step later). Every input is written, and after every op the outputs:
`SchedulerOutput`, the connector metadata (canonical form, see
`canonical_meta`), `EngineCoreOutputs`, the block pool (ref counts and free
queue order), the store jobs' pins and the lookup client calls. The Rust test
`sched::scheduler::tests::mooncake_store` replays the inputs against the Rust
scheduler with the native connector and compares each of them.

See `rust/src/engine-core/tests/README.md` for how to run it.
"""

import dataclasses
import json
import os
import random
import sys
from collections import deque

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from gen_traces import (  # noqa: E402
    eco_json,
    group_json,
    parse_args,
    pin_hash_seed,
    so_json,
)

import vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.scheduler as mss  # noqa: E402, E501
from tests.v1.core.rust_core_diff import (  # noqa: E402
    DSV41_KV_CACHE_CONFIG,
    FuzzConfig,
    build_scheduler,
    make_dsv41_kv_cache_config,
)
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E402, E501
    MooncakeLookupResult,
    MooncakeStoreWorkerMetadata,
    TailKeyBoundary,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    resolve_cache_hit_alignment_tokens,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import (  # noqa: E402
    DraftTokenIds,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mooncake_traces")
EOS = 50256
NUM_STEPS = 45


@dataclasses.dataclass
class Scenario:
    kind: str
    layout: str  # full | mamba | mamba_partial | dsv41
    kv_role: str
    load_async: bool
    lookup_async: bool
    save_decode_cache: bool = False
    enable_lookup: bool = True
    async_scheduler: bool = False
    num_workers: int = 1
    num_spec_tokens: int = 0
    num_hash_lookahead_tokens: int = 0
    block_size: int = 16
    num_blocks: int = 96
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 8


SCENARIOS = {
    "both": Scenario("both", "full", "kv_both", load_async=False, lookup_async=False),
    "producer": Scenario(
        "producer",
        "full",
        "kv_producer",
        load_async=True,
        lookup_async=True,
        async_scheduler=True,
        num_workers=2,
    ),
    "consumer": Scenario(
        "consumer",
        "full",
        "kv_consumer",
        load_async=True,
        lookup_async=False,
        save_decode_cache=True,
        num_spec_tokens=2,
    ),
    "capacity": Scenario(
        "capacity", "full", "kv_consumer", False, False, enable_lookup=False
    ),
    "mamba": Scenario(
        "mamba", "mamba", "kv_both", load_async=True, lookup_async=False, num_workers=2
    ),
    "mamba_partial": Scenario(
        "mamba_partial",
        "mamba_partial",
        "kv_both",
        load_async=False,
        lookup_async=True,
        save_decode_cache=True,
        block_size=64,
        num_blocks=64,
    ),
    "lookahead": Scenario(
        "lookahead",
        "full",
        "kv_both",
        load_async=False,
        lookup_async=False,
        num_hash_lookahead_tokens=3,
        num_workers=2,
    ),
    "dsv41": Scenario(
        "dsv41",
        "dsv41",
        "kv_both",
        load_async=True,
        lookup_async=False,
        block_size=128,
        num_blocks=160,
        max_num_batched_tokens=512,
    ),
}


class FakeLookupClient:
    """`LookupKeyClient` without the RPC: deterministic hits, and async lookups
    that answer after a few polls. Logs every call with its answer."""

    def __init__(self, vllm_config):
        self.rng = random.Random(0)
        self.align = 16
        self.hash_block_size = 16
        self.num_groups = 1
        self.states: dict[str, list] = {}  # req_id -> [polls_left, result]
        self.log: list = []

    def _result(self, num_tokens: int, num_hashes: int) -> MooncakeLookupResult:
        # Like the worker's lookup: aligned, covered by hashes, and below the
        # request end so the last token is recomputed.
        limit = min(num_tokens - 1, num_hashes * self.hash_block_size)
        rng = self.rng
        x = rng.random()
        if x < 0.3:
            hit = 0
        elif x < 0.6:
            hit = limit
        else:
            hit = rng.randrange(0, limit + 1)
        hit = hit // self.align * self.align
        if hit <= 0:
            return MooncakeLookupResult(0)
        tails = tuple(
            TailKeyBoundary(g, max(self.align, hit - rng.choice([0, self.align])))
            for g in range(self.num_groups)
        )
        return MooncakeLookupResult(hit, tails)

    @staticmethod
    def _result_json(result):
        if result is None:
            return None
        return [
            result.hit_length,
            [[b.group_id, b.num_tokens] for b in result.tail_key_boundaries],
        ]

    def prefetch(self, req_id, num_tokens, block_hashes):
        assert req_id not in self.states
        ok = self.rng.random() < 0.7
        if ok:
            self.states[req_id] = [self.rng.randrange(0, 3), None]
        self.log.append(["prefetch", req_id, num_tokens, len(block_hashes), ok])
        return ok

    def lookup(self, req_id, num_tokens, block_hashes, non_block=False):
        state = self.states.get(req_id)
        if state is None:
            state = self.states[req_id] = [
                self.rng.randrange(0, 3) if non_block else 0,
                None,
            ]
        if state[0] > 0 and non_block:
            state[0] -= 1
            result = None
        else:
            if state[1] is None:
                state[1] = self._result(num_tokens, len(block_hashes))
            result = state[1]
        if not non_block:
            self.states.pop(req_id, None)
        self.log.append(
            [
                "lookup",
                req_id,
                num_tokens,
                len(block_hashes),
                non_block,
                self._result_json(result),
            ]
        )
        return result

    def discard(self, req_id):
        # Discarding a request the client holds nothing for is a no-op, which
        # the Rust port skips.
        if self.states.pop(req_id, None) is not None:
            self.log.append(["discard", req_id])

    def reset(self):
        self.states.clear()
        self.log.append(["reset"])
        return True


def fnv(values) -> str:
    h = 0xCBF29CE484222325
    for v in values:
        for byte in (v & 0xFFFFFFFF).to_bytes(4, "little"):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def fnv_bytes(chunks) -> str:
    h = 0xCBF29CE484222325
    for chunk in chunks:
        for byte in bytes(chunk):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def pool_json(scheduler) -> dict:
    pool = scheduler.kv_cache_manager.block_pool
    return {
        "refs": fnv(b.ref_cnt for b in pool.blocks),
        "free": fnv(b.block_id for b in pool.free_block_queue.get_all_free_blocks()),
    }


def pins_json(cs) -> list:
    return sorted(
        [job, list(blocks), remaining]
        for job, (blocks, remaining) in cs._pinned_saves.items()
    )


class MetaRecorder:
    """Tracks what the canonical metadata needs beyond the metadata itself:
    the changes to the connector's `_unfinished_request_ids` (Python hands
    out the live set) and the requests that finished (whose hand-offs carry
    their block hashes)."""

    def __init__(self, connector):
        self.cs = connector.connector_scheduler
        self.ops: list = []
        self.finished: dict[str, Request] = {}
        alloc = connector.update_state_after_alloc
        build = connector.build_connector_meta
        finished = connector.request_finished_all_groups

        def update_state_after_alloc(request, blocks, num_external_tokens):
            before = set(self.cs._unfinished_request_ids)
            result = alloc(request, blocks, num_external_tokens)
            self.ops += [["add", r] for r in self.cs._unfinished_request_ids - before]
            return result

        def build_connector_meta(scheduler_output):
            before = set(self.cs._unfinished_request_ids)
            meta = build(scheduler_output)
            after = self.cs._unfinished_request_ids
            assert after <= before
            self.ops += [["discard", r] for r in sorted(before - after)]
            return meta

        def request_finished_all_groups(request, block_ids):
            self.finished[request.request_id] = request
            return finished(request, block_ids)

        connector.update_state_after_alloc = update_state_after_alloc
        connector.build_connector_meta = build_connector_meta
        connector.request_finished_all_groups = request_finished_all_groups

    def take_ops(self) -> list:
        ops, self.ops = self.ops, []
        return ops


def canonical_meta(meta, recorder: MetaRecorder, ops: list) -> dict:
    """The metadata as plain values: every field of every `ReqMeta`, which
    entries share their `block_ids` tuple, and which list of the request's
    block hashes each entry references."""
    requests = []
    for i, m in enumerate(meta.requests):
        alias = next(
            (j for j in range(i) if meta.requests[j].block_ids is m.block_ids), None
        )
        finished = recorder.finished.get(m.req_id)
        if finished is None:
            hashes = ["live", len(m.block_hashes)]
        else:
            copy = m.block_hashes is not finished.block_hashes
            assert list(m.block_hashes) == list(finished.block_hashes)
            hashes = [
                "finished",
                copy,
                len(m.block_hashes),
                fnv_bytes(m.block_hashes),
            ]
        spec = m.load_spec
        requests.append(
            {
                "req_id": m.req_id,
                "token_len_chunk": m.token_len_chunk,
                "block_ids": [list(g) for g in m.block_ids],
                "alias": alias,
                "hashes": hashes,
                "can_save": m.can_save,
                "load_spec": None
                if spec is None
                else [
                    spec.vllm_cached_tokens,
                    spec.kvpool_cached_tokens,
                    spec.can_load,
                    spec.token_len,
                    [[b.group_id, b.num_tokens] for b in spec.tail_key_boundaries],
                ],
                "token_ids": m.token_ids,
                "token_ids_start": m.token_ids_start,
                "num_hash_lookahead_tokens": m.num_hash_lookahead_tokens,
                "num_prompt_tokens": m.num_prompt_tokens,
                "prefill_end_tokens": m.prefill_end_tokens,
                "store_job_id": m.store_job_id,
                "boundary_puts": None
                if m.boundary_puts is None
                else [list(p) for p in m.boundary_puts],
                "completed_token_len": m.completed_token_len,
                "publish_partial_tail": m.publish_partial_tail,
            }
        )
    return {
        "requests": requests,
        "unfinished_ops": ops,
        "unfinished": sorted(meta.unfinished_request_ids),
        "preempted": sorted(meta.preempted_req_ids),
    }


def make_kv_cache_config(sc: Scenario) -> KVCacheConfig:
    bs = sc.block_size
    full = FullAttentionSpec(
        block_size=16 if sc.layout == "mamba_partial" else bs,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    groups = [full]
    if sc.layout in ("mamba", "mamba_partial"):
        groups.append(
            MambaSpec(
                block_size=bs,
                shapes=((2, 512), (3, 32, 32)),
                dtypes=(torch.float32, torch.float32),
                mamba_cache_mode="align",
            )
        )
    return KVCacheConfig(
        num_blocks=sc.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(groups)
        ],
    )


def make_scheduler(sc: Scenario, seed: int):
    cfg = dataclasses.replace(
        FuzzConfig.sample(seed),
        block_size=sc.block_size,
        num_blocks=sc.num_blocks,
        layout="full",
        enable_prefix_caching=True,
        max_num_seqs=sc.max_num_seqs,
        max_num_batched_tokens=sc.max_num_batched_tokens,
        long_prefill_token_threshold=0,
        num_spec_tokens=sc.num_spec_tokens,
        hash_fn="sha256",
    )
    vllm_config = build_scheduler(cfg, Scheduler).vllm_config
    if sc.layout == "dsv41":
        kv_cache_config = make_dsv41_kv_cache_config(cfg)
    else:
        kv_cache_config = make_kv_cache_config(sc)
    if sc.layout in ("mamba", "mamba_partial"):
        vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role=sc.kv_role,
        kv_connector_extra_config={
            "load_async": sc.load_async,
            "lookup_async": sc.lookup_async,
            "enable_lookup": sc.enable_lookup,
            "save_decode_cache": sc.save_decode_cache,
        },
    )
    vllm_config.num_hash_lookahead_tokens = sc.num_hash_lookahead_tokens
    block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    if sc.layout == "mamba_partial":
        hash_block_size = 16
    kv_cache_config.hash_block_size = hash_block_size
    kv_cache_config.cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
        kv_cache_config, vllm_config, block_size, hash_block_size
    )
    scheduler_cls = AsyncScheduler if sc.async_scheduler else Scheduler
    mss.LookupKeyClient = FakeLookupClient
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    cs = scheduler.connector.connector_scheduler
    cs._num_workers = sc.num_workers
    cs.client.align = cs._lookup_align()
    cs.client.hash_block_size = cs._hash_block_size
    cs.client.num_groups = len(cs._store_group_ids)
    cs.client.rng.seed(seed)
    return scheduler, kv_cache_config, cfg


def mooncake_config(cs) -> dict:
    """What the native connector is constructed from (the PyO3 constructor
    reads the same attributes)."""
    coord = cs._store_coord
    return {
        "kv_role": cs.kv_role,
        "load_async": cs.load_async,
        "lookup_async": cs.lookup_async,
        "enable_lookup": cs.enable_lookup,
        "save_decode_cache": cs.save_decode_cache,
        "store_group_ids": list(cs._store_group_ids),
        "block_size": cs._block_size,
        "hash_block_size": cs._hash_block_size,
        "boundary_state_group_ids": sorted(cs._boundary_state_group_ids),
        "num_workers": cs._num_workers,
        "store": None
        if coord is None
        else {
            "group_block_sizes": [
                g.kv_cache_spec.block_size for g in coord.kv_cache_groups
            ],
            "mamba_group_ids": sorted(coord.mamba_group_ids),
            "hash_block_size": coord.hash_block_size,
            "lcm_block_size": coord.lcm_block_size,
            "use_eagle": coord.use_eagle,
            "num_hash_lookahead_tokens": coord.num_hash_lookahead_tokens,
            "eagle_proof_margin_by_group": sorted(
                coord.eagle_proof_margin_by_group.items()
            ),
            "enable_partial_hash_hits": coord.enable_partial_hash_hits,
        },
    }


class Workload:
    """Requests sharing a few prefixes (so hits and saves overlap), small
    enough to keep traces short; aborts; a fake sampler."""

    def __init__(self, seed: int, sc: Scenario, num_steps: int):
        self.rng = random.Random(seed * 7919 + 1)
        self.sc = sc
        self.num_steps = num_steps
        bs = sc.block_size
        self.prefixes = [
            [
                self.rng.randrange(1, 30000)
                for _ in range(self.rng.randrange(bs, 6 * bs))
            ]
            for _ in range(3)
        ]
        self.next_id = 0

    def new_requests(self, step: int, hasher) -> list[Request]:
        rng = self.rng
        bs = self.sc.block_size
        out = []
        n = rng.choice([0, 0, 1, 1, 2]) if step < self.num_steps * 2 // 3 else 0
        for _ in range(n):
            prefix = rng.choice(self.prefixes) if rng.random() < 0.8 else []
            prefix = prefix[: rng.randrange(0, len(prefix) + 1)]
            suffix = [rng.randrange(1, 30000) for _ in range(rng.randrange(1, 3 * bs))]
            sp = SamplingParams(
                max_tokens=rng.choice([1, 4, 20, 40]),
                ignore_eos=True,
                stop_token_ids=[rng.randrange(1, 64)] if rng.random() < 0.2 else None,
            )
            sp.update_from_generation_config({}, EOS)
            out.append(
                Request(
                    request_id=f"req-{self.next_id}",
                    prompt_token_ids=prefix + suffix,
                    sampling_params=sp,
                    pooling_params=None,
                    arrival_time=float(self.next_id),
                    priority=0,
                    block_hasher=hasher,
                )
            )
            self.next_id += 1
        return out

    def aborts(self, live_ids) -> list[str]:
        if live_ids and self.rng.random() < 0.06:
            return [self.rng.choice(sorted(live_ids))]
        return []


class FakeWorker:
    """The worker half: each rank finishes a store job a random number of
    steps after the step that emitted it; async loads finish one step after
    they start, some of them failed."""

    def __init__(self, seed: int, num_workers: int):
        self.rng = random.Random(seed * 31 + 7)
        self.num_workers = num_workers
        # job -> ranks yet to report
        self.jobs: dict[int, int] = {}
        self.recv_reported: set[str] = set()

    def submit(self, meta) -> None:
        for m in meta.requests:
            if m.can_save and m.store_job_id is not None:
                self.jobs.setdefault(m.store_job_id, self.num_workers)

    def output(self, scheduler, step: int) -> KVConnectorOutput | None:
        rng = self.rng
        completed: dict[int, int] = {}
        for job in list(self.jobs):
            for _ in range(self.jobs[job]):
                if rng.random() < 0.5:
                    completed[job] = completed.get(job, 0) + 1
                    self.jobs[job] -= 1
            if not self.jobs[job]:
                del self.jobs[job]
        items = list(completed.items())
        rng.shuffle(items)
        receiving = {
            rid
            for rid, r in scheduler.requests.items()
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS or r.is_finished()
        }
        self.recv_reported &= receiving
        new_recv = receiving - self.recv_reported
        self.recv_reported |= new_recv
        failed = {
            rid
            for rid in sorted(new_recv)
            if not scheduler.requests[rid].is_finished() and rng.random() < 0.15
        }
        if not (items or new_recv):
            return None
        return KVConnectorOutput(
            finished_recving=new_recv or None,
            failed_recving=failed,
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(dict(items))
            if items
            else None,
        )


def model_output(seed, step, scheduler, so, num_spec_tokens):
    req_ids = list(so.num_scheduled_tokens)
    sampled = []
    drafts_ids, drafts = [], []
    for rid in req_ids:
        req = scheduler.requests[rid]
        if req.num_computed_tokens < req.num_tokens:
            sampled.append([])
            continue
        rng = random.Random(f"{seed}-{rid}-{req.num_output_tokens}-{step}")
        n_spec = len(so.scheduled_spec_decode_tokens.get(rid, ()))
        n = 1 + (rng.randrange(0, n_spec + 1) if n_spec else 0)
        sampled.append([rng.randrange(1, 64) for _ in range(n)])
        if num_spec_tokens:
            drafts_ids.append(rid)
            drafts.append([rng.randrange(1, 64) for _ in range(num_spec_tokens)])
    out = ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)
    return out, (DraftTokenIds(drafts_ids, drafts) if drafts_ids else None)


def kv_json(kv: KVConnectorOutput | None):
    if kv is None:
        return None
    meta = kv.kv_connector_worker_meta
    return {
        "finished_recving": None
        if kv.finished_recving is None
        else list(kv.finished_recving),
        "failed_recving": list(kv.failed_recving),
        "completed_saves": None
        if meta is None
        else [[k, v] for k, v in meta.completed_saves.items()],
    }


def record(
    seed: int, sc: Scenario, num_steps: int = NUM_STEPS, on_schedule=None
) -> str:
    """Run one scenario and return its trace. `on_schedule(scheduler,
    scheduler_output, canonical_meta, recorder)` runs after every step's
    schedule."""
    kv_cache_utils.init_none_hash(sha256)
    scheduler, kv_cache_config, cfg = make_scheduler(sc, seed)
    connector = scheduler.connector
    cs = connector.connector_scheduler
    client = cs.client
    recorder = MetaRecorder(connector)
    hasher = get_request_block_hasher(
        scheduler.hash_block_size, sha256, scheduler.num_hash_lookahead_tokens
    )
    vllm_config = scheduler.vllm_config
    spec = vllm_config.speculative_config
    wl = Workload(seed, sc, num_steps)
    worker = FakeWorker(seed, sc.num_workers)
    lines = [
        {
            "config": {
                **dataclasses.asdict(cfg),
                "scenario": sc.kind,
                "connector": None,
                "kv_events": False,
                "async": sc.async_scheduler,
                "async_scheduling": vllm_config.scheduler_config.async_scheduling,
                "none_hash": bytes(kv_cache_utils.NONE_HASH).hex(),
                "max_model_len": scheduler.max_model_len,
                "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
                "enable_chunked_prefill": (
                    vllm_config.scheduler_config.enable_chunked_prefill
                ),
                "watermark": vllm_config.scheduler_config.watermark,
                "num_lookahead_tokens": scheduler.num_lookahead_tokens,
                "num_prefill_lookahead": scheduler.num_prefill_lookahead,
                "num_hash_lookahead_tokens": scheduler.num_hash_lookahead_tokens,
                "use_eagle_block_drop": scheduler.use_eagle_block_drop,
                "draft_slots": 0
                if spec is None
                else spec.max_num_new_slots_for_drafting,
                "dynamic_sd": scheduler.dynamic_sd_lookup is not None,
                "max_in_flight_tokens": vllm_config.max_in_flight_tokens,
                "max_concurrent_batches": vllm_config.max_concurrent_batches,
                "hash_block_size": scheduler.hash_block_size,
                "scheduler_block_size": scheduler.block_size,
                "groups": [group_json(g) for g in kv_cache_config.kv_cache_groups],
                "retention_interval": kv_cache_config.prefix_cache_retention_interval,
                "needs_kv_cache_zeroing": kv_cache_config.needs_kv_cache_zeroing,
                "cache_hit_alignment_tokens": (
                    kv_cache_config.cache_hit_alignment_tokens
                ),
                "prefix_cacheable_group_ids": list(
                    kv_cache_config.prefix_cacheable_group_ids
                ),
                "scheduler_reserve_full_isl": scheduler.scheduler_reserve_full_isl,
                "cache_block_size": vllm_config.cache_config.block_size,
                "mamba_cache_mode": vllm_config.cache_config.mamba_cache_mode,
                "enable_mamba_shared_prefix_checkpoint": (
                    vllm_config.cache_config.enable_mamba_shared_prefix_checkpoint
                ),
                "max_num_active_reqs": scheduler.max_num_active_reqs,
                "requires_kv_delivery": scheduler.requires_kv_delivery,
                "recompute_kv_load_failures": scheduler.recompute_kv_load_failures,
                "is_kv_producer": vllm_config.kv_transfer_config.is_kv_producer,
                "is_kv_consumer": vllm_config.kv_transfer_config.is_kv_consumer,
                "mooncake": mooncake_config(cs),
            }
        }
    ]

    def emit(line: dict) -> None:
        line["client"], client.log = client.log, []
        lines.append(line)

    pending = deque()
    for step in range(num_steps):
        for request in wl.new_requests(step, hasher):
            scheduler.add_request(request)
            sp = request.sampling_params
            emit(
                {
                    "op": "add",
                    "id": request.request_id,
                    "prompt": list(request.prompt_token_ids),
                    "max_tokens": sp.max_tokens,
                    "min_tokens": sp.min_tokens,
                    "eos": sp.eos_token_id,
                    "stop": list(sp.stop_token_ids or []),
                    "priority": request.priority,
                    "arrival_time": request.arrival_time,
                }
            )
        aborts = wl.aborts(list(scheduler.requests))
        if aborts:
            result = scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
            emit(
                {
                    "op": "abort",
                    "ids": aborts,
                    "result": [r.request_id for r in result],
                    "pool": pool_json(scheduler),
                    "pins": pins_json(cs),
                }
            )
        so = scheduler.schedule()
        meta = so.kv_connector_metadata
        worker.submit(meta)
        out, draft = model_output(seed, step, scheduler, so, sc.num_spec_tokens)
        out.kv_connector_output = worker.output(scheduler, step)
        canonical = canonical_meta(meta, recorder, recorder.take_ops())
        if on_schedule is not None:
            on_schedule(scheduler, so, canonical, recorder)
        emit(
            {
                "op": "schedule",
                "out": so_json(so),
                "meta": canonical,
                "pool": pool_json(scheduler),
                "pins": pins_json(cs),
            }
        )
        pending.append((step, so, out, draft))
        if sc.async_scheduler and len(pending) < 2:
            continue
        sched_step, so0, out0, draft0 = pending.popleft()
        sampled = [list(t) for t in out0.sampled_token_ids]
        kv = kv_json(out0.kv_connector_output)
        eco = scheduler.update_from_output(so0, out0)
        emit(
            {
                "op": "update",
                "sched_step": sched_step,
                "req_ids": list(out0.req_ids),
                "sampled": sampled,
                "kv": kv,
                "eco": eco_json(eco),
                "pool": pool_json(scheduler),
                "pins": pins_json(cs),
                "push_work": connector.has_pending_push_work(),
            }
        )
        if draft0 is not None and not sc.async_scheduler:
            scheduler.update_draft_token_ids(draft0)
            lines.append(
                {
                    "op": "draft",
                    "req_ids": list(draft0.req_ids),
                    "drafts": [list(d) for d in draft0.draft_token_ids],
                }
            )
        lines.append(
            {
                "op": "check",
                "usage": scheduler.get_kv_cache_usage(),
                "counts": list(scheduler.get_request_counts()),
            }
        )
    return "\n".join(json.dumps(line) for line in lines) + "\n"


# The checked-in traces (scenario, seed, steps), kept small.
CHECKED_IN = (("both", 0, 12), ("mamba_partial", 0, 12))


def main() -> None:
    pin_hash_seed()
    out_dir, cases = parse_args(tuple(SCENARIOS), CHECKED_IN, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for kind, seed, steps in cases:
        sc = SCENARIOS[kind]
        if sc.layout == "dsv41" and not os.path.exists(DSV41_KV_CACHE_CONFIG):
            print("skipped", kind, "(no DSV4.1 KV cache config)")
            continue
        name = f"{kind}_{seed}.jsonl"
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(record(seed, sc, steps or NUM_STEPS))
        print("wrote", name)


if __name__ == "__main__":
    main()
