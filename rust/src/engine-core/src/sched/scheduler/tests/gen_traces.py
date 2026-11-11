# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record scheduler traces from the Python implementation.

Drives `vllm.v1.core.sched.scheduler.Scheduler` (or `AsyncScheduler`) with
the randomized workload of `tests/v1/core/rust_core_diff.py` (shared prefixes,
aborts, spec-decode drafts, stop tokens, block pressure, priorities) and
writes every input and every `SchedulerOutput` / `EngineCoreOutputs`,
normalized to JSON. The Rust test `sched::scheduler::tests::differential`
replays the inputs against the Rust scheduler and compares each output.

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

from tests.v1.core.rust_core_diff import (  # noqa: E402
    FuzzConfig,
    Workload,
    build_scheduler,
    fake_model_output,
    make_kv_cache_config,
    make_request,
)
from vllm.utils.hashing import sha256, sha256_cbor  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher  # noqa: E402
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def so_json(so):
    cached = so.scheduled_cached_reqs
    return {
        "new": [
            [
                r.req_id,
                r.num_computed_tokens,
                [list(g) for g in r.block_ids],
                len(r.prefill_token_ids),
                sum(r.prefill_token_ids),
                r.replay_start,
            ]
            for r in so.scheduled_new_reqs
        ],
        "cached": [
            list(cached.req_ids),
            [list(t) for t in cached.new_token_ids],
            [None if b is None else [list(g) for g in b] for b in cached.new_block_ids],
            list(cached.num_computed_tokens),
            list(cached.num_output_tokens),
        ],
        "nst": [[k, v] for k, v in so.num_scheduled_tokens.items()],
        "total": so.total_num_scheduled_tokens,
        "spec": [[k, list(v)] for k, v in so.scheduled_spec_decode_tokens.items()],
        "finished": sorted(so.finished_req_ids),
        "preempted": None
        if so.preempted_req_ids is None
        else sorted(so.preempted_req_ids),
        "zero": so.new_block_ids_to_zero,
        "copies": None
        if so.kv_cache_block_copies is None
        else [[c.src_block_id, c.dst_block_id] for c in so.kv_cache_block_copies],
        "k": so.num_spec_tokens_to_schedule,
        "skip_drafts": so.skip_drafts,
        "sync_kv": so.has_sync_kv_loads,
    }


def prefix_stats_json(s):
    return [
        s.reset,
        s.requests,
        s.queries,
        s.hits,
        s.preempted_requests,
        s.preempted_queries,
        s.preempted_hits,
    ]


def eco_json(ecos):
    out = []
    for client, eco in ecos.items():
        stats = eco.scheduler_stats
        spec = None if stats is None else stats.spec_decoding_stats
        out.append(
            [
                client,
                {
                    "outputs": [
                        [
                            o.request_id,
                            list(o.new_token_ids),
                            None if o.finish_reason is None else int(o.finish_reason),
                            o.stop_reason,
                            o.num_nans_in_logits,
                            None
                            if o.prefill_stats is None
                            else dataclasses.astuple(o.prefill_stats),
                        ]
                        for o in eco.outputs
                    ],
                    "finished": None
                    if eco.finished_requests is None
                    else sorted(eco.finished_requests),
                    "stats": None
                    if stats is None
                    else [
                        stats.num_running_reqs,
                        stats.num_waiting_reqs,
                        stats.num_skipped_waiting_reqs,
                        stats.kv_cache_usage,
                        prefix_stats_json(stats.prefix_cache_stats),
                        None
                        if spec is None
                        else [
                            spec.num_drafts,
                            spec.num_draft_tokens,
                            spec.num_accepted_tokens,
                            spec.num_accepted_tokens_per_pos,
                        ],
                    ],
                },
            ]
        )
    return out


def group_json(group):
    spec = group.kv_cache_spec
    return {
        "type": type(spec).__name__,
        "block_size": spec.block_size,
        "sliding_window": getattr(spec, "sliding_window", None),
        "extra_retained_tokens": getattr(spec, "extra_retained_tokens", 0),
        "bounded_replay": getattr(spec, "bounded_replay", False),
        "mamba_cache_mode": getattr(spec, "mamba_cache_mode", None),
        "num_speculative_blocks": getattr(spec, "num_speculative_blocks", 0),
        "num_prefill_checkpoint_blocks": getattr(
            spec, "num_prefill_checkpoint_blocks", 0
        ),
        "prefill_checkpoint_alignment": getattr(
            spec, "prefill_checkpoint_alignment", None
        ),
    }


def make_scheduler(cfg, kind: str):
    """The scheduler under test and its KV cache config. The `mamba` kind
    pairs full attention with a Mamba group in "align" cache mode."""
    scheduler_cls = AsyncScheduler if kind == "async" else Scheduler
    if kind != "mamba":
        return build_scheduler(cfg, scheduler_cls), make_kv_cache_config(cfg)
    import torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    base = build_scheduler(dataclasses.replace(cfg, layout="full"), Scheduler)
    vllm_config = base.vllm_config
    vllm_config.cache_config.mamba_cache_mode = "align"
    bs = cfg.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=bs, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=bs,
                    shapes=((2, 512), (3, 32, 32)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=bs,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    return scheduler, kv_cache_config


def log_connector_calls(connector, log: list) -> None:
    """Record the scheduler's calls into the connector, with its answers."""
    match = connector.get_num_new_matched_tokens
    alloc = connector.update_state_after_alloc
    finished = connector.request_finished_all_groups

    def get_num_new_matched_tokens(request, num_computed_tokens):
        result = match(request, num_computed_tokens)
        log.append(["match", request.request_id, num_computed_tokens, *result])
        return result

    def update_state_after_alloc(request, blocks, num_external_tokens):
        log.append(
            [
                "alloc",
                request.request_id,
                num_external_tokens,
                [list(g) for g in blocks.get_block_ids()],
            ]
        )
        return alloc(request, blocks, num_external_tokens)

    def request_finished_all_groups(request, block_ids):
        log.append(["finished", request.request_id, [list(g) for g in block_ids]])
        return finished(request, block_ids)

    connector.get_num_new_matched_tokens = get_num_new_matched_tokens
    connector.update_state_after_alloc = update_state_after_alloc
    connector.request_finished_all_groups = request_finished_all_groups


def record(seed: int, kind: str, num_steps: int | None = None) -> str:
    use_async = kind == "async"
    cfg = FuzzConfig.sample_extended(seed) if kind == "ext" else FuzzConfig.sample(seed)
    if num_steps is not None:
        cfg.num_steps = num_steps
    rng = random.Random(seed)
    if use_async:
        # Spec decode under async scheduling advances drafts in the worker.
        cfg.num_spec_tokens = rng.choice([0, 2])
    if kind == "mamba":
        cfg.layout = "mamba"
        cfg.num_spec_tokens = 0
        cfg.enable_prefix_caching = True
    hash_fn = {"sha256": sha256, "sha256_cbor": sha256_cbor}[cfg.hash_fn]
    kv_cache_utils.init_none_hash(hash_fn)
    scheduler, kv_cache_config = make_scheduler(cfg, kind)
    hasher = get_request_block_hasher(scheduler.hash_block_size, hash_fn)
    wl = Workload(cfg)
    vllm_config = scheduler.vllm_config
    spec = vllm_config.speculative_config
    conn_log: list = []
    if scheduler.connector is not None:
        log_connector_calls(scheduler.connector, conn_log)
    lines = [
        {
            "config": {
                **dataclasses.asdict(cfg),
                "async": use_async,
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
                "groups": [group_json(g) for g in kv_cache_config.kv_cache_groups],
                "retention_interval": kv_cache_config.prefix_cache_retention_interval,
                "needs_kv_cache_zeroing": kv_cache_config.needs_kv_cache_zeroing,
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
            }
        }
    ]

    def flush_conn():
        if conn_log:
            lines.append({"op": "conn", "calls": list(conn_log)})
            conn_log.clear()

    pending = deque()
    num_updates = 0
    for step in range(cfg.num_steps):
        for spec_ in wl.new_request_specs():
            request = make_request(spec_, hasher)
            sp = request.sampling_params
            lines.append(
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
            scheduler.add_request(request)
            flush_conn()
        aborts = wl.aborts(list(scheduler.requests))
        if aborts:
            result = scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
            lines.append(
                {
                    "op": "abort",
                    "ids": aborts,
                    "result": [r.request_id for r in result],
                }
            )
            flush_conn()
        so = scheduler.schedule()
        out, draft = fake_model_output(cfg, step, scheduler, so)
        lines.append({"op": "schedule", "out": so_json(so)})
        flush_conn()
        pending.append((step, so, out, draft))
        if use_async and len(pending) < 2:
            continue
        sched_step, so0, out0, draft0 = pending.popleft()
        # update_from_output trims the sampled lists in place at a stop.
        sampled = [list(t) for t in out0.sampled_token_ids]
        eco = scheduler.update_from_output(so0, out0)
        lines.append(
            {
                "op": "update",
                "sched_step": sched_step,
                "req_ids": list(out0.req_ids),
                "sampled": sampled,
                "finished_recving": None
                if out0.kv_connector_output is None
                else list(out0.kv_connector_output.finished_recving or ()),
                "eco": eco_json(eco),
            }
        )
        flush_conn()
        num_updates += 1
        if draft0 is not None and not use_async:
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


KINDS = ("sync", "async", "ext", "mamba")
# The checked-in traces (kind, seed, steps): kept small, see
# `rust/src/engine-core/tests/README.md`.
CHECKED_IN = (("async", 1, 12), ("mamba", 0, 12))


def pin_hash_seed() -> None:
    """Re-exec with a fixed `PYTHONHASHSEED`, so that string hashes (set order,
    the fake model's sampling seeds) and `NONE_HASH` repeat across runs."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def parse_args(kinds, checked_in, default_out: str):
    """Parse the CLI shared by the trace generators into `(out_dir, cases)`,
    each case `(kind, seed, steps)` with `steps` None for the default."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        help="a count or `lo-hi`: record every kind for each seed instead of "
        "the checked-in set",
    )
    parser.add_argument("--out", default=default_out, help="output directory")
    parser.add_argument("--steps", type=int, help="override the steps per trace")
    parser.add_argument("--kinds", default=",".join(kinds), help="with --seeds")
    args = parser.parse_args()
    if args.seeds is None:
        cases = [(k, s, args.steps or n) for k, s, n in checked_in]
        return args.out, cases
    lo, hi = args.seeds.split("-") if "-" in args.seeds else (0, args.seeds)
    names = args.kinds.split(",")
    for name in names:
        if name not in kinds:
            parser.error(f"unknown kind {name!r}")
    cases = [(k, s, args.steps) for s in range(int(lo), int(hi)) for k in names]
    return args.out, cases


def main() -> None:
    pin_hash_seed()
    out_dir, cases = parse_args(KINDS, CHECKED_IN, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for kind, seed, steps in cases:
        name = f"{kind}_{seed}.jsonl"
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(record(seed, kind, steps))
        print("wrote", name)


if __name__ == "__main__":
    main()
