# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record scheduler traces with the real NIXL connector (scheduler role).

Drives the Python `Scheduler` (or `AsyncScheduler`) with the randomized
workload of `tests/v1/core/rust_core_diff.py` plus P/D requests whose
`kv_transfer_params` vary over the fields the NIXL scheduler half reads, with
the worker half faked (transfers complete or fail a step after they start).
Besides the `gen_traces.py` records it writes each step's
`NixlConnectorMetadata` and the `kv_transfer_params` of finished requests,
as typed JSON (tuples, sets, floats and objects tagged). The Rust test
`sched::scheduler::tests::nixl_differential` replays the inputs against the
Rust scheduler with the native NIXL connector and compares every output.

See `rust/src/engine-core/tests/README.md` for how to run it.
"""

import dataclasses
import json
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, REPO_ROOT)
# The scheduler suite's `gen_traces.py`.
sys.path.insert(
    0, os.path.join(REPO_ROOT, "rust/src/engine-core/src/sched/scheduler/tests")
)

from gen_traces import (  # noqa: E402
    eco_json,
    group_json,
    parse_args,
    pin_hash_seed,
    so_json,
)

from tests.v1.core.rust_core_diff import fake_model_output, make_request  # noqa: E402
from tests.v1.kv_connector.unit.nixl_rust_diff import (  # noqa: E402
    CLOCK,
    KINDS,
    NixlKnobs,
    NixlWorkload,
    frozen_nixl_clock,
    make_scheduler,
    sample_config,
)
from vllm.utils.hashing import sha256, sha256_cbor  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

OUT_DIR = os.path.join(HERE, "nixl_traces")


def tj(x):
    """Typed JSON: tuples, sets, dicts, floats and objects are tagged so the
    replay compares Python types too."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return {"f": x}
    if isinstance(x, tuple):
        return {"t": [tj(v) for v in x]}
    if isinstance(x, list):
        return [tj(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return {"s": sorted(x)}
    if isinstance(x, dict):
        return {"d": [[k, tj(v)] for k, v in x.items()]}
    if dataclasses.is_dataclass(x):
        return {
            "c": type(x).__name__,
            "v": [[f.name, tj(getattr(x, f.name))] for f in dataclasses.fields(x)],
        }
    return {"c": type(x).__name__, "v": [[k, tj(v)] for k, v in vars(x).items()]}


def nixl_config_json(scheduler) -> dict:
    """The resolved settings the Rust connector is built from (the binding
    reads the same attributes)."""
    s = scheduler.connector.connector_scheduler
    pc = s.vllm_config.parallel_config
    return {
        "mode": s._TRANSFER_MODE,
        "engine_id": s.engine_id,
        "side_channel_host": s.side_channel_host,
        "side_channel_port": s.side_channel_port,
        "transfer_tp_size": s.transfer_tp_size,
        "tensor_parallel_size": pc.tensor_parallel_size,
        "decode_context_parallel_size": pc.decode_context_parallel_size,
        "pipeline_parallel_size": pc.pipeline_parallel_size,
        "kv_lease_duration": s._kv_lease_duration,
        "heartbeat_interval": s._heartbeat_interval,
        "use_host_buffer": s.use_host_buffer,
        "is_hma_required": s._is_hma_required,
        "num_kv_cache_groups": len(s.kv_cache_config.kv_cache_groups),
        "transfer_group_ids": list(s.kv_cache_config.transfer_group_ids),
        "blocks_per_sw": s.blocks_per_sw,
        "ssm_spec_blocks": s._ssm_spec_blocks,
        "has_mamba": s._has_mamba,
        "bounded_replay": s._bounded_replay,
        "num_prefill_lookahead_tokens": s.vllm_config.num_prefill_lookahead_tokens,
        "kv_recompute_threshold": s.kv_recompute_threshold,
        "bidirectional_kv_xfer": bool(s.is_bidirectional_kv_xfer_enabled),
        "decoder_kv_blocks_ttl": s.decoder_kv_blocks_ttl,
        "is_kv_producer": s.vllm_config.kv_transfer_config.is_kv_producer,
        "push_registration_timeout": getattr(s, "_push_registration_timeout", 0.0),
        "registration_generation": s._registration_generation,
    }


def config_json(cfg, scheduler, kv_cache_config, knobs) -> dict:
    """`gen_traces.record`'s config, plus the KV transfer roles and the NIXL
    settings."""
    vllm_config = scheduler.vllm_config
    spec = vllm_config.speculative_config
    kv_transfer_config = vllm_config.kv_transfer_config
    return {
        **dataclasses.asdict(cfg),
        "async": knobs.use_async,
        "async_scheduling": vllm_config.scheduler_config.async_scheduling,
        "none_hash": bytes(kv_cache_utils.NONE_HASH).hex(),
        "max_model_len": scheduler.max_model_len,
        "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
        "enable_chunked_prefill": vllm_config.scheduler_config.enable_chunked_prefill,
        "watermark": vllm_config.scheduler_config.watermark,
        "num_lookahead_tokens": scheduler.num_lookahead_tokens,
        "num_prefill_lookahead": scheduler.num_prefill_lookahead,
        "num_hash_lookahead_tokens": scheduler.num_hash_lookahead_tokens,
        "use_eagle_block_drop": scheduler.use_eagle_block_drop,
        "draft_slots": 0 if spec is None else spec.max_num_new_slots_for_drafting,
        "dynamic_sd": scheduler.dynamic_sd_lookup is not None,
        "max_in_flight_tokens": vllm_config.max_in_flight_tokens,
        "max_concurrent_batches": vllm_config.max_concurrent_batches,
        "hash_block_size": scheduler.hash_block_size,
        "groups": [group_json(g) for g in kv_cache_config.kv_cache_groups],
        "retention_interval": kv_cache_config.prefix_cache_retention_interval,
        "needs_kv_cache_zeroing": kv_cache_config.needs_kv_cache_zeroing,
        "prefix_cacheable_group_ids": list(kv_cache_config.prefix_cacheable_group_ids),
        "scheduler_reserve_full_isl": scheduler.scheduler_reserve_full_isl,
        "cache_block_size": vllm_config.cache_config.block_size,
        "mamba_cache_mode": vllm_config.cache_config.mamba_cache_mode,
        "enable_mamba_shared_prefix_checkpoint": (
            vllm_config.cache_config.enable_mamba_shared_prefix_checkpoint
        ),
        "max_num_active_reqs": scheduler.max_num_active_reqs,
        "requires_kv_delivery": scheduler.requires_kv_delivery,
        "recompute_kv_load_failures": scheduler.recompute_kv_load_failures,
        "is_kv_producer": kv_transfer_config.is_kv_producer,
        "is_kv_consumer": kv_transfer_config.is_kv_consumer,
        "nixl_config": nixl_config_json(scheduler),
    }


def kvo_json(kvo):
    if kvo is None:
        return None
    return {
        "recving": None if kvo.finished_recving is None else list(kvo.finished_recving),
        "sending": None if kvo.finished_sending is None else list(kvo.finished_sending),
        "failed": list(kvo.failed_recving or ()),
        "invalid": list(kvo.invalid_block_ids or ()),
    }


def record(kind: str, seed: int, num_steps: int | None = None) -> str:
    # The NIXL scheduler stamps deadlines and metadata with
    # `time.perf_counter()`; the trace drives it with `CLOCK` (recorded as
    # `clock` ops).
    with frozen_nixl_clock():
        return _record(kind, seed, num_steps)


def _record(kind: str, seed: int, num_steps: int | None) -> str:
    knobs = NixlKnobs.sample(kind, seed)
    cfg = sample_config(kind, seed)
    if num_steps is not None:
        cfg.num_steps = num_steps
    cfg.nixl = knobs.role
    if knobs.use_async:
        cfg.num_spec_tokens = random.Random(seed).choice([0, 2])
    hash_fn = {"sha256": sha256, "sha256_cbor": sha256_cbor}[cfg.hash_fn]
    kv_cache_utils.init_none_hash(hash_fn)
    CLOCK[0] = 1000.0
    scheduler, kv_cache_config = make_scheduler(cfg, knobs)
    hasher = get_request_block_hasher(scheduler.hash_block_size, hash_fn)
    wl = NixlWorkload(cfg, knobs, len(kv_cache_config.kv_cache_groups))
    connector = scheduler.connector
    lines = [{"config": config_json(cfg, scheduler, kv_cache_config, knobs)}]
    rng = random.Random(seed * 13 + 5)
    dt = rng.choice([0.7, 1.9, 3.3])
    pending = deque()
    nixl_state: dict = {}
    for step in range(cfg.num_steps):
        CLOCK[0] = 1000.0 + step * dt
        lines.append({"op": "clock", "t": CLOCK[0]})
        if step and step % 37 == 0:
            # The handshake listener republished a changed registration.
            connector.connector_scheduler._registration_generation += 1
            lines.append(
                {
                    "op": "generation",
                    "n": connector.connector_scheduler._registration_generation,
                }
            )
        for spec_ in wl.new_request_specs():
            request = make_request(spec_, hasher)
            sp = request.sampling_params
            line = {
                "op": "add",
                "id": request.request_id,
                "prompt": list(request.prompt_token_ids),
                "max_tokens": sp.max_tokens,
                "min_tokens": sp.min_tokens,
                "eos": sp.eos_token_id,
                "stop": list(sp.stop_token_ids or []),
                "priority": request.priority,
                "arrival_time": request.arrival_time,
                "params": spec_["kv_transfer_params"],
            }
            scheduler.add_request(request)
            # After the P-side truncation.
            line["truncated"] = [
                request.num_prompt_tokens,
                request.max_tokens,
                len(request.block_hashes),
            ]
            lines.append(line)
        aborts = wl.aborts(list(scheduler.requests))
        if aborts:
            result = scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
            lines.append(
                {"op": "abort", "ids": aborts, "result": [r.request_id for r in result]}
            )
        so = scheduler.schedule()
        out, draft = fake_model_output(cfg, step, scheduler, so, None, nixl_state)
        lines.append(
            {
                "op": "schedule",
                "out": so_json(so),
                "meta": tj(so.kv_connector_metadata),
            }
        )
        pending.append((step, so, out, draft))
        if knobs.use_async and len(pending) < 2:
            continue
        sched_step, so0, out0, draft0 = pending.popleft()
        sampled = [list(t) for t in out0.sampled_token_ids]
        eco = scheduler.update_from_output(so0, out0)
        lines.append(
            {
                "op": "update",
                "sched_step": sched_step,
                "req_ids": list(out0.req_ids),
                "sampled": sampled,
                "kvo": kvo_json(out0.kv_connector_output),
                "eco": eco_json(eco),
                "params": [
                    [o.request_id, tj(o.kv_transfer_params)]
                    for outs in eco.values()
                    for o in outs.outputs
                    if o.kv_transfer_params is not None
                ],
            }
        )
        if draft0 is not None and not knobs.use_async:
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
                "push_work": connector.has_pending_push_work(),
            }
        )
    return "\n".join(json.dumps(line) for line in lines) + "\n"


# The checked-in traces (kind, seed, steps): a pull-mode decode side and a
# push-mode producer, kept small.
CHECKED_IN = (("pull_d", 2, 10), ("push", 10, 10))


def main() -> None:
    pin_hash_seed()
    out_dir, cases = parse_args(KINDS, CHECKED_IN, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for kind, seed, steps in cases:
        name = f"{kind}_{seed}.jsonl"
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(record(kind, seed, steps))
        print("wrote", name)


if __name__ == "__main__":
    main()
