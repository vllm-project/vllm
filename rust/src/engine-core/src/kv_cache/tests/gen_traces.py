# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record KV cache manager traces from the Python implementation.

Drives `vllm.v1.core.kv_cache_manager.KVCacheManager` with a randomized,
scheduler-like workload (prefix sharing, chunked prefill, decode, preemption,
frees, prefix-cache resets) and writes every call with its result. The Rust
test `kv_cache::tests::differential` replays the traces against the Rust port
and compares each result line by line.

See `rust/src/engine-core/tests/README.md` for how to run it.
"""

import functools
import os
import random
import sys
from math import lcm

import torch

from vllm import envs
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request, RequestStatus

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def full(bs):
    return FullAttentionSpec(
        block_size=bs, num_kv_heads=1, head_size=1, dtype=torch.float32
    ), f"full {bs}"


def mla(bs):
    return MLAAttentionSpec(
        block_size=bs, num_kv_heads=1, head_size=1, dtype=torch.float32
    ), f"mla {bs}"


def swa(bs, window, extra=0):
    return SlidingWindowSpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=window,
        extra_retained_tokens=extra,
    ), f"swa {bs} {window} {extra}"


def swa_mla(bs, window, bounded_replay):
    return SlidingWindowMLASpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=window,
        bounded_replay=bounded_replay,
    ), f"swa_mla {bs} {window} {int(bounded_replay)}"


def local(bs, chunk):
    return ChunkedLocalAttentionSpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=chunk,
    ), f"local {bs} {chunk}"


def mamba(bs, mode="align", spec_blocks=0, ckpt_blocks=0, ckpt_align=None):
    return MambaSpec(
        block_size=bs,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=spec_blocks,
        num_prefill_checkpoint_blocks=ckpt_blocks,
        prefill_checkpoint_alignment=ckpt_align,
    ), f"mamba {bs} {mode} {spec_blocks} {ckpt_blocks} {ckpt_align or 'none'}"


# name -> (groups, kwargs)
CONFIGS = {
    "full": ([full(16)], dict(num_blocks=40)),
    "full_nocache": ([full(16)], dict(num_blocks=40, enable_caching=False)),
    "full_events": ([full(16)], dict(num_blocks=30, enable_kv_cache_events=True)),
    "full_swa": ([full(16), swa(16, 32)], dict(num_blocks=60)),
    "full_swa_eagle": (
        [full(16), swa(16, 48)],
        dict(num_blocks=70, use_eagle=True, num_lookahead=1),
    ),
    "full_swa_retention": (
        [full(16), swa(16, 32)],
        dict(num_blocks=70, retention_interval=64),
    ),
    "full_swa_retention0": (
        [full(16), swa(16, 32)],
        dict(num_blocks=70, retention_interval=0),
    ),
    "dsv4_like": (
        [mla(64), swa_mla(64, 128, False), swa_mla(64, 128, True)],
        dict(num_blocks=60, enable_kv_cache_events=True),
    ),
    "full_mamba": ([full(16), mamba(16)], dict(num_blocks=60)),
    "full_mamba_fine": (
        [full(16), mamba(64)],
        dict(num_blocks=80, hash_block_size=16, enable_kv_cache_events=True),
    ),
    "full_mamba_fine_lookahead": (
        [full(16), mamba(64)],
        dict(num_blocks=80, hash_block_size=16, num_hash_lookahead_tokens=1),
    ),
    "full_swa_mamba": ([full(16), swa(16, 32), mamba(16)], dict(num_blocks=90)),
    "full_local": ([full(16), local(16, 64)], dict(num_blocks=60)),
    "full32_swa16": ([full(32), swa(16, 32)], dict(num_blocks=60, hash_block_size=16)),
    "mamba_none_spec": (
        [mamba(16, mode="none", spec_blocks=2)],
        dict(num_blocks=40, enable_caching=False),
    ),
    "full_mamba_ckpt": (
        [full(16), mamba(64, ckpt_blocks=1, ckpt_align=16)],
        dict(num_blocks=90, hash_block_size=16),
    ),
    "full_mamba_eagle": (
        [full(16), mamba(16)],
        dict(num_blocks=70, use_eagle=True, num_lookahead=1),
    ),
    "full_mamba_fine_eagle_shared": (
        [full(16), mamba(64)],
        dict(
            num_blocks=90,
            hash_block_size=16,
            use_eagle=True,
            num_lookahead=1,
            shared_ckpt=True,
        ),
    ),
    "full64_mamba64_hash16": (
        [full(64), mamba(64)],
        dict(num_blocks=60, hash_block_size=16, enable_kv_cache_events=True),
    ),
    "full_swa_swa": ([full(16), swa(16, 32), swa(16, 32)], dict(num_blocks=80)),
    "swa_only": ([swa(16, 48)], dict(num_blocks=30)),
    "swa_extra": ([full(16), swa(16, 32, extra=16)], dict(num_blocks=70)),
    "full_mamba_retention": (
        [full(16), mamba(16)],
        dict(num_blocks=70, retention_interval=48),
    ),
    "dcp2": ([full(16)], dict(num_blocks=40, dcp=2)),
}


def fmt_groups(blocks):
    if blocks is None:
        return "None"
    return "|".join(",".join(str(b.block_id) for b in g) for g in blocks)


def fmt_pool_state(pool, usage):
    free = pool.free_block_queue.get_all_free_blocks()
    return f"{','.join(str(b.block_id) for b in free)} {usage:.6f}"


def fmt_hash(h):
    if h is None:
        return "-"
    return str(h) if isinstance(h, int) else h[:8].hex()


def fmt_events(events, medium=False):
    out = []
    for e in events:
        m = f" m={e.medium}" if medium else ""
        if isinstance(e, BlockStored):
            out.append(
                f"S g={e.group_idx} h={','.join(fmt_hash(h) for h in e.block_hashes)}"
                f" p={fmt_hash(e.parent_block_hash)} t={len(e.token_ids)}"
                f" bs={e.block_size} k={e.kv_cache_spec_kind}"
                f" w={e.kv_cache_spec_sliding_window}"
                f" la={e.lookahead_token_ids}{m}"
            )
        elif isinstance(e, BlockRemoved):
            out.append(f"R g={e.group_idx} h={fmt_hash(e.block_hashes[0])}{m}")
        elif isinstance(e, AllBlocksCleared):
            out.append("C")
    return ";".join(out)


def gen(name, groups, opts, seed, num_steps=80):
    rng = random.Random(seed)
    num_blocks = opts.get("num_blocks", 50)
    enable_caching = opts.get("enable_caching", True)
    use_eagle = opts.get("use_eagle", False)
    num_lookahead = opts.get("num_lookahead", 0)
    events = opts.get("enable_kv_cache_events", False)
    lookahead_hash = opts.get("num_hash_lookahead_tokens", 0)
    retention = opts.get("retention_interval")
    dcp = opts.get("dcp", 1)
    shared_ckpt = opts.get("shared_ckpt", False)
    specs = [s for s, _ in groups]
    sched_bs = lcm(*(s.block_size for s in specs)) * dcp
    hash_bs = opts.get("hash_block_size", sched_bs)
    max_model_len = 2048
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"l{i}"], s) for i, s in enumerate(specs)],
        prefix_cache_retention_interval=retention,
    )
    manager = KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=sched_bs,
        hash_block_size=hash_bs,
        enable_caching=enable_caching,
        use_eagle=use_eagle,
        num_hash_lookahead_tokens=lookahead_hash,
        enable_kv_cache_events=events,
        log_stats=True,
        dcp_world_size=dcp,
        enable_mamba_shared_prefix_checkpoint=shared_ckpt,
    )
    use_connector_lookup = config.has_mamba_layers and len(specs) > 1
    lines = [
        (
            f"config num_blocks={num_blocks} hash_block_size={hash_bs} "
            f"scheduler_block_size={sched_bs} enable_caching={int(enable_caching)} "
            f"use_eagle={int(use_eagle)} num_hash_lookahead_tokens={lookahead_hash} "
            f"max_model_len={max_model_len} events={int(events)} "
            f"retention={'none' if retention is None else retention} "
            f"zeroing={int(config.needs_kv_cache_zeroing)} "
            f"int_hashes={int(envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES)}"
            f" dcp={dcp} shared_ckpt={int(shared_ckpt)}"
        )
    ]
    lines += [f"group {desc}" for _, desc in groups]

    prefixes = [
        [rng.randrange(1000) for _ in range(rng.randrange(8, 200))] for _ in range(4)
    ]
    reqs: dict[str, Request] = {}
    synced: dict[str, tuple[int, int, int, int]] = {}
    waiting: list[str] = []
    running: list[str] = []
    counter = 0

    def sync(rid):
        req = reqs[rid]
        n_tok, n_hash, _, _ = synced.get(rid, (0, 0, -1, -1))
        state = (
            req.num_tokens,
            len(req.block_hashes),
            req.num_computed_tokens,
            req.shared_prefix_boundary,
        )
        if synced.get(rid) == state:
            return
        toks = req.all_token_ids[n_tok:]
        hashes = req.block_hashes[n_hash:]
        lines.append(
            f"sync {rid} {req.num_prompt_tokens} {req.num_computed_tokens} "
            f"{req.shared_prefix_boundary} "
            f"{','.join(map(str, toks)) or '-'} "
            # First and last 8 bytes: what events print (bytes / low 64 bits).
            f"{','.join((h[:8] + h[-8:]).hex() for h in hashes) or '-'}"
        )
        synced[rid] = state

    def emit(op, result):
        lines.append(op)
        lines.append(f"= {result}")

    def chunk_for(remaining):
        if rng.random() < 0.5:
            return remaining
        return min(remaining, sched_bs * rng.randrange(1, 4))

    def alloc(rid, num_new, n_computed=0, computed=None):
        req = reqs[rid]
        sync(rid)
        la = num_lookahead if req.num_computed_tokens >= req.num_prompt_tokens else 0
        out = manager.allocate_slots(
            req,
            num_new,
            num_new_computed_tokens=n_computed,
            new_computed_blocks=computed,
            num_lookahead_tokens=la,
        )
        emit(
            f"alloc {rid} {num_new} {n_computed} {la} {int(computed is not None)}",
            fmt_groups(None if out is None else out.blocks),
        )
        return out

    for _step in range(num_steps):
        manager.new_step_starts()
        lines.append("step")
        # New arrivals.
        for _ in range(rng.randrange(0, 3)):
            counter += 1
            rid = f"r{counter}"
            if reqs and rng.random() < 0.3:
                # Resend an earlier prompt.
                tokens = list(reqs[rng.choice(list(reqs))].prompt_token_ids)
            else:
                prefix = rng.choice(prefixes)
                cut = rng.randrange(0, len(prefix) + 1)
                tokens = prefix[:cut] + [
                    rng.randrange(1000) for _ in range(rng.randrange(1, 120))
                ]
            sp = SamplingParams(max_tokens=64)
            sp.update_from_generation_config({}, eos_token_id=100000)
            reqs[rid] = Request(
                request_id=rid,
                prompt_token_ids=tokens,
                sampling_params=sp,
                pooling_params=None,
                block_hasher=get_request_block_hasher(hash_bs, sha256, lookahead_hash),
            )
            waiting.append(rid)
        # Running requests.
        for rid in list(running):
            req = reqs[rid]
            if req.num_computed_tokens >= req.num_tokens:
                if rng.random() < 0.1 or req.num_tokens >= max_model_len - 8:
                    running.remove(rid)
                    sync(rid)
                    emit(
                        f"finalize {rid}",
                        str(manager.finalize_partial_tail_offloads(req)),
                    )
                    lines.append(f"free {rid}")
                    manager.free(req)
                    continue
                req.append_output_token_ids(rng.randrange(1000))
            remaining = req.num_tokens - req.num_computed_tokens
            num_new = chunk_for(remaining)
            out = alloc(rid, num_new)
            if out is None:
                # Preempt.
                running.remove(rid)
                lines.append(f"free {rid}")
                manager.free(req)
                req.num_computed_tokens = 0
                waiting.append(rid)
            else:
                req.num_computed_tokens += num_new
        # Waiting requests.
        for rid in list(waiting):
            req = reqs[rid]
            sync(rid)
            if use_connector_lookup and rng.random() < 0.5:
                computed, n, boundary, diverged = (
                    manager.get_computed_blocks_for_connector(req)
                )
                emit(
                    f"getc {rid}",
                    f"{n} {boundary} {int(diverged)} {fmt_groups(computed.blocks)}",
                )
            else:
                computed, n, boundary = manager.get_computed_blocks(req)
                emit(f"get {rid}", f"{n} {boundary} {fmt_groups(computed.blocks)}")
            req.shared_prefix_boundary = boundary
            lines.append(f"record {rid} {n}")
            manager.record_prefix_cache_stats(req, n)
            num_new = chunk_for(req.num_tokens - n)
            out = alloc(rid, num_new, n, computed)
            if out is None:
                break
            waiting.remove(rid)
            running.append(rid)
            req.num_computed_tokens = n + num_new
        for rid in running[:2]:
            emit(f"blocks {rid}", fmt_groups(manager.get_blocks(rid).blocks))
            sync(rid)
            emit(f"estimate {rid}", str(manager.estimate_cached_tokens(reqs[rid])))
            n = reqs[rid].num_computed_tokens
            emit(
                f"ids_computed {rid} {n}",
                "|".join(
                    ",".join(map(str, g))
                    for g in manager.get_block_ids_for_computed_tokens(rid, n)
                ),
            )
        for rid in waiting[:1]:
            sync(rid)
            emit(f"cached {rid}", str(manager.get_num_cached_tokens(reqs[rid])))
        copies, retained = manager.take_kv_cache_block_copies()
        emit(
            "copies",
            ",".join(f"{c.src_block_id}>{c.dst_block_id}" for c in copies)
            + " "
            + ",".join(str(b.block_id) for b in retained),
        )
        manager.block_pool.free_blocks(retained)
        emit(
            "offloads",
            ";".join(
                f"{k}:{v}" for k, v in manager.take_boundary_state_offloads().items()
            ),
        )
        emit("zero", ",".join(map(str, manager.take_new_block_ids())))
        if rng.random() < 0.05:
            evict_set = {rng.randrange(1, num_blocks) for _ in range(3)}
            ids = list(evict_set)
            lines.append(f"evict {','.join(map(str, ids))}")
            manager.evict_blocks(evict_set)
        emit("state", fmt_pool_state(manager.block_pool, manager.usage))
        if events:
            emit("events", fmt_events(manager.take_events()))
        if rng.random() < 0.03:
            # Free everything and reset the prefix cache.
            for rid in running:
                lines.append(f"free {rid}")
                manager.free(reqs[rid])
                reqs[rid].num_computed_tokens = 0
                waiting.append(rid)
            running.clear()
            emit("reset", str(manager.reset_prefix_cache()))
    stats = manager.make_prefix_cache_stats()
    emit("stats", f"{stats.requests} {stats.queries} {stats.hits}")
    return lines


HISPARSE_BS = 16

# name -> options of `gen_hisparse`.
HISPARSE_CONFIGS = {
    "hisparse": dict(num_blocks=48, host_blocks=80, enable_caching=True, events=True),
    "hisparse_nocache": dict(num_blocks=40, host_blocks=60, enable_caching=False),
    "hisparse_hash8": dict(
        num_blocks=48,
        host_blocks=64,
        enable_caching=True,
        events=True,
        hash_block_size=8,
    ),
    "hisparse_2res_mtp": dict(
        num_blocks=64,
        host_blocks=80,
        enable_caching=True,
        num_resident_groups=2,
        num_prefill_lookahead=3,
        import_step_tokens=3,
        max_in_flight_tokens=64,
    ),
}


def hisparse_groups(num_resident_groups):
    """The layout of `make_hisparse_kv_cache_config` in
    `tests/v1/core/test_prefix_caching.py`."""
    source = FullAttentionSpec(
        block_size=HISPARSE_BS, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    groups = [
        (
            KVCacheGroupSpec(
                ["source"],
                source,
                role=KVCacheGroupRole.HISPARSE_SOURCE,
                host_resident=True,
            ),
            f"source {HISPARSE_BS}",
        ),
        (
            KVCacheGroupSpec(
                ["indexer"], source, role=KVCacheGroupRole.HISPARSE_INDEXER
            ),
            f"indexer {HISPARSE_BS}",
        ),
    ]
    for i in range(num_resident_groups):
        groups.append(
            (
                KVCacheGroupSpec(
                    [f"resident_{i}"],
                    HiSparseResidentSpec(
                        block_size=HISPARSE_BS, page_size=HISPARSE_BS * 4
                    ),
                ),
                f"resident {HISPARSE_BS}",
            )
        )
        groups.append(
            (
                KVCacheGroupSpec(
                    [f"hot_{i}"],
                    HiSparseHotSpec(
                        block_size=HISPARSE_BS,
                        page_size=HISPARSE_BS * 4,
                        blocks_per_request=2,
                    ),
                    enable_kv_transfer=False,
                ),
                f"hot {HISPARSE_BS} 2",
            )
        )
    return groups


def gen_hisparse(name, opts, seed, num_steps=70):
    """A scheduler-like workload driving a HiSparse KV cache manager and its
    coordinator as the HiSparse connector does, with a simulated worker that
    completes host transfers and copies at random."""
    rng = random.Random(seed)
    num_blocks = opts["num_blocks"]
    host_blocks = opts["host_blocks"]
    enable_caching = opts["enable_caching"]
    events = opts.get("events", False)
    num_prefill_lookahead = opts.get("num_prefill_lookahead", 0)
    import_step_tokens = opts.get("import_step_tokens", 0)
    max_in_flight_tokens = opts.get("max_in_flight_tokens")
    groups = hisparse_groups(opts.get("num_resident_groups", 1))
    hash_bs = opts.get("hash_block_size", HISPARSE_BS)
    max_model_len = 512
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[g for g, _ in groups],
        hisparse_host_num_blocks=host_blocks,
    )
    manager = KVCacheManager(
        config,
        max_model_len=max_model_len,
        max_in_flight_tokens=max_in_flight_tokens,
        scheduler_block_size=HISPARSE_BS,
        hash_block_size=hash_bs,
        enable_caching=enable_caching,
        num_prefill_lookahead=num_prefill_lookahead,
        enable_kv_cache_events=events,
        log_stats=True,
    )
    coordinator = get_hisparse_coordinator(
        manager, import_step_tokens=import_step_tokens
    )
    host_pool = coordinator.host_manager.block_pool
    lines = [
        (
            f"config num_blocks={num_blocks} hash_block_size={hash_bs} "
            f"scheduler_block_size={HISPARSE_BS} "
            f"enable_caching={int(enable_caching)} "
            f"use_eagle=0 num_hash_lookahead_tokens=0 "
            f"max_model_len={max_model_len} events={int(events)} retention=none "
            f"zeroing={int(config.needs_kv_cache_zeroing)} "
            f"int_hashes={int(envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES)} dcp=1 "
            f"shared_ckpt=0 hisparse_host_blocks={host_blocks} "
            f"import_step_tokens={import_step_tokens} "
            f"max_in_flight_tokens={max_in_flight_tokens or 'none'} "
            f"num_prefill_lookahead={num_prefill_lookahead}"
        )
    ]
    lines += [f"group {desc}" for _, desc in groups]

    prefixes = [
        [rng.randrange(1000) for _ in range(rng.randrange(8, 160))] for _ in range(3)
    ]
    reqs: dict[str, Request] = {}
    synced: dict[str, tuple] = {}
    waiting: list[str] = []
    running: list[str] = []
    # rid -> (local computed tokens, external tokens) of an async load.
    loading: dict[str, tuple[int, int]] = {}
    in_flight: list[int] = []
    enqueued: set[int] = set()
    copies_in_flight: list[int] = []
    deferred_frees: list[list] = []
    counter = 0

    def sync(rid):
        req = reqs[rid]
        n_tok, n_hash, _, _ = synced.get(rid, (0, 0, -1, -1))
        state = (
            req.num_tokens,
            len(req.block_hashes),
            req.num_computed_tokens,
            req.shared_prefix_boundary,
        )
        if synced.get(rid) == state:
            return
        toks = req.all_token_ids[n_tok:]
        hashes = req.block_hashes[n_hash:]
        lines.append(
            f"sync {rid} {req.num_prompt_tokens} {req.num_computed_tokens} "
            f"{req.shared_prefix_boundary} "
            f"{','.join(map(str, toks)) or '-'} "
            f"{','.join((h[:8] + h[-8:]).hex() for h in hashes) or '-'}"
        )
        synced[rid] = state

    def emit(op, result):
        lines.append(op)
        lines.append(f"= {result}")

    def fmt_block(b):
        return f"h{b.block_id}" if b.pool() is host_pool else str(b.block_id)

    def free(rid, finished):
        req = reqs[rid]
        if finished:
            req.status = RequestStatus.FINISHED_STOPPED
        if rng.random() < 0.3:
            blocks = manager.pop_blocks_for_free(req)
            emit(f"pop {rid} {int(finished)}", ",".join(map(fmt_block, blocks)))
            deferred_frees.append(list(reversed(blocks)))
        else:
            lines.append(f"free {rid} {int(finished)}")
            manager.free(req)

    def alloc(rid, num_new, n_computed=0, computed=None, ext=0, cap=False):
        req = reqs[rid]
        sync(rid)
        delay = ext > 0
        out = manager.allocate_slots(
            req,
            num_new,
            num_new_computed_tokens=n_computed,
            new_computed_blocks=computed,
            num_external_computed_tokens=ext,
            delay_cache_blocks=delay,
            full_sequence_must_fit=cap,
        )
        emit(
            f"alloc {rid} {num_new} {n_computed} 0 {int(computed is not None)} "
            f"{ext} {int(cap)} {int(delay)}",
            fmt_groups(None if out is None else out.blocks),
        )
        return out

    def chunk_for(remaining):
        if rng.random() < 0.5:
            return remaining
        return min(remaining, HISPARSE_BS * rng.randrange(1, 4))

    for _step in range(num_steps):
        manager.new_step_starts()
        lines.append("step")
        for blocks in deferred_frees:
            lines.append(f"freeblocks {','.join(map(fmt_block, blocks)) or '-'}")
            manager.block_pool.free_blocks(blocks)
        deferred_frees.clear()
        for _ in range(rng.randrange(0, 3)):
            counter += 1
            rid = f"r{counter}"
            if reqs and rng.random() < 0.3:
                tokens = list(reqs[rng.choice(list(reqs))].prompt_token_ids)
            else:
                prefix = rng.choice(prefixes)
                cut = rng.randrange(0, len(prefix) + 1)
                tokens = prefix[:cut] + [
                    rng.randrange(1000) for _ in range(rng.randrange(1, 100))
                ]
            sp = SamplingParams(max_tokens=48)
            sp.update_from_generation_config({}, eos_token_id=100000)
            reqs[rid] = Request(
                request_id=rid,
                prompt_token_ids=tokens,
                sampling_params=sp,
                pooling_params=None,
                block_hasher=get_request_block_hasher(hash_bs, sha256, 0),
            )
            waiting.append(rid)
        scheduled: list[tuple[str, int, int]] = []
        # Finished async loads.
        for rid in list(loading):
            if rng.random() < 0.5:
                continue
            n, ext = loading.pop(rid)
            failed = rng.random() < 0.15
            lines.append(f"finish_import {rid} {int(failed)}")
            coordinator.finish_host_import(rid, failed=failed)
            req = reqs[rid]
            if failed:
                free(rid, False)
                req.num_computed_tokens = 0
                waiting.append(rid)
                continue
            req.num_computed_tokens = n + ext
            sync(rid)
            lines.append(f"cache {rid} {req.num_computed_tokens}")
            manager.cache_blocks(req, req.num_computed_tokens)
            running.append(rid)
        for rid in list(running):
            req = reqs[rid]
            if req.num_computed_tokens >= req.num_tokens:
                if rng.random() < 0.1 or req.num_tokens >= max_model_len - 8:
                    running.remove(rid)
                    free(rid, True)
                    continue
                req.append_output_token_ids(rng.randrange(1000))
            remaining = req.num_tokens - req.num_computed_tokens
            num_new = chunk_for(remaining)
            out = alloc(rid, num_new)
            if out is None:
                running.remove(rid)
                free(rid, False)
                req.num_computed_tokens = 0
                waiting.append(rid)
            else:
                scheduled.append((rid, req.num_computed_tokens, num_new))
                req.num_computed_tokens += num_new
        for rid in list(waiting):
            req = reqs[rid]
            sync(rid)
            computed, n, boundary = manager.get_computed_blocks(req)
            emit(f"get {rid}", f"{n} {boundary} {fmt_groups(computed.blocks)}")
            req.shared_prefix_boundary = boundary
            remaining = req.num_tokens - n
            ext = 0
            if remaining > HISPARSE_BS and rng.random() < 0.25:
                ext = rng.randrange(1, remaining)
            cap = rng.random() < 0.5
            if ext:
                out = alloc(rid, 0, n, computed, ext=ext, cap=cap)
                if out is None:
                    break
                waiting.remove(rid)
                loading[rid] = (n, ext)
                continue
            num_new = chunk_for(remaining)
            out = alloc(rid, num_new, n, computed, cap=cap)
            if out is None:
                break
            waiting.remove(rid)
            running.append(rid)
            scheduled.append((rid, n, num_new))
            req.num_computed_tokens = n + num_new
        # HiSparseConnectorScheduler.build_connector_meta.
        advance = [
            (rid, min(start + count, reqs[rid].num_tokens))
            for rid, start, count in scheduled
        ]
        lines.append("advance " + (",".join(f"{rid}:{n}" for rid, n in advance) or "-"))
        coordinator.advance_scheduled(advance)
        command = coordinator.build_offload_command()
        emit(
            "offload",
            ";".join(
                f"{t.transfer_id}:{t.host_block_id}:"
                f"{'/'.join(map(str, t.resident_block_ids))}:{int(t.restore)}"
                for t in command.page_transfers
            ),
        )
        in_flight.extend(t.transfer_id for t in command.page_transfers)
        host_copies = coordinator.take_host_block_copies()
        emit(
            "hcopies",
            ",".join(f"{c.src_block_id}>{c.dst_block_id}" for c in host_copies),
        )
        for rid, start, count in scheduled:
            mirrors = coordinator.build_row_mirrors(((rid, start, count),))
            emit(
                f"mirrors {rid} {start} {count}",
                ";".join(
                    f"{'/'.join(map(str, m.source_starts))}:"
                    f"{m.destination_start}:{m.num_rows}"
                    for m in mirrors
                ),
            )
        sched = ",".join(f"{rid}:{s}:{c}" for rid, s, c in scheduled) or "-"
        emit(f"allres {sched}", str(coordinator.all_context_pages_resident(scheduled)))
        updates = coordinator.take_residency_updates([rid for rid, _, _ in scheduled])
        emit(
            f"updates {','.join(rid for rid, _, _ in scheduled) or '-'}",
            ";".join(
                f"{rid}:{','.join(map(str, u.pages))}:"
                f"{'/'.join(','.join(map(str, g)) for g in u.block_ids)}"
                for rid, u in updates.items()
            ),
        )
        # HiSparseConnectorScheduler.update_connector_output, one transfer at
        # a time: Python publishes the requests completed by one call in set
        # order, which depends on PYTHONHASHSEED.
        if copies_in_flight:
            lines.append(f"release {','.join(map(str, copies_in_flight))}")
            coordinator.release_completed_host_copies(copies_in_flight)
        copies_in_flight = [c.dst_block_id for c in host_copies]
        for spill_id in list(in_flight):
            r = rng.random()
            if r < 0.4:
                continue
            enq = {spill_id: 1} if spill_id not in enqueued else {}
            comp = {spill_id: 1} if r < 0.85 else {}
            lines.append(f"spill {spill_id} {len(enq)} {len(comp)}")
            coordinator.update_spills(enq, comp)
            enqueued.add(spill_id)
            if comp:
                in_flight.remove(spill_id)
        emit(
            "pending",
            f"{int(coordinator.has_pending_work())} "
            f"{int(coordinator.has_pending_reclamation())}",
        )
        for rid in running[:2]:
            emit(f"blocks {rid}", fmt_groups(manager.get_blocks(rid).blocks))
            emit(
                f"row {rid}",
                "|".join(
                    ",".join(str(b.block_id) for b in m.get_residency_row(rid))
                    for m in coordinator.resident_managers
                ),
            )
            sync(rid)
            emit(f"estimate {rid}", str(manager.estimate_cached_tokens(reqs[rid])))
        emit("zero", ",".join(map(str, manager.take_new_block_ids())))
        emit("state", fmt_pool_state(manager.block_pool, manager.usage))
        emit("hstate", fmt_pool_state(host_pool, host_pool.get_usage()))
        if events:
            emit("events_m", fmt_events(manager.take_events(), medium=True))
        if rng.random() < 0.03:
            for rid in running:
                free(rid, False)
                reqs[rid].num_computed_tokens = 0
                waiting.append(rid)
            running.clear()
            emit("reset", str(manager.reset_prefix_cache()))
    stats = manager.make_prefix_cache_stats()
    emit("stats", f"{stats.requests} {stats.queries} {stats.hits}")
    return lines


# The checked-in traces (config, seed, steps), kept small.
CHECKED_IN = (
    ("full_swa_mamba", 0, 12),
    ("full_mamba_fine_eagle_shared", 0, 12),
    ("hisparse", 0, 12),
)


def main():
    """Without `--seeds`, regenerate the checked-in traces; see
    `rust/src/engine-core/tests/README.md`."""
    import argparse

    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    jobs = {
        name: functools.partial(gen, name, groups, opts)
        for name, (groups, opts) in CONFIGS.items()
    } | {
        name: functools.partial(gen_hisparse, name, opts)
        for name, opts in HISPARSE_CONFIGS.items()
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", help="a count or `lo-hi`: every config per seed")
    parser.add_argument("--out", default=OUT_DIR, help="output directory")
    parser.add_argument("--steps", type=int, help="override the steps per trace")
    parser.add_argument("names", nargs="*", help="only these configs")
    args = parser.parse_args()
    if args.seeds is None:
        cases = [(n, s, args.steps or k) for n, s, k in CHECKED_IN]
    else:
        lo, hi = args.seeds.split("-") if "-" in args.seeds else (0, args.seeds)
        cases = [(n, s, args.steps) for s in range(int(lo), int(hi)) for n in jobs]
    cases = [c for c in cases if not args.names or c[0] in args.names]
    init_none_hash(sha256)
    os.makedirs(args.out, exist_ok=True)
    for name, seed, steps in cases:
        kwargs = {} if steps is None else {"num_steps": steps}
        lines = jobs[name](seed, **kwargs)
        path = os.path.join(args.out, f"{name}_{seed}.trace")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(path, len(lines), file=sys.stderr)


if __name__ == "__main__":
    main()
