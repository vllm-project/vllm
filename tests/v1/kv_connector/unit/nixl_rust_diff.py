# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential testing of the native (Rust) NIXL scheduler half against the
Python `NixlPullConnectorScheduler` / `NixlPushConnectorScheduler`.

Two Python schedulers run the same randomized P/D workload (the
`rust_core_diff.py` workload with `kv_transfer_params` varied over the fields
the NIXL scheduler reads, the worker half faked): one with the Python NIXL
connector, one whose connector hooks are served by
`vllm._rust_core.RustNixlConnectorScheduler`. Every step's
`NixlConnectorMetadata`, every finished request's `kv_transfer_params` and the
rest of the scheduler outputs must match, down to the pickled bytes.

The workload also feeds
`rust/src/engine-core/src/sched/scheduler/tests/gen_nixl_traces.py`, whose
traces the Rust scheduler replays with the native connector.

    .venv/bin/python -m tests.v1.kv_connector.unit.nixl_rust_diff --seeds 0-20
"""

import argparse
import contextlib
import dataclasses
import io
import os
import pickle
import random
import sys
import types
from collections.abc import Iterator
from typing import Any

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

from tests.v1.core.rust_core_diff import (  # noqa: E402
    DSV41_KV_CACHE_CONFIG,
    FuzzConfig,
    Workload,
    build_scheduler,
    fake_model_output,
    first_diff,
    make_kv_cache_config,
    make_request,
    normalize,
)
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.utils import (  # noqa: E402
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (  # noqa: E402
    base_scheduler,
    pull_scheduler,
    push_scheduler,
)
from vllm.utils.hashing import sha256, sha256_cbor  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.metrics.cache_hit_source import CacheHitSource  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

KINDS = ("pull_d", "pull_p", "pull_bidir", "push", "pull_async")

# The NIXL scheduler stamps deadlines and its metadata with
# `time.perf_counter()`; `frozen_nixl_clock` makes it read `CLOCK[0]`.
CLOCK = [1000.0]


@contextlib.contextmanager
def frozen_nixl_clock() -> Iterator[None]:
    modules = (base_scheduler, pull_scheduler, push_scheduler)
    saved = [m.time for m in modules]  # type: ignore[attr-defined]
    fake = types.SimpleNamespace(perf_counter=lambda: CLOCK[0])
    for m in modules:
        m.time = fake  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for m, t in zip(modules, saved):
            m.time = t  # type: ignore[attr-defined]


@dataclasses.dataclass
class NixlKnobs:
    mode: str  # "pull" | "push"
    role: str  # "producer" | "consumer" | "both"
    bidirectional: bool = False
    host_buffer: bool = False
    use_async: bool = False
    extra: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def sample(cls, kind: str, seed: int) -> "NixlKnobs":
        rng = random.Random(seed * 1009 + sum(map(ord, kind)))
        extra = {
            "kv_lease_duration": rng.choice([30, 12, 7]),
            "decoder_kv_blocks_ttl": rng.choice([480, 3.5]),
            "kv_recompute_threshold": rng.choice([64, 0, 300]),
        }
        knobs = {
            "pull_d": cls("pull", "consumer"),
            "pull_p": cls("pull", "producer", host_buffer=rng.random() < 0.5),
            "pull_bidir": cls("pull", "both", bidirectional=True),
            "push": cls("push", "both", host_buffer=rng.random() < 0.3),
            "pull_async": cls("pull", "both", use_async=True),
        }[kind]
        if knobs.bidirectional:
            extra["bidirectional_kv_xfer"] = True
        if knobs.mode == "push" and rng.random() < 0.5:
            extra["push_registration_timeout"] = 2.5
        knobs.extra = extra
        return knobs


class NixlWorkload(Workload):
    """`Workload` with P/D params varied over the fields the NIXL scheduler
    reads. Requests Python cannot abort before scheduling (a missing remote
    field is a `KeyError` there) are never aborted."""

    def __init__(self, cfg: FuzzConfig, knobs: NixlKnobs, num_groups: int):
        super().__init__(cfg)
        self.knobs = knobs
        self.num_groups = num_groups
        self.no_abort: set[str] = set()

    def kv_transfer_params(self, num_prompt_tokens: int):
        knobs = self.knobs
        rng = random.Random(self.cfg.seed * 31 + self.next_id)
        sides = {"producer": ["decode"], "consumer": ["prefill"]}.get(
            knobs.role, ["decode", "prefill"]
        )
        if knobs.bidirectional:
            sides.append("readback")
        side = rng.choice(sides + [None])
        num_blocks = -(-num_prompt_tokens // self.cfg.block_size)
        if side is None:
            return rng.choice([None, {}, {"unrelated": 1}])
        params: dict[str, Any]
        if side == "decode":
            params = dict(do_remote_decode=True, do_remote_prefill=False)
            if rng.random() < 0.3:
                params["tp_size"] = 2
            return params
        params = dict(
            do_remote_prefill=side == "prefill",
            do_remote_decode=side == "readback",
            remote_engine_id=rng.choice(["remote-engine", "remote-engine-2"]),
            remote_request_id=f"remote-{self.next_id}",
            remote_host="remote-host",
            remote_port=rng.choice([1234, 5678]),
        )
        if rng.random() < 0.5:
            params["remote_block_ids"] = [
                list(range(g, g + num_blocks)) for g in range(self.num_groups)
            ]
        else:
            params["remote_block_ids"] = list(range(num_blocks))
        if knobs.mode == "push" or rng.random() < 0.8:
            params["tp_size"] = rng.choice([1, 2, 4])
        for key, values in (
            ("dcp_size", [1, 2]),
            ("pp_size", [1, 2]),
            ("remote_block_size", [16, None]),
            ("remote_num_tokens", [num_prompt_tokens, num_prompt_tokens // 2, 0]),
            ("remote_blocks_expiry_time", [1234.5, None, 7]),
            ("remote_registration_generation", [0, 3]),
            ("transfer_mode", [knobs.mode]),
        ):
            if rng.random() < 0.5:
                params[key] = rng.choice(values)
        if side == "readback":
            params["remote_num_tokens"] = rng.randrange(0, num_prompt_tokens + 1)
        if knobs.mode == "pull" and rng.random() < 0.1:
            # Invalid params: no transfer (Python warns), no heartbeats.
            del params[rng.choice(["remote_port", "remote_host"])]
            self.no_abort.add(f"req-{self.next_id}")
        if side == "readback" and rng.random() < 0.2:
            # Nothing to read back. (A D-side request without remote blocks
            # trips Python's `num_external_tokens == 0` assertion.)
            params["remote_block_ids"] = []
        return params

    def aborts(self, live_ids):
        return [i for i in super().aborts(live_ids) if i not in self.no_abort]


def make_mamba_kv_cache_config(cfg: FuzzConfig):
    import torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    bs = cfg.block_size
    return KVCacheConfig(
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


def make_scheduler(cfg: FuzzConfig, knobs: NixlKnobs):
    from vllm.v1.structured_output import StructuredOutputManager

    base_layout = "full" if cfg.layout == "mamba" else cfg.layout
    base = build_scheduler(
        dataclasses.replace(cfg, layout=base_layout, nixl=None, connector=None),
        Scheduler,
    )
    vllm_config = base.vllm_config
    if cfg.layout == "mamba":
        vllm_config.cache_config.mamba_cache_mode = "align"
        kv_cache_config = make_mamba_kv_cache_config(cfg)
    else:
        kv_cache_config = make_kv_cache_config(cfg)
    if kv_cache_config.hash_block_size is None:
        # Set by the engine; NIXL's P-side prompt truncation reads it.
        kv_cache_config.hash_block_size = cfg.block_size
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlPushConnector" if knobs.mode == "push" else "NixlConnector",
        kv_role=f"kv_{knobs.role}",
        engine_id="rust-core-diff",
        kv_load_failure_policy=cfg.kv_load_failure_policy,
        kv_buffer_device="cpu" if knobs.host_buffer else "cuda",
        kv_connector_extra_config=dict(knobs.extra),
    )
    scheduler_cls = AsyncScheduler if knobs.use_async else Scheduler
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=cfg.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    return scheduler, kv_cache_config


def sample_config(kind: str, seed: int) -> FuzzConfig:
    cfg = FuzzConfig.sample(seed)
    rng = random.Random(seed * 7727 + sum(map(ord, kind)))
    layouts = ["full", "full_swa", "full_swa_replay", "mamba"]
    if os.path.exists(DSV41_KV_CACHE_CONFIG):
        layouts.append("dsv41")
    cfg.layout = rng.choice(layouts)
    if cfg.layout == "dsv41":
        cfg.block_size = 128
        cfg.num_blocks = rng.choice([96, 512, 4096])
    if cfg.layout in ("mamba", "dsv41"):
        cfg.enable_prefix_caching = True
    if cfg.layout == "mamba":
        cfg.num_spec_tokens = 0
    cfg.kv_load_failure_policy = rng.choice(["recompute", "fail"])
    cfg.num_steps = rng.choice([40, 80])
    return cfg


class RustNixlShim(SupportsHMA):
    """The scheduler-side connector hooks of a Python NIXL connector, served by
    `RustNixlConnectorScheduler` the way the Rust scheduler drives it: the
    request's `kv_transfer_params` are parsed (and a P-side prompt truncated)
    when the request is added, the rest takes plain values."""

    _cache_hit_source = CacheHitSource.P2P

    def __init__(self, connector, scheduler: Scheduler):
        from vllm._rust_core import RustNixlConnectorScheduler

        self.py = connector
        self.scheduler = scheduler
        self.rust = RustNixlConnectorScheduler(connector)
        self.requires_kv_delivery = connector.requires_kv_delivery
        self.supports_divergent_local_hybrid_hits = (
            connector.supports_divergent_local_hybrid_hits
        )

    def __getattr__(self, name: str) -> Any:
        # Hooks with no NIXL state (base defaults, worker-side stats, ...).
        return getattr(self.py, name)

    def on_new_request(self, request) -> None:
        self.rust.add_request(request)
        self.rust.on_new_request(request.request_id)

    def may_load_async(self, request) -> bool:
        return self.rust.may_load_async(request.request_id)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.rust.get_num_new_matched_tokens(
            request.request_id,
            len(request.prompt_token_ids or []),
            request.num_prompt_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        num_computed_blocks = [
            sum(1 for b in group if b.block_hash is not None and not b.is_null)
            for group in blocks.blocks
        ]
        self.rust.update_state_after_alloc(
            request.request_id,
            blocks.get_unhashed_block_ids_all_groups(),
            num_computed_blocks,
            num_external_tokens,
        )

    def build_connector_meta(self, scheduler_output):
        scheduled = []
        for req_id, block_ids, _ in yield_req_data(scheduler_output):
            req = self.scheduler.requests[req_id]
            scheduled.append(
                (
                    req_id,
                    None if block_ids is None else [list(g) for g in block_ids],
                    scheduler_output.num_scheduled_tokens[req_id],
                    req.num_computed_tokens,
                    req.num_prompt_tokens,
                )
            )
        return self.rust.build_connector_meta(scheduled)

    def request_finished(self, request, block_ids):
        raise AssertionError("NIXL supports HMA")

    def request_finished_all_groups(self, request, block_ids):
        return self.rust.request_finished(
            request.request_id,
            request.status.value,
            request.num_computed_tokens,
            [list(g) for g in block_ids],
        )

    def update_connector_output(self, connector_output) -> None:
        sending = connector_output.finished_sending
        recving = connector_output.finished_recving
        self.rust.update_connector_output(
            None if sending is None else list(sending),
            None if recving is None else list(recving),
        )

    def has_pending_push_work(self) -> bool:
        return self.rust.has_pending_push_work()


def fast_pickle(obj: Any) -> bytes:
    """Pickle without the memo: the bytes depend on the values, their types
    and order, but not on which equal objects happen to be shared."""
    f = io.BytesIO()
    p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
    p.fast = True
    p.dump(obj)
    return f.getvalue()


def shared_block_ids(meta) -> list[Any]:
    """The object sharing the metadata's Python builder establishes."""
    out: list[Any] = [
        (rid, m.local_physical_block_ids is m.local_block_ids)
        for reqs in (meta.reqs_to_recv, meta.reqs_to_save)
        for rid, m in reqs.items()
    ]
    for rid, reg in meta.push_registrations.items():
        recv = meta.reqs_to_recv.get(rid)
        out.append(
            (rid, recv is not None and reg["local_block_ids"] is recv.local_block_ids)
        )
    return out


def without_set_order(obj: Any) -> Any:
    """`obj` with its sets replaced by sorted lists, for telling set-order
    differences apart."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, dict):
        return {k: without_set_order(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(without_set_order(v) for v in obj)
    if dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__"):
        return (type(obj).__name__, without_set_order(vars(obj)))
    return obj


@dataclasses.dataclass
class PickleStats:
    compared: int = 0
    identical: int = 0
    identical_unmemoized: int = 0
    set_order_only: int = 0


def compare_pickles(stats: PickleStats, what: str, a: Any, b: Any) -> str | None:
    stats.compared += 1
    if pickle.dumps(a) == pickle.dumps(b):
        stats.identical += 1
        return None
    if fast_pickle(a) == fast_pickle(b):
        stats.identical_unmemoized += 1
        return None
    if fast_pickle(without_set_order(a)) == fast_pickle(without_set_order(b)):
        stats.set_order_only += 1
        return None
    return f"{what}: pickles differ: {a!r:.600} vs {b!r:.600}"


def run_one(kind: str, seed: int, stats: PickleStats) -> tuple[bool, str]:
    """Drive a Python-connector and a Rust-connector scheduler in lockstep."""
    knobs = NixlKnobs.sample(kind, seed)
    cfg = sample_config(kind, seed)
    cfg.nixl = knobs.role
    if knobs.use_async:
        cfg.num_spec_tokens = random.Random(seed).choice([0, 2])
    hash_fn = {"sha256": sha256, "sha256_cbor": sha256_cbor}[cfg.hash_fn]
    init_none_hash(hash_fn)
    with frozen_nixl_clock():
        CLOCK[0] = 1000.0
        ref, kv_cache_config = make_scheduler(cfg, knobs)
        test, _ = make_scheduler(cfg, knobs)
        py_connector = test.connector
        shim = RustNixlShim(py_connector, test)
        test.connector = shim
        hasher = get_request_block_hasher(ref.hash_block_size, hash_fn)
        wl = NixlWorkload(cfg, knobs, len(kv_cache_config.kv_cache_groups))
        dt = random.Random(seed * 13 + 5).choice([0.7, 1.9, 3.3])
        pending: list[Any] = []
        state_ref: dict = {}
        state_test: dict = {}
        for step in range(cfg.num_steps):
            CLOCK[0] = 1000.0 + step * dt
            shim.rust.set_fixed_clock(CLOCK[0])
            if step and step % 37 == 0:
                for s in (ref.connector, py_connector):
                    s.connector_scheduler._registration_generation += 1
                shim.rust.set_registration_generation(
                    py_connector.connector_scheduler._registration_generation
                )
            for spec in wl.new_request_specs():
                a, b = make_request(spec, hasher), make_request(spec, hasher)
                ref.add_request(a)
                test.add_request(b)
                if d := first_diff(normalize(a), normalize(b), "Request"):
                    return False, f"step {step}: {d}"
                if d := first_diff(
                    normalize(a.kv_transfer_params),
                    normalize(b.kv_transfer_params),
                    "kv_transfer_params",
                ):
                    return False, f"step {step}: {d}"
            if aborts := wl.aborts(list(ref.requests)):
                ra = ref.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
                rb = test.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
                if [r.request_id for r in ra] != [r.request_id for r in rb]:
                    return False, f"step {step}: finish_requests"
            so_a, so_b = ref.schedule(), test.schedule()
            meta_a, meta_b = so_a.kv_connector_metadata, so_b.kv_connector_metadata
            if d := compare_pickles(stats, "NixlConnectorMetadata", meta_a, meta_b):
                return False, f"step {step}: {d}"
            if shared_block_ids(meta_a) != shared_block_ids(meta_b):
                return False, f"step {step}: block id sharing differs"
            if d := first_diff(normalize(so_a), normalize(so_b), "SchedulerOutput"):
                return False, f"step {step}: {d}"
            out_a, draft_a = fake_model_output(cfg, step, ref, so_a, None, state_ref)
            out_b, draft_b = fake_model_output(cfg, step, test, so_b, None, state_test)
            pending.append((so_a, so_b, out_a, out_b, draft_a, draft_b))
            if knobs.use_async and len(pending) < 2:
                continue
            so_a, so_b, out_a, out_b, draft_a, draft_b = pending.pop(0)
            eco_a = ref.update_from_output(so_a, out_a)
            eco_b = test.update_from_output(so_b, out_b)
            if d := first_diff(normalize(eco_a), normalize(eco_b), "EngineCoreOutputs"):
                return False, f"step {step}: {d}"
            for client, outs in eco_a.items():
                for oa, ob in zip(outs.outputs, eco_b[client].outputs):
                    if oa.kv_transfer_params is None:
                        continue
                    if d := compare_pickles(
                        stats,
                        f"kv_transfer_params of {oa.request_id}",
                        oa.kv_transfer_params,
                        ob.kv_transfer_params,
                    ):
                        return False, f"step {step}: {d}"
            if draft_a is not None and not knobs.use_async:
                ref.update_draft_token_ids(draft_a)
                test.update_draft_token_ids(draft_b)
            a = (ref.get_request_counts(), ref.connector.has_pending_push_work())
            b = (test.get_request_counts(), shim.has_pending_push_work())
            if a != b:
                return False, f"step {step}: counts/push work {a} != {b}"
    return True, "ok"


def _parse_seeds(s: str) -> Iterator[int]:
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            yield from range(int(lo), int(hi))
        else:
            yield int(part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-5")
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()
    stats = PickleStats()
    failures = 0
    for seed in _parse_seeds(args.seeds):
        for kind in args.kinds.split(","):
            ok, msg = run_one(kind, seed, stats)
            print(f"seed={seed} kind={kind} {'OK ' if ok else 'FAIL'} {msg}")
            failures += not ok
    print(stats)
    print(f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
