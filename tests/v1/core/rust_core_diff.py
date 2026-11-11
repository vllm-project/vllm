# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential fuzzing of the Rust engine core against the Python reference.

Two schedulers are driven with identical random workloads (requests with shared
prefixes, aborts, spec-decode drafts, stop tokens, block pressure) and every
`SchedulerOutput` and `EngineCoreOutputs` they produce must be identical once
normalized to plain Python values.

Run as a script for long fuzzing campaigns:

    .venv/bin/python -m tests.v1.core.rust_core_diff --seeds 0-200

`--async` drives `AsyncScheduler` the way EngineCore's batch queue does: step
N+1 is scheduled before the output of step N is processed.

`--nixl` and `--mooncake` add the real NIXL / Mooncake store connector
(scheduler role, worker side faked). With `--mode rust-sched`, each asserts
that the Rust scheduler runs the native port (`VLLM_RUST_CORE_NATIVE_HOOKS=0`
tests the Python hooks). `--structured` adds xgrammar structured output
requests, sampled from the grammar bitmasks, which are compared every step.
"""

import argparse
import contextlib
import copy
import dataclasses
import enum
import functools
import os
import random
import sys
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import msgspec
import numpy as np
import torch

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

from tests.v1.core.utils import create_scheduler  # noqa: E402
from vllm.config import KVEventsConfig, KVTransferConfig, LoRAConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.sampling_params import (  # noqa: E402
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils.hashing import sha256, sha256_cbor  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (  # noqa: E402
    DraftTokenIds,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

EOS = 50256


@dataclasses.dataclass
class FuzzConfig:
    seed: int
    block_size: int
    num_blocks: int
    layout: str
    enable_prefix_caching: bool
    max_num_seqs: int
    max_num_batched_tokens: int
    long_prefill_token_threshold: int
    policy: str
    num_spec_tokens: int
    hash_fn: str
    num_steps: int
    # Extended knobs (`--extended`); the defaults keep the base configs.
    kv_events: bool = False
    connector: str | None = None
    # The real NixlConnector (scheduler role) as `kv_{nixl}` (`--nixl`).
    nixl: str | None = None
    kv_load_failure_policy: str = "recompute"
    # The real MooncakeStoreConnector (scheduler role, store faked).
    mooncake: bool = False
    # Structured output requests (xgrammar), with this reasoning parser
    # ("" for none).
    structured: bool = False
    reasoning_parser: str = ""
    # Streaming-input sessions (resumable requests and their next chunks).
    streaming: bool = False
    # LoRA adapters on some requests, with `max_loras` bounding the batch.
    lora: bool = False

    @classmethod
    def sample(cls, seed: int) -> "FuzzConfig":
        rng = random.Random(seed)
        block_size = rng.choice([16, 16, 64])
        return cls(
            seed=seed,
            block_size=block_size,
            num_blocks=rng.choice([64, 256, 4096]),
            layout=rng.choice(["full", "full", "full_swa", "full_swa_replay", "swa"]),
            enable_prefix_caching=rng.random() < 0.8,
            max_num_seqs=rng.choice([4, 16, 64]),
            max_num_batched_tokens=rng.choice([256, 1024, 4096]),
            long_prefill_token_threshold=rng.choice([0, 0, 128]),
            policy=rng.choice(["fcfs", "fcfs", "priority"]),
            num_spec_tokens=rng.choice([0, 0, 2]),
            hash_fn=rng.choice(["sha256", "sha256_cbor"]),
            num_steps=rng.choice([60, 150]),
        )

    @classmethod
    def sample_extended(cls, seed: int) -> "FuzzConfig":
        """`sample` plus KV events, a KV connector and the DSV4.1 layout,
        drawn from a separate stream so the base fields stay unchanged."""
        cfg = cls.sample(seed)
        rng = random.Random(seed * 104729 + 3)
        if rng.random() < 0.3 and os.path.exists(DSV41_KV_CACHE_CONFIG):
            cfg.layout = "dsv41"
            cfg.block_size = 128
            cfg.num_blocks = rng.choice([96, 512, 4096])
        cfg.kv_events = rng.random() < 0.5
        cfg.connector = rng.choice([None, "sync", "async"])
        return cfg

    @classmethod
    def sample_nixl(cls, seed: int) -> "FuzzConfig":
        """`sample_extended` with the NixlConnector instead of the test
        connector: P/D requests on the producer, consumer or both sides, with
        the worker side faked (transfers complete or fail one step later)."""
        cfg = cls.sample_extended(seed)
        rng = random.Random(seed * 7727 + 11)
        cfg.connector = None
        cfg.nixl = rng.choice(["producer", "consumer", "both"])
        cfg.kv_load_failure_policy = rng.choice(["recompute", "fail"])
        return cfg

    @classmethod
    def sample_mooncake(cls, seed: int) -> "FuzzConfig":
        """`sample_extended` with the MooncakeStoreConnector, whose lookups
        hit a fake store holding leading blocks of the shared prefixes, and
        without KV events (which its native scheduler half does not port)."""
        cfg = cls.sample_extended(seed)
        cfg.connector = None
        cfg.kv_events = False
        cfg.mooncake = True
        return cfg

    @classmethod
    def sample_streaming(cls, seed: int) -> "FuzzConfig":
        """`sample_extended` without a KV connector, with streaming-input
        sessions."""
        cfg = cls.sample_extended(seed)
        cfg.connector = None
        cfg.streaming = True
        return cfg

    @classmethod
    def sample_lora(cls, seed: int) -> "FuzzConfig":
        """`sample` with LoRA adapters on some requests."""
        cfg = cls.sample(seed)
        cfg.lora = True
        return cfg

    @classmethod
    def sample_structured(cls, seed: int) -> "FuzzConfig":
        """`sample` with structured output requests, optionally with a
        reasoning parser."""
        cfg = cls.sample(seed)
        rng = random.Random(seed * 6151 + 5)
        cfg.structured = True
        cfg.reasoning_parser = rng.choice(["", "deepseek_r1", "qwen3"])
        return cfg


# Path to a pickled DeepSeek-V4.1 TP4 scheduler KV cache config (dict with
# `kv_cache_config`, `block_size` and `hash_block_size`), used by `--extended`
# when set.
DSV41_KV_CACHE_CONFIG = os.environ.get("RUST_CORE_DIFF_DSV41_KVCFG", "")


def make_dsv41_kv_cache_config(cfg: FuzzConfig) -> KVCacheConfig:
    import copy
    import pickle

    with open(DSV41_KV_CACHE_CONFIG, "rb") as f:
        saved = pickle.load(f)
    assert saved["block_size"] == saved["hash_block_size"] == cfg.block_size
    config = copy.deepcopy(saved["kv_cache_config"])
    config.num_blocks = cfg.num_blocks
    return config


def make_kv_cache_config(cfg: FuzzConfig) -> KVCacheConfig:
    if cfg.layout == "dsv41":
        return make_dsv41_kv_cache_config(cfg)
    bs = cfg.block_size
    full = FullAttentionSpec(
        block_size=bs, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    swa = SlidingWindowSpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * bs,
    )
    swa_replay = SlidingWindowMLASpec(
        block_size=bs,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=2 * bs,
        bounded_replay=True,
    )
    groups = {
        "full": [full],
        "swa": [swa],
        "full_swa": [full, swa],
        "full_swa_replay": [full, swa_replay],
    }[cfg.layout]
    return KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(groups)
        ],
    )


def build_scheduler(cfg: FuzzConfig, scheduler_cls: type[Scheduler]) -> Scheduler:
    base = create_scheduler(
        block_size=cfg.block_size,
        num_blocks=cfg.num_blocks,
        enable_prefix_caching=cfg.enable_prefix_caching,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        long_prefill_token_threshold=cfg.long_prefill_token_threshold,
        num_speculative_tokens=cfg.num_spec_tokens or None,
        scheduling_policy=cfg.policy,
        max_model_len=8192,
        model="Qwen/Qwen3-0.6B",
    )
    vllm_config = base.vllm_config
    vllm_config.cache_config.num_gpu_blocks = cfg.num_blocks
    if cfg.kv_events:
        vllm_config.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True, publisher="null"
        )
    if cfg.mooncake:
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector", kv_role="kv_both"
        )
    if cfg.reasoning_parser:
        vllm_config.structured_outputs_config.reasoning_parser = cfg.reasoning_parser
    if cfg.lora:
        vllm_config.lora_config = LoRAConfig(max_loras=2)
    if cfg.connector is not None:
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector=RustCoreDiffConnector.__name__,
            kv_connector_module_path=__name__,
            kv_role="kv_both",
            kv_connector_extra_config={
                "seed": cfg.seed,
                "is_async": cfg.connector == "async",
            },
        )
    kv_cache_config = make_kv_cache_config(cfg)
    if cfg.nixl is not None:
        if kv_cache_config.hash_block_size is None:
            # Set by the engine; NIXL's P-side prompt truncation reads it.
            kv_cache_config.hash_block_size = cfg.block_size
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=f"kv_{cfg.nixl}",
            engine_id="rust-core-diff",
            kv_load_failure_policy=cfg.kv_load_failure_policy,
        )
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=cfg.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


class RustCoreDiffConnectorMetadata(KVConnectorMetadata):
    def __init__(self, records: list[Any]):
        self.records = records


class RustCoreDiffConnector(KVConnectorBase_V1, SupportsHMA):
    """Scheduler-side connector that reports deterministic external hits and
    records what it sees of the KV cache manager (block views, pool state) in
    its metadata, so the `SchedulerOutput` comparison covers it."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        self.seed = extra["seed"]
        self.is_async = extra["is_async"]
        self.block_size = vllm_config.cache_config.block_size
        self.records: list[Any] = []
        self.pool = None

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self.pool = gpu_block_pool

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        rng = random.Random(
            hash(
                (self.seed, request.request_id, num_computed_tokens, request.num_tokens)
            )
        )
        if rng.random() < 0.5:
            return 0, False
        avail = max(request.num_tokens - 1 - num_computed_tokens, 0)
        n = rng.randrange(0, avail // self.block_size + 1) * self.block_size
        return n, self.is_async and n > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.records.append(
            (
                "alloc",
                request.request_id,
                num_external_tokens,
                blocks.get_block_ids(),
                blocks.get_unhashed_block_ids_all_groups(),
                [
                    [(b.block_id, b.block_hash, b.ref_cnt, b.is_null) for b in g]
                    for g in blocks.blocks
                ],
            )
        )

    def build_connector_meta(self, scheduler_output):
        records, self.records = self.records, []
        if self.pool is not None:
            records.append(
                ("pool", self.pool.get_num_free_blocks(), self.pool.get_usage())
            )
        return RustCoreDiffConnectorMetadata(records)

    def request_finished_all_groups(self, request, block_ids):
        self.records.append(("finished", request.request_id, block_ids))
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_SKIP_FIELDS = {
    # Opaque objects shared by identity; their content is the request's.
    "sampling_params",
    "pooling_params",
    "lora_request",
    "mm_features",
    "prompt_embeds",
    "trace_headers",
    # Wall-clock dependent.
    "timestamp",
    "events",
    "scheduler_stats",
}
# Fields of `Request` objects not compared: the NIXL connector keeps
# per-request flags in `request.kv_transfer_params`, which its native port
# keeps to itself, and the structured output manager its grammar state in
# `request.structured_output_request`. The outputs' `kv_transfer_params` and
# the grammar bitmasks are compared.
_SKIP_REQUEST_FIELDS = {"kv_transfer_params", "structured_output_request"}


def normalize(obj: Any) -> Any:
    """Convert scheduler outputs into plain, order-preserving Python values."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (set, frozenset)):
        return sorted(normalize(x) for x in obj)
    if isinstance(obj, dict):
        return [(normalize(k), normalize(v)) for k, v in obj.items()]
    if isinstance(obj, (list, tuple)) or type(obj).__name__ == "TokenBuffer":
        return [normalize(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return {
            f.name: normalize(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name not in _SKIP_FIELDS
        }
    if isinstance(obj, msgspec.Struct):
        return {
            name: normalize(getattr(obj, name))
            for name in obj.__struct_fields__
            if name not in _SKIP_FIELDS
        }
    if hasattr(obj, "__dict__"):
        skip = (
            _SKIP_FIELDS | _SKIP_REQUEST_FIELDS
            if isinstance(obj, Request)
            else _SKIP_FIELDS
        )
        return {
            k: normalize(v)
            for k, v in vars(obj).items()
            if k not in skip and not k.startswith("_")
        }
    return repr(obj)


def first_diff(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}: {a!r} vs {b!r}"
    if isinstance(a, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            if d := first_diff(a[k], b[k], f"{path}.{k}"):
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}: {a!r:.400} vs {b!r:.400}"
        for i, (x, y) in enumerate(zip(a, b)):
            if d := first_diff(x, y, f"{path}[{i}]"):
                return d
        return None
    return None if a == b else f"{path}: {a!r:.400} != {b!r:.400}"


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


class Workload:
    """Deterministic request stream and fake model, keyed only by the seed and
    the scheduler-visible state so both schedulers see the same inputs as long
    as they agree."""

    def __init__(self, cfg: FuzzConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed * 7919 + 1)
        self.prefixes = [
            [self.rng.randrange(1, 30000) for _ in range(self.rng.randrange(1, 900))]
            for _ in range(4)
        ]
        self.next_id = 0

    def structured_outputs(self) -> dict[str, Any] | None:
        """The structured output constraint of a new request, from a separate
        stream so the other fields stay unchanged."""
        if not self.cfg.structured:
            return None
        rng = random.Random(self.cfg.seed * 37 + self.next_id)
        if rng.random() < 0.3:
            return None
        return rng.choice(STRUCTURED_OUTPUTS)

    def new_request_specs(self) -> list[dict[str, Any]]:
        rng = self.rng
        specs = []
        for _ in range(rng.choice([0, 0, 0, 1, 1, 2, 5])):
            prefix = rng.choice(self.prefixes) if rng.random() < 0.7 else []
            prefix = prefix[: rng.randrange(0, len(prefix) + 1)]
            suffix = [rng.randrange(1, 30000) for _ in range(rng.randrange(1, 600))]
            specs.append(
                dict(
                    request_id=f"req-{self.next_id}",
                    prompt_token_ids=prefix + suffix,
                    max_tokens=rng.choice([1, 2, 8, 32, 100]),
                    stop_token_ids=[rng.randrange(1, 64)]
                    if rng.random() < 0.3
                    else None,
                    ignore_eos=rng.random() < 0.5,
                    min_tokens=rng.choice([0, 0, 3]),
                    priority=rng.randrange(0, 3),
                    arrival_time=float(self.next_id),
                    kv_transfer_params=self.kv_transfer_params(len(prefix + suffix)),
                    structured_outputs=self.structured_outputs(),
                    resumable=self.resumable(),
                    lora=self.lora(),
                )
            )
            self.next_id += 1
        return specs

    def lora(self) -> int | None:
        """The LoRA adapter of a new request (of 4), from a separate stream so
        the other fields stay unchanged."""
        if not self.cfg.lora:
            return None
        rng = random.Random(self.cfg.seed * 59 + self.next_id)
        return rng.choice([None, 0, 1, 2, 3])

    def resumable(self) -> bool:
        """Whether a new request opens a streaming-input session, from a
        separate stream so the other fields stay unchanged."""
        if not self.cfg.streaming:
            return False
        return random.Random(self.cfg.seed * 53 + self.next_id).random() < 0.5

    def session_chunks(self, step: int, sessions: list[str]) -> list[dict[str, Any]]:
        """The next input chunks of some live sessions: resumable ones, or a
        final non-resumable one (the finished sentinel)."""
        if not sessions:
            return []
        rng = random.Random(self.cfg.seed * 7 + step * 1009)
        chunks = []
        for request_id in sorted(sessions):
            if rng.random() >= 0.15:
                continue
            chunks.append(
                dict(
                    request_id=request_id,
                    prompt_token_ids=[
                        rng.randrange(1, 30000) for _ in range(rng.randrange(1, 80))
                    ],
                    max_tokens=rng.choice([1, 2, 8, 32]),
                    stop_token_ids=None,
                    ignore_eos=rng.random() < 0.5,
                    min_tokens=0,
                    priority=0,
                    arrival_time=float(1000 + step),
                    resumable=rng.random() < 0.8,
                )
            )
        return chunks

    def kv_transfer_params(self, num_prompt_tokens: int) -> dict[str, Any] | None:
        """NIXL P/D params of a new request, from a separate stream so the
        other fields stay unchanged."""
        if self.cfg.nixl is None:
            return None
        rng = random.Random(self.cfg.seed * 31 + self.next_id)
        sides = {"producer": ["decode"], "consumer": ["prefill"]}.get(
            self.cfg.nixl, ["decode", "prefill"]
        )
        side = rng.choice(sides + [None])
        if side == "decode":
            return dict(do_remote_decode=True, do_remote_prefill=False)
        if side == "prefill":
            num_blocks = -(-num_prompt_tokens // self.cfg.block_size)
            return dict(
                do_remote_prefill=True,
                do_remote_decode=False,
                remote_engine_id="remote-engine",
                remote_request_id=f"prefill-{self.next_id}",
                remote_block_ids=list(range(num_blocks)),
                remote_host="remote-host",
                remote_port=1234,
            )
        return None

    def aborts(self, live_ids: list[str]) -> list[str]:
        if live_ids and self.rng.random() < 0.05:
            return [self.rng.choice(sorted(live_ids))]
        return []


STRUCTURED_OUTPUTS: list[dict[str, Any]] = [
    dict(
        json={
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 8},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            },
            "required": ["name", "age"],
        }
    ),
    dict(json={"type": "array", "items": {"enum": ["a", "b", 3]}, "maxItems": 4}),
    dict(json_object=True),
    dict(regex=r"(yes|no) [0-9]{1,3}( items)?"),
    dict(regex=r"[a-z]{2,6}@[a-z]{2,5}\.(com|org)"),
    dict(choice=["alpha", "beta", "gamma delta"]),
]


def make_request(spec: dict[str, Any], hasher: Callable) -> Request:
    sp = SamplingParams(
        max_tokens=spec["max_tokens"],
        stop_token_ids=spec["stop_token_ids"],
        ignore_eos=spec["ignore_eos"],
        min_tokens=min(spec["min_tokens"], spec["max_tokens"]),
    )
    if spec.get("structured_outputs") is not None:
        sp.structured_outputs = StructuredOutputsParams(
            **copy.deepcopy(spec["structured_outputs"])
        )
        sp.structured_outputs._backend = "xgrammar"
        # Let grammars end requests.
        sp.ignore_eos = False
        sp.max_tokens = max(sp.max_tokens, 32)
    sp.update_from_generation_config({}, EOS)
    if spec.get("kv_transfer_params") is not None:
        params = copy.deepcopy(spec["kv_transfer_params"])
        sp.extra_args = {"kv_transfer_params": params}
    lora_request = None
    if (adapter := spec.get("lora")) is not None:
        lora_request = LoRARequest(
            f"adapter-{adapter}", adapter + 1, f"/adapters/{adapter}"
        )
    return Request(
        request_id=spec["request_id"],
        prompt_token_ids=list(spec["prompt_token_ids"]),
        sampling_params=sp,
        pooling_params=None,
        lora_request=lora_request,
        arrival_time=spec["arrival_time"],
        priority=spec["priority"],
        block_hasher=hasher,
        resumable=spec.get("resumable", False),
    )


def _allowed_tokens(row: np.ndarray) -> np.ndarray:
    """The token ids a grammar bitmask row allows."""
    bits = np.unpackbits(row.view(np.uint8), bitorder="little")
    return np.flatnonzero(bits)


def fake_model_output(
    cfg: FuzzConfig,
    step: int,
    scheduler: Scheduler,
    so,
    reported: set[str] | None = None,
    nixl_state: dict[str, set[str]] | None = None,
    grammar_output=None,
) -> tuple[ModelRunnerOutput, DraftTokenIds | None]:
    """With `reported` (async scheduling, where a load may be reported before
    the previous report is processed), each async KV load is reported once:
    `reported` tracks the loads reported so far. `nixl_state` tracks the NIXL
    (or Mooncake) transfers reported so far. Structured output requests
    sample from their rows of `grammar_output`."""
    req_ids = list(so.num_scheduled_tokens)
    sampled: list[list[int]] = []
    drafts_ids, drafts = [], []
    grammar_rows = {}
    if grammar_output is not None:
        stride = cfg.num_spec_tokens + 1
        for i, rid in enumerate(grammar_output.structured_output_request_ids):
            grammar_rows[rid] = grammar_output.grammar_bitmask[i * stride :]
    for rid in req_ids:
        req = scheduler.requests[rid]
        if req.num_computed_tokens < req.num_tokens:
            sampled.append([])  # partial prefill
            continue
        rng = random.Random(hash((cfg.seed, rid, req.num_output_tokens, step)))
        spec_tokens = so.scheduled_spec_decode_tokens.get(rid, ())
        n_spec = len(spec_tokens)
        n = 1 + (rng.randrange(0, n_spec + 1) if n_spec else 0)
        if rid in grammar_rows:
            # Accept a prefix of the drafts, then sample the next token from
            # the grammar (or end reasoning).
            tokens = [t for t in spec_tokens[: n - 1] if t >= 0]
            allowed = _allowed_tokens(grammar_rows[rid][len(tokens)])
            if len(allowed) > 100000 and rng.random() < 0.3:
                token = THINK_END
            elif len(allowed):
                token = int(allowed[rng.randrange(len(allowed))])
            else:
                token = EOS
            sampled.append(tokens + [token])
        else:
            sampled.append([rng.randrange(1, 64) for _ in range(n)])
        if cfg.num_spec_tokens:
            drafts_ids.append(rid)
            drafts.append([rng.randrange(1, 64) for _ in range(cfg.num_spec_tokens)])
    out = ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)
    if cfg.connector == "async":
        # Async KV loads complete one step after they start.
        recving = {
            rid
            for rid, r in scheduler.requests.items()
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        }
        if reported is not None:
            reported &= recving
            recving -= reported
            reported |= recving
        if recving:
            out.kv_connector_output = KVConnectorOutput(finished_recving=recving)
    if cfg.nixl is not None:
        assert nixl_state is not None
        out.kv_connector_output = _fake_nixl_output(cfg, step, scheduler, nixl_state)
    if cfg.mooncake:
        assert nixl_state is not None
        out.kv_connector_output = _fake_mooncake_output(scheduler, nixl_state)
    draft = DraftTokenIds(drafts_ids, drafts) if drafts_ids else None
    return out, draft


# `</think>` of the Qwen3 tokenizer.
THINK_END = 151668


def _fake_mooncake_output(
    scheduler: Scheduler, state: dict[str, set[str]]
) -> KVConnectorOutput:
    """The Mooncake store worker side: each load completes once, a step
    after it starts, and so does each finished request's save."""
    recv, send = state.setdefault("recv", set()), state.setdefault("send", set())
    receiving = {
        rid
        for rid, r in scheduler.requests.items()
        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    }
    new_recv = receiving - recv
    recv |= new_recv
    # A load reported in the previous output may not be processed yet (async
    # scheduling): an aborted request it frees has nothing to save.
    loading = new_recv | state.get("last_recv", set())
    state["last_recv"] = new_recv
    sending = {rid for rid, r in scheduler.requests.items() if r.is_finished()}
    new_send = sending - send - loading
    send |= new_send
    return KVConnectorOutput(
        finished_recving=new_recv or None, finished_sending=new_send or None
    )


class _FakeMooncakeLookup:
    """`LookupKeyClient` with the store faked: a lookup hits the leading
    blocks found in `store`."""

    store: set[bytes] = set()
    block_size = 16

    def __init__(self, *args, **kwargs):
        pass

    def lookup(self, req_id, num_tokens, block_hashes, non_block=False):
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
            MooncakeLookupResult,
        )

        n = 0
        for h in block_hashes:
            if h not in self.store:
                break
            n += 1
        bs = self.block_size
        return MooncakeLookupResult(min(n * bs, num_tokens - 1) // bs * bs)

    def prefetch(self, *args, **kwargs):
        pass

    def discard(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def _fake_mooncake_store(cfg: FuzzConfig, hasher: Callable):
    """Fake the Mooncake store: it holds leading blocks of each prefix."""
    from unittest.mock import patch

    rng = random.Random(cfg.seed * 92821 + 7)
    store: set[bytes] = set()
    for prefix in Workload(cfg).prefixes:
        request = Request("store", prefix + [1], SamplingParams(), None)
        hashes = hasher(request)
        store.update(hashes[: rng.randrange(0, len(hashes) + 1)])
    _FakeMooncakeLookup.store = store
    _FakeMooncakeLookup.block_size = cfg.block_size
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.scheduler."
        "LookupKeyClient",
        _FakeMooncakeLookup,
    ):
        yield


def _wait_for_grammar(scheduler: Scheduler, request: Request) -> None:
    """Wait until the request's grammar compiled, so both schedulers see it
    ready at the same step."""
    from concurrent.futures import Future

    structured_req = request.structured_output_request
    if structured_req is None:
        return
    grammar = structured_req._grammar
    if isinstance(grammar, Future):
        grammar.exception()


def _compare_grammar_outputs(a, b) -> str | None:
    if (a is None) != (b is None):
        return f"GrammarOutput: {a!r:.200} vs {b!r:.200}"
    if a is None:
        return None
    if a.structured_output_request_ids != b.structured_output_request_ids:
        return (
            f"GrammarOutput.ids: {a.structured_output_request_ids} != "
            f"{b.structured_output_request_ids}"
        )
    if a.grammar_bitmask.shape != b.grammar_bitmask.shape:
        return (
            f"GrammarOutput.bitmask shape: {a.grammar_bitmask.shape} != "
            f"{b.grammar_bitmask.shape}"
        )
    if not np.array_equal(a.grammar_bitmask, b.grammar_bitmask):
        rows = np.flatnonzero((a.grammar_bitmask != b.grammar_bitmask).any(axis=1))
        return f"GrammarOutput.bitmask: rows {rows.tolist()} differ"
    return None


def _fake_nixl_output(
    cfg: FuzzConfig, step: int, scheduler: Scheduler, state: dict[str, set[str]]
) -> KVConnectorOutput | None:
    """The NIXL worker side: each receive completes once, a step after it
    starts (also for a request aborted while waiting for it), and so does
    each send of a finished prefill (`do_remote_decode`); some receives fail
    as a whole (`failed_recving`) or by block (`invalid_block_ids`,
    single-group layouts only)."""
    recv, send = state.setdefault("recv", set()), state.setdefault("send", set())
    receiving: set[str] = set()
    sending: set[str] = set()
    for rid, r in scheduler.requests.items():
        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            receiving.add(rid)
        elif r.is_finished():
            params = r.kv_transfer_params or {}
            (sending if params.get("do_remote_decode") else receiving).add(rid)
    recv &= receiving
    new_recv = receiving - recv
    recv |= new_recv
    send &= sending
    new_send = sending - send
    send |= new_send
    failed: set[str] = set()
    invalid: set[int] = set()
    single_group = len(scheduler.kv_cache_config.kv_cache_groups) == 1
    for rid in sorted(new_recv):
        if scheduler.requests[rid].is_finished():
            continue
        rng = random.Random(f"{cfg.seed}-{rid}-{step}")
        x = rng.random()
        if x < 0.15:
            failed.add(rid)
        elif x < 0.3 and single_group:
            block_ids = scheduler.kv_cache_manager.get_block_ids(rid)[0]
            if block_ids:
                invalid.add(rng.choice(block_ids))
    if not (new_recv or new_send):
        return None
    return KVConnectorOutput(
        finished_recving=new_recv or None,
        finished_sending=new_send or None,
        failed_recving=failed,
        invalid_block_ids=invalid,
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_one(
    cfg: FuzzConfig,
    make_ref: Callable[[FuzzConfig], Scheduler],
    make_test: Callable[[FuzzConfig], Scheduler],
    test_ctx: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
    expect_manager_module: str | None = None,
    use_async: bool = False,
) -> tuple[bool, str]:
    """Drive both schedulers in lockstep. Returns (ok, message).

    With `use_async`, the output of each step is processed after the next
    step is scheduled, and drafts stay with the (fake) worker."""
    hash_fn = {"sha256": sha256, "sha256_cbor": sha256_cbor}[cfg.hash_fn]
    init_none_hash(hash_fn)
    if cfg.nixl is not None:
        _freeze_nixl_clock()
    ref_hasher = get_request_block_hasher(cfg.block_size, hash_fn)
    store = (
        _fake_mooncake_store(cfg, ref_hasher)
        if cfg.mooncake
        else contextlib.nullcontext()
    )
    with store:
        ref = make_ref(cfg)
        with test_ctx():
            test = make_test(cfg)
            test_hasher = get_request_block_hasher(cfg.block_size, hash_fn)
    if (core := getattr(test, "_core", None)) is not None and _expect_native():
        # The native ports whenever they support the configuration.
        expected = "nixl" if cfg.nixl else "mooncake_store" if cfg.mooncake else None
        if expected is not None and core.connector_impl != expected:
            return False, f"connector_impl is {core.connector_impl}, not {expected}"
        if cfg.nixl is not None:
            core.native_nixl_connector.set_fixed_clock(NIXL_CLOCK)
    if expect_manager_module is not None:
        module = type(test.kv_cache_manager).__module__
        if module != expect_manager_module:
            return False, f"test scheduler uses a KV cache manager from {module}"
    events_ref: list[Any] = []
    events_test: list[Any] = []
    if cfg.kv_events:
        ref.kv_event_publisher.publish = lambda b: events_ref.append(b.events)
        test.kv_event_publisher.publish = lambda b: events_test.append(b.events)
    wl = Workload(cfg)
    pending: deque[Any] = deque()
    reported_ref: set[str] | None = set() if use_async else None
    reported_test: set[str] | None = set() if use_async else None
    nixl_ref: dict[str, set[str]] = {}
    nixl_test: dict[str, set[str]] = {}

    def both(fn_name: str, *args_ref, args_test=None):
        with test_ctx():
            b = getattr(test, fn_name)(*(args_test if args_test else args_ref))
        a = getattr(ref, fn_name)(*args_ref)
        return a, b

    for step in range(cfg.num_steps):
        for spec in wl.new_request_specs():
            req_ref = make_request(spec, ref_hasher)
            req_test = make_request(spec, test_hasher)
            if req_ref.use_structured_output:
                # As EngineCore does before adding the request.
                ref.structured_output_manager.grammar_init(req_ref)
                with test_ctx():
                    test.structured_output_manager.grammar_init(req_test)
            ref.add_request(req_ref)
            with test_ctx():
                test.add_request(req_test)
            _wait_for_grammar(ref, req_ref)
            _wait_for_grammar(test, req_test)
        if cfg.streaming:
            sessions = [rid for rid, r in ref.requests.items() if r.resumable]
            for spec in wl.session_chunks(step, sessions):
                ref.add_request(make_request(spec, ref_hasher))
                with test_ctx():
                    test.add_request(make_request(spec, test_hasher))
        if cfg.mooncake and step == 40:
            a, b = both("reset_prefix_cache")
            if a != b:
                return False, f"step {step}: reset_prefix_cache {a} != {b}"
        aborts = wl.aborts(list(ref.requests))
        if aborts:
            from vllm.v1.request import RequestStatus

            a, b = both("finish_requests", aborts, RequestStatus.FINISHED_ABORTED)
            if d := first_diff(normalize(a), normalize(b), "finish_requests"):
                return False, f"step {step}: {d}"
        so_ref, so_test, raised = _call_both(ref.schedule, test.schedule, test_ctx)
        if raised is not None:
            return raised[0], f"step {step}: schedule: {raised[1]}"
        if d := first_diff(normalize(so_ref), normalize(so_test), "SchedulerOutput"):
            return False, f"step {step}: {d}"
        go_ref = go_test = None
        if cfg.structured:
            go_ref = ref.get_grammar_bitmask(so_ref)
            with test_ctx():
                go_test = test.get_grammar_bitmask(so_test)
            if d := _compare_grammar_outputs(go_ref, go_test):
                return False, f"step {step}: {d}"
        out, draft = fake_model_output(
            cfg, step, ref, so_ref, reported_ref, nixl_ref, go_ref
        )
        out_t, draft_t = fake_model_output(
            cfg, step, test, so_test, reported_test, nixl_test, go_test
        )
        pending.append((step, so_ref, so_test, out, out_t, draft, draft_t))
        if use_async and len(pending) < 2:
            continue
        sched_step, so_ref, so_test, out, out_t, draft, draft_t = pending.popleft()
        eco_ref, eco_test, raised = _call_both(
            functools.partial(ref.update_from_output, so_ref, out),
            functools.partial(test.update_from_output, so_test, out_t),
            test_ctx,
        )
        if raised is not None:
            return raised[0], f"step {step}: update_from_output: {raised[1]}"
        if d := first_diff(
            normalize(eco_ref), normalize(eco_test), "EngineCoreOutputs"
        ):
            return False, f"step {step} (output of {sched_step}): {d}"
        if d := first_diff(normalize(events_ref), normalize(events_test), "KVEvents"):
            return False, f"step {step}: {d}"
        a, b = _cache_stats(eco_ref), _cache_stats(eco_test)
        if d := first_diff(a, b, "SchedulerStats"):
            return False, f"step {step}: {d}"
        if draft is not None and not use_async:
            ref.update_draft_token_ids(draft)
            with test_ctx():
                test.update_draft_token_ids(draft_t)
        a = (ref.get_kv_cache_usage(), ref.get_request_counts())
        with test_ctx():
            b = (test.get_kv_cache_usage(), test.get_request_counts())
        if a != b:
            return False, f"step {step}: usage/counts {a} != {b}"
    return True, "ok"


NIXL_CLOCK = 1000.0


def _call_both(
    call_ref: Callable[[], Any],
    call_test: Callable[[], Any],
    test_ctx: Callable[[], contextlib.AbstractContextManager],
) -> tuple[Any, Any, tuple[bool, str] | None]:
    """`(call_ref(), call_test(), None)`; if the reference raises, the test
    must raise the same exception type, which ends the run (ok or not)."""
    try:
        a = call_ref()
    except Exception as e:
        try:
            with test_ctx():
                call_test()
        except Exception as e2:
            if type(e2) is type(e):
                return None, None, (True, f"both raised {type(e).__name__}")
            return None, None, (False, f"reference raised {e!r}, test {e2!r}")
        return None, None, (False, f"reference raised {e!r}, test did not")
    with test_ctx():
        b = call_test()
    return a, b, None


def _expect_native() -> bool:
    return os.environ.get("VLLM_RUST_CORE_NATIVE_HOOKS", "1") != "0"


def _freeze_nixl_clock() -> None:
    """NIXL stamps send deadlines and its metadata with `time.perf_counter()`;
    a fixed clock makes both schedulers' metadata comparable."""
    import types

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        base_scheduler,
        pull_scheduler,
    )

    frozen = types.SimpleNamespace(perf_counter=lambda: NIXL_CLOCK)
    base_scheduler.time = frozen  # type: ignore[assignment]
    pull_scheduler.time = frozen  # type: ignore[assignment]


def _cache_stats(eco: dict[int, Any]) -> Any:
    """The KV cache fields of the scheduler stats (the rest is timing)."""
    return [
        normalize(
            (
                o.scheduler_stats.kv_cache_usage,
                o.scheduler_stats.prefix_cache_stats,
                o.scheduler_stats.connector_prefix_cache_stats,
            )
        )
        for o in eco.values()
        if o.scheduler_stats is not None
    ]


def _parse_seeds(s: str) -> Iterator[int]:
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            yield from range(int(lo), int(hi))
        else:
            yield int(part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-20")
    parser.add_argument(
        "--mode",
        default="py-py",
        choices=["py-py", "rust-kv", "rust-sched"],
        help="py-py validates the harness; rust-kv swaps in the Rust KV manager; "
        "rust-sched tests the Rust scheduler (RustScheduler)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also fuzz KV events, a KV connector and the DSV4.1 layout",
    )
    parser.add_argument(
        "--nixl",
        action="store_true",
        help="fuzz P/D through the real NixlConnector (scheduler role)",
    )
    parser.add_argument(
        "--mooncake",
        action="store_true",
        help="fuzz the real MooncakeStoreConnector (scheduler role)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="fuzz xgrammar structured output requests",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="fuzz streaming-input sessions",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="test AsyncScheduler, driven like EngineCore's batch queue",
    )
    args = parser.parse_args()

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    ref_cls = AsyncScheduler if args.use_async else Scheduler
    test_cls: type = ref_cls
    make_ref = lambda cfg: build_scheduler(cfg, ref_cls)  # noqa: E731
    test_ctx: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext
    if args.mode == "rust-kv":
        from unittest.mock import patch

        test_ctx = lambda: patch.dict(  # noqa: E731
            os.environ, {"VLLM_USE_RUST_CORE": "1"}
        )
    elif args.mode == "rust-sched":
        from vllm.v1.core.sched.rust_scheduler import (
            RustAsyncScheduler,
            RustScheduler,
        )

        test_cls = RustAsyncScheduler if args.use_async else RustScheduler
    make_test = lambda cfg: build_scheduler(cfg, test_cls)  # noqa: E731

    failures = 0
    for seed in _parse_seeds(args.seeds):
        if args.nixl:
            cfg = FuzzConfig.sample_nixl(seed)
        elif args.mooncake:
            cfg = FuzzConfig.sample_mooncake(seed)
        elif args.structured:
            cfg = FuzzConfig.sample_structured(seed)
        elif args.streaming:
            cfg = FuzzConfig.sample_streaming(seed)
        elif args.extended:
            cfg = FuzzConfig.sample_extended(seed)
        else:
            cfg = FuzzConfig.sample(seed)
        ok, msg = run_one(
            cfg,
            make_ref,
            make_test,
            test_ctx,
            expect_manager_module="vllm._rust_core" if args.mode == "rust-kv" else None,
            use_async=args.use_async,
        )
        note = "" if msg == "ok" else msg
        print(f"seed={seed} {'OK ' if ok else 'FAIL'} {cfg} {note}")
        failures += not ok
    print(f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
