# SPDX-License-Identifier: Apache-2.0
"""Genesis prealloc_v2 — first-call profile + replay activation memory plan.

Closes the **5-7 GB activation/fragmentation gap** between vLLM and llama.cpp
on the same hardware/model (per cross-engine research 2026-05-05). vLLM trusts
PyTorch's dynamic caching allocator; llama.cpp's `ggml_gallocr_reserve()`
plans static offsets for the entire compute buffer at boot. We can't replicate
llama.cpp's compile-time analysis (PyTorch eager has no static graph), but we
CAN approximate it with **first-call profiling + replay**:

================================================================
TWO-PHASE DESIGN (mirrors llama.cpp's reserve / alloc_graph)
================================================================

PHASE 1 — RECORDER (one-time warmup):
  1. Operator runs `genesis prealloc-record --model X --max-ctx Y` (or it
     runs automatically during vLLM's profile_run when v2 is enabled).
  2. PreallocV2Recorder context manager monkey-patches torch.empty,
     torch.zeros, Tensor.new_empty IN FLAGGED MODULES ONLY (default scope:
     `vllm.model_executor.layers.fla`, `vllm.model_executor.layers.mamba`,
     `vllm.distributed.kv_transfer`).
  3. Each call records `(call_site, shape, dtype, device)` keyed by
     filename:lineno of the calling frame.
  4. After warmup: per call_site we compute `shape_envelope = max along
     each dim across all observed calls`.
  5. Manifest written to `~/.cache/genesis/prealloc_v2/{key_hash}.json`.

PHASE 2 — REPLAY (every subsequent boot with same key):
  1. PreallocV2Replay reads manifest from cache.
  2. For each call_site → instantiate `GenesisPreallocBuffer` with the
     envelope shape (one boot-time `torch.empty()` per site).
  3. Re-monkey-patches torch.empty at recorded sites: returns
     `GPB.slice_to(cached_buf, requested_shape)` instead of new alloc.
  4. After install: zero `torch.empty` calls during forward at recorded
     sites. PyTorch caching allocator only sees the boot-time pool
     allocations + small transient activations — fragmentation drops to
     near-zero.

================================================================
INVALIDATION RULES
================================================================

Manifest cache key = sha256 of:
  - model_arch (e.g. "qwen3_5")
  - max_model_len
  - max_num_batched_tokens
  - tp_size
  - kv_cache_dtype
  - vllm_pin_sha
  - genesis_pin_sha

When ANY of these change → cache miss → recorder re-runs. Stale manifests
are silently ignored (no false re-use).

================================================================
SAFETY
================================================================

- Monkey-patch is **scope-restricted** — only modules matching the scope
  patterns get the wrapped torch.empty. Out-of-scope sites continue to
  use stock allocator.
- Replay site-patches use `inspect`-based resolution + hash check on the
  source line — if the upstream code changed (e.g. shape construction
  refactored), the patch refuses to apply and the call falls back to
  stock allocator.
- `GENESIS_ENABLE_PREALLOC_V2_RECORDER=1` (off by default) — operator
  opt-in for the recorder pass.
- `GENESIS_ENABLE_PREALLOC_V2_REPLAY=1` (off by default) — operator
  opt-in for replay.
- Both off → zero behavioral change vs stock vLLM.

================================================================
RELATIONSHIP TO PN59 / PN12 / P22 / P26 / P37 / P39a / GdnScratchPool
================================================================

The hand-instrumented per-kernel pools (PN12 FFN, P22 dequant, P37 MoE,
PN59 GdnScratchPool with acquire_h_window / acquire_o_output) are the
**reference implementations** of this pattern at specific call sites.
prealloc_v2 generalizes the same pattern to the LONG TAIL of un-
instrumented allocations across FLA / Mamba / Marlin / kv_transfer that
together account for the 5-7 GB delta vs llama.cpp.

When prealloc_v2 lands, hand-instrumented pools become **redundant in
principle** but stay in production because they're already battle-tested.
prealloc_v2 takes care of everything they don't cover.

================================================================
STATUS
================================================================

This module is the **skeleton + manifest schema** (Level 3 sprint #1,
2026-05-05). The Recorder + Replay implementations follow incrementally:

- v0.1 (this commit): manifest schema, key derivation, dirs, type stubs
- v0.2: PreallocV2Recorder context manager (monkey-patch + record)
- v0.3: envelope reduction + manifest writer
- v0.4: PreallocV2Replay (manifest reader + GPB instantiator)
- v0.5: vLLM profile_run hook integration
- v0.6: empirical validation (recorder pass on PROD config + memory
        delta measurement vs stock)

Author: Sandermage 2026-05-05.
Per Sander's design call after 4-agent deep research:
  "вариант Е но другой более качественный и правильный логически и
   технически. Нам надо что бы все фиксы работали всегда".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.prealloc_v2")


# ─── Manifest schema ──────────────────────────────────────────────────


_MANIFEST_VERSION = 1
"""Bump on any breaking schema change. Recorder writes; Replay refuses
to load older versions."""


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "genesis" / "prealloc_v2"
"""Per-user manifest cache. Override via GENESIS_PREALLOC_V2_CACHE_DIR
env. Created on first write."""


_DEFAULT_SCOPE = (
    "vllm.model_executor.layers.fla",
    "vllm.model_executor.layers.mamba",
    "vllm.distributed.kv_transfer",
)
"""Module path prefixes whose torch.empty/zeros/new_empty calls are
recorded + replayed. Out-of-scope sites use stock allocator."""


@dataclass
class CallSiteEnvelope:
    """Envelope shape for one allocation site.

    Recorded across all calls observed during a warmup pass. Replayed by
    pre-allocating ONE buffer of `shape_envelope` size at boot, then
    serving all subsequent calls as slice views.
    """
    file: str          # e.g. "/usr/.../fla/ops/chunk_o.py"
    lineno: int        # source line of the torch.empty() call
    shape_envelope: list[int]   # max along each dim
    dtype: str         # e.g. "torch.bfloat16"
    device_kind: str   # e.g. "cuda" (index abstracted; one buffer per device)
    n_calls_observed: int = 0
    source_hash: str = ""   # sha256 of source line text — invalidate on drift

    def site_key(self) -> str:
        """Stable identifier for this call site across invocations."""
        return f"{self.file}:{self.lineno}"


@dataclass
class PreallocV2Manifest:
    """Top-level manifest written by recorder, read by replay."""
    version: int = _MANIFEST_VERSION
    model_arch: str = ""
    max_model_len: int = 0
    max_num_batched_tokens: int = 0
    tp_size: int = 1
    kv_cache_dtype: str = ""
    vllm_pin_sha: str = ""
    genesis_pin_sha: str = ""
    recorded_at: str = ""   # ISO-8601 UTC
    scope: list[str] = field(default_factory=list)
    call_sites: dict[str, CallSiteEnvelope] = field(default_factory=dict)

    # ── Cache key derivation ──

    def cache_key(self) -> str:
        """Deterministic hash combining all invalidation-relevant fields."""
        relevant = (
            self.model_arch, str(self.max_model_len),
            str(self.max_num_batched_tokens), str(self.tp_size),
            self.kv_cache_dtype, self.vllm_pin_sha, self.genesis_pin_sha,
            ",".join(sorted(self.scope)),
        )
        h = hashlib.sha256("|".join(relevant).encode("utf-8"))
        return h.hexdigest()[:16]   # 64-bit truncation is plenty

    # ── Serialization ──

    def to_json(self) -> str:
        """Serialize to JSON. Custom encoder for CallSiteEnvelope."""
        d = asdict(self)
        # asdict handles nested dataclasses but call_sites dict needs
        # explicit per-key conversion since asdict() turns the values into
        # plain dicts already.
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "PreallocV2Manifest":
        d = json.loads(raw)
        if d.get("version") != _MANIFEST_VERSION:
            raise ValueError(
                f"prealloc_v2 manifest version mismatch: file={d.get('version')}"
                f" expected={_MANIFEST_VERSION}"
            )
        # Reconstruct CallSiteEnvelope dataclasses
        call_sites_raw = d.pop("call_sites", {})
        m = cls(**d)
        m.call_sites = {
            k: CallSiteEnvelope(**v) for k, v in call_sites_raw.items()
        }
        return m


# ─── Cache directory management ──────────────────────────────────────


def get_cache_dir() -> Path:
    """Return the prealloc_v2 manifest cache directory.

    Override with GENESIS_PREALLOC_V2_CACHE_DIR env. Created if missing.
    """
    raw = os.environ.get(
        "GENESIS_PREALLOC_V2_CACHE_DIR", str(_DEFAULT_CACHE_DIR),
    )
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path(cache_key: str) -> Path:
    """Path of the JSON file for a given cache key."""
    return get_cache_dir() / f"{cache_key}.json"


def write_manifest(manifest: PreallocV2Manifest) -> Path:
    """Atomically write manifest to disk. Returns the written path."""
    target = manifest_path(manifest.cache_key())
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(manifest.to_json())
    os.replace(tmp, target)
    log.info(
        "[prealloc_v2] wrote manifest %s — %d call sites, key=%s",
        target.name, len(manifest.call_sites), manifest.cache_key(),
    )
    return target


def load_manifest(cache_key: str) -> PreallocV2Manifest | None:
    """Load manifest by cache key, or None if missing / corrupt /
    version-mismatched."""
    target = manifest_path(cache_key)
    if not target.is_file():
        return None
    try:
        return PreallocV2Manifest.from_json(target.read_text())
    except (ValueError, json.JSONDecodeError, OSError) as e:
        log.warning(
            "[prealloc_v2] failed to load manifest %s (%s) — will re-record",
            target.name, e,
        )
        return None


# ─── Public env-flag helpers ──────────────────────────────────────────


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def recorder_enabled() -> bool:
    """True iff GENESIS_ENABLE_PREALLOC_V2_RECORDER is set truthy."""
    return _env_flag("GENESIS_ENABLE_PREALLOC_V2_RECORDER")


def replay_enabled() -> bool:
    """True iff GENESIS_ENABLE_PREALLOC_V2_REPLAY is set truthy."""
    return _env_flag("GENESIS_ENABLE_PREALLOC_V2_REPLAY")


# ─── Recorder + Replay (skeleton — implementations follow in v0.2-v0.4) ──
# See module docstring "STATUS" section for the implementation roadmap.
# This v0.1 commit ships only the manifest schema + cache layer so
# downstream commits can land in TDD-discoverable chunks.
