# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN57 — TurboQuant centroids disk-persistent cache.

Inspired by vllm#41418 (TheTom, OPEN). Upstream PR pre-bakes 9 (d,bits)
centroid tables inline (~1500 LOC of constants). Genesis approach: make
centroids cache **disk-persistent** instead of in-process `@lru_cache`.

Why this is better than copying pre-baked tables
------------------------------------------------
- Upstream: 9 hardcoded (d, bits) pairs; new shape = back to solver
- Genesis disk cache: ANY (d, bits) cached after first call
- Cold start cost: 200ms × N first-time shapes per fresh container
  (acceptable — happens only on cache-empty boot)
- Subsequent boots / worker restarts: instant lookup from disk
- Memory `feedback_v794_pn14_validated`: container restarts are common
  in our prod (PN25 fork-safe op + monkey-patch survival)

What this replaces
------------------
`vllm/model_executor/layers/quantization/turboquant/centroids.py:80-87`:

    @lru_cache(maxsize=32)
    def get_centroids(d: int, bits: int) -> torch.Tensor:
        centroids, _ = solve_lloyd_max(d, bits)
        return centroids

We text-patch the `get_centroids` body to first check disk cache at
`~/.cache/genesis/turboquant_centroids.pkl`, miss → solve → store.

Cache invariants
----------------
- Lloyd-Max solver is **fully deterministic** given (d, bits) — pre-baked
  values are bit-identical to solver output (verified by upstream test
  `test_prebaked_matches_solver` max_abs_diff=0.0)
- Cache never poisoned by partial writes (atomic rename pattern)
- Cache portable across machines (CPU-side floats, no GPU state)
- Defensive: fall through to solver on any cache failure (corrupt file,
  permission error, missing dir)

Models affected
---------------
ALL configs that use TurboQuant KV cache:
- 27B Lorbus + TQ k8v4 (calls get_centroids(128, 4))
- 35B FP8 + TQ k8v4 (same)
- 27B FP8 e5m2 short/long (no TQ, patch no-op)

Default OFF until live verified — risk is minimal (cache-only optimization)
but conservative until we measure cold-start savings on our workload.

Author: Sandermage backport of upstream pattern (TheTom vllm#41418).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn57_tq_centroids_disk_cache")

GENESIS_PN57_MARKER = "Genesis PN57 TQ centroids disk cache (vllm#41418 inspiration)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on get_centroids body (8 lines incl @lru_cache)
ANCHOR_OLD = (
    "@lru_cache(maxsize=32)\n"
    "def get_centroids(d: int, bits: int) -> torch.Tensor:\n"
    "    \"\"\"Get precomputed Lloyd-Max centroids (cached).\"\"\"\n"
    "    centroids, _ = solve_lloyd_max(d, bits)\n"
    "    return centroids"
)

ANCHOR_NEW = (
    "# [Genesis PN57 vllm#41418-inspired] disk-persistent cache for centroids.\n"
    "# Lloyd-Max solver is fully deterministic given (d, bits) — same input\n"
    "# always gives bit-identical output. Pre-baked tables in upstream PR\n"
    "# #41418 are equivalent to disk cache populated lazily.\n"
    "# [Audit A-02 fix 2026-05-05] All `os.*` calls live inside helper bodies\n"
    "# with LOCAL `import os`. Module-level constants do NOT use `os` so we\n"
    "# never require `import os` at target module top level.\n"
    "_GENESIS_PN57_CACHE_PATH = None  # lazily resolved on first use\n"
    "_GENESIS_PN57_DISK = None  # lazily loaded\n"
    "\n"
    "def _genesis_pn57_path():\n"
    "    import os as _os  # LOCAL — no top-level dependency\n"
    "    global _GENESIS_PN57_CACHE_PATH\n"
    "    if _GENESIS_PN57_CACHE_PATH is None:\n"
    "        _GENESIS_PN57_CACHE_PATH = _os.path.expanduser(\n"
    "            \"~/.cache/genesis/turboquant_centroids.pkl\"\n"
    "        )\n"
    "    return _GENESIS_PN57_CACHE_PATH\n"
    "\n"
    "def _genesis_pn57_load_disk():\n"
    "    import os as _os  # LOCAL\n"
    "    global _GENESIS_PN57_DISK\n"
    "    if _GENESIS_PN57_DISK is not None:\n"
    "        return _GENESIS_PN57_DISK\n"
    "    try:\n"
    "        import pickle\n"
    "        _path = _genesis_pn57_path()\n"
    "        if _os.path.isfile(_path):\n"
    "            with open(_path, \"rb\") as _f:\n"
    "                _GENESIS_PN57_DISK = pickle.load(_f)\n"
    "        else:\n"
    "            _GENESIS_PN57_DISK = {}\n"
    "    except Exception:\n"
    "        _GENESIS_PN57_DISK = {}\n"
    "    return _GENESIS_PN57_DISK\n"
    "\n"
    "def _genesis_pn57_save_disk():\n"
    "    try:\n"
    "        import os as _os  # LOCAL\n"
    "        import pickle, tempfile\n"
    "        _path = _genesis_pn57_path()\n"
    "        _os.makedirs(_os.path.dirname(_path), exist_ok=True)\n"
    "        _dirn = _os.path.dirname(_path)\n"
    "        with tempfile.NamedTemporaryFile(\"wb\", dir=_dirn, delete=False) as _tf:\n"
    "            pickle.dump(_GENESIS_PN57_DISK, _tf)\n"
    "            _tmp = _tf.name\n"
    "        _os.replace(_tmp, _path)\n"
    "    except Exception:\n"
    "        pass  # cache write failure non-fatal\n"
    "\n"
    "@lru_cache(maxsize=32)\n"
    "def get_centroids(d: int, bits: int) -> torch.Tensor:\n"
    "    \"\"\"Get precomputed Lloyd-Max centroids (in-memory + disk cached).\"\"\"\n"
    "    _disk = _genesis_pn57_load_disk()\n"
    "    _key = (int(d), int(bits))\n"
    "    if _key in _disk:\n"
    "        try:\n"
    "            return torch.tensor(_disk[_key], dtype=torch.float32)\n"
    "        except Exception:\n"
    "            pass\n"
    "    centroids, _ = solve_lloyd_max(d, bits)\n"
    "    try:\n"
    "        _disk[_key] = tuple(centroids.cpu().tolist())\n"
    "        _genesis_pn57_save_disk()\n"
    "    except Exception:\n"
    "        pass\n"
    "    return centroids"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/quantization/turboquant/centroids.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN57 TQ centroids disk cache (vllm#41418 inspiration)",
        target_file=str(target),
        marker=GENESIS_PN57_MARKER,
        sub_patches=[TextPatch(
            name="pn57_disk_cache",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "_PREBAKED_CENTROIDS",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN57")
    log_decision("PN57", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "centroids.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "PN57 applied: TQ centroids disk-cached"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged #41418"
    return "failed", failure.reason if failure else "unknown failure"
