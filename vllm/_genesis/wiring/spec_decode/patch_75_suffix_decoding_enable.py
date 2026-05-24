# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 75 — auto-enable Suffix Decoding (Arctic Inference).

Activates upstream PR #25784 (Suffix Decoding, MERGED 2025-11-03, present in
our pin `07351e088`). Auto-rewrites `speculative_config.method` from "ngram"
to "suffix" when `GENESIS_ENABLE_P75_SUFFIX_DECODING=1`. Operator convenience
patch — equivalent to manually setting `--speculative-config '{"method":"suffix",...}'`.

================================================================
WHY USE SUFFIX DECODING
================================================================

vLLM stock ngram (numba CPU + KMP) on free-form Qwen3.6 workload: 46 tok/s.
Suffix Decoding (per arxiv 2411.04975, NeurIPS 2025): up to 2.8× over EAGLE
on agentic workloads. Per-prompt suffix tree with branch-frequency stats,
**dynamic K per step** (no fixed num_speculative_tokens truncation), cross-
request response cache (FIFO eviction).

Empirical expected gains on our config:
  - Tool-call (heavy repeats): +40-60% over our 75 tok/s strict-ngram
  - Free-form text: +15-25% over our 46 tok/s baseline (suffix tree handles
    short repeats that pure ngram misses)
  - Mixed agentic (tool + free-form): +30-50% combined

================================================================
DEPENDENCY
================================================================

`pip install arctic-inference` — must be in container build OR runtime entrypoint.
The class `vllm.v1.spec_decode.suffix_decoding.SuffixDecodingProposer` does
a lazy import at `__init__` so failure is loud and recoverable (operator
sees clear error and falls back to ngram by removing P75 env).

================================================================
SAFETY MODEL
================================================================

- If `arctic_inference` not installed → log warning + keep method=ngram (no boot failure)
- If user explicitly set method=suffix already → no-op (idempotent)
- If user explicitly set method != ngram (e.g. mtp) → no-op (we only swap from ngram)
- Suffix tree memory: bounded by suffix_decoding_max_cached_requests (default 10000)

================================================================
TUNABLE ENV
================================================================

GENESIS_ENABLE_P75_SUFFIX_DECODING=1                    # master switch
GENESIS_P75_TREE_DEPTH=24                               # suffix tree max depth
GENESIS_P75_SPEC_FACTOR=2.0                             # max draft length factor
GENESIS_P75_MIN_PROB=0.10                               # branch prob threshold
GENESIS_P75_CACHE_REQS=10000                            # cross-request cache cap

References:
- PR #25784 (vllm-project/vllm) MERGED 2025-11-03
- RFC #18037
- arXiv 2411.04975 (Suffix Decoding paper)
- Arctic Inference: github.com/snowflakedb/ArcticInference

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p75_suffix_decoding")

GENESIS_P75_MARKER = "Genesis P75 suffix-decoding auto-enable v7.56_local_os_import"


# ─── Sub-patch: inject method swap BEFORE the ngram default-value branch ────
# Anchor on the "Set default values if not provided" block (line ~482) which
# is the start of the ngram-only configuration branch. We insert P75 swap
# logic immediately BEFORE that branch so the swap (if it fires) reroutes
# downstream method-dependent code to the suffix path.

P75_OLD = (
    "        if self.method in (\"ngram\", \"[ngram]\"):\n"
    "            self.method = \"ngram\"\n"
    "\n"
    "        if self.method in (\"ngram\", \"ngram_gpu\"):\n"
)

P75_NEW = (
    "        if self.method in (\"ngram\", \"[ngram]\"):\n"
    "            self.method = \"ngram\"\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P75 vllm#25784] Auto-swap ngram -> suffix when env enabled.\n"
    "        # Suffix Decoding (Arctic Inference, arxiv 2411.04975) gives 30-50%\n"
    "        # higher TPS than ngram on agentic/tool-call workloads via per-prompt\n"
    "        # suffix tree + dynamic K speculation. Falls back gracefully if\n"
    "        # arctic_inference not installed.\n"
    "        # NOTE: local `import os` because vllm/config/speculative.py does NOT\n"
    "        # import `os` at module level — we add it scoped to this block to\n"
    "        # avoid mutating import order or risking circular-import issues.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        import os as _genesis_p75_os\n"
    "        if (\n"
    "            _genesis_p75_os.environ.get(\"GENESIS_ENABLE_P75_SUFFIX_DECODING\", \"\").strip().lower()\n"
    "            in (\"1\", \"true\", \"yes\", \"on\")\n"
    "            and self.method == \"ngram\"\n"
    "        ):\n"
    "            try:\n"
    "                import arctic_inference  # noqa: F401\n"
    "                _genesis_p75_orig = self.method\n"
    "                self.method = \"suffix\"\n"
    "                # Sane defaults tuned for Qwen3.6-A3B mixed workload.\n"
    "                # Each is overridable via env. Only set if upstream default (None or 0).\n"
    "                if not getattr(self, \"suffix_decoding_max_tree_depth\", 0):\n"
    "                    self.suffix_decoding_max_tree_depth = int(\n"
    "                        _genesis_p75_os.environ.get(\"GENESIS_P75_TREE_DEPTH\", \"24\"))\n"
    "                if not getattr(self, \"suffix_decoding_max_spec_factor\", 0):\n"
    "                    self.suffix_decoding_max_spec_factor = float(\n"
    "                        _genesis_p75_os.environ.get(\"GENESIS_P75_SPEC_FACTOR\", \"2.0\"))\n"
    "                if not getattr(self, \"suffix_decoding_min_token_prob\", 0):\n"
    "                    self.suffix_decoding_min_token_prob = float(\n"
    "                        _genesis_p75_os.environ.get(\"GENESIS_P75_MIN_PROB\", \"0.10\"))\n"
    "                if not getattr(self, \"suffix_decoding_max_cached_requests\", 0):\n"
    "                    self.suffix_decoding_max_cached_requests = int(\n"
    "                        _genesis_p75_os.environ.get(\"GENESIS_P75_CACHE_REQS\", \"10000\"))\n"
    "                logger.warning(\n"
    "                    \"[Genesis P75] Auto-swapped speculative method '%s' -> 'suffix' \"\n"
    "                    \"(tree_depth=%d, spec_factor=%.2f, min_prob=%.3f, cache_reqs=%d). \"\n"
    "                    \"Disable via GENESIS_ENABLE_P75_SUFFIX_DECODING=0.\",\n"
    "                    _genesis_p75_orig, self.suffix_decoding_max_tree_depth,\n"
    "                    self.suffix_decoding_max_spec_factor,\n"
    "                    self.suffix_decoding_min_token_prob,\n"
    "                    self.suffix_decoding_max_cached_requests,\n"
    "                )\n"
    "            except ImportError:\n"
    "                logger.warning(\n"
    "                    \"[Genesis P75] arctic_inference not installed -- \"\n"
    "                    \"keeping method=ngram. Install with: pip install arctic-inference\"\n"
    "                )\n"
    "\n"
    "        if self.method in (\"ngram\", \"ngram_gpu\"):\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("config/speculative.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P75 config/speculative.py — auto-enable suffix decoding",
        target_file=str(target),
        marker=GENESIS_P75_MARKER,
        sub_patches=[
            TextPatch(
                name="p75_suffix_swap",
                anchor=P75_OLD,
                replacement=P75_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P75",
            "GENESIS_ENABLE_P75_SUFFIX_DECODING",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P75 — suffix decoding auto-enable."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P75")
    log_decision("P75", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/config/speculative.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P75] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P75" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P75 applied: when GENESIS_ENABLE_P75_SUFFIX_DECODING=1 AND user "
        "config sets method=ngram, automatically swap to method=suffix "
        "(Arctic Inference suffix tree, dynamic K). Per arxiv 2411.04975 "
        "expected +30-50% TPS on agentic workloads."
    )
