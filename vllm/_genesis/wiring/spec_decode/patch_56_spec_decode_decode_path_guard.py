# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 56 — TurboQuant spec-decode safe-path guard.

================================================================
SUPERSEDED 2026-04-25 — see "Status update" section below before
adopting this patch. The simpler upstream workaround is:

    --compilation-config '{"cudagraph_mode":"NONE"}'

posted by @noonghunna in
https://github.com/noonghunna/qwen36-27b-single-3090/commit/de1d1afab324c8467dfd80f70da2e55567e3e841

Use that instead. P56 stays in the tree as a research artifact
documenting a partial-fix dead-end.
================================================================


Problem
-------
The TurboQuant attention backend declares
`_init_reorder_batch_threshold(1, supports_spec_as_decode=False)` at
`turboquant_attn.py:192`. Combined with the absence of a varlen kernel
analogous to FlashAttention's, this routes any speculative-decode batch
(MTP n>1 or ngram n>1, where each request has q_len = 1 + num_spec) into
`_prefill_attention`'s synthetic-decode fast path (lines 622-646).

The fast path constructs `synth_seq_lens = arange(cached+1, seq+1)` and
`synth_bt = block_table.expand(q_len, -1)`, then calls
`triton_turboquant_decode_attention` treating each row of the q_len-row
query as an *independent* decode request. That kernel's online softmax
runs **per row**, with no cross-row causal rescale.

For true single-token decode (q_len == 1) this is correct. For spec-decode
batches where rows are causally linked (row p must attend to rows 0..p-1
of the SAME request through the just-written-but-not-committed KV slots),
the per-row fast path silently breaks the causal semantics. The result:
degenerate token loops on structured outputs (`<tool_call><tool_call>...`,
`</parameter></parameter>...`, needle-recall first-token-then-loop), as
documented in vllm-project/vllm#40831 by @noonghunna.

Independently confirmed on Genesis pin `fe9c3d6c5` with:
  - Qwen3-Next-35B-A3B-FP8 (MoE hybrid, 30 MoE + 10 full-attn)
  - 2× RTX A5000 (Ampere SM 8.6), TP=2
  - kv_cache_dtype=turboquant_k8v4
  - --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}'

Symptom on our config: tool-call output `<tool_call>\n<function<parameter=
city>\nParis\n</parameter>\n<parameter=parameter=unit>\nCelsius\n</parameter>\n
</function>\n</parameter>\n</parameter>\n</parameter>\n</parameter>` — same
shape as @noonghunna's reproduction on Qwen3.6-27B Lorbus int4 single-3090.
PR #40074 (IOOB clamp) backport applied — bug persists. Hypothesis 2
(decode kernel non-deterministic dequant) thereby refuted on our rig.

Fix (this patch)
----------------
Tighten the fast-path entry condition from `q_len <= _CONTINUATION_DECODE_
THRESHOLD` to `q_len == 1`. Anything with q_len > 1 (spec-decode batches,
chunked prefill continuation chunks) falls through to the
`_continuation_prefill` path which uses `flash_attn_varlen_func` with
`causal=True` — that path correctly preserves cross-row causal semantics.

Performance cost: spec-decode requests pay the dequant overhead of
`_continuation_prefill` instead of the decode fast-path. Worth it for
correctness; without the guard, spec-decode is functionally broken on
TurboQuant.

Status: opt-in (`GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1`) until upstream
either flips `supports_spec_as_decode=True` with a varlen kernel, or
acknowledges the routing limitation.

Credit
------
- Root-cause analysis: independent investigation on top of @noonghunna's
  isolation matrix in #40831.
- Bug surface: @noonghunna (vllm-project/vllm#40807, #40831), Qwen3.6-27B
  single-3090 production stack at github.com/noonghunna/qwen36-27b-single-3090.
- Backend design: @vibhavagarwal5 et al. (TurboQuant author, line 192
  `supports_spec_as_decode=False` is theirs; the routing guard here just
  works around it pending a proper varlen kernel).

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

log = logging.getLogger("genesis.wiring.p56_spec_decode_guard")

GENESIS_P56_MARKER = "Genesis P56 spec-decode safe-path guard v7.11"


def _is_enabled() -> bool:
    """Env-gate. Off by default — opt-in until upstream flips
    supports_spec_as_decode=True. Enable with GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1.
    """
    return os.environ.get(
        "GENESIS_ENABLE_P56_SPEC_DECODE_GUARD", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor: the exact `if` condition entering the synthetic-decode fast path.
# We tighten `q_len <= _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`
# so spec-decode batches (q_len > 1 per request) are routed through the
# correct flash_attn_varlen path inside _continuation_prefill instead.

OLD_GATE = "                if q_len <= _CONTINUATION_DECODE_THRESHOLD:"

NEW_GATE = (
    "                # [Genesis P56 spec-decode safe-path guard v7.11]\n"
    "                # Original condition was `q_len <= _CONTINUATION_DECODE_THRESHOLD`,\n"
    "                # but the synthetic-decode fast path's per-row online softmax\n"
    "                # does not preserve causal semantics across spec-decode draft\n"
    "                # rows. Restrict to true single-token decode (q_len == 1).\n"
    "                # Spec-decode batches fall through to _continuation_prefill\n"
    "                # which uses flash_attn_varlen_func with causal=True. See\n"
    "                # vllm-project/vllm#40831 (@noonghunna) for the bug repro and\n"
    "                # this docstring for the routing analysis.\n"
    "                if q_len == 1 and q_len <= _CONTINUATION_DECODE_THRESHOLD:"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P56 spec-decode safe-path guard",
        target_file=str(target),
        marker=GENESIS_P56_MARKER,
        sub_patches=[
            TextPatch(
                name="p56_spec_decode_guard",
                anchor=OLD_GATE,
                replacement=NEW_GATE,
                required=True,
            )
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P56 wiring. Never raises.

    DEPRECATED 2026-04-25 by noonghunna's six-probe ladder
    (https://github.com/noonghunna/qwen36-27b-single-3090/commit/de1d1afa).
    His Probe 4 = `_CONTINUATION_DECODE_THRESHOLD = 0` is architecturally
    equivalent to what this patch does and was empirically shown not to
    fix #40831. Real bug is in CUDA graph capture/replay layer, not
    routing. Use `--compilation-config '{"cudagraph_mode":"NONE"}'`
    instead. P56 stays in the tree as research artifact.

    v7.13: routes the env+config decision through the unified
    `dispatcher.should_apply("P56")` gate (Dispatcher v2). The previous
    config_detect.should_apply() probe + local _is_enabled() pair is now
    folded into one call so the apply matrix can record a single verdict.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P56")
    log_decision("P56", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "spec-decode fast-path guard wired; q_len > 1 batches now route "
            "through _continuation_prefill (causal-correct flash_attn_varlen)",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
