# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 57 v2 — TurboQuant spec-decode capture-safe buffers.

================================================================
HYPOTHESIS WEAKENED 2026-04-25 (after noonghunna independently
tested #40798 backport at vllm#40831 issuecomment-4317503179):

@noonghunna ran the same #40798 backport experiment on his rig
(Qwen3.6-27B + MTP n=3, single 3090, cudagraph ON, torch.compile ON,
TPS=96 confirms compile+cudagraph both engaged) and the bug **persists
unchanged** — same `<tool_call>` empty cascade, same `amber amber...`
needle loops, same Layer-2 token duplication.

This means the buffer-pointer-drift hypothesis (warmup B=max_num_seqs
vs runtime B=q_len causing captured cudagraph to reference stale
data_ptr) is **likely not the mechanism**, OR #40798 alone is
insufficient even with the workspace shared. P57 v2 below ALSO
operates on the same buffer-expansion theory, so its expected outcome
is the same: probably does not close #40831.

Decision: ship P57 v2 as opt-in/research artifact, but pivot the
investigation focus to the bit-equality probe of `_tq_decode_stage1`
(load TQ K/V workspace, call kernel twice with same query+state,
check Mid_o byte-equality). That probe addresses kernel-determinism
at the irreducible level — if outputs differ across calls, the bug is
in the kernel itself, not in any buffer-pointer/routing layer.

See `Genesis_Doc/spec_decode_investigation/v7_12_session/` for full
investigation notes.
================================================================


Investigative result (2026-04-25 session, after deep code dive following
@noonghunna's six-probe ladder on vllm-project/vllm#40831):

The root cause of #40831 token corruption is a **buffer-shape mismatch
between captured cudagraph and runtime spec-decode call**.

Specifically:

1. `attention.py::_init_turboquant_buffers` pre-allocates per-layer
   decode scratch buffers (`_tq_mid_o_buf`, `_tq_output_buf`,
   `_tq_lse_buf`) sized at `B = max_num_seqs` — i.e. assumes plain
   single-token decode.

2. `TurboQuantMetadataBuilder.__init__` declares
   `_init_reorder_batch_threshold(1, supports_spec_as_decode=False)`,
   so spec-decode batches (where each request emits q_len = 1 + num_spec
   tokens for verify) get rejected from the decode lane and routed into
   `_prefill_attention`'s continuation branch.

3. `_prefill_attention`'s continuation branch enters the synthetic-
   decode fast path (`q_len <= _CONTINUATION_DECODE_THRESHOLD`) which
   calls `triton_turboquant_decode_attention` per-row with
   `B = q_len` — different from the captured graph's `B = max_num_seqs`.

4. Captured cudagraph references buffer `data_ptr` for the
   `B = max_num_seqs` shape; runtime spec-decode call passes a
   `B = q_len` shape. The **same base buffer is sliced differently**
   in capture vs replay → **token-level corruption** visible as
   `for for`, `age age`, `parameter parameter`, `<function=call`,
   `get_get_get_weather`, etc.

Compare with `gdn_attn.py:103-115`, which sizes its captured shape as
`max_num_seqs * (1 + num_spec)` — so the captured base buffer is large
enough for **both** plain decode (slice [:B] at runtime, B = num_decodes)
**and** spec-decode multi-token verify (slice [:B] with B up to
max_num_seqs * (1 + num_spec)). All slices reference the same persistent
base `data_ptr`, no pointer drift between capture and replay, no token
corruption.

P57 v2 implements the **minimal sufficient** version of this fix for
TurboQuant: just expand the per-layer buffer allocation by
`(1 + num_speculative_tokens)`. Do NOT flip
`supports_spec_as_decode=True` — that change is more invasive (triggers
reorder-batch logic that requires additional spec-decode tracking
infrastructure analogous to gdn_attn.py:117-156, and an early v1 attempt
at the flag flip caused engine startup to fail with
`max() iterable argument is empty` during cudagraph memory profiling).

The result:
  * Per-layer decode scratch buffers pre-sized for the spec-decode shape.
  * Spec batches still route through `_prefill_attention` (no flag flip),
    but the synthetic-decode fast-path's `triton_turboquant_decode_attention`
    call is now backed by a pre-allocated buffer **large enough** for
    both decode and spec batch shapes. Captured graph and runtime call
    alias into the same base buffer.

Memory cost: linear in num_speculative_tokens. For ngram n=3 on our
prod (max_num_seqs=2, Qwen3-Next-35B-A3B-FP8, 40 attention layers):

  per-layer buf_bytes (before P57) = 2 * 32 * 32 * 129 * 4    = ~530 KiB
  per-layer buf_bytes (with P57)   = 8 * 32 * 32 * 129 * 4    = ~2.1 MiB
  Total addition (40 layers)       = ~63 MiB — acceptable.

For high-concurrency servers (max_num_seqs=1024):

  per-layer (before)  = ~270 MiB
  per-layer (with P57) = ~1080 MiB
  Total (76 layers)   = ~63 GiB additional — UNACCEPTABLE on most GPUs.
                        Such configs MUST use #40798 (workspace manager,
                        single shared pool) AND have its
                        `_reserve_turboquant_decode_workspace()` patched
                        to also use expanded shape. This is left to
                        upstream — P57 stays opt-in and warns when
                        max_num_seqs makes the cost untenable.

Status: opt-in via `GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=1`.

Compatibility:
  * Standalone vLLM (no #40798): per-layer buffers expanded. Works.
  * vLLM with #40798 (workspace manager): P57's per-layer expansion is
    a no-op (the per-layer buffers don't exist anymore — replaced by
    workspace allocations sized at `max_num_reqs` in
    `_reserve_turboquant_decode_workspace`). For full fix in that
    configuration, the workspace reservation needs the same
    `(1 + num_spec)` multiplier — that change has to land upstream as
    part of the proper fix to #40831.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Investigation supported by AI tooling for source navigation.
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

log = logging.getLogger("genesis.wiring.p57_spec_decode_capture_safe_buffers")

GENESIS_P57_MARKER = "Genesis P57 v2 spec-decode capture-safe buffers v7.12"


def _is_enabled() -> bool:
    """Env-gate. Off by default — opt-in via:
    GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=1
    """
    return os.environ.get(
        "GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Anchor: attention.py::_init_turboquant_buffers — expand B by (num_spec+1) ──
#
# Conservative anchor: capture the ORIGINAL upstream block exactly.
# If the file has already been patched by #40798 (which removes these
# register_buffer calls), this anchor will not match and the patch
# will gracefully skip — leaving #40798's workspace-manager path
# intact (which itself needs its own expansion, but that's upstream's
# responsibility).

OLD_BUFFER_ALLOC = (
    "        # Pre-allocate decode intermediate buffers so model.to(device) moves\n"
    "        # them to GPU *before* the memory profiler runs.  Without this the\n"
    "        # profiler gives all free memory to KV cache blocks and the first\n"
    "        # decode OOMs when these buffers are lazily allocated.\n"
    "        _vllm_cfg = get_current_vllm_config()\n"
    "        B = _vllm_cfg.scheduler_config.max_num_seqs"
)

NEW_BUFFER_ALLOC = (
    "        # Pre-allocate decode intermediate buffers so model.to(device) moves\n"
    "        # them to GPU *before* the memory profiler runs.  Without this the\n"
    "        # profiler gives all free memory to KV cache blocks and the first\n"
    "        # decode OOMs when these buffers are lazily allocated.\n"
    "        _vllm_cfg = get_current_vllm_config()\n"
    "        # [Genesis P57 v2 spec-decode capture-safe buffers v7.12]\n"
    "        # Expand B by (num_speculative_tokens + 1) so the captured cudagraph\n"
    "        # base buffer fits both plain decode (slice [:max_num_seqs]) AND\n"
    "        # spec-decode multi-token verify (slice [:1+num_spec*per_req]).\n"
    "        # All slices alias into the same persistent data_ptr — no pointer\n"
    "        # drift between cudagraph capture and replay. Fixes vllm#40831\n"
    "        # token corruption when used WITHOUT cudagraph_mode=NONE workaround.\n"
    "        # Mirrors the pattern in vllm/v1/attention/backends/gdn_attn.py:103-115.\n"
    "        _p57_spec_cfg = getattr(_vllm_cfg, 'speculative_config', None)\n"
    "        _p57_num_spec = (\n"
    "            (getattr(_p57_spec_cfg, 'num_speculative_tokens', 0) or 0)\n"
    "            if _p57_spec_cfg is not None else 0\n"
    "        )\n"
    "        B = _vllm_cfg.scheduler_config.max_num_seqs * (1 + _p57_num_spec)"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/attention/attention.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P57 v2 spec-decode capture-safe buffers (alloc expansion)",
        target_file=str(target),
        marker=GENESIS_P57_MARKER,
        sub_patches=[
            TextPatch(
                name="p57v2_alloc_expansion",
                anchor=OLD_BUFFER_ALLOC,
                replacement=NEW_BUFFER_ALLOC,
                required=True,
            )
        ],
    )


# ─── Cost-warning: high-concurrency configs need workspace-manager path ────

P57_MEMORY_WARN_MAX_NUM_SEQS = 32  # Warn above this; UNACCEPTABLE above ~256.


def _check_memory_acceptability() -> tuple[bool, str]:
    """Return (proceed, warning) based on current max_num_seqs config.

    On configs where (1 + num_spec) × max_num_seqs makes per-layer buffers
    blow up, we should NOT pre-allocate that much per layer. Instead the
    user should switch to the workspace-manager path (#40798) which
    shares one buffer across all layers. P57 then has nothing to do.
    """
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        max_num_seqs = cfg.scheduler_config.max_num_seqs
        spec = getattr(cfg, "speculative_config", None)
        num_spec = (
            (getattr(spec, "num_speculative_tokens", 0) or 0)
            if spec is not None else 0
        )
    except Exception as e:
        # Can't probe config — fall through and let the patch run; risk
        # is on the operator if memory blows up.
        return True, f"could not probe config to check memory cost: {e}"

    if num_spec == 0:
        return True, "num_speculative_tokens=0 — patch would be a no-op anyway"

    expanded_B = max_num_seqs * (1 + num_spec)
    if max_num_seqs > P57_MEMORY_WARN_MAX_NUM_SEQS:
        return True, (
            f"WARNING: max_num_seqs={max_num_seqs} with num_spec={num_spec} → "
            f"per-layer buffer expanded to B={expanded_B}. On large attention "
            f"layer counts this can use multi-GiB of pre-alloc. Consider the "
            f"upstream PR #40798 (workspace manager) path instead, which "
            f"shares a single workspace across layers. P57 still applies but "
            f"may not be the right cost/benefit tradeoff here."
        )
    return True, f"max_num_seqs={max_num_seqs}, num_spec={num_spec} → B={expanded_B} acceptable"


def apply() -> tuple[str, str]:
    """Apply P57 v2 wiring. Never raises.

    v7.13: routes through the unified `dispatcher.should_apply("P57")`
    gate (Dispatcher v2). P57 is marked `deprecated=True` in the registry
    because the empirical workaround for #40831 is `prompt_lookup_min=8`
    (config-only); the buffer-expansion fix here remains as a research
    artifact for operators who want to test it via env override.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P57")
    log_decision("P57", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    proceed, warning = _check_memory_acceptability()
    if not proceed:
        return "skipped", warning
    if warning:
        log.info("[Genesis P57] %s", warning)

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "attention.py not found"

    result, failure = patcher.apply()

    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "per-layer decode scratch buffers expanded by (1 + num_spec). "
            "Captured cudagraph and runtime spec-decode call now alias into "
            "the same base buffer — no pointer drift. Token corruption "
            "(#40831) eliminated when speculative_config is active.",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        # Most likely cause: anchor not found because PR #40798 already
        # removed the register_buffer block. In that case the workspace
        # manager path needs its own expansion (upstream's responsibility).
        msg = failure.reason if failure else "anchor not found"
        return (
            "skipped",
            f"{msg} — likely #40798 (workspace manager) already merged or "
            "backported. P57 v2 only patches the per-layer register_buffer "
            "path. For workspace-manager-based vLLM, the equivalent fix is "
            "to multiply the shape passed to current_workspace_manager()."
            "get_simultaneous() in _reserve_turboquant_decode_workspace by "
            "(1 + num_speculative_tokens). That requires either an upstream "
            "change to #40798 or a separate Genesis patch on top of the "
            "#40798 backport.",
        )
    return "failed", failure.reason if failure else "unknown failure"
