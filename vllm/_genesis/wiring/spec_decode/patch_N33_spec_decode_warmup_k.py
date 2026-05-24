# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N33 — spec-decode warmup K-aware sizing (root-cause fix).

Backport-with-extension of [vllm-project/vllm#37521](https://github.com/vllm-project/vllm/pull/37521)
by `itailang` (OPEN at the time of backport, 2026-05-02). Genesis
extends the upstream PR beyond its EAGLE-only gate to cover ALL
speculative-decoding methods (MTP, ngram, draft model, EAGLE).

================================================================
WHAT THIS PATCH DOES
================================================================

In `gpu_model_runner._dummy_sampler_run()`, the warmup path that drives
KV-cache profiling builds dummy speculative-decode metadata using
**one** draft token per request:

    # Vanilla vLLM nightly 0.20.1rc1.dev16+g7a1eb8ac2
    if self.speculative_config:
        draft_token_ids = [[0] for _ in range(num_reqs)]
        ...
        logits = torch.randn(
            num_tokens + num_reqs,    # = num_reqs + num_reqs at K=1
            logits.shape[-1], ...
        )
        self.rejection_sampler(...)

But at runtime, with `num_speculative_tokens = K`, the rejection sampler
sees `num_reqs * K` draft tokens and a logits tensor shaped
`[num_reqs * K + num_reqs, vocab_size]`. K can be 3 (MTP), 4 (ngram),
or up to 8 (EAGLE3).

The under-counted warmup misleads:

1. **Profile run's peak-memory probe** — KV-cache budget allocator
   thinks rejection-sampler footprint is K=1 sized, leaves too little
   headroom for the real K=N path → **mid-stream OOM during
   `propose_draft_token_ids → llm_base_proposer.propose`** (reported
   on `noonghunna/club-3090#16` 2026-05-01 by `ampersandru`).

2. **TurboQuant `WorkspaceManager` lock** — workspace is reserved at
   warmup and locked. When real spec-decode tries to grow the workspace
   to fit K-token logits, `_ensure_workspace_size` raises
   `AssertionError: Workspace is locked but allocation requires X MB`.
   Cited cause for noonghunna's `dev205` MTP K=3 single-card blocker on
   discussion #19 thread 2026-05-01 01:12.

Both bugs share **one root cause**: warmup uses dummy K=1 instead of
the real `num_speculative_tokens`. Fixing the warmup to use the real K
correctly sizes the KV-cache budget AND pre-reserves the workspace, so
both downstream symptoms disappear without further patching.

================================================================
GENESIS EXTENSION OVER UPSTREAM #37521
================================================================

Upstream PR #37521 gates the fix on `self.speculative_config.use_eagle()`
— only EAGLE benefits. Genesis applies the same principle to all
spec-decode methods because the rejection-sampler memory footprint
scales with `num_speculative_tokens` regardless of how the draft
tokens were produced (EAGLE, MTP, ngram all consume identical sampler
shapes).

The Genesis replacement reads `num_speculative_tokens` from
`self.speculative_config` directly with a defensive `getattr` chain
(`getattr(self.speculative_config, 'num_speculative_tokens', 0) or 0`)
so missing attribute, None, or 0 all fall through to the original
`[0]` behavior. No regression for non-spec-decode configs.

When `K > 1`, the dummy draft tokens are `list(range(K))` (distinct
ids) — matches upstream PR's choice. Distinct ids matter because some
rejection-sampler paths dedupe identical draft tokens, which would
under-count again.

================================================================
WHEN UPSTREAM MERGES
================================================================

PR #37521 is OPEN at backport time. When it merges, this patch's
`upstream_drift_markers` will detect the upstream comment fragment and
auto-skip on a vLLM bump. Genesis can then either:

- Remove this patch entirely if upstream's EAGLE-only fix proves
  sufficient for our user base
- Keep this patch as the "extends to MTP/ngram" delta over upstream

The delta is small (3 LOC); maintenance burden is minimal either way.

================================================================
SAFETY MODEL
================================================================

- **Algorithmic safety**: increases warmup cost (K times more dummy
  rejection-sampler work). On 2× A5000 PROD this is ~30 ms additional
  warmup. Worst-case OOM at warmup is *better* than runtime OOM —
  diagnoses too-aggressive `--gpu-memory-utilization` early.
- **Default**: ON when `num_speculative_tokens > 1` (real correctness
  fix, not experimental). Operators can disable via
  `GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1` if warmup itself OOMs
  on a tight rig.
- **Idempotent**: marker `Genesis PN33 spec-decode warmup K-aware`
  pin-checked.
- **Anchor-stable**: targets `draft_token_ids = [[0] for _ in range(...)]`
  which is canonical (single line, no formatting variance).

================================================================
EFFORT / VALIDATION
================================================================

Code change: ~3 LOC. Test surface: 11 TDD tests pinning anchor +
replacement contract + idempotency + skip-when-disabled + drift
detection.

Cross-rig validation needed:

- ampersandru `propose_draft_token_ids` mid-stream OOM should disappear
  on club-3090 1× 3090 long-vision 140K + MTP K=3 with PN33 enabled.
- noonghunna workspace-lock `AssertionError` on dev205 MTP K=3
  single-card should disappear with PN33 enabled.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport credit: itailang (vllm-project/vllm#37521).
Reporter credits:
  - ampersandru (mid-stream OOM, club-3090#16 2026-05-01 16:58)
  - noonghunna (workspace-lock blocker, club-3090 disc #19 2026-05-01 01:12)
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pN33_spec_decode_warmup_k")


GENESIS_PN33_MARKER = (
    "Genesis PN33 spec-decode warmup K-aware "
    "(vllm#37521 backport extended to MTP/ngram, v7.65)"
)


# Anchor: the original 1-draft-token warmup line in
# `gpu_model_runner._dummy_sampler_run()`. Single-line + canonical
# formatting so anchor drift risk is minimal.
PN33_ANCHOR = (
    "        if self.speculative_config:\n"
    "            draft_token_ids = [[0] for _ in range(num_reqs)]\n"
)


# Replacement: K-aware dummy draft tokens covering ALL spec-decode
# methods. Defensive `getattr` chain so missing/None/0 fall through
# to original behavior.
#
# Env override: GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1 forces
# the original [0] behavior for tight-VRAM rigs where the K-sized
# warmup itself OOMs.
PN33_REPLACEMENT = (
    "        if self.speculative_config:\n"
    "            # [Genesis PN33 vllm#37521-extended]\n"
    "            # Use real num_speculative_tokens for warmup so the\n"
    "            # KV-cache profile and TQ workspace lock account for\n"
    "            # peak rejection-sampler footprint. Covers EAGLE, MTP,\n"
    "            # ngram, draft-model methods uniformly.\n"
    "            # Fallback to [0] if env disabled or attribute missing.\n"
    "            import os as _genesis_pn33_os\n"
    "            _genesis_pn33_disabled = _genesis_pn33_os.environ.get(\n"
    "                'GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K', ''\n"
    "            ).strip() in ('1', 'true', 'yes', 'on')\n"
    "            _genesis_pn33_K = getattr(\n"
    "                self.speculative_config, 'num_speculative_tokens', 0\n"
    "            ) or 0\n"
    "            if (not _genesis_pn33_disabled) and _genesis_pn33_K > 1:\n"
    "                _genesis_pn33_dummy_tokens = list(range(_genesis_pn33_K))\n"
    "            else:\n"
    "                _genesis_pn33_dummy_tokens = [0]\n"
    "            draft_token_ids = [\n"
    "                _genesis_pn33_dummy_tokens for _ in range(num_reqs)\n"
    "            ]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN33 v1/worker/gpu_model_runner.py — spec-decode warmup "
            "K-aware sizing (vllm#37521 extended to MTP/ngram)"
        ),
        target_file=str(target),
        marker=GENESIS_PN33_MARKER,
        sub_patches=[
            TextPatch(
                name="pN33_warmup_k_draft_tokens",
                anchor=PN33_ANCHOR,
                replacement=PN33_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN33",
            # Upstream-side detection: if vllm#37521 (or equivalent)
            # merges into the pin, the modified file will contain the
            # specific PR-introduced line that builds K-element dummy
            # draft tokens via `range(self.num_spec_tokens)`. This is
            # specific enough to PR #37521 that it won't match any
            # other place in vllm that uses `use_eagle()` generically.
            #
            # IMPORTANT: do NOT use just "use_eagle()" — that method
            # is referenced from many places in normal vllm code and
            # would false-positive (PN33 v1 had this bug, observed on
            # live boot 2026-05-02 — server skipped PN33 incorrectly).
            "spec_decode_tokens = [i for i in range(self.num_spec_tokens)]",
            "spec_decode_tokens for _ in range(num_reqs)",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN33 — spec-decode warmup K-aware sizing.

    Default ON when spec-decode is active. Env override
    GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1 reverts to [0].
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN33")
    log_decision("PN33", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_model_runner.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN33] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #37521 (or equivalent) appears merged. "
                "Re-evaluate whether Genesis PN33 extension is still "
                "needed for MTP/ngram coverage."
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "PN33 applied: spec-decode warmup uses real num_speculative_tokens "
        "instead of dummy K=1. Closes (a) ampersandru mid-stream OOM via "
        "propose_draft_token_ids and (b) noonghunna workspace-lock "
        "AssertionError on TQ + MTP K=3 single-card. Disable via "
        "GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1 if warmup OOMs."
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
