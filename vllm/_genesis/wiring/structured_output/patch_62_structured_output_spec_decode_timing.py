# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 62 — structured-output + spec-decode reasoning-end timing fix.

Backport of vllm-project/vllm#36138 (sfbemerk).

================================================================
HIGH-PRIORITY candidate for closing residual 30-50% broken tool-call
output that P60+P60b+P61 didn't fully resolve.
================================================================

What this fixes
---------------
When `</think>` (or implicit reasoning-end via `<tool_call>`) arrives WITHIN
a speculative-decode token batch:

  1. Old `should_advance()` checks `delta_from = num_computed_tokens -
     num_output_placeholders` which is computed BEFORE the model runs and
     INCLUDES speculative tokens. After the model runs, `all_token_ids` is
     shorter than expected — `delta = all_token_ids[delta_from:]` is empty
     → `is_reasoning_end_streaming()` never sees `</think>` → reasoning_ended
     stays False forever.

  2. Result: grammar bypass for ALL POST-REASONING tokens. Model emits
     unconstrained text after reasoning ended. For tool-call requests this
     manifests as garbage XML that doesn't conform to qwen3_coder schema.

This was identified by sfbemerk as a special case of more general bugs in
handling structured output when reasoning_end appears in speculative tokens.

Mechanism (per sfbemerk + cicirori #34650 analysis)
---------------------------------------------------
Per-step timing:
  Step k: schedule()  → num_computed_tokens += (1 + num_spec)  [BEFORE model]
  Step k: model runs  → outputs M tokens (1 ≤ M ≤ 1+num_spec)
  Step k: update_from_output() → all_token_ids extends by M
  Step k: should_advance(req)  → delta_from > len(all_token_ids) → empty
                                  delta → reasoning_end check fails

Fix
---
Replace `should_advance(request)` with `update_reasoning_ended(request,
new_token_ids)` — explicitly pass the tokens THAT WERE ACCEPTED THIS STEP,
not a derived delta. Then check `is_reasoning_end_streaming(all_token_ids,
new_token_ids)` against the actual accepted tokens.

Also add:
  - `validate_tokens_reasoning_aware()` — split spec tokens at reasoning
    boundary, validate ONLY post-boundary against grammar
  - `identify_constrained_draft_tokens()` — same split returning unconstrained
    + constrained partitions
  - `_find_reasoning_end_in_tokens()` — find index of reasoning-end marker
    using `is_reasoning_end_streaming` on progressive prefixes

Update `grammar_bitmask` to apply post-reasoning bitmask only to positions
after the discovered reasoning-end-idx within `req_tokens`.

What this patch DOES NOT do
---------------------------
- Does NOT remove `should_advance` (kept as no-op compat wrapper for any
  external callers we haven't found). PR #36138 deletes it; we keep it
  alive to reduce blast radius.
- Does NOT touch `enable_in_reasoning` path (separate code branch, not
  affected by spec-decode timing).

Risks
-----
- Multiple call sites updated. If any one anchor drifts, that call site
  reverts to old `should_advance` behavior. Other call sites continue to
  use the new logic. Patch group becomes partial.
- New methods rely on `is_reasoning_end_streaming` interface — if that
  changes upstream, our new methods break.
- All-or-nothing not enforced (per file). Operator should grep boot logs
  for `[P62 ...] applied N sub-patches` to verify all expected anchors
  matched.

Status: opt-in (`GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1`).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher, TextPatchResult, TextPatch,
)

log = logging.getLogger("genesis.wiring.p62_struct_out_spec_timing")

GENESIS_P62_MARKER = "Genesis P62 structured-output spec-decode timing v7.70 (post-PR41199 update — uses _get_reasoner)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── File 1: structured_output/__init__.py ────────────────────────────────

# Sub-patch 1a: rewrite grammar_bitmask req_tokens loop to be reasoning-aware
GRAMMAR_BITMASK_OLD = (
    "                state_advancements = 0\n"
    "                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())\n"
    "                for token in itertools.chain(req_tokens, (-1,)):\n"
    "                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))\n"
    "                    if token == -1:\n"
    "                        # Stop advancing the grammar once we hit a padding token.\n"
    "                        apply_bitmask = False\n"
    "                    if apply_bitmask and not grammar.is_terminated():\n"
    "                        accepted = grammar.accept_tokens(req_id, [token])\n"
    "                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)\n"
    "                        state_advancements += 1\n"
    "                    cumulative_index += 1"
)

GRAMMAR_BITMASK_NEW = (
    "                # [Genesis P62 vllm#36138, v7.70 post-PR41199] reasoning-aware bitmask + advance\n"
    "                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())\n"
    "                reasoning_end_idx = None\n"
    "                if req_tokens and not apply_bitmask:\n"
    "                    # When reasoning hasn't ended yet, only positions AFTER\n"
    "                    # reasoning_end inside this spec batch must be constrained.\n"
    "                    # PR #41199 made reasoner per-request via _get_reasoner;\n"
    "                    # grammar_bitmask() has `requests` dict in scope, use it.\n"
    "                    _p62_req = requests.get(req_id)\n"
    "                    _p62_reasoner = (\n"
    "                        self._get_reasoner(_p62_req) if _p62_req is not None else None\n"
    "                    )\n"
    "                    reasoning_end_idx = self._find_reasoning_end_in_tokens(\n"
    "                        _p62_reasoner, list(req_tokens)\n"
    "                    )\n"
    "                state_advancements = 0\n"
    "                for tok_idx, token in enumerate(itertools.chain(req_tokens, (-1,))):\n"
    "                    if reasoning_end_idx is not None:\n"
    "                        pos_apply_bitmask = tok_idx > reasoning_end_idx\n"
    "                    else:\n"
    "                        pos_apply_bitmask = apply_bitmask\n"
    "                    self._fill_bitmasks(\n"
    "                        ((grammar, cumulative_index, pos_apply_bitmask),)\n"
    "                    )\n"
    "                    if token == -1:\n"
    "                        # Stop advancing the grammar once we hit a padding token.\n"
    "                        pos_apply_bitmask = False\n"
    "                    if pos_apply_bitmask and not grammar.is_terminated():\n"
    "                        accepted = grammar.accept_tokens(req_id, [token])\n"
    "                        assert accepted, (\n"
    "                            token, req_id, scheduled_spec_decode_tokens,\n"
    "                        )\n"
    "                        state_advancements += 1\n"
    "                    cumulative_index += 1"
)

# Sub-patch 1b: Add new helper methods AFTER should_advance (keep should_advance!)
# Anchor: end of should_advance method body (just before `def clear_backend`)
NEW_METHODS_OLD = (
    "        return False\n"
    "\n"
    "    def clear_backend(self) -> None:"
)

NEW_METHODS_NEW = (
    "        return False\n"
    "\n"
    "    # ── [Genesis P62 vllm#36138, v7.70 post-PR41199] reasoning-aware spec-decode helpers ──\n"
    "    # Updated 2026-05-04: vllm PR #41199 (MERGED 2026-05-01) refactored\n"
    "    # `self.reasoner` (manager-level instance) into `self._get_reasoner(request)`\n"
    "    # (per-request lazy build with chat_template_kwargs). All P62 helpers updated\n"
    "    # to call `self._get_reasoner(request)` instead of dereferencing the dead\n"
    "    # `self.reasoner` attribute.\n"
    "\n"
    "    def update_reasoning_ended(\n"
    "        self,\n"
    "        request: \"Request\",\n"
    "        new_token_ids: list[int],\n"
    "    ) -> None:\n"
    "        \"\"\"Update reasoning_ended flag based on actually-accepted tokens.\n"
    "\n"
    "        Replaces should_advance() flag-mutation logic that used a derived\n"
    "        delta which becomes empty when speculative tokens are involved.\n"
    "        \"\"\"\n"
    "        if not request.use_structured_output:\n"
    "            return\n"
    "        reasoner = self._get_reasoner(request)\n"
    "        if reasoner is None or self.enable_in_reasoning:\n"
    "            return\n"
    "        structured_req = request.structured_output_request\n"
    "        assert structured_req is not None\n"
    "        if structured_req.reasoning_ended:\n"
    "            return\n"
    "        all_token_ids = request.all_token_ids\n"
    "        if reasoner.is_reasoning_end_streaming(all_token_ids, new_token_ids):\n"
    "            structured_req.reasoning_ended = True\n"
    "\n"
    "    def validate_tokens_reasoning_aware(\n"
    "        self,\n"
    "        request: \"Request\",\n"
    "        spec_token_ids: list[int],\n"
    "    ) -> list[int]:\n"
    "        \"\"\"Validate spec tokens against grammar, splitting at reasoning end.\n"
    "        Tokens before reasoning end pass through unvalidated.\n"
    "        \"\"\"\n"
    "        unconstrained_tokens, constrained_tokens = (\n"
    "            self.identify_constrained_draft_tokens(request, spec_token_ids)\n"
    "        )\n"
    "        if constrained_tokens:\n"
    "            assert request.structured_output_request is not None\n"
    "            assert request.structured_output_request.grammar is not None\n"
    "            grammar = request.structured_output_request.grammar\n"
    "            grammar_validated = grammar.validate_tokens(constrained_tokens)\n"
    "            return unconstrained_tokens + grammar_validated\n"
    "        return spec_token_ids\n"
    "\n"
    "    def identify_constrained_draft_tokens(\n"
    "        self,\n"
    "        request: \"Request\",\n"
    "        spec_token_ids: list[int],\n"
    "    ) -> tuple[list[int], list[int]]:\n"
    "        \"\"\"Split spec tokens into (unconstrained, constrained) partitions\n"
    "        based on whether reasoning has ended at each position.\n"
    "        \"\"\"\n"
    "        if not request.use_structured_output:\n"
    "            return spec_token_ids, []\n"
    "        reasoner = self._get_reasoner(request)\n"
    "        if reasoner is None:\n"
    "            return [], spec_token_ids\n"
    "        if self.enable_in_reasoning:\n"
    "            return [], spec_token_ids\n"
    "        structured_req = request.structured_output_request\n"
    "        assert structured_req is not None\n"
    "        assert structured_req.grammar is not None\n"
    "        if structured_req.reasoning_ended:\n"
    "            return [], spec_token_ids\n"
    "        split_idx = self._find_reasoning_end_in_tokens(reasoner, spec_token_ids)\n"
    "        if split_idx is None:\n"
    "            return spec_token_ids, []\n"
    "        return spec_token_ids[: split_idx + 1], spec_token_ids[split_idx + 1 :]\n"
    "\n"
    "    def _find_reasoning_end_in_tokens(self, reasoner, token_ids: list[int]):\n"
    "        \"\"\"Find index of reasoning-end token within a list, or None.\n"
    "\n"
    "        Iterates progressive prefixes and uses is_reasoning_end_streaming\n"
    "        to detect multi-token end markers correctly. `reasoner` is a\n"
    "        per-request `ReasoningParser` instance from `_get_reasoner(request)`.\n"
    "        \"\"\"\n"
    "        if reasoner is None or self.enable_in_reasoning:\n"
    "            return None\n"
    "        for i, token in enumerate(token_ids):\n"
    "            prefix = token_ids[: i + 1]\n"
    "            if reasoner.is_reasoning_end_streaming(prefix, [token]):\n"
    "                return i\n"
    "        return None\n"
    "\n"
    "    def clear_backend(self) -> None:"
)


def _make_struct_out_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/structured_output/__init__.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P62 structured_output/__init__.py",
        target_file=str(target),
        marker=GENESIS_P62_MARKER + " :: structured_output",
        sub_patches=[
            TextPatch(name="p62_grammar_bitmask", anchor=GRAMMAR_BITMASK_OLD,
                      replacement=GRAMMAR_BITMASK_NEW, required=True),
            TextPatch(name="p62_new_methods", anchor=NEW_METHODS_OLD,
                      replacement=NEW_METHODS_NEW, required=True),
        ],
        upstream_drift_markers=[
            "def update_reasoning_ended",  # upstream-merged version present
        ],
    )


# ─── File 2: scheduler.py ──────────────────────────────────────────────────

# Sub-patch 2a: update_from_output — replace whole should_advance block
# (including inner `if not accept_tokens` body to maintain indentation).
SCHED_UPDATE_FROM_OUTPUT_OLD = (
    "            if new_token_ids and self.structured_output_manager.should_advance(request):\n"
    "                struct_output_request = request.structured_output_request\n"
    "                assert struct_output_request is not None\n"
    "                assert struct_output_request.grammar is not None\n"
    "                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]\n"
    "                    req_id, new_token_ids\n"
    "                ):\n"
    "                    logger.error(\n"
    "                        \"Unexpected: grammar rejected tokens %s for request %s. \"\n"
    "                        \"Terminating request.\",\n"
    "                        new_token_ids,\n"
    "                        req_id,\n"
    "                    )\n"
    "                    request.status = RequestStatus.FINISHED_ERROR\n"
    "                    request.resumable = False\n"
    "                    stopped = True"
)

SCHED_UPDATE_FROM_OUTPUT_NEW = (
    "            # [Genesis P62 vllm#36138] reasoning-aware grammar acceptance\n"
    "            if new_token_ids and self.structured_output_manager is not None:\n"
    "                _, _p62_tokens_for_grammar = (\n"
    "                    self.structured_output_manager.identify_constrained_draft_tokens(\n"
    "                        request, new_token_ids\n"
    "                    )\n"
    "                )\n"
    "                if _p62_tokens_for_grammar:\n"
    "                    struct_output_request = request.structured_output_request\n"
    "                    assert struct_output_request is not None\n"
    "                    assert struct_output_request.grammar is not None\n"
    "                    if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]\n"
    "                        req_id, _p62_tokens_for_grammar\n"
    "                    ):\n"
    "                        logger.warning(\n"
    "                            \"Unexpected: grammar rejected tokens %s for request %s.\",\n"
    "                            _p62_tokens_for_grammar,\n"
    "                            req_id,\n"
    "                        )\n"
    "                        request.status = RequestStatus.FINISHED_ERROR\n"
    "                        request.resumable = False\n"
    "                        stopped = True\n"
    "                # [Genesis P62 vllm#36138] update reasoning_ended after accept\n"
    "                self.structured_output_manager.update_reasoning_ended(\n"
    "                    request, new_token_ids=new_token_ids\n"
    "                )"
)

# Sub-patch 2c: update_draft_token_ids — use validate_tokens_reasoning_aware
# (anchor matches BOTH original and post-P58 layout because P58 doesn't
# touch this exact line block)
SCHED_UDTI_OLD = (
    "            # Add newly generated spec token ids to the request.\n"
    "            if self.structured_output_manager.should_advance(request):\n"
    "                metadata = request.structured_output_request\n"
    "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]\n"
    "            request.spec_token_ids = spec_token_ids"
)

SCHED_UDTI_NEW = (
    "            # [Genesis P62 vllm#36138] reasoning-aware spec validation\n"
    "            spec_token_ids = (\n"
    "                self.structured_output_manager.validate_tokens_reasoning_aware(\n"
    "                    request, spec_token_ids\n"
    "                )\n"
    "            )\n"
    "            request.spec_token_ids = spec_token_ids"
)

# Sub-patch 2d: update_draft_token_ids_in_output — same replacement
SCHED_UDTIO_OLD = (
    "            # Filter out spec tokens which do not adhere to the grammar.\n"
    "            if self.structured_output_manager.should_advance(request):\n"
    "                metadata = request.structured_output_request\n"
    "                assert metadata is not None and metadata.grammar is not None\n"
    "                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)"
)

SCHED_UDTIO_NEW = (
    "            # [Genesis P62 vllm#36138] reasoning-aware spec validation\n"
    "            spec_token_ids = (\n"
    "                self.structured_output_manager.validate_tokens_reasoning_aware(\n"
    "                    request, spec_token_ids\n"
    "                )\n"
    "            )"
)


def _make_scheduler_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P62 scheduler.py",
        target_file=str(target),
        marker=GENESIS_P62_MARKER + " :: scheduler.py",
        sub_patches=[
            # Sub-patch 2a includes both the should_advance replacement AND
            # the update_reasoning_ended call (formerly sub-patch 2b).
            TextPatch(name="p62_sched_update_from_output", anchor=SCHED_UPDATE_FROM_OUTPUT_OLD,
                      replacement=SCHED_UPDATE_FROM_OUTPUT_NEW, required=True),
            TextPatch(name="p62_sched_udti", anchor=SCHED_UDTI_OLD,
                      replacement=SCHED_UDTI_NEW, required=True),
            TextPatch(name="p62_sched_udtio", anchor=SCHED_UDTIO_OLD,
                      replacement=SCHED_UDTIO_NEW, required=True),
        ],
        upstream_drift_markers=[
            "validate_tokens_reasoning_aware",  # upstream-merged version present
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P62")
    log_decision("P62", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = [_make_struct_out_patcher(), _make_scheduler_patcher()]
    if any(p is None for p in patchers):
        return "skipped", "target file not found"

    # Pre-flight all anchors before writing
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"
        with open(p.target_file) as f:
            content = f.read()
        if p.marker in content:
            continue
        for m in p.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {p.target_file} — "
                    "vllm#36138 likely already merged.",
                )
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — anchor drifted, P62 cannot apply.",
                )

    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'}"
            )
        results.append((p.patch_name, result))

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)
    skipped = sum(1 for _, r in results if r == TextPatchResult.SKIPPED)

    if skipped > 0:
        return "skipped", (
            f"{skipped} of 2 patchers skipped — anchor drift. "
            f"{applied} applied + {idempotent} idempotent."
        )

    return "applied", (
        f"P62 applied: {applied} files modified, {idempotent} idempotent. "
        "Reasoning-aware grammar acceptance + spec-token validation now active. "
        "Should reduce residual broken tool-call rate when </think> arrives "
        "in spec batch."
    )
