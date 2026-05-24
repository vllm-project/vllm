# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN58 — vllm#40962 backport: spec-decode reasoning boundary validation.

Backport of upstream PR #40962 (OPEN, AI-assisted by author). Narrower
opt-in safety net for spec-decode + reasoning + structured output.

## Engineering rationale (read before enabling)

P62 (vllm#36138 sfbemerk) — наш existing **broader** pipeline-level fix.
Modifies `update_from_output()`, `update_draft_token_ids()`,
`update_draft_token_ids_in_output()`, `grammar_bitmask()`. Splits draft
batches unconstrained/constrained. Per-position bitmasks. Default OFF
in registry but **ENABLED in our PROD scripts** (`GENESIS_ENABLE_P62=1`).

PN58 (vllm#40962) — alternative **narrower** fix. Modifies ONLY scheduler
commit-time validation. Doesn't touch bitmask generation. **Author
warns**: "significant performance drop" with custom multi-token
reasoning markers (because per-token boundary scan is expensive).

### Why both exist as separate patches

PR #40962 author wrote: "This PR intentionally takes a narrower approach.
Instead of changing draft-token validation and bitmask generation, it
handles the issue at the accepted-token commit point." Two distinct
engineering tradeoffs:
- **P62 (broader)**: more correct (per-position grammar masks), more
  invasive, slightly slower hot-path (more validation calls per step)
- **PN58 (narrower)**: less correct in edge cases (only commit-time check,
  may miss some reasoning-boundary tokens that should have been rejected
  at draft-time), but cheaper hot-path

### Mutual exclusion (REQUIRED)

Both patches modify `if new_token_ids and ... should_advance(request):`
block in `scheduler.py`. They CANNOT coexist textually. Apply check
enforces: PN58 SKIPS if P62 active.

Genesis recommended:
- DEFAULT: P62 ON (broader safety, our PROD validated)
- ALTERNATIVE: P62 OFF + PN58 ON (if you measure perf hit from P62
  on YOUR specific reasoning parser)
- BOTH OFF: upstream baseline (broken on certain spec+reasoning combos)

### What if upstream merges either?

When upstream merges #36138 (P62 source) → P62 detects drift and SKIPS.
When upstream merges #40962 (PN58 source) → PN58 detects drift and SKIPS.
When BOTH merge: maintainers will pick winner; whichever wins, the other
backport SKIPS. Self-healing.

## Architecture (5 sub-patches across 5 files)

1. `envs.py` — register VLLM_SPEC_REASONING_BOUNDARY_VALIDATION
2. `reasoning/abs_reasoning_parsers.py` — base class methods
3. `reasoning/basic_parsers.py` — efficient single-token override
4. `v1/structured_output/__init__.py` — validation helper function
5. `v1/core/sched/scheduler.py` — import + scheduler init flag +
   validation block in update_from_output

Default OFF. Requires:
- GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY=1
- GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=0 (mutual exclusion)
- VLLM_SPEC_REASONING_BOUNDARY_VALIDATION=1 (upstream native flag)

Author: Sandermage backport (ToastyTheBot/Claude-assisted, vllm#40962).
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

log = logging.getLogger("genesis.wiring.pn58_spec_reasoning_boundary")

GENESIS_PN58_MARKER = "Genesis PN58 spec-decode reasoning boundary (vllm#40962)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def _is_p62_active() -> bool:
    """P62 (vllm#36138 broader) — mutually exclusive with PN58."""
    return os.environ.get(
        "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-A: envs.py — register VLLM_SPEC_REASONING_BOUNDARY_VALIDATION ───────
ENVS_OLD = (
    "    # Whether to enable dual cuda streams for LoRA computation\n"
    "    \"VLLM_LORA_ENABLE_DUAL_STREAM\": lambda: bool(\n"
    "        int(os.getenv(\"VLLM_LORA_ENABLE_DUAL_STREAM\", \"0\"))\n"
    "    ),\n"
    "}"
)
ENVS_NEW = (
    "    # Whether to enable dual cuda streams for LoRA computation\n"
    "    \"VLLM_LORA_ENABLE_DUAL_STREAM\": lambda: bool(\n"
    "        int(os.getenv(\"VLLM_LORA_ENABLE_DUAL_STREAM\", \"0\"))\n"
    "    ),\n"
    "    # [Genesis PN58 vllm#40962] Reasoning-boundary validation for accepted\n"
    "    # speculative tokens. Opt-in to avoid regressions on parsers not yet\n"
    "    # adapted. Default OFF.\n"
    "    \"VLLM_SPEC_REASONING_BOUNDARY_VALIDATION\": lambda: bool(\n"
    "        int(os.getenv(\"VLLM_SPEC_REASONING_BOUNDARY_VALIDATION\", \"0\"))\n"
    "    ),\n"
    "}"
)


# ─── Sub-B: abs_reasoning_parsers.py — add 2 base methods ───────────────────
ABS_PARSER_OLD = (
    "        return self.is_reasoning_end(input_ids)\n"
    "\n"
    "    @abstractmethod\n"
    "    def extract_content_ids(self, input_ids: list[int]) -> list[int]:"
)
ABS_PARSER_NEW = (
    "        return self.is_reasoning_end(input_ids)\n"
    "\n"
    "    # [Genesis PN58 vllm#40962] reasoning-boundary detection in spec tokens\n"
    "    def find_reasoning_end_index(\n"
    "        self, prefix_ids, delta_ids,\n"
    "    ):\n"
    "        \"\"\"Find where reasoning ends inside streaming token delta.\n"
    "\n"
    "        Returns the index in delta_ids where reasoning-end marker completes,\n"
    "        or None if marker doesn't complete inside delta_ids.\n"
    "        \"\"\"\n"
    "        current_input_ids = list(prefix_ids)\n"
    "        for end_index, token_id in enumerate(delta_ids):\n"
    "            current_input_ids.append(token_id)\n"
    "            if self.is_reasoning_end_streaming(current_input_ids, (token_id,)):\n"
    "                return end_index\n"
    "        return None\n"
    "\n"
    "    def may_have_reasoning_end_in_delta(self, delta_ids) -> bool:\n"
    "        \"\"\"Cheap precheck. Default conservative for multi-token markers.\"\"\"\n"
    "        return bool(delta_ids)\n"
    "\n"
    "    @abstractmethod\n"
    "    def extract_content_ids(self, input_ids: list[int]) -> list[int]:"
)


# ─── Sub-C: basic_parsers.py — efficient single-token overrides ─────────────
BASIC_PARSER_OLD = (
    "    def is_reasoning_end_streaming(\n"
    "        self, input_ids: Sequence[int], delta_ids: Iterable[int]\n"
    "    ) -> bool:\n"
    "        end_token_id = self.end_token_id\n"
    "        return end_token_id in delta_ids\n"
    "\n"
    "    def extract_content_ids(self, input_ids: list[int]) -> list[int]:"
)
BASIC_PARSER_NEW = (
    "    def is_reasoning_end_streaming(\n"
    "        self, input_ids: Sequence[int], delta_ids: Iterable[int]\n"
    "    ) -> bool:\n"
    "        end_token_id = self.end_token_id\n"
    "        return end_token_id in delta_ids\n"
    "\n"
    "    # [Genesis PN58 vllm#40962] efficient overrides for single-token markers\n"
    "    def find_reasoning_end_index(self, prefix_ids, delta_ids):\n"
    "        end_token_id = self.end_token_id\n"
    "        try:\n"
    "            return delta_ids.index(end_token_id)\n"
    "        except ValueError:\n"
    "            return None\n"
    "\n"
    "    def may_have_reasoning_end_in_delta(self, delta_ids) -> bool:\n"
    "        return self.end_token_id in delta_ids\n"
    "\n"
    "    def extract_content_ids(self, input_ids: list[int]) -> list[int]:"
)


# ─── Sub-D: structured_output/__init__.py — add validation helper ───────────
STRUCT_OUT_OLD = (
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "class StructuredOutputManager:"
)
STRUCT_OUT_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "# [Genesis PN58 vllm#40962] reasoning-boundary validation for spec tokens\n"
    "def validate_spec_tokens_with_reasoning_boundary(\n"
    "    request,\n"
    "    token_ids,\n"
    "    reasoner,\n"
    "):\n"
    "    \"\"\"Validate accepted spec tokens across reasoning boundary.\n"
    "\n"
    "    Reasoning tokens unconstrained. Once reasoning-end marker accepted,\n"
    "    only post-boundary answer suffix is grammar-validated and committed.\n"
    "    \"\"\"\n"
    "    structured_req = request.structured_output_request\n"
    "    assert token_ids\n"
    "    assert request.use_structured_output\n"
    "    assert structured_req is not None\n"
    "    assert structured_req.grammar is not None\n"
    "    assert structured_req.reasoning_ended is False\n"
    "    grammar = structured_req.grammar\n"
    "\n"
    "    boundary_end = reasoner.find_reasoning_end_index(\n"
    "        request.all_token_ids, token_ids,\n"
    "    )\n"
    "    if boundary_end is None:\n"
    "        return token_ids\n"
    "\n"
    "    keep = token_ids[: boundary_end + 1]\n"
    "    suffix = token_ids[boundary_end + 1:]\n"
    "    structured_req.reasoning_ended = True\n"
    "\n"
    "    if not suffix:\n"
    "        return keep\n"
    "\n"
    "    valid_suffix = grammar.validate_tokens(suffix)\n"
    "    if valid_suffix:\n"
    "        grammar.accept_tokens(request.request_id, valid_suffix)\n"
    "    return keep + valid_suffix\n"
    "\n"
    "\n"
    "class StructuredOutputManager:"
)


# ─── Sub-E: scheduler.py — multi-anchor (import + should_advance replace) ───
SCHED_IMPORT_OLD = (
    "from vllm.v1.structured_output import StructuredOutputManager"
)
SCHED_IMPORT_NEW = (
    "from vllm.v1.structured_output import (\n"
    "    StructuredOutputManager,\n"
    "    # [Genesis PN58 vllm#40962]\n"
    "    validate_spec_tokens_with_reasoning_boundary,\n"
    ")"
)

# Sub-E2: replace `if new_token_ids and ... should_advance(request):` block.
# This is the SAME block P62 replaces — mutual exclusion enforced by
# apply check above (P62 OFF required).
SCHED_VALIDATE_OLD = (
    "            if new_token_ids and self.structured_output_manager.should_advance(request):\n"
    "                struct_output_request = request.structured_output_request\n"
    "                assert struct_output_request is not None\n"
    "                assert struct_output_request.grammar is not None\n"
    "                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]\n"
    "                    req_id, new_token_ids\n"
    "                ):"
)

SCHED_VALIDATE_NEW = (
    "            # [Genesis PN58 vllm#40962] Reasoning-boundary validation for spec tokens.\n"
    "            # When accepted spec tokens cross </think>, post-boundary suffix\n"
    "            # is grammar-validated BEFORE commit. Narrower than P62 broader fix.\n"
    "            _pn58_advanced_with_boundary = False\n"
    "            _pn58_struct_req = request.structured_output_request\n"
    "            _pn58_validate = (\n"
    "                new_token_ids\n"
    "                and request.use_structured_output\n"
    "                and _pn58_struct_req is not None\n"
    "                and self.structured_output_manager.reasoner is not None\n"
    "                and not getattr(self.structured_output_manager, \"enable_in_reasoning\", False)\n"
    "                and getattr(_pn58_struct_req, \"reasoning_ended\", False) is False\n"
    "                # [Genesis PN58 audit A-01 fix] use vllm.envs (already imported as `envs` in scheduler)\n"
    "                # instead of os.environ.get → no extra `import os` needed.\n"
    "                and bool(getattr(envs, \"VLLM_SPEC_REASONING_BOUNDARY_VALIDATION\", False))\n"
    "            )\n"
    "            if _pn58_validate:\n"
    "                _pn58_reasoner = self.structured_output_manager.reasoner\n"
    "                if _pn58_reasoner.may_have_reasoning_end_in_delta(new_token_ids):\n"
    "                    _pn58_n = len(new_token_ids)\n"
    "                    new_token_ids = validate_spec_tokens_with_reasoning_boundary(\n"
    "                        request, new_token_ids, _pn58_reasoner,\n"
    "                    )\n"
    "                    _pn58_advanced_with_boundary = (\n"
    "                        getattr(_pn58_struct_req, \"reasoning_ended\", False) is True\n"
    "                    )\n"
    "                    _pn58_rejected = _pn58_n - len(new_token_ids)\n"
    "                    if _pn58_rejected:\n"
    "                        if request.num_computed_tokens > 0:\n"
    "                            request.num_computed_tokens -= _pn58_rejected\n"
    "                        if request.num_output_placeholders > 0:\n"
    "                            request.num_output_placeholders -= _pn58_rejected\n"
    "\n"
    "            if (\n"
    "                new_token_ids\n"
    "                and not _pn58_advanced_with_boundary\n"
    "                and self.structured_output_manager.should_advance(request)\n"
    "            ):\n"
    "                struct_output_request = request.structured_output_request\n"
    "                assert struct_output_request is not None\n"
    "                assert struct_output_request.grammar is not None\n"
    "                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]\n"
    "                    req_id, new_token_ids\n"
    "                ):"
)


def _make_envs_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("envs.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN58 envs.py",
        target_file=str(target),
        marker=GENESIS_PN58_MARKER + " (envs)",
        sub_patches=[TextPatch(name="pn58_envs", anchor=ENVS_OLD,
                                replacement=ENVS_NEW, required=True)],
    )


def _make_abs_parser_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/abs_reasoning_parsers.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN58 abs_reasoning_parsers.py",
        target_file=str(target),
        marker=GENESIS_PN58_MARKER + " (abs)",
        sub_patches=[TextPatch(name="pn58_abs_methods", anchor=ABS_PARSER_OLD,
                                replacement=ABS_PARSER_NEW, required=True)],
    )


def _make_basic_parser_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("reasoning/basic_parsers.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN58 basic_parsers.py",
        target_file=str(target),
        marker=GENESIS_PN58_MARKER + " (basic)",
        sub_patches=[TextPatch(name="pn58_basic_override", anchor=BASIC_PARSER_OLD,
                                replacement=BASIC_PARSER_NEW, required=True)],
    )


def _make_struct_out_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/structured_output/__init__.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN58 structured_output/__init__.py",
        target_file=str(target),
        marker=GENESIS_PN58_MARKER + " (struct_out)",
        sub_patches=[TextPatch(name="pn58_validate_helper", anchor=STRUCT_OUT_OLD,
                                replacement=STRUCT_OUT_NEW, required=True)],
    )


def _make_sched_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN58 scheduler.py (import + validate block)",
        target_file=str(target),
        marker=GENESIS_PN58_MARKER + " (sched)",
        sub_patches=[
            TextPatch(name="pn58_sched_import", anchor=SCHED_IMPORT_OLD,
                      replacement=SCHED_IMPORT_NEW, required=True),
            TextPatch(name="pn58_sched_validate", anchor=SCHED_VALIDATE_OLD,
                      replacement=SCHED_VALIDATE_NEW, required=True),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN58")
    log_decision("PN58", decision, reason)
    if not decision:
        return "skipped", reason

    # Mutual exclusion gate — P62 patches same scheduler block.
    if _is_p62_active():
        return "skipped", (
            "PN58 SKIPPED — P62 (vllm#36138 broader equivalent) is active. "
            "Mutually exclusive: both patch the same `should_advance` block. "
            "To use PN58 narrower variant: set "
            "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=0 first. See PN58 "
            "docstring for tradeoff details."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # [Audit A-04/A-05 fix 2026-05-05] Use MultiFilePatchTransaction —
    # validate-all-then-write-all atomicity across 5 files (envs, abs_parser,
    # basic_parser, struct_out, scheduler). Phase 1 dry-run prevents
    # partial state if P62 left scheduler.py modified.
    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    patchers = [
        _make_envs_patcher(),
        _make_abs_parser_patcher(),
        _make_basic_parser_patcher(),
        _make_struct_out_patcher(),
        _make_sched_patcher(),
    ]
    txn = MultiFilePatchTransaction(patchers, name="PN58")
    status, reason = txn.apply_or_skip()
    if status == "applied":
        return (
            "applied",
            f"{reason}. Runtime requires VLLM_SPEC_REASONING_BOUNDARY_VALIDATION=1.",
        )
    return status, reason
