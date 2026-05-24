# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 60 — GDN + ngram speculative decode state recovery.

Backport of vllm-project/vllm#40738 (Thomas Parnell, OPEN at time of writing).

================================================================
TOP CANDIDATE for #40831 / our degenerate-output bug after empirical
disproof of P58 (#40768 backport) and P59 (#39055 backport).
================================================================

Empirical isolation 2026-04-25 (blue/green tests on Genesis pin):

    Config                              Result
    -----------------------------       --------------------------
    spec=ngram + tools                  4-5/10 broken (degenerate)
    spec=ngram + tools + P58            4/5  broken
    spec=ngram + tools + P59            3/5  broken
    spec=ngram + tools + cudagraph=NONE 3/5  broken
    spec=ngram_gpu + P58 (Path B)       worse (5/5 broken)
    NO spec-decode + tools              5/5  CLEAN ✅
    spec=ngram + NO tools               5/5  CLEAN

Bug requires both spec-decode AND structured-output (tools). All four
P58/P59/cudagraph workarounds disproven empirically.

What PR #40738 fixes
--------------------
On hybrid GDN models (Qwen3.5/3.6 with linear-attention layers + ngram
speculative decode), SSM state and conv state were read from incorrect
locations after a spec decode step accepted multiple tokens:

  1. Spec decode step proposes k draft tokens, writes per-token SSM
     states to blocks 0..k, conv history into extended buffer in block 0.
  2. Scorer determines num_accepted_tokens (e.g., accepts 5 of 6).
  3. Next non-spec decode step must:
       - SSM state: read initial state from block[num_accepted-1], NOT block[0]
       - Conv state: read history from offset position within extended buffer
  4. Two bugs prevented this:
       a. num_accepted_tokens not passed on non-spec steps (gpu_model_runner.py)
       b. Conv state read from wrong offset (causal_conv1d.py Triton kernel)

This patch (P60 Phase 1) backports the PYTHON-ONLY portions:
  - File 1 gdn_attn.py: add spec_decode_src_indices field + build() logic
  - File 2 gdn_linear_attn.py: SSM state pre-copy in _forward_core +
    _forward_core_decode_non_spec
  - File 3 gpu_model_runner.py: passthrough num_accepted_tokens on
    non-spec steps when speculative_config is set

Phase 2 (Triton kernel patch in causal_conv1d.py) is DEFERRED. PR author
notes the kernel fix is necessary for full correctness ("Verified kernel
fix is necessary: disabling conv offset while keeping SSM pre-copy
causes Prompt 0 to fail"). Phase 1 may partially relieve the bug; if
empirical test shows 5/5+ clean, Phase 2 not needed; if still some
broken, Phase 2 (Triton signature change) is the next step.

Status: opt-in (`GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1`).

Compatibility
-------------
- All-or-nothing: 3 sub-patchers, all required. If any anchor drifts,
  whole group skips cleanly (no half-applied state).
- Idempotent (marker check).
- Auto-no-op once #40738 lands upstream (drift marker:
  `spec_decode_src_indices` in gdn_attn.py).

Risks acknowledged
------------------
- Without Phase 2 Triton fix, conv state may still be read from wrong
  position. Symptom: SOME spec-decode batches still produce token
  corruption (less than current rate but not zero).
- `extra_attn_metadata_args` dict construction in gpu_model_runner.py
  is in a hot-path (every step, every layer). The new branch adds one
  isinstance() check per non-spec step; expected overhead < 0.1%.

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

log = logging.getLogger("genesis.wiring.p60_gdn_ngram_state_recovery")

GENESIS_P60_MARKER = "Genesis P60 GDN+ngram state recovery v7.13"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── File 1 anchors: vllm/v1/attention/backends/gdn_attn.py ────────────────

# Sub-patch 1a: add spec_decode_src_indices to GDNAttentionMetadata
GDN_ATTN_FIELD_OLD = (
    "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n"
    "\n"
    "    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)"
)

GDN_ATTN_FIELD_NEW = (
    "    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]\n"
    "\n"
    "    # [Genesis P60 vllm#40738] 1D source block indices for state recovery\n"
    "    # after spec decode. When set, conv/ssm state must be copied from\n"
    "    # these blocks to non_spec_state_indices_tensor before decode kernel.\n"
    "    spec_decode_src_indices: torch.Tensor | None = None\n"
    "\n"
    "    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)"
)

# Sub-patch 1b: modify build() non-spec branch
GDN_ATTN_BUILD_OLD = (
    "        if spec_sequence_masks is None:\n"
    "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n"
    "                split_decodes_and_prefills(m, decode_threshold=1)\n"
    "            )\n"
    "            num_spec_decode_tokens = 0\n"
    "            spec_token_indx = None\n"
    "            non_spec_token_indx = None\n"
    "            spec_state_indices_tensor = None\n"
    "            non_spec_state_indices_tensor = block_table_tensor[:, 0]\n"
    "            spec_query_start_loc = None\n"
    "            non_spec_query_start_loc = query_start_loc\n"
    "            non_spec_query_start_loc_cpu = query_start_loc_cpu\n"
    "            num_accepted_tokens = None\n"
    "        else:"
)

GDN_ATTN_BUILD_NEW = (
    "        # [Genesis P60 vllm#40738] init recovery indices\n"
    "        spec_decode_src_indices = None\n"
    "        if spec_sequence_masks is None:\n"
    "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n"
    "                split_decodes_and_prefills(m, decode_threshold=1)\n"
    "            )\n"
    "            num_spec_decode_tokens = 0\n"
    "            spec_token_indx = None\n"
    "            non_spec_token_indx = None\n"
    "            spec_state_indices_tensor = None\n"
    "            non_spec_state_indices_tensor = block_table_tensor[:, 0]\n"
    "            spec_query_start_loc = None\n"
    "            non_spec_query_start_loc = query_start_loc\n"
    "            non_spec_query_start_loc_cpu = query_start_loc_cpu\n"
    "            # [Genesis P60 vllm#40738] gate num_accepted_tokens on spec activity\n"
    "            if (\n"
    "                self.use_spec_decode\n"
    "                and num_accepted_tokens is not None\n"
    "                and num_decodes > 0\n"
    "            ):\n"
    "                col_indices = (num_accepted_tokens[:num_decodes] - 1).clamp(min=0)\n"
    "                spec_decode_src_indices = block_table_tensor[\n"
    "                    torch.arange(num_decodes, device=block_table_tensor.device),\n"
    "                    col_indices,\n"
    "                ]\n"
    "                num_accepted_tokens = num_accepted_tokens[:num_decodes]\n"
    "                if num_prefills > 0:\n"
    "                    num_accepted_tokens = torch.cat(\n"
    "                        [\n"
    "                            num_accepted_tokens,\n"
    "                            torch.ones(\n"
    "                                num_prefills,\n"
    "                                dtype=num_accepted_tokens.dtype,\n"
    "                                device=num_accepted_tokens.device,\n"
    "                            ),\n"
    "                        ]\n"
    "                    )\n"
    "            else:\n"
    "                num_accepted_tokens = None\n"
    "        else:"
)

# Sub-patch 1c: include spec_decode_src_indices in metadata constructor
GDN_ATTN_CTOR_OLD = (
    "            num_accepted_tokens=num_accepted_tokens,\n"
    "            nums_dict=nums_dict,"
)

GDN_ATTN_CTOR_NEW = (
    "            num_accepted_tokens=num_accepted_tokens,\n"
    "            spec_decode_src_indices=spec_decode_src_indices,\n"
    "            nums_dict=nums_dict,"
)


def _make_gdn_attn_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/gdn_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P60 gdn_attn.py — spec_decode_src_indices",
        target_file=str(target),
        marker=GENESIS_P60_MARKER + " :: gdn_attn.py",
        sub_patches=[
            TextPatch(name="p60_field", anchor=GDN_ATTN_FIELD_OLD,
                      replacement=GDN_ATTN_FIELD_NEW, required=True),
            TextPatch(name="p60_build", anchor=GDN_ATTN_BUILD_OLD,
                      replacement=GDN_ATTN_BUILD_NEW, required=True),
            TextPatch(name="p60_ctor", anchor=GDN_ATTN_CTOR_OLD,
                      replacement=GDN_ATTN_CTOR_NEW, required=True),
        ],
        # Drift detection via a distinctive UPSTREAM-only phrasing. PR #40738
        # uses "1D source block indices for state recovery" as the dataclass
        # field comment; we use "[Genesis P60 ...]" prefix. This way our
        # P60b's reference to `spec_decode_src_indices` doesn't false-trigger.
        upstream_drift_markers=[
            "1D source block indices for state recovery after spec decode"
        ],
    )


# ─── File 2 anchors: vllm/model_executor/layers/mamba/gdn_linear_attn.py ───

# Sub-patch 2a: SSM pre-copy in _forward_core
GDN_LINATTN_CORE_OLD = (
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n"
    "        b = b[:num_actual_tokens]\n"
    "        a = a[:num_actual_tokens]\n"
    "\n"
    "        # 1. Convolution sequence transformation"
)

GDN_LINATTN_CORE_NEW = (
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "\n"
    "        # [Genesis P60 vllm#40738] SSM state pre-copy from accepted block\n"
    "        spec_decode_src_indices = getattr(\n"
    "            attn_metadata, 'spec_decode_src_indices', None\n"
    "        )\n"
    "        if spec_decode_src_indices is not None:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            n_correct = spec_decode_src_indices.shape[0]\n"
    "            dst_indices = non_spec_state_indices_tensor[:n_correct]\n"
    "            ssm_state[dst_indices] = ssm_state[spec_decode_src_indices]\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n"
    "        b = b[:num_actual_tokens]\n"
    "        a = a[:num_actual_tokens]\n"
    "\n"
    "        # 1. Convolution sequence transformation"
)

# Sub-patch 2b: SSM pre-copy in _forward_core_decode_non_spec
GDN_LINATTN_DEC_OLD = (
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n"
    "        b = b[:num_actual_tokens]\n"
    "        a = a[:num_actual_tokens]\n"
    "\n"
    "        conv_weights = self.conv1d.weight.view(\n"
    "            self.conv1d.weight.size(0), self.conv1d.weight.size(2)\n"
    "        )\n"
    "        mixed_qkv_non_spec = causal_conv1d_update("
)

GDN_LINATTN_DEC_NEW = (
    "        ssm_state = self_kv_cache[1]\n"
    "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
    "\n"
    "        # [Genesis P60 vllm#40738] SSM state pre-copy from accepted block\n"
    "        spec_decode_src_indices = getattr(\n"
    "            attn_metadata, 'spec_decode_src_indices', None\n"
    "        )\n"
    "        num_accepted_tokens = attn_metadata.num_accepted_tokens\n"
    "        if spec_decode_src_indices is not None:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            n_correct = spec_decode_src_indices.shape[0]\n"
    "            dst_indices = non_spec_state_indices_tensor[:n_correct]\n"
    "            ssm_state[dst_indices] = ssm_state[spec_decode_src_indices]\n"
    "\n"
    "        mixed_qkv = mixed_qkv[:num_actual_tokens]\n"
    "        b = b[:num_actual_tokens]\n"
    "        a = a[:num_actual_tokens]\n"
    "\n"
    "        conv_weights = self.conv1d.weight.view(\n"
    "            self.conv1d.weight.size(0), self.conv1d.weight.size(2)\n"
    "        )\n"
    "        mixed_qkv_non_spec = causal_conv1d_update("
)


def _make_gdn_linattn_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P60 gdn_linear_attn.py — SSM state pre-copy",
        target_file=str(target),
        marker=GENESIS_P60_MARKER + " :: gdn_linear_attn.py",
        sub_patches=[
            TextPatch(name="p60_core", anchor=GDN_LINATTN_CORE_OLD,
                      replacement=GDN_LINATTN_CORE_NEW, required=True),
            TextPatch(name="p60_decode", anchor=GDN_LINATTN_DEC_OLD,
                      replacement=GDN_LINATTN_DEC_NEW, required=True),
        ],
        # Drift detection via a distinctive UPSTREAM-only phrasing. PR #40738
        # uses "1D source block indices for state recovery" as the dataclass
        # field comment; we use "[Genesis P60 ...]" prefix. This way our
        # P60b's reference to `spec_decode_src_indices` doesn't false-trigger.
        upstream_drift_markers=[
            "1D source block indices for state recovery after spec decode"
        ],
    )


# ─── File 3 anchors: vllm/v1/worker/gpu_model_runner.py ────────────────────

GMR_OLD = (
    "            extra_attn_metadata_args = {}\n"
    "            if use_spec_decode and isinstance(\n"
    "                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)\n"
    "            ):\n"
    "                assert ubid is None, \"UBatching not supported with GDN yet\"\n"
    "                extra_attn_metadata_args = dict(\n"
    "                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],\n"
    "                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[\n"
    "                        :num_reqs_padded\n"
    "                    ],\n"
    "                )"
)

GMR_NEW = (
    "            extra_attn_metadata_args = {}\n"
    "            if use_spec_decode and isinstance(\n"
    "                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)\n"
    "            ):\n"
    "                assert ubid is None, \"UBatching not supported with GDN yet\"\n"
    "                extra_attn_metadata_args = dict(\n"
    "                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],\n"
    "                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[\n"
    "                        :num_reqs_padded\n"
    "                    ],\n"
    "                )\n"
    "            elif (\n"
    "                # [Genesis P60 vllm#40738] non-spec steps with spec_config: pass num_accepted\n"
    "                not use_spec_decode\n"
    "                and self.speculative_config is not None\n"
    "                and isinstance(\n"
    "                    builder,\n"
    "                    (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),\n"
    "                )\n"
    "            ):\n"
    "                extra_attn_metadata_args = dict(\n"
    "                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],\n"
    "                )"
)


def _make_gmr_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P60 gpu_model_runner.py — non-spec passthrough",
        target_file=str(target),
        marker=GENESIS_P60_MARKER + " :: gpu_model_runner.py",
        sub_patches=[
            TextPatch(name="p60_passthrough", anchor=GMR_OLD,
                      replacement=GMR_NEW, required=True),
        ],
        upstream_drift_markers=["[Genesis P60 vllm#40738] non-spec steps"],
    )


def apply() -> tuple[str, str]:
    """Apply P60 Phase 1 (Python-only) backport. All-or-nothing."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P60")
    log_decision("P60", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = [
        _make_gdn_attn_patcher(),
        _make_gdn_linattn_patcher(),
        _make_gmr_patcher(),
    ]
    if any(p is None for p in patchers):
        return "skipped", "one or more target files not found"

    # Pre-flight: confirm anchors before write to avoid half-applied state.
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target disappeared: {p.target_file}"
        with open(p.target_file) as f:
            content = f.read()
        if p.marker in content:
            continue  # idempotent
        for m in p.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {p.target_file} — "
                    "vllm#40738 likely already merged or backported.",
                )
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — anchor drifted, P60 cannot apply.",
                )

    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''}) — partial state risk; "
                "container should be torn down (compose down + up -d)."
            )
        results.append((p.patch_name, result))

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)
    skipped = sum(1 for _, r in results if r == TextPatchResult.SKIPPED)

    if skipped > 0:
        return "skipped", (
            f"{skipped} of 3 sub-patchers skipped — anchor drift on some "
            f"files. {applied} applied + {idempotent} idempotent."
        )

    return "applied", (
        f"P60 Phase 1 applied: {applied} files modified, {idempotent} "
        "idempotent. SSM state pre-copy active for GDN+ngram spec decode. "
        "Conv state Triton kernel fix (Phase 2) NOT included — if "
        "reproducer still shows broken output, Phase 2 is the next step."
    )
