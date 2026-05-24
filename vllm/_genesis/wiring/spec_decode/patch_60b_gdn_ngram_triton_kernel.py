# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 60b — GDN+ngram Triton kernel offset (P60 Phase 2).

Backport of vllm-project/vllm#40738 (Thomas Parnell), Triton kernel portion.

================================================================
DEPENDS ON P60 (Phase 1). Apply P60 BEFORE P60b. Without Phase 1,
the SSM state pre-copy isn't done so the conv state offset alone
won't help.

P60 Phase 1 result on dev205: 13/30 clean (43%) vs baseline 6/30 (~20%)
PR author notes both Phase 1 + Phase 2 are needed for full correctness.
Expected combined: 28/30+ clean (~95%).
================================================================

What Phase 2 does
-----------------
Modifies `_causal_conv1d_fwd_kernel` Triton kernel to accept
`num_accepted_tokens_ptr` parameter and `IS_SPEC_DECODING` constexpr.
When active, computes `conv_state_token_offset = num_accepted - 1`
and applies it to:

  - STEP 1 (read prior history): `state_len-1+offset` instead of `state_len-1`
  - STEP 2 (write conv state):    `idx_tokens_conv + seqlen + offset` instead of
                                  `idx_tokens_conv + seqlen`

Also modifies `causal_conv1d_fn` Python wrapper to accept and pass through
`num_accepted_tokens` parameter.

And modifies `gdn_linear_attn.py` call sites to actually pass the parameter
(currently they don't even though the kernel parameter is there from P60
Phase 1).

Risk acknowledged
-----------------
1. **Triton signature change invalidates JIT cache.** Existing cached PTX
   for `_causal_conv1d_fwd_kernel` references the OLD signature. After
   patch applied, first call recompiles (~5-10s, profiler-visible spike).
   Mitigation: clear `/root/.triton/cache/` before boot if needed.

2. **One additional positional kernel arg.** `_causal_conv1d_fwd_kernel`
   only has ONE caller (`causal_conv1d_fn` itself), so we update both
   atomically. Verified via `grep -rn _causal_conv1d_fwd_kernel\\[`.

3. **Backward compat preserved at Python level.** New kwarg
   `num_accepted_tokens=None` defaults to no-op behavior. Callers that
   don't pass it continue working as before.

Status: opt-in via `GENESIS_ENABLE_P60B_TRITON_KERNEL=1`. Should be enabled
together with `GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1` (P60 Phase 1).

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

log = logging.getLogger("genesis.wiring.p60b_gdn_ngram_triton_kernel")

GENESIS_P60B_MARKER = "Genesis P60b GDN+ngram Triton kernel offset v7.13"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P60B_TRITON_KERNEL", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── File 1: causal_conv1d.py ──────────────────────────────────────────────

# Sub-patch 1a: Add num_accepted_tokens_ptr to kernel signature
KERNEL_SIG_OLD = (
    "    initial_state_idx,  # (batch,)\n"
    "    num_computed_tokens,  # (batch,)\n"
    "    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr"
)

KERNEL_SIG_NEW = (
    "    initial_state_idx,  # (batch,)\n"
    "    num_computed_tokens,  # (batch,)\n"
    "    num_accepted_tokens_ptr,  # (batch,) or None  [Genesis P60b vllm#40738]\n"
    "    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr"
)

# Sub-patch 1b: Add IS_SPEC_DECODING constexpr
KERNEL_CONSTEXPR_OLD = (
    "    IS_APC_ENABLED: tl.constexpr,\n"
    "    HAS_NULL_BLOCK: tl.constexpr,"
)

KERNEL_CONSTEXPR_NEW = (
    "    IS_APC_ENABLED: tl.constexpr,\n"
    "    IS_SPEC_DECODING: tl.constexpr,  # [Genesis P60b vllm#40738]\n"
    "    HAS_NULL_BLOCK: tl.constexpr,"
)

# Sub-patch 1c: Compute conv_state_token_offset at top of kernel body
KERNEL_OFFSET_OLD = (
    "    # single-sequence id\n"
    "    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)\n"
    "    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))"
)

KERNEL_OFFSET_NEW = (
    "    # single-sequence id\n"
    "    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)\n"
    "\n"
    "    # [Genesis P60b vllm#40738] Compute conv state token offset.\n"
    "    if IS_SPEC_DECODING:\n"
    "        conv_state_token_offset = (\n"
    "            tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1\n"
    "        )\n"
    "    else:\n"
    "        conv_state_token_offset = 0\n"
    "\n"
    "    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))"
)

# Sub-patch 1d: STEP 1 — apply offset to prior_tokens load
KERNEL_STEP1_OLD = (
    "        if load_init_state:\n"
    "            # load from conv_states\n"
    "            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok"
)

KERNEL_STEP1_NEW = (
    "        if load_init_state:\n"
    "            # load from conv_states  [Genesis P60b vllm#40738: + conv_state_token_offset]\n"
    "            prior_tokens = (\n"
    "                conv_states_base\n"
    "                + (state_len - 1 + conv_state_token_offset) * stride_conv_state_tok\n"
    "            )"
)

# Sub-patch 1e: STEP 2 — apply offset to conv_states write
KERNEL_STEP2_OLD = (
    "                conv_states_ptrs_source = (\n"
    "                    conv_states_ptr\n"
    "                    + (conv_states_input_coord * stride_conv_state_seq)\n"
    "                    + (idx_feats * stride_conv_state_dim)[None, :]\n"
    "                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]\n"
    "                )  # [BLOCK_M, BLOCK_N]"
)

KERNEL_STEP2_NEW = (
    "                conv_states_ptrs_source = (\n"
    "                    conv_states_ptr\n"
    "                    + (conv_states_input_coord * stride_conv_state_seq)\n"
    "                    + (idx_feats * stride_conv_state_dim)[None, :]\n"
    "                    # [Genesis P60b vllm#40738: + conv_state_token_offset]\n"
    "                    + (\n"
    "                        (idx_tokens_conv + seqlen + conv_state_token_offset)\n"
    "                        * stride_conv_state_tok\n"
    "                    )[:, None]\n"
    "                )  # [BLOCK_M, BLOCK_N]"
)

# Sub-patch 1f: Add num_accepted_tokens parameter to causal_conv1d_fn wrapper
WRAPPER_SIG_OLD = (
    "    block_idx_first_scheduled_token: torch.Tensor | None = None,\n"
    "    block_idx_last_scheduled_token: torch.Tensor | None = None,\n"
    "    initial_state_idx: torch.Tensor | None = None,\n"
    "    num_computed_tokens: torch.Tensor | None = None,\n"
    "    block_size_to_align=0,\n"
    "    metadata=None,\n"
    "    validate_data=False,\n"
    "):"
)

WRAPPER_SIG_NEW = (
    "    block_idx_first_scheduled_token: torch.Tensor | None = None,\n"
    "    block_idx_last_scheduled_token: torch.Tensor | None = None,\n"
    "    initial_state_idx: torch.Tensor | None = None,\n"
    "    num_computed_tokens: torch.Tensor | None = None,\n"
    "    num_accepted_tokens: torch.Tensor | None = None,  # [Genesis P60b vllm#40738]\n"
    "    block_size_to_align=0,\n"
    "    metadata=None,\n"
    "    validate_data=False,\n"
    "):"
)

# Sub-patch 1g: Pass num_accepted_tokens + IS_SPEC_DECODING into kernel call
KERNEL_CALL_OLD = (
    "        initial_state_idx,\n"
    "        num_computed_tokens,\n"
    "        out,"
)

KERNEL_CALL_NEW = (
    "        initial_state_idx,\n"
    "        num_computed_tokens,\n"
    "        num_accepted_tokens,  # [Genesis P60b vllm#40738]\n"
    "        out,"
)

# Sub-patch 1h: Add IS_SPEC_DECODING in kernel call kwargs
KERNEL_KWARG_OLD = (
    "        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,\n"
    "        HAS_NULL_BLOCK=null_block_id is not None,"
)

KERNEL_KWARG_NEW = (
    "        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,\n"
    "        IS_SPEC_DECODING=num_accepted_tokens is not None,  # [Genesis P60b vllm#40738]\n"
    "        HAS_NULL_BLOCK=null_block_id is not None,"
)


def _make_kernel_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/ops/causal_conv1d.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P60b causal_conv1d.py — Triton kernel + wrapper",
        target_file=str(target),
        marker=GENESIS_P60B_MARKER + " :: causal_conv1d.py",
        sub_patches=[
            TextPatch(name="p60b_kernel_sig", anchor=KERNEL_SIG_OLD,
                      replacement=KERNEL_SIG_NEW, required=True),
            TextPatch(name="p60b_constexpr", anchor=KERNEL_CONSTEXPR_OLD,
                      replacement=KERNEL_CONSTEXPR_NEW, required=True),
            TextPatch(name="p60b_offset_calc", anchor=KERNEL_OFFSET_OLD,
                      replacement=KERNEL_OFFSET_NEW, required=True),
            TextPatch(name="p60b_step1", anchor=KERNEL_STEP1_OLD,
                      replacement=KERNEL_STEP1_NEW, required=True),
            TextPatch(name="p60b_step2", anchor=KERNEL_STEP2_OLD,
                      replacement=KERNEL_STEP2_NEW, required=True),
            TextPatch(name="p60b_wrapper_sig", anchor=WRAPPER_SIG_OLD,
                      replacement=WRAPPER_SIG_NEW, required=True),
            TextPatch(name="p60b_kernel_call", anchor=KERNEL_CALL_OLD,
                      replacement=KERNEL_CALL_NEW, required=True),
            TextPatch(name="p60b_kernel_kwarg", anchor=KERNEL_KWARG_OLD,
                      replacement=KERNEL_KWARG_NEW, required=True),
        ],
        # Drift marker is the SPECIFIC line combination unique to fwd_kernel
        # post-PR. `IS_SPEC_DECODING` alone matches `_causal_conv1d_update_kernel`
        # too broadly (already in dev205). Use the new positional parameter
        # in fwd_kernel signature as the discriminator.
        upstream_drift_markers=[
            "num_computed_tokens,  # (batch,)\n    num_accepted_tokens_ptr,"
        ],
    )


# ─── File 2: gdn_linear_attn.py — pass num_accepted_tokens to call sites ──

# Sub-patch 2a: causal_conv1d_fn call (prefill path)
GDN_CONV_FN_OLD = (
    "            mixed_qkv_non_spec = causal_conv1d_fn(\n"
    "                mixed_qkv_non_spec_T,\n"
    "                conv_weights,\n"
    "                self.conv1d.bias,\n"
    "                activation=self.activation,\n"
    "                conv_states=conv_state,\n"
    "                has_initial_state=has_initial_state,\n"
    "                cache_indices=non_spec_state_indices_tensor,\n"
    "                query_start_loc=non_spec_query_start_loc,\n"
    "                metadata=attn_metadata,\n"
    "            ).transpose(0, 1)"
)

GDN_CONV_FN_NEW = (
    "            # [Genesis P60b vllm#40738] gate num_accepted_tokens on spec activity\n"
    "            _p60b_spec_src = getattr(\n"
    "                attn_metadata, 'spec_decode_src_indices', None\n"
    "            )\n"
    "            _p60b_conv_num_accepted = (\n"
    "                num_accepted_tokens if _p60b_spec_src is not None else None\n"
    "            )\n"
    "            mixed_qkv_non_spec = causal_conv1d_fn(\n"
    "                mixed_qkv_non_spec_T,\n"
    "                conv_weights,\n"
    "                self.conv1d.bias,\n"
    "                activation=self.activation,\n"
    "                conv_states=conv_state,\n"
    "                has_initial_state=has_initial_state,\n"
    "                cache_indices=non_spec_state_indices_tensor,\n"
    "                query_start_loc=non_spec_query_start_loc,\n"
    "                num_accepted_tokens=_p60b_conv_num_accepted,\n"
    "                metadata=attn_metadata,\n"
    "            ).transpose(0, 1)"
)


def _make_gdn_caller_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P60b gdn_linear_attn.py — pass num_accepted to conv_fn",
        target_file=str(target),
        marker=GENESIS_P60B_MARKER + " :: gdn_linear_attn.py",
        sub_patches=[
            TextPatch(name="p60b_gdn_conv_fn", anchor=GDN_CONV_FN_OLD,
                      replacement=GDN_CONV_FN_NEW, required=True),
        ],
        upstream_drift_markers=["_p60b_conv_num_accepted"],
    )


def _clear_triton_cache() -> tuple[bool, str]:
    """Best-effort: clear Triton JIT cache to force kernel recompile.
    Returns (cleared, reason). Failure is non-fatal — kernel will still
    work, just may use stale cached PTX which expects OLD signature."""
    cache_dirs = [
        os.path.expanduser("~/.triton/cache"),
        "/root/.triton/cache",
        os.environ.get("TRITON_CACHE_DIR", ""),
    ]
    for d in cache_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            import shutil
            # Just clear the causal_conv1d-related cache subdirs
            removed = 0
            for entry in os.listdir(d):
                full = os.path.join(d, entry)
                if os.path.isdir(full) and "causal_conv1d" in entry.lower():
                    shutil.rmtree(full)
                    removed += 1
            if removed:
                return True, f"cleared {removed} causal_conv1d cache entries from {d}"
        except Exception as e:
            log.debug("[Genesis P60b] cache clear failed for %s: %s", d, e)
    return False, "no cache dir cleared (may be empty already or permission denied)"


def apply() -> tuple[str, str]:
    """Apply P60b (Phase 2) Triton kernel patch + gdn caller passthrough."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P60b")
    log_decision("P60b", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = [_make_kernel_patcher(), _make_gdn_caller_patcher()]
    if any(p is None for p in patchers):
        return "skipped", "target file not found"

    # Pre-flight
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
                    "vllm#40738 likely already merged.",
                )
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — anchor drifted.",
                )

    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'}"
            )
        results.append((p.patch_name, result))

    # Clear Triton cache to force kernel recompile with new signature
    _cleared, cache_msg = _clear_triton_cache()
    log.info("[Genesis P60b] Triton cache: %s", cache_msg)

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)
    skipped = sum(1 for _, r in results if r == TextPatchResult.SKIPPED)

    if skipped > 0:
        return "skipped", (
            f"{skipped} of 2 patchers skipped — anchor drift. "
            f"{applied} applied + {idempotent} idempotent."
        )

    return "applied", (
        f"P60b Phase 2 applied: {applied} files modified, {idempotent} "
        f"idempotent. Triton kernel offset active. {cache_msg} "
        "First spec-decode call will trigger kernel recompile (~5-10s)."
    )
