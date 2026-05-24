# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N32 v2 — GDN _forward_core chunked prefill (Cliff 2 fix).

================================================================
v7.69 REDESIGN (club-3090#19 finding 3, 2026-05-02)
================================================================

The v7.65 PN32 design chunked at the WRONG level. It patched the outer
`GatedDeltaNetAttention.forward_cuda` and sliced `mixed_qkv/b/a` before
calling `torch.ops.vllm.gdn_attention_core`. Inside `_forward_core`, the
chunked call to `self.chunk_gated_delta_rule(...)` still received
`attn_metadata.non_spec_query_start_loc` (FULL prompt cu_seqlens),
`chunk_indices`, and `chunk_offsets` describing the FULL prompt — so
the FLA kernel allocated `h = (B, NT_FULL, H, V, K)` regardless of the
chunked outer slice.

Cross-rig empirical evidence: noonghunna 2026-05-02 reported PN32 v1
OOMing EARLIER (30K) than baseline (50-60K) because the chunked outer
loop added control-flow overhead WITHOUT reducing the inner allocation.

================================================================
v7.69 v2 DESIGN
================================================================

Patch `_forward_core` directly. Wrap the prefill branch's
`self.chunk_gated_delta_rule(...)` call with a chunk-aware loop that:

1. Detects single-sequence prefill (`non_spec_query_start_loc.shape[0]
   == 2`, i.e. one [0, T] entry). Multi-sequence prefill bypasses to
   original (correct chunking would require slicing across sequence
   boundaries — out of scope).

2. For T > THRESHOLD (default 16384), splits into chunks of CHUNK_SIZE
   (default 8192). Per chunk:
   - Slice query/key/value/g/beta along T dim (dim=1, since shape is
     `(1, T, H, D)` after unsqueeze in the caller).
   - Build chunk-local `cu_seqlens=[0, chunk_len]` (1-tensor on the
     same device as query).
   - Pass `chunk_indices=None, chunk_offsets=None` — FLA recomputes
     internally from cu_seqlens.
   - Thread `initial_state`: first chunk uses computed initial_state;
     subsequent chunks use the prior chunk's `last_recurrent_state`.
   - Set `output_final_state=True` always (we need it for chaining;
     last chunk's final_state goes into ssm_state cache as before).

3. Concatenate chunk outputs along dim=1 → `core_attn_out_non_spec`
   matching the original full-shape output.

4. Persist `last_recurrent_state` (from final chunk) into `ssm_state`
   cache exactly as the original.

================================================================
COMPOSITION WITH P103 (RECOMMENDED — both default OFF)
================================================================

PN32 v2 chunks at the OUTER FLA boundary (call to
`chunk_gated_delta_rule`). P103 chunks INSIDE FLA's
`chunk_gated_delta_rule_fwd` (the kernel orchestrator), splitting the
inner `h = k.new_empty(B, NT, H, V, K)` allocation.

The two are COMPLEMENTARY, not redundant:

- **PN32 alone**: reduces the OUTPUT buffer size per call
  (`core_attn_out_non_spec`: 8K × 64 × 128 × 2 = 131 MiB per chunk vs
  full-prompt 819 MiB at 50K). Reduces transient peak inside one layer.

- **P103 alone**: reduces the INNER `h` tensor inside
  `chunk_gated_delta_rule_fwd` (the recurrent state buffer:
  200 MiB per FLA-internal sub-T at MAX_T=16K vs full 805 MiB at 64K).

- **PN32 + P103**: PN32 calls FLA with chunk-sized inputs, FLA's
  `chunk_gated_delta_rule_fwd` (now wrapped by P103) further splits if
  the chunk is still > P103's MAX_T. Best memory profile.

Recommended for single-24GB-GPU users hitting Cliff 2:

    GENESIS_ENABLE_P103=1                              # required
    GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1          # recommended
    GENESIS_PN32_GDN_CHUNK_SIZE=8192                   # default
    GENESIS_PN32_GDN_CHUNK_THRESHOLD=16384             # default
    GENESIS_FLA_FWD_H_MAX_T=16384                      # P103 default

================================================================
DEPENDENCIES
================================================================

- **Hard requirement**: NONE (PN32 v2 functions standalone — chunks
  the outer call independently).
- **Strong recommendation**: enable P103 simultaneously. Without P103,
  each PN32 chunk still allocates the inner `h` tensor at chunk-T
  size; for chunk_size=8K that's 200 MiB which is fine, but P103
  composes for additional safety on very long contexts (>200K).
- **Conflict**: P28 (legacy persistent buffer pool). P28 caches
  `core_attn_out` per shape; PN32 allocates per-chunk transient. Pick
  one — both modify overlapping code paths.

================================================================
THRESHOLD SEMANTICS
================================================================

`num_tokens` in `_forward_core` is `hidden_states.size(0)` — total
batched tokens across all sequences in this forward call (continuous
batching sum). PN32 v2 fires when:

  num_tokens > GENESIS_PN32_GDN_CHUNK_THRESHOLD (default 16384)
  AND single-sequence prefill (cu_seqlens shape == [2])
  AND prefill branch (attn_metadata.num_prefills > 0)

Multi-sequence prefill (continuous batching of N short prompts
totaling > THRESHOLD) bypasses to original — correct chunking across
sequence boundaries requires inner state-cache management not exposed
at this layer.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1`)
- Pure text-patch on `_forward_core` (idempotent via marker)
- Single-sequence prefill only — multi-sequence bypasses (no risk of
  wrong cross-sequence state mixing)
- Anchor matches the EXACT prefill `if attn_metadata.num_prefills > 0:`
  block from upstream (v0.20.1rc1.dev16+g7a1eb8ac2 reference)
- Drift-aware: if upstream rewrites _forward_core, anchor won't match
  → SKIPPED, source stays vanilla
- Numerical correctness: chained `last_recurrent_state` propagation
  preserves recurrent state across chunks (same mechanism FLA uses
  internally for chunk_indices/chunk_offsets). Validated by tiny-
  tensor unit test (synthetic dims, single-sequence comparison vs
  unchunked reference).

================================================================
HISTORY
================================================================

- v7.65 (2026-05-01): initial PN32 patching `forward_cuda` outer.
  Tested only on 2× A5000 PROD (TP=2, doesn't hit Cliff 2). Cross-rig
  validation by noonghunna on 1× 3090 (2026-05-02) revealed the
  metadata-mismatch bug: chunking outer didn't propagate to the inner
  FLA call's cu_seqlens/chunk_indices.

- v7.69 v2 (2026-05-02): rewritten to chunk `_forward_core` directly,
  with chunk-local cu_seqlens and threaded initial_state. Composes
  with P103 (P103 = inner FLA-kernel chunking; PN32 = outer
  FLA-call chunking). Documented dependencies + composition matrix.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (CLIFF2_INVESTIGATION_20260430.md +
                       club-3090#19 cross-rig finding 3, 2026-05-02).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN32_gdn_chunked_prefill")

GENESIS_PN32_MARKER = (
    "Genesis PN32 v2 GDN _forward_core chunked-prefill (Cliff 2 fix) v7.69"
)


# ─── Anchor: prefill branch in _forward_core (v0.20.1rc1.dev16) ─────
# Matches the EXACT 27-line block of the prefill `if
# attn_metadata.num_prefills > 0:` branch from
# `model_executor/layers/mamba/gdn_linear_attn.py:_forward_core`.

PN32_ANCHOR = (
    "        # 2.2: Process the remaining part\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]\n"
    "            assert has_initial_state is not None\n"
    "            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = self.chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=non_spec_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "            # Init cache\n"
    "            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(\n"
    "                ssm_state.dtype\n"
    "            )\n"
)

PN32_REPLACEMENT = (
    "        # 2.2: Process the remaining part\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]\n"
    "            assert has_initial_state is not None\n"
    "            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]\n"
    "\n"
    "            # [Genesis PN32 v2 v7.69 chunked-prefill] Cliff 2 fix:\n"
    "            # chunk the FLA call so the inner core_attn_out_non_spec\n"
    "            # buffer is allocated per-chunk (not full-prompt). Composes\n"
    "            # with P103 (P103 chunks INSIDE chunk_gated_delta_rule_fwd's\n"
    "            # h tensor). Single-sequence prefill only — multi-sequence\n"
    "            # bypasses to original since chunking across cu_seqlens\n"
    "            # boundaries requires inner state-cache surgery not exposed\n"
    "            # at this layer. See vllm/_genesis/wiring/hybrid/\n"
    "            # patch_N32_gdn_chunked_prefill.py for full design notes.\n"
    "            import os as _genesis_pn32_os\n"
    "            _genesis_pn32_enabled = _genesis_pn32_os.environ.get(\n"
    "                'GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL', ''\n"
    "            ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            try:\n"
    "                _genesis_pn32_threshold = int(\n"
    "                    _genesis_pn32_os.environ.get(\n"
    "                        'GENESIS_PN32_GDN_CHUNK_THRESHOLD', '16384'\n"
    "                    )\n"
    "                )\n"
    "            except (ValueError, TypeError):\n"
    "                _genesis_pn32_threshold = 16384\n"
    "            try:\n"
    "                _genesis_pn32_chunk_size = int(\n"
    "                    _genesis_pn32_os.environ.get(\n"
    "                        'GENESIS_PN32_GDN_CHUNK_SIZE', '8192'\n"
    "                    )\n"
    "                )\n"
    "            except (ValueError, TypeError):\n"
    "                _genesis_pn32_chunk_size = 8192\n"
    "\n"
    "            # Single-sequence detection: cu_seqlens shape [2] = one\n"
    "            # [0, T] entry. Multi-seq has shape [N+1] for N>1.\n"
    "            _genesis_pn32_T_full = int(query_non_spec.shape[1])\n"
    "            _genesis_pn32_is_single_seq = (\n"
    "                non_spec_query_start_loc is not None\n"
    "                and non_spec_query_start_loc.shape[0] == 2\n"
    "            )\n"
    "            _genesis_pn32_should_chunk = (\n"
    "                _genesis_pn32_enabled\n"
    "                and _genesis_pn32_is_single_seq\n"
    "                and _genesis_pn32_T_full > _genesis_pn32_threshold\n"
    "            )\n"
    "\n"
    "            if _genesis_pn32_should_chunk:\n"
    "                # ─── Chunked path: split FLA call into chunks ───\n"
    "                _genesis_pn32_chunks = []\n"
    "                _genesis_pn32_state = initial_state\n"
    "                _genesis_pn32_last_state = None\n"
    "                for _genesis_pn32_start in range(\n"
    "                    0, _genesis_pn32_T_full, _genesis_pn32_chunk_size\n"
    "                ):\n"
    "                    _genesis_pn32_end = min(\n"
    "                        _genesis_pn32_start + _genesis_pn32_chunk_size,\n"
    "                        _genesis_pn32_T_full,\n"
    "                    )\n"
    "                    _genesis_pn32_chunk_len = (\n"
    "                        _genesis_pn32_end - _genesis_pn32_start\n"
    "                    )\n"
    "                    # Slice along T dim (dim=1) — shape (1, T, H, D)\n"
    "                    _genesis_pn32_q_chunk = query_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_k_chunk = key_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_v_chunk = value_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_g_chunk = g_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_beta_chunk = beta_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    # Chunk-local cu_seqlens — single-seq, length = chunk_len\n"
    "                    _genesis_pn32_chunk_cu_seqlens = torch.tensor(\n"
    "                        [0, _genesis_pn32_chunk_len],\n"
    "                        device=query_non_spec.device,\n"
    "                        dtype=non_spec_query_start_loc.dtype,\n"
    "                    )\n"
    "                    # FLA call on chunk; output_final_state=True for chaining\n"
    "                    (\n"
    "                        _genesis_pn32_o_chunk,\n"
    "                        _genesis_pn32_last_state,\n"
    "                    ) = self.chunk_gated_delta_rule(\n"
    "                        q=_genesis_pn32_q_chunk,\n"
    "                        k=_genesis_pn32_k_chunk,\n"
    "                        v=_genesis_pn32_v_chunk,\n"
    "                        g=_genesis_pn32_g_chunk,\n"
    "                        beta=_genesis_pn32_beta_chunk,\n"
    "                        initial_state=_genesis_pn32_state,\n"
    "                        output_final_state=True,\n"
    "                        cu_seqlens=_genesis_pn32_chunk_cu_seqlens,\n"
    "                        chunk_indices=None,\n"
    "                        chunk_offsets=None,\n"
    "                        use_qk_l2norm_in_kernel=False,\n"
    "                    )\n"
    "                    _genesis_pn32_chunks.append(_genesis_pn32_o_chunk)\n"
    "                    # Thread state for next chunk\n"
    "                    _genesis_pn32_state = _genesis_pn32_last_state\n"
    "                    # Free chunk references (allocator can reuse)\n"
    "                    del (\n"
    "                        _genesis_pn32_q_chunk,\n"
    "                        _genesis_pn32_k_chunk,\n"
    "                        _genesis_pn32_v_chunk,\n"
    "                        _genesis_pn32_g_chunk,\n"
    "                        _genesis_pn32_beta_chunk,\n"
    "                    )\n"
    "\n"
    "                core_attn_out_non_spec = torch.cat(\n"
    "                    _genesis_pn32_chunks, dim=1\n"
    "                )\n"
    "                last_recurrent_state = _genesis_pn32_last_state\n"
    "                del _genesis_pn32_chunks\n"
    "            else:\n"
    "                # ─── Original path (multi-seq, below threshold, or env off) ───\n"
    "                (\n"
    "                    core_attn_out_non_spec,\n"
    "                    last_recurrent_state,\n"
    "                ) = self.chunk_gated_delta_rule(\n"
    "                    q=query_non_spec,\n"
    "                    k=key_non_spec,\n"
    "                    v=value_non_spec,\n"
    "                    g=g_non_spec,\n"
    "                    beta=beta_non_spec,\n"
    "                    initial_state=initial_state,\n"
    "                    output_final_state=True,\n"
    "                    cu_seqlens=non_spec_query_start_loc,\n"
    "                    chunk_indices=attn_metadata.chunk_indices,\n"
    "                    chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                    use_qk_l2norm_in_kernel=False,\n"
    "                )\n"
    "            # Init cache\n"
    "            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(\n"
    "                ssm_state.dtype\n"
    "            )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN32 v2 model_executor/layers/mamba/gdn_linear_attn.py — "
            "_forward_core chunked-prefill (Cliff 2 fix v7.69)"
        ),
        target_file=str(target),
        marker=GENESIS_PN32_MARKER,
        sub_patches=[
            TextPatch(
                name="pN32_v2_forward_core_chunked_prefill",
                anchor=PN32_ANCHOR,
                replacement=PN32_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Specific to v2's own insertion. Generic '[Genesis PN32'
            # would false-positive if v1 ever existed alongside (it
            # doesn't currently — v2 supersedes — but defensive).
            "[Genesis PN32 v2 v7.69 chunked-prefill]",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN32 v2 — _forward_core chunked-prefill (text-patch).

    v7.69 v2 supersedes v7.65 v1. v1 chunked at the wrong level
    (forward_cuda outer) and didn't propagate cu_seqlens — empirically
    OOM'd EARLIER than baseline on club-3090 cross-rig. v2 chunks
    inside _forward_core where chunk-local cu_seqlens can be built
    correctly.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN32")
    log_decision("PN32", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN32 v2 v7.69 applied: GDN _forward_core prefill branch "
            "now uses single-seq chunked path for long prompts (>16K "
            "tokens default). Chunks query/key/value/g/beta along T, "
            "builds chunk-local cu_seqlens, threads initial_state via "
            "last_recurrent_state. Multi-seq bypasses to original. "
            "Default OFF — opt-in via GENESIS_ENABLE_PN32_GDN_"
            "CHUNKED_PREFILL=1. Composes with P103 for full Cliff 2 "
            "coverage on single-24GB-GPU (P103 chunks the inner FLA h "
            "tensor; PN32 chunks the outer FLA call output buffer)."
        ),
        patch_name="PN32 v2 GDN _forward_core chunked-prefill",
    )
