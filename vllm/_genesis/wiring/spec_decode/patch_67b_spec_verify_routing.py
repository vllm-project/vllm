# SPDX-License-Identifier: Apache-2.0
"""P67b — architectural routing for spec-verify K+1 batches.

Companion to P67 kernel (multi-query attention). Phase B of the spec-decode
FULL cudagraph fix per architectural research (path b).

Adds a P67 dispatch branch at the top of `TurboQuantAttentionImpl.forward()`
BEFORE the existing prefill/decode/mixed dispatch. For uniform K+1 spec-verify
batches (max_query_len in [2, 16], total_seq_len > max_query_len), routes
through P67 kernel directly, completely BYPASSING `_prefill_attention`.

WHY THIS MATTERS:
- `_prefill_attention` has the upstream `tolist_cudagraph_fix` bypass which
  crashes with `cudaErrorStreamCaptureInvalidated` under FULL cudagraph capture
- Routing K+1 through P67 at forward() level avoids the bypass entirely
- Combined with restoring P65 (UNIFORM_BATCH cudagraph_support), enables
  FULL_AND_PIECEWISE cudagraph mode for spec-decode → +20-30% TPS

This is the "minimal viable" version of Phase B path (b) per the research:
- No metadata schema change (no is_spec_verify field added)
- No supports_spec_as_decode flip (avoids reorder_batch_threshold change)
- Just an extra branch at top of forward() that intercepts K+1 batches

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

log = logging.getLogger("genesis.wiring.p67b_spec_verify_routing")

# ─── B2 fix v7.62.12: bake env reads at module load ─────────────────────
# Hot-path env reads in the original emit ran on EVERY spec-verify forward()
# call. While env reads are not GPU syncs (so not strictly illegal under
# cudagraph capture), they incur ~50-100ns Python overhead × ~800 dispatch/sec
# = ~0.04-0.08% TPS waste. More importantly: they're inconsistent with the
# H2 fix already applied to P67. Bake at apply-time for symmetry + perf.
#
# Trade-off: GENESIS_P67_USE_UPSTREAM and GENESIS_P67_NUM_KV_SPLITS become
# container-launch-time tunables. Operators set them the same way (env vars
# at start), the snapshot just happens earlier.
_BAKED_USE_UPSTREAM = (
    os.environ.get("GENESIS_P67_USE_UPSTREAM", "1") == "1"
)
# For NUM_KV_SPLITS: empty string → fall back to self.max_num_kv_splits at
# runtime (instance attr). Non-empty → bake as int literal.
_BAKED_NUM_KV_SPLITS_RAW = os.environ.get("GENESIS_P67_NUM_KV_SPLITS", "").strip()
try:
    _BAKED_NUM_KV_SPLITS: int | None = (
        int(_BAKED_NUM_KV_SPLITS_RAW) if _BAKED_NUM_KV_SPLITS_RAW else None
    )
except ValueError:
    _BAKED_NUM_KV_SPLITS = None

log.info(
    "[Genesis P67b B2] baked env at module load: USE_UPSTREAM=%s NUM_KV_SPLITS=%s "
    "(no per-dispatch env reads)",
    _BAKED_USE_UPSTREAM,
    _BAKED_NUM_KV_SPLITS if _BAKED_NUM_KV_SPLITS is not None else "(default = self.max_num_kv_splits)",
)

GENESIS_P67B_MARKER = "Genesis P67b spec-verify forward() routing v7.63.x_nopow2_gqa"


# Anchor: existing `forward()` method just before the dispatch branches.
# We inject our P67 branch RIGHT AFTER the metadata sanity checks and BEFORE
# the `if not attn_metadata.is_prefill:` block.

P67B_OLD = (
    "        # Compute attention (KV cache was already updated by do_kv_cache_update)\n"
    "        # With reorder_batch_threshold=1, decodes come first in the batch.\n"
    "        # num_decodes/num_decode_tokens from metadata give the split point.\n"
    "        num_decodes = attn_metadata.num_decodes\n"
    "        num_decode_tokens = attn_metadata.num_decode_tokens\n"
    "\n"
    "        if not attn_metadata.is_prefill:\n"
)

P67B_NEW = (
    "        # Compute attention (KV cache was already updated by do_kv_cache_update)\n"
    "        # With reorder_batch_threshold=1, decodes come first in the batch.\n"
    "        # num_decodes/num_decode_tokens from metadata give the split point.\n"
    "        num_decodes = attn_metadata.num_decodes\n"
    "        num_decode_tokens = attn_metadata.num_decode_tokens\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P67b vllm-genesis] Spec-verify K+1 dispatch BEFORE prefill.\n"
    "        # Routes uniform K+1 spec-verify batches through P67 kernel directly,\n"
    "        # bypassing _prefill_attention's broken cudagraph capture path.\n"
    "        # Conditions: env P67=1, multi-query (1<max_q≤16), has prior cache,\n"
    "        # uniform layout, output buffer cached on instance.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        try:\n"
    "            from vllm._genesis.kernels.p67_multi_query_kernel import (\n"
    "                is_active as _genesis_p67b_active,\n"
    "                call_p67_attention as _genesis_p67b_call,\n"
    "            )\n"
    "            _genesis_p67b_Hk = key.shape[1]\n"
    "            # v7.63.x_nopow2: HEADS_PER_KV pow-2 requirement DROPPED.\n"
    "            # Split-M kernel was generalized via BLOCK_QH = next_power_of_2(hpk)\n"
    "            # padding + lane_valid mask in p67_multi_query_kernel.py. P67b\n"
    "            # routes ANY GQA factor >= 2 through the kernel; non-pow-2 hpk\n"
    "            # (e.g. Qwen3.6-27B GQA=24/4=6 → BLOCK_QH=8 with 2 lanes masked)\n"
    "            # compiles cleanly and runs bit-exact to the pow-2 case for the\n"
    "            # valid lanes. Fused-M (GENESIS_P67_USE_FUSED=1) still requires\n"
    "            # pow-2 — its launcher raises ValueError on non-pow-2, which the\n"
    "            # outer try/except below catches and falls through to upstream.\n"
    "            _genesis_p67b_hpk = (\n"
    "                self.num_heads // _genesis_p67b_Hk if _genesis_p67b_Hk else 0\n"
    "            )\n"
    "            _genesis_p67b_shape_ok = (\n"
    "                self.num_heads >= 8\n"
    "                and self.head_size in (128, 256)\n"
    "                and _genesis_p67b_hpk >= 2\n"
    "            )\n"
    "            # NO prior_len threshold — P67b MUST fire for ALL K+1 batches\n"
    "            # under cudagraph capture, otherwise bypass gets captured (broken).\n"
    "            # max_seq_len in metadata is upper bound, not actual; can't use for filter.\n"
    "            _genesis_p67b_dispatch = (\n"
    "                _genesis_p67b_active()\n"
    "                and _genesis_p67b_shape_ok\n"
    "                and 1 < attn_metadata.max_query_len <= 16\n"
    "                and N > 0\n"
    "                and (N % attn_metadata.max_query_len) == 0\n"
    "                and attn_metadata.query_start_loc is not None\n"
    "            )\n"
    "            if _genesis_p67b_dispatch:\n"
    "                _genesis_p67b_K1 = attn_metadata.max_query_len\n"
    "                _genesis_p67b_B = N // _genesis_p67b_K1\n"
    "                if attn_metadata.query_start_loc.shape[0] == _genesis_p67b_B + 1:\n"
    "                    # ROUTE: env GENESIS_P67_USE_UPSTREAM=1 → upstream kernel\n"
    "                    # (drift-free, multi-CTA), else → P67 (custom Triton kernel).\n"
    "                    # [Genesis P67b B2 v7.62.12] baked at module load (no per-dispatch env read)\n"
    f"                    _use_upstream = {_BAKED_USE_UPSTREAM}\n"
    "                    import torch as _genesis_p67b_torch\n"
    "                    if _use_upstream:\n"
    "                        # ─── Upstream kernel path (drift-free, multi-CTA) ───\n"
    "                        from vllm.v1.attention.ops.triton_turboquant_decode import (\n"
    "                            triton_turboquant_decode_attention as _ts_decode_attn,\n"
    "                        )\n"
    "                        # Build synth args mirroring upstream's _continuation_prefill:\n"
    "                        # query: (N=B*K1, Hq, D)\n"
    "                        # synth_seq_lens[req*K1+i] = base_seq_lens[req] - K1 + 1 + i\n"
    "                        # synth_bt[req*K1+i] = block_table[req]\n"
    "                        _q_flat = q[:N].view(N, self.num_heads, self.head_size)\n"
    "                        _genesis_p67b_offs = _genesis_p67b_torch.arange(\n"
    "                            _genesis_p67b_K1, device=q.device, dtype=attn_metadata.seq_lens.dtype\n"
    "                        )\n"
    "                        _genesis_p67b_synth_sl = (\n"
    "                            attn_metadata.seq_lens[:_genesis_p67b_B, None] - _genesis_p67b_K1 + 1 + _genesis_p67b_offs[None, :]\n"
    "                        ).reshape(-1)\n"
    "                        _genesis_p67b_synth_bt = attn_metadata.block_table[:_genesis_p67b_B].repeat_interleave(\n"
    "                            _genesis_p67b_K1, dim=0\n"
    "                        )\n"
    "                        # Pre-allocated output buffer (cudagraph-safe).\n"
    "                        _genesis_p67b_buf_key = (N, self.num_heads, self.head_size)\n"
    "                        if not hasattr(self, '_genesis_p67b_out_buffers'):\n"
    "                            self._genesis_p67b_out_buffers = {}\n"
    "                        _genesis_p67b_out_buf = self._genesis_p67b_out_buffers.get(_genesis_p67b_buf_key)\n"
    "                        if _genesis_p67b_out_buf is None or _genesis_p67b_out_buf.dtype != q.dtype:\n"
    "                            _genesis_p67b_out_buf = _genesis_p67b_torch.empty(\n"
    "                                _genesis_p67b_buf_key, dtype=q.dtype, device=q.device\n"
    "                            )\n"
    "                            self._genesis_p67b_out_buffers[_genesis_p67b_buf_key] = _genesis_p67b_out_buf\n"
    "                        # [v7.63.x P67b illegal-address fix — credit @Quentin-M]\n"
    "                        # layer._tq_mid_o_buf is allocated by the decode path for\n"
    "                        # max_num_seqs rows. K+1 synthetic rows (B*K1) exceed that\n"
    "                        # (e.g. max_num_seqs=4, K1=5 → need 20 rows) → OOB write\n"
    "                        # → cudaErrorIllegalAddress surfacing at next CPU sync.\n"
    "                        # Fix: per-K1 SimpleNamespace holder on `self` so decode-\n"
    "                        # path buffers are never touched. The kernel self-allocates\n"
    "                        # the correct size on first K+1 dispatch (via buf_holder)\n"
    "                        # and caches it there. Grows automatically if B increases.\n"
    "                        # Affected configs: any Hq/Hk not power-of-2 where the\n"
    "                        # custom P67 Triton kernel can't compile (e.g. Qwen3.6-27B\n"
    "                        # with Hq/Hk=5) → fallback to upstream path here.\n"
    "                        # Defensive on Genesis PROD (Lorbus 27B INT4 our P67 builds\n"
    "                        # cleanly), but engages whenever USE_UPSTREAM=1 forces it.\n"
    "                        _genesis_p67b_n_synth = _genesis_p67b_B * _genesis_p67b_K1\n"
    "                        if not hasattr(self, '_genesis_p67b_syn_holders'):\n"
    "                            self._genesis_p67b_syn_holders = {}\n"
    "                        _genesis_p67b_holder = self._genesis_p67b_syn_holders.get(\n"
    "                            _genesis_p67b_K1\n"
    "                        )\n"
    "                        if _genesis_p67b_holder is None:\n"
    "                            import types as _genesis_p67b_types\n"
    "                            _genesis_p67b_holder = _genesis_p67b_types.SimpleNamespace()\n"
    "                            self._genesis_p67b_syn_holders[_genesis_p67b_K1] = _genesis_p67b_holder\n"
    "                        _genesis_p67b_cached_mid_o = getattr(\n"
    "                            _genesis_p67b_holder, '_tq_mid_o_buf', None\n"
    "                        )\n"
    "                        if (\n"
    "                            _genesis_p67b_cached_mid_o is not None\n"
    "                            and _genesis_p67b_cached_mid_o.shape[0] < _genesis_p67b_n_synth\n"
    "                        ):\n"
    "                            _genesis_p67b_holder._tq_mid_o_buf = None\n"
    "                            _genesis_p67b_holder._tq_lse_buf = None\n"
    "                        _genesis_p67b_attn_out = _ts_decode_attn(\n"
    "                            query=_q_flat,\n"
    "                            kv_cache=kv_cache,\n"
    "                            block_table=_genesis_p67b_synth_bt,\n"
    "                            seq_lens=_genesis_p67b_synth_sl,\n"
    "                            Pi=Pi,\n"
    "                            centroids=centroids,\n"
    "                            scale=self.scale,\n"
    "                            mse_bits=self.tq_config.key_mse_bits,\n"
    "                            key_packed_size=self.tq_config.key_packed_size,\n"
    "                            value_quant_bits=self.tq_config.effective_value_quant_bits,\n"
    "                            key_fp8=self.tq_config.key_fp8,\n"
    "                            norm_correction=self.tq_config.norm_correction,\n"
    "                            PiT=PiT,\n"
    "                            output_buf=_genesis_p67b_out_buf,\n"
    "                            mid_o_buf=getattr(_genesis_p67b_holder, '_tq_mid_o_buf', None),\n"
    "                            lse_buf=getattr(_genesis_p67b_holder, '_tq_lse_buf', None),\n"
    "                            buf_holder=_genesis_p67b_holder,\n"
    # [Genesis P67b B2 v7.62.12] baked num_kv_splits at module load.
    # If env was unset, use instance attr at runtime (no env read either).
    + (
        f"                            max_num_kv_splits={int(_BAKED_NUM_KV_SPLITS)},\n"
        if _BAKED_NUM_KV_SPLITS is not None
        else "                            max_num_kv_splits=self.max_num_kv_splits,\n"
    ) +
    "                        )\n"
    "                    else:\n"
    "                        # ─── P67 kernel path (custom, may drift on long ctx) ───\n"
    "                        _genesis_p67b_q = q.contiguous().view(\n"
    "                            _genesis_p67b_B, _genesis_p67b_K1, self.num_heads, self.head_size\n"
    "                        )\n"
    "                        k_view = key[:N].view(N, self.num_kv_heads, self.head_size)\n"
    "                        v_view = value[:N].view(N, self.num_kv_heads, self.head_size)\n"
    "                        _genesis_p67b_k = k_view.contiguous().view(\n"
    "                            _genesis_p67b_B, _genesis_p67b_K1, self.num_kv_heads, self.head_size\n"
    "                        )\n"
    "                        _genesis_p67b_v = v_view.contiguous().view(\n"
    "                            _genesis_p67b_B, _genesis_p67b_K1, self.num_kv_heads, self.head_size\n"
    "                        )\n"
    "                        _genesis_p67b_block_size = kv_cache.shape[1]\n"
    "                        _genesis_p67b_kps = self.tq_config.key_packed_size\n"
    "                        _genesis_p67b_vdb = (\n"
    "                            self.tq_config.value_data_bytes\n"
    "                            if hasattr(self.tq_config, 'value_data_bytes')\n"
    "                            else (self.head_size // 2)\n"
    "                        )\n"
    "                        _genesis_p67b_buf_key = (\n"
    "                            _genesis_p67b_B, _genesis_p67b_K1,\n"
    "                            self.num_heads, self.head_size,\n"
    "                        )\n"
    "                        if not hasattr(self, '_genesis_p67b_out_buffers'):\n"
    "                            self._genesis_p67b_out_buffers = {}\n"
    "                        _genesis_p67b_out_buf = self._genesis_p67b_out_buffers.get(_genesis_p67b_buf_key)\n"
    "                        if _genesis_p67b_out_buf is None or _genesis_p67b_out_buf.dtype != q.dtype:\n"
    "                            _genesis_p67b_out_buf = _genesis_p67b_torch.empty(\n"
    "                                _genesis_p67b_buf_key, dtype=q.dtype, device=q.device\n"
    "                            )\n"
    "                            self._genesis_p67b_out_buffers[_genesis_p67b_buf_key] = _genesis_p67b_out_buf\n"
    "                        _genesis_p67b_attn_out = _genesis_p67b_call(\n"
    "                            q=_genesis_p67b_q,\n"
    "                            kv_cache=kv_cache,\n"
    "                            block_table=attn_metadata.block_table,\n"
    "                            seq_lens=attn_metadata.seq_lens,\n"
    "                            k_chunk=_genesis_p67b_k,\n"
    "                            v_chunk=_genesis_p67b_v,\n"
    "                            scale=self.scale,\n"
    "                            block_size=_genesis_p67b_block_size,\n"
    "                            kps=_genesis_p67b_kps,\n"
    "                            val_data_bytes=_genesis_p67b_vdb,\n"
    "                            output=_genesis_p67b_out_buf,\n"
    "                        )\n"
    "                    # Write to engine output buffer in (N, Hq*D) or (N, Hq, D) layout.\n"
    "                    _genesis_p67b_attn_flat = _genesis_p67b_attn_out.view(\n"
    "                        N, self.num_heads, self.head_size\n"
    "                    )\n"
    "                    if output.ndim == 3:\n"
    "                        output[:N] = _genesis_p67b_attn_flat\n"
    "                    else:\n"
    "                        output[:N] = _genesis_p67b_attn_flat.reshape(N, -1)\n"
    "                    return output\n"
    "        except Exception as _genesis_p67b_err:\n"
    "            import logging as _genesis_p67b_logging\n"
    "            _genesis_p67b_logging.getLogger('genesis.wiring.p67b').warning(\n"
    "                'P67b dispatch failed (%s), falling through to upstream',\n"
    "                _genesis_p67b_err, exc_info=True,\n"
    "            )\n"
    "\n"
    "        if not attn_metadata.is_prefill:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P67b turboquant_attn.py forward() spec-verify routing",
        target_file=str(target),
        marker=GENESIS_P67B_MARKER,
        sub_patches=[
            TextPatch(
                name="p67b_forward_spec_verify_branch",
                anchor=P67B_OLD,
                replacement=P67B_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P67b",
            "_genesis_p67b_call",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P67b — forward() spec-verify routing."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    # Reuse P67 env flag — P67b is meaningless without P67 kernel
    decision, reason = should_apply("P67")
    log_decision("P67b", decision, reason)
    if not decision:
        return "skipped", "P67 kernel disabled — P67b dispatch unused"

    # 2026-04-27 v756 bisect SAFETY GATE (mirrors P67's gate): without
    # speculative_config in vllm config, P67b's forward() routing would
    # try to dispatch into the disabled P67 kernel for any non-decode
    # batch (chunked-prefill included), reproducing the v756 IndexKernel
    # overflow. Refuse to apply P67b unless spec-decode is actually
    # configured. See V756_STABILITY_INVESTIGATION_20260427.md.
    try:
        from vllm._genesis.config_detect import recommend
        cd_verdict, cd_reason = recommend("P67")
        if cd_verdict.startswith("skip"):
            return "skipped", (
                f"P67b SAFETY GATE — config_detect for P67 says {cd_verdict}: "
                f"{cd_reason} | env flag IGNORED to prevent v756-class "
                "IndexKernel overflow. P67b cannot be safely applied "
                "without P67 kernel and spec-decode."
            )
    except Exception as e:
        log.warning("[P67b] safety gate config_detect probe failed: %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        pass
    else:
        for m in patcher.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} in {patcher.target_file} — "
                    "P67b likely already injected.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                "required anchor (forward() dispatch preamble) not found — "
                "upstream forward() may have changed; P67b cannot apply.",
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
        "P67b forward() spec-verify routing injected. K+1 batches now bypass "
        "_prefill_attention entirely (cudagraph-safe)."
    )
