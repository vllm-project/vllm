# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 67 — TurboQuant multi-query kernel for spec-decode K+1 batches.

Genesis-original. The PROPER fix for noonghunna's #40880 bug class
(MTP × TurboQuant × FULL cudagraph degenerate output). P65 was a
workaround (downgrades cudagraph_mode to PIECEWISE → ~30% throughput hit);
P67 replaces it by handling K+1 batches in a Triton kernel that supports
FULL cudagraph capture.

================================================================
WHAT THIS PATCH DOES

Inserts a hook at the top of `TurboQuantAttentionImpl._prefill_attention`.
For K+1 spec-verify continuation-prefill batches (multi-query AND has
prior cached KV) AND when env GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1,
routes the batch through our P67 multi-query Triton kernel instead of
the buggy bypass / slow per-request continuation loop.

For all other shapes (true first-chunk prefill, pure decode, non-spec-decode
continuation) the existing upstream logic runs unchanged.

================================================================
SAFETY / FALLBACK

- env-gated (default OFF) — operator must explicitly enable
- on any P67 exception, falls through to existing eager continuation branch
- if Triton kernel fails to build (CPU-only host etc.), kernel.is_active()
  returns False → hook is no-op
- preserves bit-for-bit behavior when env flag is off

After P67 is empirically validated:
  1. Restore P65 to declare UNIFORM_BATCH (no longer need cudagraph downgrade)
  2. Spec-decode batches regain FULL cudagraph speedup
  3. Net effect: P64+P65v2+P66+P67 = correct + fast

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

log = logging.getLogger("genesis.wiring.p67_tq_multi_query_kernel")

GENESIS_P67_MARKER = "Genesis P67 TQ multi-query kernel for spec-decode K+1 v7.63.x_nopow2_gqa"


# ─── H2 fix: bake env reads at module-load (eager) ───────────────────────
# Hot-path env reads in the original emit (~800 dispatches/sec on K=3 spec
# decode workload) cost ~0.5% TPS. Snapshot the env values once at module
# load and inline them as literals into the text-patch body. Operators set
# these the same way (env vars at container start) — the snapshot just
# happens earlier (plugin register time) instead of per-dispatch.
#
# Trade-off: changing GENESIS_P67_MAX_PRIOR_LEN or
# GENESIS_P67_DEBUG_COMPARE at runtime no longer takes effect — must
# restart container. Acceptable: these are container-launch-time tunables.
_BAKED_MAX_PRIOR = int(os.environ.get("GENESIS_P67_MAX_PRIOR_LEN", "4096"))
_BAKED_DEBUG_COMPARE = (
    os.environ.get("GENESIS_P67_DEBUG_COMPARE", "0") == "1"
)
log.info(
    "[Genesis P67 H2] baked env at module load: MAX_PRIOR_LEN=%d "
    "DEBUG_COMPARE=%s (no per-dispatch env reads)",
    _BAKED_MAX_PRIOR, _BAKED_DEBUG_COMPARE,
)


# ─── Sub-patch: insert P67 hook at top of _prefill_attention ────────────────
# Anchor on the function signature + initial fast-path check. Insert P67
# dispatch BEFORE the fast path so we can intercept K+1 continuation cases
# that the fast path's `max_query_len == max_seq_len` guard skips.

P67_OLD = (
    "    ) -> torch.Tensor:\n"
    "        N, Hq, D = query.shape\n"
    "\n"
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # max_query_len == max_seq_len means no request has prior cached KV.\n"
    "        # Both are Python ints — no GPU sync.\n"
    "        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:\n"
)

P67_NEW = (
    "    ) -> torch.Tensor:\n"
    "        N, Hq, D = query.shape\n"
    "\n"
    "        # [Genesis P67 vllm-genesis] Multi-query continuation prefill hook.\n"
    "        # If batch is K+1 spec-verify continuation (multi-query AND has prior\n"
    "        # cached KV) AND env GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1,\n"
    "        # route to our Triton kernel that handles compressed cache directly\n"
    "        # under FULL cudagraph (proper fix vs P65 workaround).\n"
    "        try:\n"
    "            from vllm._genesis.kernels.p67_multi_query_kernel import (\n"
    "                is_active as _genesis_p67_is_active,\n"
    "                call_p67_attention as _genesis_p67_call,\n"
    "            )\n"
    "            import logging as _genesis_p67_logging\n"
    "            _genesis_p67_log = _genesis_p67_logging.getLogger('genesis.kernels.p67')\n"
    "            if not hasattr(self, '_genesis_p67_entry_count'):\n"
    "                self._genesis_p67_entry_count = 0\n"
    "            self._genesis_p67_entry_count += 1\n"
    "            if self._genesis_p67_entry_count <= 3:\n"
    "                # [Genesis 2026-05-02] Was log.warning — downgraded to\n"
    "                # log.info: this is an INFO-level diagnostic for first 3\n"
    "                # P67 dispatches per layer, not an actual warning. Was\n"
    "                # creating ~120 fake-WARNING entries per boot under\n"
    "                # VLLM_LOGGING_LEVEL=WARNING which obscured real warnings.\n"
    "                _genesis_p67_log.info(\n"
    "                    'P67 hook ENTRY #%d: N=%d Hq=%d D=%d Hk=%d max_q=%s max_s=%s active=%s',\n"
    "                    self._genesis_p67_entry_count, N, Hq, D, key.shape[1],\n"
    "                    attn_metadata.max_query_len, attn_metadata.max_seq_len,\n"
    "                    _genesis_p67_is_active(),\n"
    "                )\n"
    "            # Shape guard: P67 supports D in {128, 256}, Hq>=8, GQA>=2.\n"
    "            # v7.63.x_nopow2: DROPPED the (hpk & (hpk-1))==0 power-of-2\n"
    "            # requirement. Split-M kernel now compiles for any GQA factor\n"
    "            # via BLOCK_QH = next_power_of_2(HEADS_PER_KV) padding + lane_valid\n"
    "            # mask in the kernel body. Qwen3.6-27B GQA=24/4=6 → BLOCK_QH=8 with\n"
    "            # 2 lanes masked, bit-exact to pow-2 case when hpk is itself pow-2.\n"
    "            # Hpk=1 still excluded (degenerate; upstream MQA path handles it).\n"
    "            # Fused-M (GENESIS_P67_USE_FUSED=1) still requires pow-2 — its\n"
    "            # BLOCK_M = K_PLUS_1*hpk has not been generalized; the kernel\n"
    "            # launcher raises ValueError on non-pow-2 + fused, caller falls\n"
    "            # through cleanly to upstream.\n"
    "            _genesis_p67_Hk = key.shape[1]\n"
    "            _genesis_p67_hpk = Hq // _genesis_p67_Hk if _genesis_p67_Hk else 0\n"
    "            _genesis_p67_shape_ok = (\n"
    "                Hq >= 8\n"
    "                and D in (128, 256)\n"
    "                and _genesis_p67_hpk >= 2\n"
    "            )\n"
    "            # v7.34: split-M architecture eliminates per-row epilogue drift.\n"
    "            # Default threshold MUCH HIGHER (32K) since split-M is bit-exact\n"
    "            # to per-query precision. Tunable via GENESIS_P67_MAX_PRIOR_LEN.\n"
    "            # [Genesis P67 H2 v7.62.6] baked at module load instead of\n"
    "            # per-dispatch env read (~0.5% TPS recovered).\n"
    f"            _genesis_p67_max_prior = {_BAKED_MAX_PRIOR}\n"
    "            _genesis_p67_max_kp1 = 16\n"
    "            _genesis_p67_prior_len = (\n"
    "                attn_metadata.max_seq_len - attn_metadata.max_query_len\n"
    "            )\n"
    "            _genesis_p67_dispatch = (\n"
    "                _genesis_p67_is_active()\n"
    "                and _genesis_p67_shape_ok\n"
    "                and attn_metadata.max_query_len > 1\n"
    "                and attn_metadata.max_query_len <= _genesis_p67_max_kp1\n"
    "                and attn_metadata.max_seq_len > attn_metadata.max_query_len\n"
    "                and _genesis_p67_prior_len <= _genesis_p67_max_prior\n"
    "                and N > 0\n"
    "                and (N % attn_metadata.max_query_len) == 0\n"
    "            )\n"
    "            if not hasattr(self, '_genesis_p67_call_count'):\n"
    "                self._genesis_p67_call_count = 0\n"
    "                self._genesis_p67_dispatch_count = 0\n"
    "            self._genesis_p67_call_count += 1\n"
    "            if _genesis_p67_dispatch:\n"
    "                self._genesis_p67_dispatch_count += 1\n"
    "                _genesis_p67_K_PLUS_1 = attn_metadata.max_query_len\n"
    "                _genesis_p67_B = N // _genesis_p67_K_PLUS_1\n"
    "                _genesis_p67_qsl = attn_metadata.query_start_loc\n"
    "                if (\n"
    "                    _genesis_p67_qsl is not None\n"
    "                    and _genesis_p67_qsl.shape[0] == _genesis_p67_B + 1\n"
    "                ):\n"
    "                    Hk_genesis = key.shape[1]\n"
    "                    # Skip .contiguous() if already contiguous to avoid alloc inside cudagraph capture.\n"
    "                    _genesis_p67_q_src = query if query.is_contiguous() else query.contiguous()\n"
    "                    _genesis_p67_k_src = key if key.is_contiguous() else key.contiguous()\n"
    "                    _genesis_p67_v_src = value if value.is_contiguous() else value.contiguous()\n"
    "                    _genesis_p67_q = _genesis_p67_q_src.view(\n"
    "                        _genesis_p67_B, _genesis_p67_K_PLUS_1, Hq, D\n"
    "                    )\n"
    "                    _genesis_p67_k_chunk = _genesis_p67_k_src.view(\n"
    "                        _genesis_p67_B, _genesis_p67_K_PLUS_1, Hk_genesis, D\n"
    "                    )\n"
    "                    _genesis_p67_v_chunk = _genesis_p67_v_src.view(\n"
    "                        _genesis_p67_B, _genesis_p67_K_PLUS_1, Hk_genesis, D\n"
    "                    )\n"
    "                    _genesis_p67_cfg = self.tq_config\n"
    "                    _genesis_p67_block_size = kv_cache.shape[1]\n"
    "                    _genesis_p67_kps = _genesis_p67_cfg.key_packed_size\n"
    "                    _genesis_p67_vdb = _genesis_p67_cfg.value_data_bytes if hasattr(_genesis_p67_cfg, 'value_data_bytes') else (D // 2)\n"
    "                    if self._genesis_p67_dispatch_count <= 3:\n"
    "                        _genesis_p67_log.warning(\n"
    "                            'P67 dispatch #%d: B=%d K_PLUS_1=%d Hq=%d Hk=%d D=%d block_size=%d kps=%d vdb=%d max_seq=%d',\n"
    "                            self._genesis_p67_dispatch_count, _genesis_p67_B,\n"
    "                            _genesis_p67_K_PLUS_1, Hq, Hk_genesis, D,\n"
    "                            _genesis_p67_block_size, _genesis_p67_kps, _genesis_p67_vdb,\n"
    "                            attn_metadata.max_seq_len,\n"
    "                        )\n"
    "                    # Cudagraph-safe pre-allocated output buffer per shape on self.\n"
    "                    # Avoids torch.empty_like inside cudagraph capture.\n"
    "                    _genesis_p67_buf_key = (_genesis_p67_B, _genesis_p67_K_PLUS_1, Hq, D)\n"
    "                    if not hasattr(self, '_genesis_p67_out_buffers'):\n"
    "                        self._genesis_p67_out_buffers = {}\n"
    "                    _genesis_p67_out_buf = self._genesis_p67_out_buffers.get(_genesis_p67_buf_key)\n"
    "                    if _genesis_p67_out_buf is None or _genesis_p67_out_buf.dtype != query.dtype or _genesis_p67_out_buf.device != query.device:\n"
    "                        import torch as _genesis_p67_torch_alloc\n"
    "                        _genesis_p67_out_buf = _genesis_p67_torch_alloc.empty(\n"
    "                            _genesis_p67_buf_key, dtype=query.dtype, device=query.device\n"
    "                        )\n"
    "                        self._genesis_p67_out_buffers[_genesis_p67_buf_key] = _genesis_p67_out_buf\n"
    "                    _genesis_p67_out = _genesis_p67_call(\n"
    "                        q=_genesis_p67_q,\n"
    "                        kv_cache=kv_cache,\n"
    "                        block_table=attn_metadata.block_table,\n"
    "                        seq_lens=attn_metadata.seq_lens,\n"
    "                        k_chunk=_genesis_p67_k_chunk,\n"
    "                        v_chunk=_genesis_p67_v_chunk,\n"
    "                        scale=self.scale,\n"
    "                        block_size=_genesis_p67_block_size,\n"
    "                        kps=_genesis_p67_kps,\n"
    "                        val_data_bytes=_genesis_p67_vdb,\n"
    "                        output=_genesis_p67_out_buf,\n"
    "                    )\n"
    "                    # DEBUG MODE v7.26: log stats + fall through to upstream.\n"
    "                    # [Genesis P67 H2 v7.62.6] DEBUG_COMPARE baked at module load.\n"
    "                    # If GENESIS_P67_DEBUG_COMPARE=1 was set at container start,\n"
    "                    # log P67 output statistics\n"
    "                    # for first 5 dispatches but DON'T return — let upstream\n"
    "                    # produce clean output. Allows direct correctness verification\n"
    "                    # without poisoning the engine.\n"
    f"                    _debug_compare = {_BAKED_DEBUG_COMPARE}\n"
    "                    # [Genesis P67 B1 fix v7.62.12] Telemetry .item()/.tolist() are\n"
    "                    # GPU->CPU syncs that break under FULL cudagraph capture (capture\n"
    "                    # gets invalidated). Gate on is_current_stream_capturing() so the\n"
    "                    # debug stats run ONLY in eager mode (warmup + non-captured paths).\n"
    "                    import torch as _genesis_p67_torch\n"
    "                    _genesis_p67_capturing = (\n"
    "                        _genesis_p67_torch.cuda.is_available()\n"
    "                        and _genesis_p67_torch.cuda.is_current_stream_capturing()\n"
    "                    )\n"
    "                    if self._genesis_p67_dispatch_count <= 5 and not _genesis_p67_capturing:\n"
    "                        _stats_out = _genesis_p67_out.float().detach()\n"
    "                        _amax = float(_stats_out.abs().max().item())\n"
    "                        _amean = float(_stats_out.abs().mean().item())\n"
    "                        _has_nan = bool(_genesis_p67_torch.isnan(_stats_out).any().item())\n"
    "                        _has_inf = bool(_genesis_p67_torch.isinf(_stats_out).any().item())\n"
    "                        _per_t_amax = _stats_out.abs().amax(dim=-1).amax(dim=-1).flatten().tolist()\n"
    "                        _genesis_p67_log.warning(\n"
    "                            'P67 dispatch #%d STATS: shape=%s amax=%.4f amean=%.4f nan=%s inf=%s per_t_amax=%s',\n"
    "                            self._genesis_p67_dispatch_count, tuple(_genesis_p67_out.shape),\n"
    "                            _amax, _amean, _has_nan, _has_inf, _per_t_amax,\n"
    "                        )\n"
    "                    if not _debug_compare:\n"
    "                        return _genesis_p67_out.view(N, Hq, D)\n"
    "                    # Else: fall through to upstream, log only.\n"
    "        except Exception as _genesis_p67_err:\n"
    "            import logging as _genesis_p67_logging\n"
    "            _genesis_p67_logging.getLogger('genesis.kernels.p67').warning(\n"
    "                'P67 dispatch failed (%s), falling through to upstream',\n"
    "                _genesis_p67_err, exc_info=True,\n"
    "            )\n"
    "\n"
    "        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).\n"
    "        # max_query_len == max_seq_len means no request has prior cached KV.\n"
    "        # Both are Python ints — no GPU sync.\n"
    "        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P67 turboquant_attn.py — multi-query kernel hook",
        target_file=str(target),
        marker=GENESIS_P67_MARKER,
        sub_patches=[
            TextPatch(
                name="p67_kernel_hook",
                anchor=P67_OLD,
                replacement=P67_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P67",
            "_genesis_p67_call",
            "TQ multi-query kernel",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P67 hook injection."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P67")
    log_decision("P67", decision, reason)
    if not decision:
        return "skipped", reason

    # 2026-04-27 v756 bisect SAFETY GATE: P67's `max_query_len > 1` dispatch
    # heuristic ALSO matches chunked-prefill batches (not just spec-verify
    # K+1). Without spec-decode, P67 misroutes prefill batches through the
    # multi-query kernel which assumes uniform K+1 layout per request,
    # causing scrambled output and downstream
    # `hidden_states[logits_indices]` overflow under sustained burst.
    # v756 reproducer 100% reliable; B5 confirmed P67=0 stable.
    # See docs/reference/V756_STABILITY_INVESTIGATION_20260427.md.
    # This safety gate refuses to apply P67 even when env flag is set if the
    # config lacks speculative_config — operator may not know this is unsafe.
    try:
        from vllm._genesis.config_detect import recommend
        cd_verdict, cd_reason = recommend("P67")
        if cd_verdict.startswith("skip"):
            return "skipped", (
                f"P67 SAFETY GATE — config_detect says {cd_verdict}: "
                f"{cd_reason} | env flag IGNORED to prevent v756-class "
                "IndexKernel overflow under chunked-prefill + sustained "
                "burst. To force-enable, you must also enable spec-decode "
                "(--speculative-config '{\"method\":\"...\"}')."
            )
    except Exception as e:
        log.warning("[P67] safety gate config_detect probe failed: %s", e)

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
                    "P67 likely already injected.",
                )
        if patcher.sub_patches[0].anchor not in content:
            return (
                "skipped",
                "required anchor (_prefill_attention top + fast-path check) not "
                "found — upstream drift; P67 cannot apply.",
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

    # Diagnostic info for log
    diag = ""
    try:
        from vllm._genesis.kernels.p67_multi_query_kernel import diagnostic_info
        diag = f" Diag: {diagnostic_info()}"
    except Exception:
        # Diagnostic helper is optional — apply already succeeded, log without diag
        pass

    return "applied", (
        "P67 hook injected at top of _prefill_attention. Multi-query continuation "
        "prefill batches (spec-verify K+1 with prior cached KV) will route to "
        "Genesis Triton kernel when GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1." + diag
    )
