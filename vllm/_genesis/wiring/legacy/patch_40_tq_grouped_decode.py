# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 40 — TurboQuant GQA-grouped decode stage1 dispatch.

Strategy
--------
Upstream PR #40792 modifies `triton_turboquant_decode_attention` to
branch between the scalar `_tq_decode_stage1` kernel (used pre-PR and
for all MSE-quantized key presets) and the new `_tq_grouped_decode_stage1`
kernel (k8v4 + GQA only).

We monkey-patch the module-level
`vllm.v1.attention.ops.triton_turboquant_decode.triton_turboquant_decode_attention`
with a wrapper that:
  1. Checks if the grouped path is eligible via
     `FlaKktBufferManager.should_use_grouped_kernel` (dispatcher).
  2. If yes → builds the grouped kernel via
     `tq_grouped_decode.get_grouped_kernel()` and launches with
     upstream-matching grid / constants.
  3. Else → delegates to original `triton_turboquant_decode_attention`
     (preserves upstream correctness on MSE presets, non-GQA, and on
     the no-P40-env deployment).

Opt-in gate
-----------
Active only when `GENESIS_ENABLE_P40=1` in the env (resolved at
`tq_grouped_decode` module import). OFF by default so operators
explicitly enable after a per-GPU correctness + throughput bench.

Upstream drift self-retirement
------------------------------
When #40792 merges into the vLLM release we integrate against, the
upstream `triton_turboquant_decode_attention` will already contain the
dispatch. At that point our wrapper becomes redundant — we detect by
checking if the target module exports `_tq_grouped_decode_stage1`, and
skip apply() with a `self-retired (upstream PR #40792 landed)`
reason. Our drift check verifies by symbol existence, not text content.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.4 implementation (opt-in)
"""
from __future__ import annotations

import logging
from typing import Any

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least

log = logging.getLogger("genesis.wiring.p40_tq_grouped_decode")

_GENESIS_P40_MARKER_ATTR = "_genesis_p40_wrapped"

_MODULE_PATH = "vllm.v1.attention.ops.triton_turboquant_decode"
_FN_NAME = "triton_turboquant_decode_attention"
# When this symbol appears on the upstream module, their PR #40792 has
# merged and we self-retire.
_UPSTREAM_DRIFT_SYMBOL = "_tq_grouped_decode_stage1"


def should_apply() -> bool:
    """Platform + env gate. Must mirror `tq_grouped_decode.should_apply`."""
    from vllm._genesis.kernels.tq_grouped_decode import should_apply as k_should
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return k_should()


def _import_target() -> tuple[Any, Any] | None:
    """Return (module, original_fn) or None on import failure."""
    import importlib
    try:
        mod = importlib.import_module(_MODULE_PATH)
    except ImportError:
        return None
    except Exception as e:
        log.warning("[Genesis P40] import %s: %s", _MODULE_PATH, e)
        return None
    fn = getattr(mod, _FN_NAME, None)
    if fn is None:
        return None
    return mod, fn


def apply() -> tuple[str, str]:
    """Rebind `triton_turboquant_decode_attention` with P40 dispatcher wrapper.

    Never raises. Returns (status, reason).

    v7.12: consults `config_detect.should_apply("P40")` first.
    Skipped automatically if upstream PR #40792 active.
    """
    try:
        from vllm._genesis import config_detect
        ok, reason = config_detect.should_apply("P40")
        if not ok:
            return "skipped", reason
    except Exception:
        pass  # config_detect optional; fall through to legacy logic

    if not should_apply():
        if not is_nvidia_cuda():
            return "skipped", "platform: NVIDIA SM 8.0+ required"
        if not is_sm_at_least(8, 0):
            return "skipped", "SM < 8.0 — need Ampere+"
        # If platform OK but env off, that's the normal opt-in case.
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_P40=1 to enable GQA-grouped "
            "decode stage1 (port of upstream PR #40792)"
        )

    target = _import_target()
    if target is None:
        return "skipped", (
            f"target module {_MODULE_PATH!r} or symbol {_FN_NAME!r} "
            "not available — TurboQuant backend not compiled in"
        )
    mod, original = target

    # Self-retirement on upstream merge.
    if hasattr(mod, _UPSTREAM_DRIFT_SYMBOL):
        return "skipped", (
            f"upstream drift: {_UPSTREAM_DRIFT_SYMBOL!r} already present "
            "in target module → PR #40792 has merged, P40 self-retired"
        )

    if getattr(original, _GENESIS_P40_MARKER_ATTR, False):
        return "applied", "already wrapped (idempotent)"

    # Pre-import our kernel builder to surface compile errors EARLY
    # (rather than on first request).
    try:
        from vllm._genesis.kernels.tq_grouped_decode import (
            get_grouped_kernel,
            should_use_grouped_kernel,
            BLOCK_H, BLOCK_KV, NUM_WARPS, NUM_STAGES,
        )
    except Exception as e:
        return "failed", f"kernel import failed: {e}"

    def _genesis_grouped_decode_wrapper(
        query,
        kv_cache,
        block_table,
        seq_lens,
        Pi,
        centroids,
        scale,
        mse_bits,
        key_packed_size,
        value_quant_bits,
        key_fp8: bool = False,
        norm_correction: bool = False,
        PiT=None,
        mid_o_buf=None,
        output_buf=None,
        lse_buf=None,
        buf_holder=None,
        max_num_kv_splits: int = 32,
    ):
        """P40 dispatcher: grouped kernel for k8v4+GQA, scalar for rest."""
        import torch
        import triton

        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        kv_group_size = Hq // Hk

        if not should_use_grouped_kernel(
            kv_group_size=kv_group_size,
            key_fp8=key_fp8,
            value_quant_bits=value_quant_bits,
        ):
            # Fall through to original upstream kernel — untouched.
            return original(
                query=query, kv_cache=kv_cache, block_table=block_table,
                seq_lens=seq_lens, Pi=Pi, centroids=centroids,
                scale=scale, mse_bits=mse_bits,
                key_packed_size=key_packed_size,
                value_quant_bits=value_quant_bits, key_fp8=key_fp8,
                norm_correction=norm_correction, PiT=PiT,
                mid_o_buf=mid_o_buf, output_buf=output_buf, lse_buf=lse_buf,
                buf_holder=buf_holder,
                max_num_kv_splits=max_num_kv_splits,
            )

        # Grouped path — use upstream-equivalent prep + our kernel
        grouped_kernel = get_grouped_kernel()
        if grouped_kernel is None:
            # Triton unavailable — fall back to upstream
            return original(
                query=query, kv_cache=kv_cache, block_table=block_table,
                seq_lens=seq_lens, Pi=Pi, centroids=centroids,
                scale=scale, mse_bits=mse_bits,
                key_packed_size=key_packed_size,
                value_quant_bits=value_quant_bits, key_fp8=key_fp8,
                norm_correction=norm_correction, PiT=PiT,
                mid_o_buf=mid_o_buf, output_buf=output_buf, lse_buf=lse_buf,
                buf_holder=buf_holder,
                max_num_kv_splits=max_num_kv_splits,
            )

        device = query.device
        block_size = kv_cache.shape[1]
        cfg = mod._get_layout(
            D, mse_bits, value_quant_bits, key_packed_size,
        )

        # key_fp8 path: pass query through — kernel casts inline.
        q_rot = query.contiguous()

        NUM_KV_SPLITS = max_num_kv_splits

        # v7.48 (2026-04-27): fallback path now uses shared singleton via
        # GenesisPreallocBuffer instead of per-call torch.empty + per-instance
        # buf_holder attach. Eliminates allocator churn when upstream's
        # buf_holder mechanism doesn't pre-attach (e.g. spec-decode verify
        # path where layer's mid_o_buf is None on first call). Toggle via
        # GENESIS_BUFFER_MODE=per_layer to revert to legacy.
        from vllm._genesis.buffer_mode import buffer_mode_for
        _p40_mode = buffer_mode_for("P40")

        if (
            mid_o_buf is not None
            and mid_o_buf.shape[0] >= B
            and mid_o_buf.shape[2] >= NUM_KV_SPLITS
        ):
            mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
        elif _p40_mode == "shared":
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            # MAX-size singleton: pre-allocate worst-case (max_num_seqs × Hq
            # × max_num_kv_splits × (D+1)) once, reuse forever via slicing.
            # Hq + D are model-fixed, so namespace by them only.
            ns = f"p40_mid_o|Hq={Hq}|D={D}|splits={NUM_KV_SPLITS}|fp32"
            max_B = max(B, 8)  # 4× headroom over our max_num_seqs=2
            shape = (max_B, Hq, NUM_KV_SPLITS, D + 1)
            buf = GPB.get_or_create(ns, shape, torch.float32, device, zero_init=False)
            mid_o = buf[:B, :Hq, :NUM_KV_SPLITS, :]
        else:
            # Legacy per-call alloc + per-instance attach (rollback path).
            mid_o = torch.empty(
                B, Hq, NUM_KV_SPLITS, D + 1,
                dtype=torch.float32, device=device,
            )
            if buf_holder is not None:
                buf_holder._tq_mid_o_buf = mid_o

        fp8_e4b15 = mod._use_fp8_e4b15(device.index or 0)
        heads_per_kv_head = triton.cdiv(kv_group_size, BLOCK_H)
        head_groups = Hk * heads_per_kv_head

        grid = (B, head_groups, NUM_KV_SPLITS)
        grouped_kernel[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            Q_HEAD_NUM=Hq,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            BLOCK_H=BLOCK_H,
            FP8_E4B15=fp8_e4b15,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Stage 2 — reduce across KV splits. Reuse upstream's stage2
        # directly (unchanged by PR #40792).
        # v7.48: same shared-singleton pattern as mid_o above.
        if output_buf is not None and output_buf.shape[0] >= B:
            output = output_buf[:B, :Hq, :D]
        elif _p40_mode == "shared":
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            ns = f"p40_output|Hq={Hq}|D={D}|fp32"
            max_B = max(B, 8)
            shape = (max_B, Hq, D)
            buf = GPB.get_or_create(ns, shape, torch.float32, device, zero_init=False)
            output = buf[:B, :Hq, :D]
        else:
            output = torch.empty(
                B, Hq, D, dtype=torch.float32, device=device,
            )
            if buf_holder is not None:
                buf_holder._tq_output_buf = output

        if lse_buf is not None and lse_buf.shape[0] >= B:
            lse = lse_buf[:B, :Hq]
        elif _p40_mode == "shared":
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            ns = f"p40_lse|Hq={Hq}|fp32"
            max_B = max(B, 8)
            shape = (max_B, Hq)
            buf = GPB.get_or_create(ns, shape, torch.float32, device, zero_init=False)
            lse = buf[:B, :Hq]
        else:
            lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_lse_buf = lse

        stage2 = mod._fwd_kernel_stage2
        grid2 = (B, Hq)
        stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=cfg["BLOCK_D"],
            Lv=D,
            OUTPUT_FP16=1 if query.dtype == torch.float16 else 0,
            num_warps=4,
            num_stages=2,
        )

        return output.to(query.dtype)

    # Stamp marker + preserve original for revert() / fallback.
    setattr(
        _genesis_grouped_decode_wrapper,
        _GENESIS_P40_MARKER_ATTR, True,
    )
    if not getattr(
        _genesis_grouped_decode_wrapper, "_genesis_p40_original", None,
    ):
        setattr(
            _genesis_grouped_decode_wrapper,
            "_genesis_p40_original", original,
        )

    setattr(mod, _FN_NAME, _genesis_grouped_decode_wrapper)

    log.info(
        "[Genesis P40] rebound %s.%s (opt-in GQA-grouped decode stage1 "
        "for k8v4; fallback to scalar for MSE presets)",
        _MODULE_PATH, _FN_NAME,
    )
    return "applied", "module-level fn wrapped (dispatcher to grouped kernel)"


def is_applied() -> bool:
    target = _import_target()
    if target is None:
        return False
    _mod, fn = target
    return getattr(fn, _GENESIS_P40_MARKER_ATTR, False)


def revert() -> bool:
    target = _import_target()
    if target is None:
        return False
    mod, fn = target
    if not getattr(fn, _GENESIS_P40_MARKER_ATTR, False):
        return False
    original = getattr(fn, "_genesis_p40_original", None)
    if original is None:
        return False
    setattr(mod, _FN_NAME, original)
    return True
