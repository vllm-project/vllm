# SPDX-License-Identifier: Apache-2.0
"""P40 — GQA-grouped TurboQuant k8v4 decode stage1 kernel.

Ports the `_tq_grouped_decode_stage1` Triton kernel from upstream PR
vllm-project/vllm#40792 (OPEN as of 2026-04-24) so Genesis can harvest
the +10-27% decode throughput on Qwen3-32B-class GQA configurations
without waiting for merge.

Motivation (upstream PR body)
-----------------------------
The stock `_tq_decode_stage1` kernel (in
`vllm/v1/attention/ops/triton_turboquant_decode.py`) launches ONE CTA
per `(batch, head, kv_split)` — `Hq = 64` on Qwen3.6-35B-A3B means 64
CTAs per batch per split, with every CTA redundantly loading the same
KV tile because `Hq = KV_GROUP_SIZE × Hk`. The grouped kernel batches
up to `BLOCK_H = 16` Q heads that share one KV head, loading K/V
ONCE and computing QK/PV via `tl.dot` on float16 — roughly 4× fewer
KV loads, 2× arithmetic intensity on tensor cores.

PR author measured:
  - Qwen3-32B @ A100 PCIe: +27% decode tok/s (k8v4)
  - Qwen3-32B @ H100:      +16% decode tok/s (k8v4)

Our target (2×A5000 SM 8.6 Qwen3.6-35B-A3B-FP8 k8v4) should see a
similar directional win. `BLOCK_H=16` fits our per-rank `Hq/TP=64/2=32`
cleanly (2 head-groups per KV head). `BLOCK_KV=16` + `num_warps=4` +
`num_stages=2` is within the A5000 shared-memory budget
(~100 KB / SM available, kernel needs ≤ 64 KB).

Scope limitations
-----------------
Upstream kernel hard-codes `tl.static_assert(VQB == 4)` — only
applies to `turboquant_k8v4` preset. MSE-quantized key presets
(`turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`)
continue using the scalar `_tq_decode_stage1`. Our prod config runs
k8v4, so this constraint doesn't bite us.

Opt-in gate
-----------
Enabled via `GENESIS_ENABLE_P40=1`. OFF by default so the first
production deployment must explicitly benchmark correctness and
throughput before flipping on. Once we have GPU bench data confirming
+10% or better on our setup, we flip the default to on.

Correctness guardrails
----------------------
- `tl.static_assert(VQB == 4)` fires at compile-time if misused.
- Dispatcher only enters the grouped path when `kv_group_size > 1 and
  key_fp8` — matches upstream exactly; MSE paths retain scalar kernel.
- `torch.empty`-allocated scratch (mid_o, output, lse) matches stage2
  output layout byte-for-byte — no change to stage2 kernel or return
  path required.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.4 implementation (opt-in)
"""
from __future__ import annotations

import logging
import os


log = logging.getLogger("genesis.tq_grouped_decode")

_ENV_ENABLE = "GENESIS_ENABLE_P40"


def _read_env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


_ENABLED_AT_IMPORT: bool = _read_env_enabled()


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA + SM ≥ 8.0 (Ampere+) + opt-in env."""
    from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    if not _ENABLED_AT_IMPORT:
        return False
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def _build_grouped_kernel():
    """Define the Triton kernel lazily so import on CPU-only hosts works.

    Triton import is guarded — on hosts without CUDA the import fails;
    we catch and return None so the wiring layer can fall through to
    the upstream scalar kernel.
    """
    try:
        from vllm.triton_utils import tl, triton
    except Exception:
        try:
            import triton
            import triton.language as tl
        except Exception:
            return None

    @triton.jit
    def _tq_grouped_decode_stage1(
        Q_rot_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        Mid_o_ptr,
        stride_qb,
        stride_qh,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_bt_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        Q_HEAD_NUM: tl.constexpr,
        KPS: tl.constexpr,
        VQB: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        ATTN_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_H: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
    ):
        """GQA-grouped TQ decode stage1 for the FP8 key path (k8v4).

        Each CTA processes up to BLOCK_H Q heads that share one KV head,
        loading K/V once and computing scores via `tl.dot`.

        Constrained to VQB==4 (4-bit uniform values); MSE key presets
        keep the upstream scalar kernel.
        """
        bid = tl.program_id(0)
        head_group_id = tl.program_id(1)
        sid = tl.program_id(2)

        heads_per_kv_head: tl.constexpr = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
        kv_head = head_group_id // heads_per_kv_head
        group_idx = head_group_id % heads_per_kv_head
        cur_head = (
            kv_head * KV_GROUP_SIZE
            + group_idx * BLOCK_H
            + tl.arange(0, BLOCK_H)
        )
        mask_h = (cur_head < (kv_head + 1) * KV_GROUP_SIZE) & (
            cur_head < Q_HEAD_NUM
        )

        seq_len = tl.load(Seq_lens_ptr + bid)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_start >= split_end:
            out_base = (
                bid * stride_mid_b
                + cur_head * stride_mid_h
                + sid * stride_mid_s
            )
            tl.store(
                Mid_o_ptr + out_base + HEAD_DIM,
                float("-inf"),
                mask=mask_h,
            )
            return

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)

        q_base = (
            bid * stride_qb + cur_head[:, None] * stride_qh + d_offs[None, :]
        )
        q_rot = tl.load(
            Q_rot_ptr + q_base,
            mask=mask_h[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        m_prev = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        for start_n in range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end

            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=kv_mask, other=0,
            ).to(tl.int64)

            slot_bases = (
                block_nums * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_head, tl.int64) * stride_cache_head
            )

            # K dequant (FP8 only — enforced by dispatcher gate)
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

            scores = tl.dot(
                q_rot.to(tl.float16), tl.trans(k_float.to(tl.float16))
            )
            scores = (scores * ATTN_SCALE).to(tl.float32)
            scores = tl.where(
                mask_h[:, None] & kv_mask[None, :], scores, -float("inf"),
            )

            n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max[:, None])

            tl.static_assert(
                VQB == 4,
                "grouped kernel only supports 4-bit values",
            )
            val_bases = slot_bases + KPS

            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(
                KV_cache_ptr + sc_bases, mask=kv_mask, other=0,
            ).to(tl.uint16)
            sc_hi = tl.load(
                KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0,
            ).to(tl.uint16)
            v_scales = (
                (sc_lo | (sc_hi << 8))
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            zr_lo = tl.load(
                KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0,
            ).to(tl.uint16)
            zr_hi = tl.load(
                KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0,
            ).to(tl.uint16)
            v_zeros = (
                (zr_lo | (zr_hi << 8))
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

            acc = acc * re_scale[:, None] + tl.dot(
                p.to(tl.float16), values.to(tl.float16),
            ).to(tl.float32)
            l_prev = l_prev * re_scale + tl.sum(p, 1)
            m_prev = n_e_max

        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        out_base = (
            bid * stride_mid_b
            + cur_head[:, None] * stride_mid_h
            + sid * stride_mid_s
        )
        tl.store(
            Mid_o_ptr + out_base + d_offs[None, :],
            acc / safe_l[:, None],
            mask=mask_h[:, None] & d_mask[None, :],
        )
        lse = m_prev + tl.log(safe_l)
        tl.store(
            Mid_o_ptr
            + bid * stride_mid_b
            + cur_head * stride_mid_h
            + sid * stride_mid_s
            + HEAD_DIM,
            lse,
            mask=mask_h,
        )

    return _tq_grouped_decode_stage1


# Lazy accessor — built once per process on first use, cached.
_CACHED_KERNEL = None


def get_grouped_kernel():
    """Return the compiled grouped-decode Triton kernel (None on non-CUDA).

    Build is deferred to first call so CPU-only test environments don't
    fail at import time (Triton + CUDA only available on GPU hosts).
    """
    global _CACHED_KERNEL
    if _CACHED_KERNEL is None:
        _CACHED_KERNEL = _build_grouped_kernel()
    return _CACHED_KERNEL


def should_use_grouped_kernel(
    kv_group_size: int,
    key_fp8: bool,
    value_quant_bits: int,
) -> bool:
    """Dispatcher decision: route to grouped kernel iff all match.

    Mirrors upstream #40792 branch condition plus our env opt-in gate.
    Returns False on any upstream-incompatible config so caller falls
    back to the original scalar kernel (correctness preserved).
    """
    if not should_apply():
        return False
    if kv_group_size <= 1:
        return False
    if not key_fp8:
        return False
    # Grouped kernel has tl.static_assert(VQB == 4) — only k8v4 path.
    if value_quant_bits != 4:
        return False
    return True


# Launch-parameter constants — match upstream PR #40792 hard-coded values.
BLOCK_H = 16
BLOCK_KV = 16
NUM_WARPS = 4
NUM_STAGES = 2
