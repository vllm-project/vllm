# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL launcher for the optimized UltraQuant 4-bit D=256 decode kernel.

The production kernel uses scaled FP4×E4M3 QK MFMA, native V conversion,
query hoisting, in-kernel Walsh-Hadamard rotation for GQA-6/8/16, and strided
tile-group scheduling. These choices are fixed and are not environment-tunable.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)

_FLYDSL_AVAILABLE: bool | None = None
_ULTRAQUANT_MOD: Any = None
_FLYC: Any = None
_FX: Any = None
_TYPING_T: Any = None
_CC: Any = None
_IR: Any = None


def is_flydsl_available() -> bool:
    """Return whether the gfx950 UltraQuant FlyDSL kernel can be loaded."""
    global _FLYDSL_AVAILABLE, _ULTRAQUANT_MOD
    global _FLYC, _FX, _TYPING_T, _CC, _IR
    if _FLYDSL_AVAILABLE is not None:
        return _FLYDSL_AVAILABLE
    try:
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            _FLYDSL_AVAILABLE = False
            return False

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl.expr.typing import T

        from vllm.v1.attention.ops.flydsl_kernels import (
            ultraquant_decode_hd256 as ultraquant_mod,
        )

        _FLYC = flyc
        _FX = fx
        _TYPING_T = T
        _CC = CompilationContext
        _IR = ir
        _ULTRAQUANT_MOD = ultraquant_mod
        _FLYDSL_AVAILABLE = True
        logger.info_once("FlyDSL UltraQuant D=256 decode is available")
    except Exception as ex:  # noqa: BLE001
        _FLYDSL_AVAILABLE = False
        logger.warning_once(
            "FlyDSL UltraQuant decode is unavailable (%s); using Triton fallback.",
            ex,
        )
    return _FLYDSL_AVAILABLE


_ULTRAQUANT_FLYDSL_GQA = (6, 8, 16)


def is_flydsl_hd256_available(query_group_size: int) -> bool:
    return query_group_size in _ULTRAQUANT_FLYDSL_GQA and is_flydsl_available()


def ultraquant_flydsl_decode_eligible(
    *,
    head_size: int,
    num_kv_groups: int,
    has_sinks: bool,
    sliding_window: int | None,
    flydsl_loaded: bool,
) -> bool:
    """Return whether decode should use the gfx950 FlyDSL kernel.

    Production FlyDSL covers D=256 and GQA in {6, 8, 16}. Sinks and sliding
    window fall back to unified Triton. ``flydsl_loaded`` is the process-level
    compiler/hardware probe so this policy can be unit-tested without a GPU.
    """
    return (
        flydsl_loaded
        and head_size == 256
        and num_kv_groups in _ULTRAQUANT_FLYDSL_GQA
        and not has_sinks
        and not (sliding_window and sliding_window > 0)
    )


# -- Kernel module cache -------------------------------------------------------
# Compiling a FlyDSL kernel is expensive; cache by the constexprs that
# parameterize stride math: num_kv_heads, num_partitions, max_blocks_per_seq,
# scale, query_group_size, kv_block_size, padded_slot, hw_v_transpose,
# tile_groups_per_partition.
_KERN_CACHE: dict[tuple, Any] = {}

_LOG_INVOKED_ONCE: bool = False

# Per-device cached FP4 E2M1 value table (used as the kernel "centroids" LUT).
_FP4_LUT_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _fp4_value_lut(device: torch.device) -> torch.Tensor:
    """Return the fixed 16-entry FP4 E2M1 value table as fp32 [16] on device.

    Indexed by the stored 4-bit E2M1 code, so ``lut[nibble]`` is the dequant
    value directly (matches ``format.FP4_BITS_TO_VALUE``).
    """
    from vllm.v1.attention.ops.ultraquant.format import FP4_BITS_TO_VALUE

    key = (str(device), torch.float32)
    t = _FP4_LUT_CACHE.get(key)
    if t is None:
        t = torch.tensor(
            FP4_BITS_TO_VALUE, dtype=torch.float32, device=device
        ).contiguous()
        _FP4_LUT_CACHE[key] = t
    return t

    # Per-device cached qperm head-dim permutation for scaled QK MFMA.


# qperm[phys] = the natural head-dim placed at physical operand position phys.
# It realigns the fixed FP4xFP8 scaled-MFMA contraction so that, with K fed
# native-contiguous, Q's columns pair correctly. Folded into PiT on the host
# (PiT[:, qperm]) so the Q rotation matmul emits already-permuted q_rot at zero
# extra runtime cost. Derived + validated in tmp_qk_scaled_mfma.py (contigK).
_QPERM_CACHE: dict[str, torch.Tensor] = {}


def _kmap_phys(group: int, s: int) -> int:
    # Physical operand position of group's element s for the K=128 fp4/fp8
    # scaled MFMA (matches tmp_qk_scaled_mfma.py::_kmap).
    return 32 * (s >> 4) + 64 * (group % 2) + 16 * (group // 2) + (s & 15)


def _qperm_index(
    device: torch.device, D: int = 128, group_size: int = 32
) -> torch.Tensor:
    # _kmap_phys describes ONE K=128 contraction (4 groups of 32). A head dim
    # wider than 128 is covered by ceil(D/128) back-to-back MFMA issues, each
    # contracting its own independent 128-wide slice, so the same map repeats
    # per slice: qperm = [qperm128, qperm128 + 128, ...]. Applying _kmap_phys
    # with group >= 4 instead yields positions outside the slice and is not a
    # permutation at all.
    key = f"{device}:{D}:{group_size}"
    t = _QPERM_CACHE.get(key)
    if t is None:
        k_per_mfma = 128
        assert D % k_per_mfma == 0, (
            f"scaled QK MFMA needs D to be a multiple of {k_per_mfma}, got {D}"
        )
        groups_per_mfma = k_per_mfma // group_size
        qperm = [0] * D
        for half in range(D // k_per_mfma):
            base = half * k_per_mfma
            for g in range(groups_per_mfma):
                for s in range(group_size):
                    qperm[base + _kmap_phys(g, s)] = base + g * group_size + s
        assert sorted(qperm) == list(range(D)), "qperm is not a permutation"
        t = torch.tensor(qperm, dtype=torch.long, device=device).contiguous()
        _QPERM_CACHE[key] = t
    return t


def _detect_max_capture_B() -> int:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        candidates: list[int] = []
        sizes = cfg.compilation_config.cudagraph_capture_sizes
        if sizes:
            candidates.append(int(max(sizes)))
        sched = getattr(cfg, "scheduler_config", None)
        if sched is not None and getattr(sched, "max_num_seqs", None):
            candidates.append(int(sched.max_num_seqs))
        if candidates:
            return max(candidates)
    except Exception:  # noqa: BLE001
        pass
    return 512


class _SegmBufPool:
    """Single-bucket buffer pool for segm_out/segm_max/segm_sum/output + the
    pooled Q-rotation intermediates. Identical to the TQ pool except that
    ultraquant needs an extra ``q_fp8`` (float8_e4m3fn) slot for the Q precision
    haircut (``q_rot_fp32 -> fp8_e4m3 -> bf16``) so that no fresh allocation
    lands in the HIP graph memory pool post-capture.
    """

    __slots__ = ("_bufs", "_max_B")

    def __init__(self) -> None:
        self._bufs: dict[tuple, dict[str, torch.Tensor]] = {}
        self._max_B: int | None = None

    def get(
        self,
        B: int,
        Hk: int,
        Hq: int,
        num_partitions: int,
        QG: int,
        D: int,
        device: torch.device,
        q_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self._max_B is None:
            self._max_B = _detect_max_capture_B()
        B_bucket = max(self._max_B, int(B))
        key = (
            int(Hk),
            int(Hq),
            int(num_partitions),
            int(QG),
            int(D),
            str(device),
            q_dtype,
        )
        bufs = self._bufs.get(key)
        if bufs is None or bufs["segm_out"].shape[0] < B_bucket:
            if bufs is not None and bufs["segm_out"].shape[0] < B_bucket:
                logger.warning_once(
                    "FlyDSL UltraQuant buffer pool growing from %d "
                    "to %d (B=%d). If this happens AFTER cudagraph warmup the "
                    "captured graphs hold stale pointers and will GPU-fault.",
                    bufs["segm_out"].shape[0],
                    B_bucket,
                    B,
                )
            bufs = {
                "segm_out": torch.empty(
                    (B_bucket, Hk, num_partitions, QG, D),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "segm_max": torch.empty(
                    (B_bucket, Hk, num_partitions, QG),
                    dtype=torch.float32,
                    device=device,
                ),
                "segm_sum": torch.empty(
                    (B_bucket, Hk, num_partitions, QG),
                    dtype=torch.float32,
                    device=device,
                ),
                "output": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=q_dtype,
                    device=device,
                ),
                "q_rot": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=q_dtype,
                    device=device,
                ),
                "q_float": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=torch.float32,
                    device=device,
                ),
                "q_rot_fp32": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=torch.float32,
                    device=device,
                ),
                # ultraquant-specific: E4M3 haircut intermediate (pooled so the
                # cast does not allocate a fresh tensor post-capture).
                "q_fp8": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            }
            self._bufs[key] = bufs
            self._max_B = B_bucket
            logger.info_once(
                "FlyDSL UltraQuant buffer pool allocated "
                "shape=(Hk=%d, Hq=%d, P=%d, QG=%d, D=%d, dtype=%s) "
                "B_bucket=%d. VRAM = %.1f MiB / shape.",
                Hk,
                Hq,
                num_partitions,
                QG,
                D,
                q_dtype,
                B_bucket,
                sum(t.numel() * t.element_size() for t in bufs.values()) / (1 << 20),
            )
        return {
            "segm_out": bufs["segm_out"][:B],
            "segm_max": bufs["segm_max"][:B],
            "segm_sum": bufs["segm_sum"][:B],
            "output": bufs["output"][:B],
            "q_rot": bufs["q_rot"][:B],
            "q_float": bufs["q_float"][:B],
            "q_rot_fp32": bufs["q_rot_fp32"][:B],
            "q_fp8": bufs["q_fp8"][:B],
        }


_SEGM_POOL = _SegmBufPool()

_GET_KERNEL_STATS = {"hits": 0, "misses": 0, "build_total_s": 0.0}


def _get_kernel(
    num_kv_heads: int,
    num_partitions: int,
    max_blocks_per_seq: int,
    scale: float,
    query_group_size: int,
    kv_block_size: int,
    padded_slot: int,
    fuse_qrot: bool,
    num_seqs_hint: int,
    tile_groups_per_partition: int,
    stride_q_seq: int,
    stride_q_head: int,
):
    key = (
        num_kv_heads,
        num_partitions,
        max_blocks_per_seq,
        round(scale, 8),
        query_group_size,
        kv_block_size,
        padded_slot,
        fuse_qrot,
        tile_groups_per_partition,
        stride_q_seq,
        stride_q_head,
    )
    cached = _KERN_CACHE.get(key)
    if cached is not None:
        _GET_KERNEL_STATS["hits"] += 1
        return cached

    assert is_flydsl_available()
    import time

    build_start = time.perf_counter()
    kmod = _ULTRAQUANT_MOD
    assert kmod is not None
    kfn = kmod.build_ultraquant_decode_hd256_module(
        num_seqs=num_seqs_hint,
        num_kv_heads=num_kv_heads,
        num_partitions=num_partitions,
        padded_slot=padded_slot,
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=scale,
        query_group_size=query_group_size,
        kv_block_size=kv_block_size,
        tile_groups_per_partition=tile_groups_per_partition,
        fuse_qrot=fuse_qrot,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
    )
    allocator = kmod.allocator
    flyc = _FLYC
    fx = _FX
    T = _TYPING_T
    CompilationContext = _CC
    ir_mod = _IR

    @flyc.jit
    def _launch(
        out,
        es,
        ml,
        q,
        kvc,
        cents,
        bt,
        sl,
        gx: fx.Int32,  # type: ignore[name-defined]
        gy: fx.Int32,  # type: ignore[name-defined]
        gz: fx.Int32,  # type: ignore[name-defined]
        stream: fx.Stream,  # type: ignore[name-defined]
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir_mod.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        from flydsl.expr import arith

        grid_x = arith.index_cast(T.index, gx.ir_value())
        grid_y = arith.index_cast(T.index, gy.ir_value())
        grid_z = arith.index_cast(T.index, gz.ir_value())
        kfn(out, es, ml, q, kvc, cents, bt, sl).launch(
            grid=(grid_x, grid_y, grid_z),
            block=(kmod.BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    _KERN_CACHE[key] = _launch
    build_time = time.perf_counter() - build_start
    _GET_KERNEL_STATS["misses"] += 1
    _GET_KERNEL_STATS["build_total_s"] += build_time
    logger.info(
        "Built FlyDSL UltraQuant kernel #%d in %.2fs (total %.1fs), key=%s",
        _GET_KERNEL_STATS["misses"],
        build_time,
        _GET_KERNEL_STATS["build_total_s"],
        key,
    )
    return _launch


# -- Partition reducer ---------------------------------------------------------
from vllm.triton_utils import tl  # noqa: E402


@triton.jit
def _reduce_partitions(
    output_ptr,  # [N, Hq, D] in OUT_DTYPE
    segm_out_ptr,  # [N, Hk, P, QG, D] bf16
    segm_max_ptr,  # [N, Hk, P, QG] fp32
    segm_sum_ptr,  # [N, Hk, P, QG] fp32
    out_stride_n: tl.int64,
    out_stride_h: tl.int64,
    NUM_KV_HEADS: tl.constexpr,
    QG: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    hq = tl.program_id(1)
    kv_h = hq // QG
    qg = hq % QG
    # Splitting the head-dim across programs is what gives this kernel enough
    # parallelism to use the GPU: the reduction runs over PARTITIONS, so every
    # head-dim column is independent. A (B, Hq) grid is only 32 workgroups at
    # B=4 -- ~12% of a 256-CU part -- which is why it moved 8.4 MB at just
    # ~995 GB/s. The partition reduction itself is untouched (same values, same
    # order), so the result is bit-identical; only the columns each program
    # owns change. The [P] max/sum vectors get re-read per D-block, which is
    # 4 KB of duplicated traffic against a 262 KB tile.

    msum_base = n * (NUM_KV_HEADS * NUM_PARTS * QG) + kv_h * (NUM_PARTS * QG) + qg
    so_base = (
        n * (NUM_KV_HEADS * NUM_PARTS * QG * HEAD_SIZE)
        + kv_h * (NUM_PARTS * QG * HEAD_SIZE)
        + qg * HEAD_SIZE
    )

    p_off = tl.arange(0, NUM_PARTS)
    d_off = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    m_idx = msum_base + p_off * QG
    seg_max = tl.load(segm_max_ptr + m_idx)
    seg_sum = tl.load(segm_sum_ptr + m_idx)

    valid = seg_max > float("-inf")
    overall_max = tl.max(tl.where(valid, seg_max, float("-inf")))

    rescale = tl.where(valid, tl.exp(seg_max - overall_max), 0.0)
    seg_sum_rescaled = seg_sum * rescale
    overall_sum = tl.sum(seg_sum_rescaled)

    so_idx = so_base + p_off[:, None] * (QG * HEAD_SIZE) + d_off[None, :]
    seg_out = tl.load(segm_out_ptr + so_idx).to(tl.float32)
    weighted = seg_out * seg_sum_rescaled[:, None]
    acc_sum = tl.sum(weighted, axis=0)
    acc = tl.where(overall_sum > 0.0, acc_sum / overall_sum, 0.0)

    out_off = n * out_stride_n + hq * out_stride_h + d_off
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty))


# -- Fused Q rotation + E4M3 haircut (Triton) ---------------------------------
# Replaces the 4-op torch prologue (copy bf16->fp32, mm by PiT, copy fp32->e4m3,
# copy e4m3->bf16) with ONE kernel. The win is kernel COUNT, not bytes: each of
# those small ops costs ~2-4us of fixed GPU-side cost here largely independent of
# element count (verified: sweeping the pool's B_bucket 512->1, i.e. 512x fewer
# elements in the copies, moved the floor 0%). Under CUDA graphs the host launch
# cost is already gone, so only per-kernel GPU cost remains and collapsing 4
# kernels into 1 is the lever that is left.
@triton.jit
def _fused_q_rot_haircut(
    q_ptr,  # [B, Hq, D] query dtype (bf16/fp16), strided
    pit_ptr,  # [D, D] fp32 contiguous (already qperm-permuted)
    out_ptr,  # [B_bucket, Hq, D] out dtype, contiguous
    M,  # B * Hq (runtime)
    q_stride_n: tl.int64,
    q_stride_h: tl.int64,
    HQ: tl.constexpr,
    D: tl.constexpr,
    HAIRCUT: tl.constexpr,  # round through E4M3 (fp8-consuming QK MFMA paths)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < M
    n = m_off // HQ
    h = m_off % HQ

    # Splitting the OUTPUT columns across programs (not just the rows) is what
    # gives this kernel parallelism at small batch: M = B*Hq is only 8 at B=1,
    # so a row-only grid launches ONE workgroup and serializes the whole
    # 256x256 rotation on a single CU. The K reduction is untouched by the
    # split, so the result stays bit-identical.
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        q_idx = n[:, None] * q_stride_n + h[:, None] * q_stride_h + k_off[None, :]
        q_blk = tl.load(q_ptr + q_idx, mask=m_mask[:, None], other=0.0)
        p_blk = tl.load(pit_ptr + k_off[:, None] * D + d_off[None, :])
        acc = tl.dot(q_blk.to(tl.float32), p_blk, acc)

    if HAIRCUT:
        # fp32 -> E4M3 -> fp32. Mirrors the torch path's
        # q_rot_fp32 -> float8_e4m3fn -> bf16 round trip (e4m3 is a subset of
        # bf16, so the second hop is exact and folds away here).
        acc = acc.to(tl.float8e4nv).to(tl.float32)

    out_idx = m_off[:, None] * D + d_off[None, :]
    tl.store(out_ptr + out_idx, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


# Prologue Q-rotation fusion (fallback for non-hd256 / in-kernel-WHT-off paths).
# Cleared the accuracy ladder (offline parity -> autoregressive drift -> paired
# GSM8K); pinned ON as part of the validated production config.
_FUSE_Q_ROT = True

# Stronger form: do the rotation INSIDE the decode kernel as a Walsh-Hadamard
# butterfly and drop the prologue dispatch entirely. The prologue is ~7us at
# B=4 and that is almost all fixed per-dispatch cost (sweeping its grid 4x moves
# it 0%), so deleting the launch -- not optimizing it -- is the win.
# Default ON: verified bit-exact (torch.equal) against the prologue path across
# B={1,2,3,4,8,16} x seq={4k,8k,16k,32k,50021,75k}, including non-power-of-2
# batch and seq.
_FUSE_QROT_INKERNEL = True


def _run_fused_q_rot(query, PiT_used, out, B, Hq, D, haircut):
    """Launch the fused prologue. ``out`` is the pooled [B_bucket, Hq, D] buf."""
    M = B * Hq
    BLOCK_M = 16
    BLOCK_K = 32
    BLOCK_D = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _fused_q_rot_haircut[grid](
        query,
        PiT_used,
        out,
        M,
        query.stride(0),
        query.stride(1),
        HQ=int(Hq),
        D=int(D),
        HAIRCUT=bool(haircut),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )


# -- Public launcher ----------------------------------------------------------
def flydsl_ultraquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    PiT: torch.Tensor | None = None,
    max_seq_len: int = 0,
    output_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the optimized FlyDSL UltraQuant D=256 decode."""
    if not is_flydsl_available():
        raise RuntimeError(
            "FlyDSL UltraQuant decode requires gfx950 and importable FlyDSL."
        )
    if sinks is not None:
        raise NotImplementedError("FlyDSL UltraQuant decode does not support sinks")

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = int(kv_cache.shape[3])
    QG = Hq // Hk
    assert D == 256, f"UltraQuant FlyDSL requires head_dim=256, got {D}"
    assert block_size in (16, 32, 64, 128, 256), (
        f"UltraQuant FlyDSL supports kv_block_size 16/32/64/128/256, got {block_size}"
    )
    assert QG in (6, 8, 16), (
        f"UltraQuant FlyDSL supports GQA factor 6, 8 or 16, got {QG}"
    )
    assert is_flydsl_hd256_available(QG)

    device = query.device

    # ---- PiT (Hadamard) — cache the contiguous fp32 form on the layer -----
    PiT_f32: torch.Tensor
    if buf_holder is not None:
        PiT_f32 = getattr(buf_holder, "_ultraquant_PiT_f32", None)
        if PiT_f32 is None:
            assert PiT is not None, (
                "UltraQuant launcher requires PiT (the Hadamard rotation)"
            )
            PiT_f32 = PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
            if not PiT_f32.is_contiguous():
                PiT_f32 = PiT_f32.contiguous()
            buf_holder._ultraquant_PiT_f32 = PiT_f32
    else:
        assert PiT is not None, (
            "UltraQuant launcher requires PiT (the Hadamard rotation)"
        )
        PiT_f32 = PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
        if not PiT_f32.is_contiguous():
            PiT_f32 = PiT_f32.contiguous()

    centroids_c = _fp4_value_lut(device)

    # ---- Partition count (FA2 split-KV) -----------------------------------
    kv_compute_block = _ULTRAQUANT_MOD.KV_COMPUTE_BLOCK
    worst_case_max_seq_len = int(block_table.shape[1]) * int(block_size)
    if max_seq_len <= 0:
        max_seq_len = worst_case_max_seq_len
    sizing_max_seq_len = worst_case_max_seq_len

    # Split-KV parallelism cap. The old default (32) silently serialized long
    # contexts: once required_num_partitions exceeds the cap, TGPP grows and each
    # workgroup walks MORE tile-groups instead of the grid getting wider. The
    # decode kernel is latency-bound at ~1-2 waves/CU (grid = B*Hk*P workgroups
    # of a single wavefront), so that serialization dominated long-context decode.
    # Raising the cap to 256 keeps TGPP ~1 and measured (B=16, rocprof, per step
    # decode+reduce+store): 16k 82.7->40.0us, 32k 152.0->55.1us, 64k 339.4->95.7us,
    # 131k 671.3->179.0us (2.1x - 3.8x, growing with context). 512 regresses
    # because _reduce_partitions cost grows with partition count.
    # Cost: the segm pool scales with P (~8x vs P=32).
    # BATCH-ADAPTIVE split-KV cap (low-concurrency occupancy fix).
    # rocprof (B=4, 75k): the decode kernel is occupancy/latency-bound, not
    # compute-bound -- OccupancyPercent 5.8%, MfmaUtil 4.5%, VALUBusy 14.5%
    # (stalled ~85% on HBM), VALUUtilization 90% (no divergence). The grid is
    # B*Hk*num_partitions single-wavefront workgroups; at B=4/Hk=1/P=256 that's
    # only 1024 wg = 4/CU, too few to hide memory latency. Splitting MORE at low
    # batch adds resident wavefronts and cuts ITL; at high batch the grid is
    # already full so extra partitions only inflate _reduce_partitions + empty-
    # partition overhead. Measured end-to-end (graphs on, TP=8, uniform 75k,
    # median ITL vs kv8): cap512 vs cap256 -> C4 +9.0%->+4.9%, C8 +6.6%->+2.5%
    # (better), C16 +2.8%->+4.3% (worse). So use 512 at low batch, 256 at high.
    # This is only SAFE because the build-local-allocator fix in
    # ultraquant_decode_hd256.py lets the two num_partitions variants (256 & 512)
    # coexist across CUDA-graph batch buckets without the @flyc.jit
    # "global 'allocator' changed since first compile" drift crash.
    MAX_PARTITIONS = 512 if B <= 8 else 256
    required_num_partitions = (
        sizing_max_seq_len + kv_compute_block - 1
    ) // kv_compute_block
    parallelism_floor = min(MAX_PARTITIONS, max(1, max_num_kv_splits))
    num_partitions_actual = max(
        parallelism_floor,
        min(MAX_PARTITIONS, required_num_partitions),
    )
    # Cap partitions by how much work there actually IS to split. Nothing above
    # forces a partition to hold a useful number of tokens -- max_num_kv_splits
    # sets a parallelism FLOOR -- so a short context gets split into partitions
    # of a handful of tokens each, where every partition is almost entirely
    # fixed cost (and each one still writes a full QG x D partial for the
    # reduce to read back). Measured sweet spot is ~130-147 tokens/partition,
    # roughly flat in between; next_power_of_2 below rounds this target up, so
    # the effective range lands at ~96-192 tokens. At B=4 this is worth 27.9 ->
    # 24.4 us at seq=2048 and 27.0 -> 24.8 us at seq=16384, and is inert once
    # the context is long enough to want every partition (seq >= ~64k).
    _tok_per_part = 192
    if _tok_per_part > 0:
        _useful_parts = max(
            1, (int(sizing_max_seq_len) + _tok_per_part - 1) // _tok_per_part
        )
        num_partitions_actual = min(num_partitions_actual, _useful_parts)
    num_partitions = max(2, triton.next_power_of_2(num_partitions_actual))
    _tgpp_required = max(
        1,
        (required_num_partitions + num_partitions - 1) // num_partitions,
    )
    tile_groups_per_partition = int(triton.next_power_of_2(_tgpp_required))

    # ---- Pooled buffers ---------------------------------------------------
    pool_bufs = _SEGM_POOL.get(
        B,
        Hk,
        Hq,
        num_partitions,
        QG,
        D,
        device,
        query.dtype,
    )
    segm_out = pool_bufs["segm_out"]
    segm_max = pool_bufs["segm_max"]
    segm_sum = pool_bufs["segm_sum"]
    if output_buf is None:
        output = pool_bufs["output"]
    else:
        output = output_buf[:B] if output_buf.shape[0] != B else output_buf

    # Fold the scaled-MFMA head-dim permutation into PiT's columns so the
    # WHT / prologue produces native operand order. Always apply the E4M3
    # haircut: scaled QK consumes Q as E4M3.
    PiT_used = PiT_f32
    qperm_idx = _qperm_index(device, D=D)
    if buf_holder is not None:
        PiT_used = getattr(buf_holder, "_ultraquant_PiT_perm_f32", None)
        if PiT_used is None:
            PiT_used = PiT_f32.index_select(1, qperm_idx).contiguous()
            buf_holder._ultraquant_PiT_perm_f32 = PiT_used
    else:
        PiT_used = PiT_f32.index_select(1, qperm_idx).contiguous()
    _q_float = pool_bufs["q_float"]
    _q_rot_f32 = pool_bufs["q_rot_fp32"]
    _q_fp8 = pool_bufs["q_fp8"]
    _q_rot_out = pool_bufs["q_rot"]

    _inkernel_qrot = (
        _FUSE_QROT_INKERNEL
        and int(D) == 256
        and QG in _ULTRAQUANT_FLYDSL_GQA
        and query.dim() == 3
        and query.stride(2) == 1
        and query.stride(1) == int(D)
    )
    if _inkernel_qrot:
        q_for_kernel = query
    # The fused kernel indexes the head-dim contiguously; fall back if not.
    elif _FUSE_Q_ROT and query.stride(-1) == 1:
        # One kernel for the whole prologue instead of four ops.
        _run_fused_q_rot(query, PiT_used, _q_rot_out, B, Hq, D, haircut=True)
    else:
        _q_float.copy_(query)
        torch.mm(
            _q_float.view(B * Hq, D),
            PiT_used,
            out=_q_rot_f32.view(B * Hq, D),
        )
        _q_fp8.copy_(_q_rot_f32)  # fp32 -> e4m3
        _q_rot_out.copy_(_q_fp8)  # e4m3 -> bf16
    if not _inkernel_qrot:
        q_for_kernel = _q_rot_out

    # ---- FlyDSL kernel launch --------------------------------------------
    max_bps = int(block_table.shape[1])
    launch = _get_kernel(
        Hk,
        num_partitions,
        max_bps,
        scale,
        QG,
        block_size,
        padded_slot,
        fuse_qrot=_inkernel_qrot,
        num_seqs_hint=int(B),
        tile_groups_per_partition=int(tile_groups_per_partition),
        stride_q_seq=int(q_for_kernel.stride(0)),
        stride_q_head=int(q_for_kernel.stride(1)),
    )
    global _LOG_INVOKED_ONCE
    if not _LOG_INVOKED_ONCE:
        _LOG_INVOKED_ONCE = True
        logger.info(
            "FlyDSL UltraQuant launcher invoked: B=%d Hk=%d Hq=%d D=%d QG=%d "
            "num_partitions=%d (actual=%d, cap=%d) TGPP=%d max_bps=%d "
            "block_size=%d padded_slot=%d max_seq_len=%d "
            "(coverage=%d tokens, worst_case=%d tokens)",
            B,
            Hk,
            Hq,
            D,
            QG,
            num_partitions,
            num_partitions_actual,
            MAX_PARTITIONS,
            tile_groups_per_partition,
            max_bps,
            int(block_size),
            padded_slot,
            int(max_seq_len),
            num_partitions * tile_groups_per_partition * kv_compute_block,
            worst_case_max_seq_len,
        )
    launch(
        segm_out,
        segm_sum,
        segm_max,
        q_for_kernel,
        kv_cache,
        centroids_c,
        block_table,
        seq_lens,
        B,
        Hk,
        num_partitions,
        torch.cuda.current_stream(),
    )

    # ---- Reduce partitions -> [B, Hq, D] ---------------------------------
    # 64 splits the head-dim 4 ways, taking the grid from 32 to 128 workgroups.
    # Bit-identical by construction (the partition reduction is per-column).
    _red_block_d = min(int(D), 64)
    _reduce_partitions[(B, Hq, triton.cdiv(int(D), _red_block_d))](
        output_ptr=output,
        segm_out_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_sum_ptr=segm_sum,
        out_stride_n=output.stride(0),
        out_stride_h=output.stride(1),
        NUM_KV_HEADS=Hk,
        QG=QG,
        NUM_PARTS=num_partitions,
        HEAD_SIZE=D,
        BLOCK_D=_red_block_d,
    )
    return output
