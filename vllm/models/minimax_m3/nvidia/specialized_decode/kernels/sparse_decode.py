# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sparse_decode stage: FP8-KV sparse decode over the 16 pages `page_topk`
selected, one cluster of CL CTAs per (row, KV head), MXFP8 output for the
o_proj GEMV.

CuTe DSL kernel (`sparse_decode_kernel.py`, fp16-P online softmax). One kernel is
compiled per (cluster width,
Q heads per KV head, pre-wait order, min blocks, PDL) at build.

Residency: the kernel only waits inside its own cluster (DSMEM mbarrier), so
correctness needs one cluster to be schedulable. `build()` probes
`cuOccupancyMaxActiveClusters` of the compiled kernel for every cluster
width (16, 8) and picks the width per row count
(`geometry.sparse_cluster_size`, recorded in `cluster_sizes`): up to
`one_wave_rows` rows (the decode chain was tuned for 8) the grid must be
co-resident (one wave), 16 CTAs per (row, KV head) narrowed to 8 when 16 does
not fit, else `build()` fails closed; larger grids (up to `max_rows`, 64)
only need to be schedulable and take the width with the fewest waves
(`multi_wave_rows` lists those that still need more than one).
`strict_one_wave=False` drops the one-wave requirement.
The kernel itself takes up to 128 rows (one MXFP8 scale tile).
"""

from __future__ import annotations

import threading
from typing import Any

from . import _arch, _dtypes, geometry

HEADS_PER_KV = 16
# Q heads per KV head the kernel is built for: TP4 16, TP8 8 (the MMA's 16 rows padded;
# the grid, one cluster per (row, KV head), is TP4's, so the cluster widths and one-wave
# rules are too).
HEADS_PER_KV_BUILDS = (16, 8)
HEAD_DIM = 128
PAGE = 128
TOPK = 16
# Rows of the largest launch (the executor's largest step).
MAX_ROWS = 64
# Rows up to which the grid must fit in one wave of clusters.
ONE_WAVE_ROWS = 8
# CTAs per SM the kernel's register allocation targets (``__launch_bounds__`` min
# blocks; instantiated: 1, 2, 3). 1 is the default build. The 3 build holds more
# clusters at once but is slower per CTA, so it only runs the row counts it takes from
# two waves to one (from the occupancy the build probes).
MIN_BLOCKS = 1
ALT_MIN_BLOCKS = 3
# Residency cap (CTAs per SM, by padding shared memory) for the row counts that take
# more than one wave either way: with fewer CTAs resident the two waves are more
# balanced, which measured faster there (only where the cap adds no wave; the build
# checks). 0 disables.
CAP_CTAS = 2
# Launches above this many rows read lens (and q, ``q_prewait``) before the PDL grid
# wait, which shortens the decode chain at long context. Up to 8 rows (one request)
# every input is read after the wait: the early reads did not help there.
PREWAIT_MIN_ROWS = 9


class CuteDslUnavailable(RuntimeError):
    pass


def _kernel():
    """The CuTe DSL kernel module (imports cutlass)."""
    try:
        from . import sparse_decode_kernel
    except ImportError as e:  # pragma: no cover - depends on the install
        raise CuteDslUnavailable(
            f"sparse_decode needs the CuTe DSL (nvidia-cutlass-dsl): {e}"
        ) from e
    return sparse_decode_kernel


def _ext():
    """Load-time check (the kernel registry calls it before building): the CuTe
    DSL kernel module imports. Kernels compile in `build()`."""
    return _kernel()


_lock = threading.Lock()


def _dump_dir() -> str:
    """Where the DSL writes the kept cubins (read back for the occupancy probe): a
    per-process temp dir."""
    import tempfile

    global _DUMP
    if _DUMP is None:
        _DUMP = tempfile.mkdtemp(prefix="minimax_m3_sd_sparse_decode_")
    return _DUMP


_DUMP: str | None = None
_compiled: dict[tuple, Any] = {}


def _compile(cl: int, hq: int, pre: bool, minb: int, pdl: bool, cap: int = 0):
    """The compiled kernel of one build (cached per process); outside capture. `cap` >
    0 pads the shared memory
    so that at most `cap` CTAs are resident per SM."""
    key = (int(cl), int(hq), bool(pre), int(minb), bool(pdl), int(cap))
    fn = _compiled.get(key)
    if fn is not None:
        return fn
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"sparse_decode: kernel {key} must be compiled outside CUDA-graph capture "
            "(build())"
        )
    k = _kernel()
    i64, i32, f32 = cutlass.Int64, cutlass.Int32, cutlass.Float32
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute.compile(
                k.sparse_decode_launch,
                *([i64(0)] * 16),
                f32(0.0),
                f32(0.0),
                *([i64(0)] * 6),
                i32(0),
                i32(0),
                i32(0),
                i32(1),
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
                stream,
                key[5],
                options=f"--keep-cubin --dump-dir={_dump_dir()}",
            )
    return fn


_functions: dict[tuple, tuple] = {}


def _kernel_function(key: tuple):
    """(CUfunction, dynamic smem bytes) of a compiled build, loaded from its cubin on
    the current device (the DSL
    launches through its own runtime and exposes no handle; the occupancy probe needs
    one)."""
    import cuda.bindings.driver as drv

    hit = _functions.get(key)
    if hit is not None:
        return hit
    cubin = _compiled[key].__cubin__
    if not cubin:
        raise RuntimeError(f"sparse_decode: no cubin kept for {key}")
    err, lib = drv.cuLibraryLoadData(cubin, None, None, 0, None, None, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"sparse_decode: cuLibraryLoadData: {err}")
    err, kernels = drv.cuLibraryEnumerateKernels(1, lib)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"sparse_decode: cuLibraryEnumerateKernels: {err}")
    err, kf = drv.cuKernelGetFunction(kernels[0])
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"sparse_decode: cuKernelGetFunction: {err}")
    err, static = drv.cuFuncGetAttribute(
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kf
    )
    need = _kernel().smem_alloc(key[5])
    dyn = 0 if (err == drv.CUresult.CUDA_SUCCESS and static >= need) else need
    if dyn:
        drv.cuFuncSetAttribute(
            kf,
            drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dyn,
        )
    drv.cuFuncSetAttribute(
        kf,
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
        1,
    )
    hit = _functions[key] = (kf, dyn, lib)
    return hit


def max_clusters(cl: int, minb: int = 1, hq: int = HEADS_PER_KV, cap: int = 0) -> int:
    """Clusters of width `cl` of the (min blocks `minb`, pre-wait) build that
    can be resident at once (cuOccupancyMaxActiveClusters); negative = -CUresult."""
    import cuda.bindings.driver as drv

    key = (int(cl), int(hq), True, int(minb), True, int(cap))
    _compile(*key)
    kf, smem, _ = _kernel_function(key)
    cfg = drv.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cl * 64, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = _kernel().THREADS, 1, 1
    cfg.sharedMemBytes = smem
    attr = drv.CUlaunchAttribute()
    attr.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    val = drv.CUlaunchAttributeValue()
    val.clusterDim.x, val.clusterDim.y, val.clusterDim.z = cl, 1, 1
    attr.value = val
    cfg.attrs = [attr]
    cfg.numAttrs = 1
    err, n = drv.cuOccupancyMaxActiveClusters(kf, cfg)
    return int(n) if err == drv.CUresult.CUDA_SUCCESS else -int(err)


def num_regs(cl: int, minb: int = 1, hq: int = HEADS_PER_KV, pre: bool = True) -> int:
    """Registers per thread of one build (for the port's resource comparison)."""
    import cuda.bindings.driver as drv

    key = (int(cl), int(hq), bool(pre), int(minb), True, 0)
    _compile(*key)
    kf, _, _ = _kernel_function(key)
    err, n = drv.cuFuncGetAttribute(
        drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, kf
    )
    return int(n) if err == drv.CUresult.CUDA_SUCCESS else -1


def smem_bytes() -> int:
    return _kernel().SMEM_BYTES


def _f32(x: float) -> float:
    import struct

    return struct.unpack("f", struct.pack("f", x))[0]


def cluster_size(rows: int, num_kv_heads: int) -> int:
    """The preferred width before the occupancy probe: 16 CTAs per (row, KV
    head) unless that makes the grid 32 clusters or wider (then 8). The stage
    launches with the width `build()` chose (`cluster_sizes`)."""
    return geometry.sparse_cluster_size(rows, num_kv_heads)


def split_kv(kv_cache):
    """(k, v) uint8 views [pages, Hkv, 128, 128] of an fp8 KV cache whose
    last dim is K|V (256 bytes per token per head). The cache must be laid
    out (page, head, token, 256) up to strides; pass a permuted view if the
    allocator's dim order differs."""
    import torch

    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * HEAD_DIM:
        raise ValueError(
            "sparse_decode: kv cache [pages, Hkv, 128, 256] expected, got "
            f"{tuple(kv_cache.shape)}"
        )
    kv = kv_cache.view(torch.uint8)
    return kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]


def sparse_lens(seq_lens, *, page: int = PAGE, topk: int = TOPK):
    """int32 tokens covered by the selected pages (graph-safe torch ops):
    L if ceil(L / page) <= topk else (topk - 1) * page + tokens of the last
    page. Assumes page_topk always keeps the last (partial) page
    (local_blocks >= 1)."""
    import torch

    s = seq_lens.to(torch.int32)
    npages = (s + (page - 1)) // page
    last = s - (npages - 1).clamp_min(0) * page
    return torch.where(npages <= topk, s, (topk - 1) * page + last)


class ClusterSparseDecodeStage:
    """`__call__(q_fp8, k, v, tables, lens, out, q8, sf, sfp)`

    q_fp8   e4m3 [T, Hq, 128]             (Hq = heads_per_kv * Hkv per rank: 16 at TP4, 8 at TP8)
    k, v    e4m3/uint8 [P, Hkv, 128, 128] views (`split_kv`)
    tables  int32 [Hkv, T, >= 16]         page_topk output (physical pages)
    lens    int32 [Hkv, T]                `sparse_lens` of each row
    out     bf16 [T, Hq * 128]
    q8      uint8 [T, Hq * 128]           MXFP8 values for the o_proj GEMV
    sf      uint8 [128 * Hq * 4]          UE8M0, 128x4 swizzled, rows padded to 128
    sfp     uint8 [T, Hq * 4]             UE8M0, row-major copy
    """  # noqa: E501

    stage = "sparse_decode"

    def __init__(
        self,
        *,
        num_kv_heads: int,
        sm_scale: float,
        heads_per_kv: int = HEADS_PER_KV,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        kv_cache_dtype: str = "fp8",
        max_rows: int = 8,
        pdl: bool = True,
        strict_one_wave: bool = True,
        one_wave_rows: int = ONE_WAVE_ROWS,
        min_blocks: int = MIN_BLOCKS,
    ):
        self.fmt = _dtypes.kv_cache_format(
            kv_cache_dtype
        )  # rejects nvfp4 / unimplemented
        self.num_kv_heads = int(num_kv_heads)
        if heads_per_kv not in HEADS_PER_KV_BUILDS:
            raise ValueError(
                f"sparse_decode: {heads_per_kv} Q heads per KV head, not one of "
                f"{HEADS_PER_KV_BUILDS}"
            )
        self.heads_per_kv = int(heads_per_kv)
        self.qk_scale = float(sm_scale) * float(q_scale) * float(k_scale)
        # The kernel's fp32 log2-domain scale, rounded as the CUDA launcher did (double
        # product -> float).
        self.qk_scale_log2 = float(_f32(self.qk_scale * 1.4426950408889634))
        self.v_scale = float(_f32(float(v_scale)))
        if not 1 <= max_rows <= MAX_ROWS:
            raise ValueError(
                f"sparse_decode: max_rows {max_rows} outside 1..{MAX_ROWS}"
            )
        self.max_rows, self.pdl, self.strict = (
            int(max_rows),
            bool(pdl),
            bool(strict_one_wave),
        )
        self.one_wave_rows = int(one_wave_rows)
        if min_blocks not in (1, 2, 3):
            raise ValueError(f"sparse_decode: min_blocks {min_blocks} not in (1, 2, 3)")
        self.min_blocks = int(min_blocks)
        # rows -> min-blocks build (ALT_MIN_BLOCKS where it saves a wave; only
        # when the base build is MIN_BLOCKS), set by build()
        self.row_min_blocks: dict[int, int] = {}
        self.alt_max_active_clusters: dict[int, int] = {}
        # rows -> residency cap (CTAs per SM, 0 = none; set by build()): rows that stay
        # multi-wave run with at
        # most CAP_CTAS CTAs per SM when that does not add a wave (balanced waves; see
        # CAP_CTAS)
        self.row_cap: dict[int, int] = {}
        self.cap_max_active_clusters: dict[int, int] = {}
        self._built = False
        self.max_active_clusters: dict[int, int] = {}
        # Row counts whose grid exceeds one wave of clusters (built, slower).
        self.multi_wave_rows: list[int] = []
        self.cluster_sizes: dict[int, int] = {}  # rows -> cluster width, set by build()

    def build(self):
        spec = _arch.device_spec()
        smem = smem_bytes()
        if smem > spec.smem_per_block_opt_in:
            raise _arch.ResidencyError(
                f"sparse_decode: {smem} B smem per CTA > "
                f"{spec.smem_per_block_opt_in} B on {spec.name}"
            )
        for cl in geometry.SPARSE_CLUSTER_SIZES:
            n = max_clusters(cl, self.min_blocks, self.heads_per_kv)
            if n < 0:
                raise RuntimeError(
                    f"sparse_decode: cudaOccupancyMaxActiveClusters(CL={cl}) failed: "
                    f"error {-n}"
                )
            self.max_active_clusters[cl] = n
        for rows in range(1, self.max_rows + 1):
            need = rows * self.num_kv_heads
            one_wave = self.strict and rows <= self.one_wave_rows
            try:
                cl = geometry.sparse_cluster_size(
                    rows, self.num_kv_heads, self.max_active_clusters, one_wave=one_wave
                )
            except ValueError as err:
                raise _arch.ResidencyError(
                    f"sparse_decode: {rows} rows x {self.num_kv_heads} KV heads "
                    f"({smem} B smem per CTA) on "
                    f"{spec.name} ({spec.sm_count} SMs): {err}"
                ) from None
            self.cluster_sizes[rows] = cl
            self.row_min_blocks[rows] = self.min_blocks
            if self.max_active_clusters[cl] < need:
                self.multi_wave_rows.append(rows)
        if self.min_blocks == MIN_BLOCKS and ALT_MIN_BLOCKS != MIN_BLOCKS:
            alt = {
                c: max_clusters(c, ALT_MIN_BLOCKS, self.heads_per_kv)
                for c in geometry.SPARSE_CLUSTER_SIZES
            }
            self.alt_max_active_clusters = alt
            for rows in list(self.multi_wave_rows):
                cl = self.cluster_sizes[rows]
                waves = geometry.sparse_waves(
                    rows, self.num_kv_heads, cl, self.max_active_clusters
                )
                acl = geometry.sparse_cluster_size(
                    rows, self.num_kv_heads, alt, one_wave=False
                )
                if geometry.sparse_waves(rows, self.num_kv_heads, acl, alt) < waves:
                    self.cluster_sizes[rows] = acl
                    self.row_min_blocks[rows] = ALT_MIN_BLOCKS
                    if alt[acl] >= rows * self.num_kv_heads:
                        self.multi_wave_rows.remove(rows)
        if CAP_CTAS:
            capped = {
                c: max_clusters(c, MIN_BLOCKS, self.heads_per_kv, CAP_CTAS)
                for c in geometry.SPARSE_CLUSTER_SIZES
            }
            self.cap_max_active_clusters = capped
            for rows in self.multi_wave_rows:
                cl, mb = self.cluster_sizes[rows], self.row_min_blocks[rows]
                have = (
                    self.alt_max_active_clusters
                    if mb == ALT_MIN_BLOCKS
                    else self.max_active_clusters
                )
                if (
                    mb == MIN_BLOCKS
                    and 0 < capped[cl] < have[cl]
                    and geometry.sparse_waves(rows, self.num_kv_heads, cl, capped)
                    <= geometry.sparse_waves(rows, self.num_kv_heads, cl, have)
                ):
                    self.row_cap[rows] = CAP_CTAS
        # Compile every build a call can take, so none compiles under capture.
        for rows in range(1, self.max_rows + 1):
            for prewait in self._prewaits(rows):
                self._fn(rows, prewait)
        self._built = True
        return self

    @staticmethod
    def _prewaits(rows: int) -> tuple[int, ...]:
        return (1, 3) if rows >= PREWAIT_MIN_ROWS else (0,)

    def _fn(self, rows: int, prewait: int):
        minb = self.row_min_blocks.get(rows, self.min_blocks)
        # prewait 0 at min blocks 1: the build with every input read after the wait (no
        # pre-wait code)
        pre = not (minb == 1 and prewait == 0)
        return _compile(
            self.cluster_sizes[rows],
            self.heads_per_kv,
            pre,
            minb,
            self.pdl,
            self.row_cap.get(rows, 0),
        )

    def __call__(
        self,
        q_fp8,
        k,
        v,
        tables,
        lens,
        out,
        q8,
        sf,
        sfp,
        *,
        done_cnt=None,
        q_prewait=False,
    ):
        """`q_prewait`: q was not written by the kernel launched right before
        this one (e.g. the front, with the index scorer and page top-k in
        between), so the kernel may read it before its PDL grid wait. From
        `PREWAIT_MIN_ROWS` rows the lens (written at the start of the step,
        `sparse_lens`) are read before the wait, and q with them when
        `q_prewait`; below, every input is read after it."""
        if not self._built:
            self.build()
        t = q_fp8.shape[0]
        if not 1 <= t <= self.max_rows:
            raise ValueError(f"sparse_decode: {t} rows outside 1..{self.max_rows}")
        hq = self.heads_per_kv
        if q_fp8.shape[1] != hq * self.num_kv_heads or k.shape[1] != self.num_kv_heads:
            raise ValueError(f"sparse_decode: {hq} Q heads per KV head")
        _dtypes.check_cache_tensor("KV", k, self.fmt)
        _dtypes.check_cache_tensor("KV", v, self.fmt)
        launch(
            q_fp8,
            k,
            v,
            tables,
            lens,
            out,
            q8,
            sf,
            sfp,
            done_cnt=done_cnt,
            qk_scale_log2=self.qk_scale_log2,
            v_scale=self.v_scale,
            cl=self.cluster_sizes[t],
            min_blocks=self.row_min_blocks.get(t, self.min_blocks),
            prewait=(3 if q_prewait else 1) if t >= PREWAIT_MIN_ROWS else 0,
            heads_per_kv=hq,
            pdl=self.pdl,
            cap=self.row_cap.get(t, 0),
        )


def launch(
    q_fp8,
    k,
    v,
    tables,
    lens,
    out,
    q8,
    sf,
    sfp,
    *,
    done_cnt=None,
    qk_scale_log2: float,
    v_scale: float,
    cl: int,
    min_blocks: int,
    prewait: int,
    heads_per_kv: int = HEADS_PER_KV,
    pdl: bool = True,
    cap: int = 0,
):
    """One launch of the (cl, min_blocks, prewait) build; prewait 0 at min blocks 1
    reads every input after the wait.
    q_fp8 e4m3 [T, Hq, 128]; k, v e4m3 [pages, Hkv, 128, 128] views; tables int32 [Hkv,
    >= T, >= 16]; lens int32
    [Hkv, >= T]; out bf16 [T, Hq * 128]; q8 [T, Hq * 128]; sf [128 * Hq * 4] (128x4
    swizzle); sfp [T, Hq * 4];
    done_cnt optional int64 [1] completion counter."""
    import cuda.bindings.driver as cuda_driver
    import torch

    t = q_fp8.shape[0]
    if not 1 <= t <= 128:
        raise ValueError("sparse_decode: rows 1..128 (the MXFP8 scale tile)")
    if (
        prewait not in (0, 1, 3)
        or cl not in geometry.SPARSE_CLUSTER_SIZES
        or min_blocks not in (1, 2, 3)
    ):
        raise ValueError(
            f"sparse_decode: build (cl {cl}, min blocks {min_blocks}, prewait "
            f"{prewait}) not supported"
        )
    if not (
        q_fp8.dim() == 3
        and q_fp8.shape[2] == HEAD_DIM
        and q_fp8.stride(2) == 1
        and q_fp8.element_size() == 1
    ):
        raise ValueError(
            "sparse_decode: q must be e4m3 [T, Hq, 128] with a contiguous last dim"
        )
    if not (
        k.dim() == 4
        and v.dim() == 4
        and k.shape[2] == PAGE
        and k.shape[3] == HEAD_DIM
        and k.stride(3) == 1
        and v.stride(3) == 1
        and k.element_size() == 1
        and v.element_size() == 1
    ):
        raise ValueError(
            "sparse_decode: k/v must be uint8/e4m3 [pages, Hkv, 128, 128] views"
        )
    hkv = k.shape[1]
    if heads_per_kv not in HEADS_PER_KV_BUILDS or q_fp8.shape[1] != heads_per_kv * hkv:
        raise ValueError(f"sparse_decode: {heads_per_kv} Q heads per KV head")
    if not (
        tables.dtype == torch.int32
        and tables.shape[0] == hkv
        and tables.shape[1] >= t
        and tables.shape[2] >= TOPK
        and tables.stride(2) == 1
    ):
        raise ValueError(
            "sparse_decode: tables must be int32 [Hkv, >= T, >= 16] with a contiguous "
            "last dim"
        )
    if not (
        lens.dtype == torch.int32
        and lens.shape[0] == hkv
        and lens.shape[1] >= t
        and lens.stride(1) == 1
    ):
        raise ValueError(
            "sparse_decode: lens must be int32 [Hkv, >= T] with contiguous rows"
        )
    width = q_fp8.shape[1] * HEAD_DIM
    if not (out.dtype == torch.bfloat16 and out.stride(1) == 1):
        raise ValueError(
            "sparse_decode: out must be bf16 [T, Hq * 128] with a contiguous last dim"
        )
    if not (
        q8.numel() >= t * width
        and q8.is_contiguous()
        and sf.numel() >= 128 * width // 32
        and sf.is_contiguous()
        and sfp.numel() >= t * width // 32
        and sfp.is_contiguous()
    ):
        raise ValueError("sparse_decode: q8 / sf / sfp too small or not contiguous")
    cnt = 0
    if done_cnt is not None:
        if done_cnt.dtype != torch.int64 or done_cnt.numel() < 1:
            raise ValueError("sparse_decode: done_cnt must be int64 [>= 1]")
        cnt = done_cnt.data_ptr()
    fn = _compile(
        cl, heads_per_kv, not (min_blocks == 1 and prewait == 0), min_blocks, pdl, cap
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q_fp8.device).cuda_stream)
    fn(
        q_fp8.data_ptr(),
        q_fp8.stride(0),
        q_fp8.stride(1),
        k.data_ptr(),
        v.data_ptr(),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        tables.data_ptr(),
        4 * tables.stride(0),
        4 * tables.stride(1),
        lens.data_ptr(),
        4 * lens.stride(0),
        qk_scale_log2,
        v_scale,
        out.data_ptr(),
        2 * out.stride(0),
        q8.data_ptr(),
        sf.data_ptr(),
        sfp.data_ptr(),
        cnt,
        q_fp8.shape[1],
        hkv,
        prewait,
        t * hkv * cl,
        stream,
    )


__all__ = [
    "ClusterSparseDecodeStage",
    "cluster_size",
    "launch",
    "max_clusters",
    "sparse_lens",
    "split_kv",
]
