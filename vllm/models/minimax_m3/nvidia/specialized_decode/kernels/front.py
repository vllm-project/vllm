# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage `front`: the attention front of one sparse layer in one launch.

The device code is CuTe DSL (`front_kernel.py`, compiled per specialisation
at build). One cluster-of-4 grid does, per decode row:

1. boundary: TP all-reduce of the previous sublayer's bf16 partials (Lamport
   push exchange), + residual (-> `out_residual`), Gemma RMSNorm, MXFP8
   quantization of the normed row;
2. QKV + index-Q + index-K MXFP8 GEMV (tcgen05 `kind::mxf8f6f4`), split-K over
   the 4 CTAs of a cluster, DSMEM reduce;
3. epilogue: per-head Gemma RMSNorm + NeoX RoPE, bf16/FP8 Q, FP8 index-Q, FP8 K/V
   insert into the KV cache, FP8 index-K insert into the indexer cache;
4. optionally the step metadata of the layer (one extra cluster).

Exchange modes (`prepushed`):

- `False` (0): this front pushes its own partial `x` into every rank's
  exchange, then polls. Uses the `"front"` exchange of the transport.
- `True` (2): the producer of the previous sublayer (the fused MoE engine)
  already pushed into the shared boundary exchange (`"g3"`, see
  `_exchange.py`); the front only polls. `x` must be None.

Every CTA of the grid spins (the GEMV clusters poll the boundary clusters'
hand-off; the boundary polls peers), so the whole grid must be co-resident:
the constructor checks that against the live device and raises
`ResidencyError` otherwise.

The per-rank head geometry (nq, nkv, niq) is a compile-time parameter: TP4 is
16 / 1 / 1 (20 head clusters), TP8 8 / 1 / 1 (12 head clusters).

Exchange buffers come from `_comm.PeerTransport`; cache formats are
checked (FP8 e4m3 only; NVFP4 rejected).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from . import _arch, _dtypes, _exchange
from ._comm import PeerBuffer, PeerTransport

HIDDEN = 6144
HEAD_DIM = 128
CLUSTER = 4
THREADS = 512
# Shared memory of the kernel: the 1024-aligned tile buffers + the small arrays (rounded
# up).
STATIC_SMEM = 128
TMEM_COLS_ASMEM = 128  # A operand in smem (bucket 8): acc | SFA | SFB
TMEM_COLS_FULL = 512  # A operand in TMEM (bucket 16): acc | SFA | SFB | A
# Rows of one launch: the kernel's activation tile. Bucket 8 runs the A-in-smem build,
# bucket 16 (9..16 rows) the A-in-TMEM one; a step with more rows runs the stock front.
MAX_ROWS = 16
M1_MAX_ROWS = 8  # the bucket-8 build (the default)
MAX_WORLD = 16  # kernels/_comm.py MAX_PEERS: the push unrolls over the peer table


class CuteDslUnavailable(RuntimeError):
    pass


def _ext(fmt: _dtypes.CacheFormat = _dtypes.CacheFormat.FP8_E4M3, world: int = 4):
    """The kernel module (CuTe DSL). Only the FP8 e4m3 cache format has device code;
    `world` is a compile-time
    parameter of each launch, accepted here for the registry's build phase."""
    if fmt is not _dtypes.CacheFormat.FP8_E4M3:
        raise _dtypes.CacheDtypeError(
            f"front: no device code for the {fmt.value} cache format"
        )
    if not 1 <= world <= MAX_WORLD:
        raise ValueError(f"front: world {world} outside 1..{MAX_WORLD}")
    try:
        import cutlass.experimental.cuda  # noqa: F401
        from cutlass.experimental import primitives  # noqa: F401

        from . import front_kernel
    except ImportError as e:
        raise CuteDslUnavailable(
            f"front: needs the CuTe DSL (nvidia-cutlass-dsl): {e}"
        ) from e
    return front_kernel


_lock = threading.Lock()
_compiled: dict[tuple, object] = {}


def _spec_rows(rows: int, maxt: int) -> int:
    """The rows a launch is specialised for (loop trip counts, MMA N and owner rows
    constant), 0 = runtime rows."""
    if maxt == 8 and rows in (4, 8):
        return rows
    if rows == 16:
        return 16
    return 0


@dataclass
class StepMetaIO:
    """Step metadata the front writes for the indexer of its layer (one extra
    cluster). Inputs: `query_start_loc` int32 [reqs + 1], `seq_lens` int32
    [reqs], `block_table` int32 [reqs, max_pages]. Outputs: `n_valid` int32
    [rows] (pos // 128 + 1, 0 for an unresolved row), `sparse_len` int32
    [heads, rows] (any strides) or None, `pt_rows` int32 [rows, pt_maxp] or
    None (the first pages of each row's block table)."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_decode_tokens: int
    n_valid: torch.Tensor
    sparse_len: torch.Tensor | None = None
    pt_rows: torch.Tensor | None = None


@dataclass
class FinalizeIn:
    """Unfused MoE finalize as the boundary input (prepushed False only):
    x = sum_k scales[t, k] * fc2[perm[t, k]] (+ x2, the shared expert)."""

    fc2: torch.Tensor  # bf16 [expanded rows, H]
    perm: torch.Tensor  # int32 [rows * topk]
    scales: torch.Tensor  # fp32 or bf16 [rows * topk]
    topk: int


class QkvFrontStage:
    """`front` for one layer (one instance per layer; all layers may share
    the transport's exchange of their bucket).

    Geometry per rank: `nq` query heads, `nkv` KV heads, `niq` index
    query heads, one index-K head; the fused weight `w` is e4m3
    [(nq + 2 nkv + niq + 1) * 128, 6144] with 128x4-swizzled UE8M0 scales
    `wsf` (rows: q | k | v | index-q | index-k).
    """

    stage = "front"

    def __init__(
        self,
        *,
        transport: PeerTransport,
        nq: int,
        nkv: int,
        niq: int,
        rot: int,
        block_size: int,
        kv_cache_dtype: str = "fp8",
        indexer_kv_dtype: str = "fp8",
        prepushed: bool = False,
        max_rows: int = M1_MAX_ROWS,
        eps: float = 1e-6,
        head_eps: float = 1e-6,
        q_scale: float = 1.0,
        with_step_meta: bool = True,
        early_trigger: bool = True,
        pdl: bool = True,
        workspace: _exchange.PushWorkspace | None = None,
        ndst: int | None = None,
    ):
        self.kv_fmt = _dtypes.kv_cache_format(kv_cache_dtype)
        self.ix_fmt = _dtypes.indexer_cache_format(indexer_kv_dtype)
        if self.kv_fmt != self.ix_fmt:
            # One build define covers both caches today.
            raise _dtypes.CacheDtypeError(
                f"front: KV ({self.kv_fmt.value}) and indexer ({self.ix_fmt.value}) "
                "cache formats must match"
            )
        if not 1 <= max_rows <= MAX_ROWS:
            raise ValueError(f"front: max_rows {max_rows} outside 1..{MAX_ROWS}")
        self.nq, self.nkv, self.niq = nq, nkv, niq
        self.heads = nq + 2 * nkv + niq + 1
        self.rot, self.block_size = rot, block_size
        self.eps, self.head_eps, self.q_scale = eps, head_eps, q_scale
        self.prepushed = bool(prepushed)
        self.with_step_meta = with_step_meta
        self.early_trigger, self.pdl = early_trigger, pdl
        self.max_rows = max_rows
        self.maxt = _exchange.bucket(max_rows)

        self.spec = _arch.device_spec()
        world = workspace.world if workspace is not None else transport.world
        self.kmod = _ext(self.kv_fmt, world)
        k = self.kmod
        self.smem = k.SMEM
        self.xq_buf, self.xsf_buf = k.XQ_BUF, k.XSF_BUF
        self.grid = (self.heads + (1 if with_step_meta else 0)) * CLUSTER
        if max_rows > self.heads:
            # Head cluster c runs boundary row c (the step-metadata cluster runs none):
            # a row without a cluster is never pushed and the GEMV clusters would wait
            # for it forever.
            raise ValueError(
                f"front: {max_rows} rows > {self.heads} head clusters (one boundary "
                "row each)"
            )
        if self.grid > k.CTR_LEN:
            raise ValueError(
                f"front: grid {self.grid} exceeds the kernel's counter table "
                f"{k.CTR_LEN}"
            )
        if self.heads * CLUSTER * THREADS < k.XQ_BUF // 16:
            raise ValueError(
                "front: too few head clusters to re-arm a hand-off parity in one pass"
            )
        self._check_residency()

        maxt = self.maxt
        if workspace is None:
            workspace = _exchange.push_workspace(
                transport, maxt, "g3" if self.prepushed else "front"
            )
        if workspace.maxt != maxt:
            raise ValueError(f"front: workspace bucket {workspace.maxt} != {maxt}")
        self.ws_py = workspace
        dev = workspace.exch.local.device
        # Private hand-off buffers: two parities each, sentinel-armed (activation bytes
        # 0x80, scale bytes 0xFF), call counters at 0.
        self.xq = torch.full((2 * self.xq_buf,), 0x80, dtype=torch.uint8, device=dev)
        self.xsf = torch.full((2 * self.xsf_buf,), 0xFF, dtype=torch.uint8, device=dev)
        self.ctr = torch.zeros(k.CTR_LEN, dtype=torch.int32, device=dev)
        self.sync = torch.zeros(k.SYNC_LEN, dtype=torch.int32, device=dev)
        ex: PeerBuffer = workspace.exch
        # The front's own push (prepushed off): unicast to ndst ranks, or one store
        # through the exchange's multicast address.
        ndst = workspace.ndst if ndst is None else ndst
        multicast = int(ex.mc_ptr) if workspace.multicast and ndst == 1 else 0
        rank, world, peers = int(ex.rank), int(ex.world), [int(p) for p in ex.peer_ptrs]
        if not (1 <= world <= MAX_WORLD and maxt in (8, 16)):
            raise ValueError(
                f"front: world 1..{MAX_WORLD}, bucket 8 or 16 (got {world}, {maxt})"
            )
        if (
            not 0 <= rank < world
            or len(peers) != world
            or peers[rank] != int(ex.local.data_ptr())
        ):
            raise ValueError(
                "front: one peer pointer per rank, peers[rank] the local exchange"
            )
        if not 1 <= ndst <= world or (multicast and ndst != 1):
            raise ValueError(
                "front: ndst in 1..world; a multicast push has one destination"
            )
        table = [peers[r] if r < world else 0 for r in range(MAX_WORLD)]
        if multicast:
            table[0] = multicast  # one store reaches every rank
        self.rank, self.world, self.ndst = rank, world, ndst
        self.peer_table = table
        self.ws_local = int(ex.local.data_ptr())
        self.flags_ptr = int(workspace.flags.data_ptr())
        self.ws = 1  # bound (kept for callers that test it)

    def _check_residency(self) -> None:
        occ = _arch.ctas_per_sm(
            self.spec,
            smem_bytes=self.smem + STATIC_SMEM,
            threads=THREADS,
            tmem_columns=TMEM_COLS_ASMEM if self.maxt <= 8 else TMEM_COLS_FULL,
        )
        _arch.check_persistent_grid("front", self.grid, occ, self.spec)
        # Clusters of 4 at one CTA per SM: the runtime's cluster occupancy query needs
        # the compiled kernel; it runs on the first compile (``_check_clusters``).
        self._clusters_checked = False

    def _check_clusters(self, fn) -> None:
        if self._clusters_checked:
            return
        n = _max_active_clusters(fn, self.grid, self.smem + STATIC_SMEM)
        if n is not None and n * CLUSTER < self.grid:
            raise _arch.ResidencyError(
                f"front: {self.grid // CLUSTER} clusters of {CLUSTER} must be "
                "co-resident, the device "
                f"reports at most {n} active clusters"
            )
        self._clusters_checked = True

    def close(self) -> None:
        self.ws = 0

    def _compiled(self, rows: int, step_meta: bool, args: tuple):
        import cuda.bindings.driver as cuda_driver
        import cutlass.cute as cute

        key = (
            self.maxt,
            _spec_rows(rows, self.maxt),
            self.world,
            self.nq,
            self.nkv,
            self.niq,
            self.rot,
            self.block_size,
            self.prepushed,
            self.early_trigger,
            step_meta,
            bool(self.pdl),
        )
        fn = _compiled.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "front: each specialisation must run once outside CUDA-graph "
                    "capture first"
                )
            with _lock:
                fn = _compiled.get(key)
                if fn is None:
                    stream = cuda_driver.CUstream(
                        torch.cuda.current_stream().cuda_stream
                    )
                    fn = _compiled[key] = cute.compile(
                        self.kmod.front_launch, *args, stream, *key[:-1], key[-1]
                    )
        self._check_clusters(fn)
        return fn

    def precompile(self) -> None:
        """Compile every specialisation this stage launches (one per distinct
        `_spec_rows` over its rows), outside CUDA-graph capture: the executor
        first calls the front inside a FULL graph capture, where compiling is
        refused. The kernel arguments are scalars (device addresses, sizes,
        strides), so zeros of the same types compile the same code; nothing is
        launched. Also runs the cluster co-residency check."""
        n_lead = 34  # the pointer arguments of ``__call__`` before the peer table
        reps: dict[int, int] = {}
        for rows in range(1, self.max_rows + 1):
            reps.setdefault(_spec_rows(rows, self.maxt), rows)
        for rows in reps.values():
            args = (
                (0,) * n_lead
                + tuple(int(p) for p in self.peer_table)
                + (int(rows), self.rank, self.ndst, 0, 0, 0)
                + (float(self.eps), float(self.head_eps), float(1.0 / self.q_scale))
                + (0,) * 12
            )
            self._compiled(rows, self.with_step_meta, args)

    def __call__(
        self,
        rows: int,
        *,
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        out_residual: torch.Tensor,
        w: torch.Tensor,
        wsf: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        iq_w: torch.Tensor,
        ik_w: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        q_fp8: torch.Tensor,
        index_q: torch.Tensor,
        x: torch.Tensor | None = None,
        x2: torch.Tensor | None = None,
        finalize: FinalizeIn | None = None,
        q_out: torch.Tensor | None = None,
        step_meta: StepMetaIO | None = None,
        single_round: bool = False,
    ) -> None:
        """One front launch (graph-capturable; no host sync).

        residual bf16 [rows, 6144]; norm_w bf16 [6144] (Gemma: x * (1 + w));
        out_residual bf16 [rows, 6144]; cos_sin bf16 [max_pos, rot] ([cos | sin]
        halves); positions / slot_mapping / index_slot_mapping int64 [rows]
        (slot -1 skips the insert); kv_cache [blocks, nkv, block_size, 2 * 128]
        (fp8, k then v); index_cache [slots, 128] (fp8); q_fp8 [rows, nq * 128];
        index_q [rows, niq * 128]; q_out bf16 [rows, nq * 128] optional.
        single_round: residual_mid = bf16(sum of the partials + residual)
        with one rounding, as the stock deferred-MoE finalize + all-reduce +
        norm rounds it; otherwise bf16(bf16(sum) + residual), as FlashInfer's
        fused all-reduce + norm does.
        """
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"front: rows {rows} outside 1..{self.max_rows}")
        if self.prepushed:
            if x is not None or finalize is not None:
                raise ValueError("front: prepushed stage takes no partial input")
        elif x is None and finalize is None:
            raise ValueError("front: the partial x (or a finalize input) is required")
        if (step_meta is not None) != self.with_step_meta:
            raise ValueError(
                "front: step_meta must match with_step_meta (the grid size is fixed at "
                "build)"
            )
        _dtypes.check_cache_tensor("KV", kv_cache, self.kv_fmt)
        _dtypes.check_cache_tensor("indexer K", index_cache, self.ix_fmt)
        if not (self.rot > 0 and self.rot <= HEAD_DIM and self.rot % 8 == 0):
            raise ValueError("front: rotary dim")
        if (
            tuple(w.shape) != (self.heads * HEAD_DIM, HIDDEN)
            or not w.is_contiguous()
            or w.element_size() != 1
        ):
            raise ValueError(
                f"front: weight {tuple(w.shape)} must be [{self.heads * HEAD_DIM}, "
                f"{HIDDEN}] E4M3"
            )
        if (
            kv_cache.dim() != 4
            or kv_cache.stride(3) != 1
            or kv_cache.element_size() != 1
        ):
            raise ValueError(
                "front: fp8 KV cache [blocks, heads, block_size, 2*D] expected"
            )
        if index_cache.element_size() != 1 or not index_cache.is_contiguous():
            raise ValueError("front: fp8 index cache expected")
        for name, t in (
            ("positions", positions),
            ("slot_mapping", slot_mapping),
            ("index_slot_mapping", index_slot_mapping),
        ):
            if t.dtype != torch.int64:
                raise ValueError(f"front: {name} must be int64")
        fin = finalize
        m = step_meta

        def ptr(t):
            return 0 if t is None else int(t.data_ptr())

        sl = m.sparse_len if m is not None else None
        pt = m.pt_rows if m is not None else None
        args = (
            int(w.data_ptr()),
            ptr(x),
            ptr(x2),
            ptr(fin.fc2) if fin else 0,
            ptr(fin.perm) if fin else 0,
            ptr(fin.scales) if fin else 0,
            ptr(residual),
            ptr(norm_w),
            ptr(out_residual),
            self.ws_local,
            self.flags_ptr,
            ptr(self.xq),
            ptr(self.xsf),
            ptr(self.ctr),
            ptr(wsf),
            ptr(q_w),
            ptr(k_w),
            ptr(iq_w),
            ptr(ik_w),
            ptr(cos_sin),
            ptr(positions),
            ptr(slot_mapping),
            ptr(index_slot_mapping),
            ptr(kv_cache),
            ptr(index_cache),
            ptr(q_out),
            ptr(q_fp8),
            ptr(index_q),
            ptr(m.query_start_loc) if m else 0,
            ptr(m.seq_lens) if m else 0,
            ptr(m.block_table) if m else 0,
            ptr(m.n_valid) if m else 0,
            ptr(sl),
            ptr(pt),
            *self.peer_table,
            int(rows),
            self.rank,
            self.ndst,
            int(fin.topk) if fin else 0,
            1 if (fin is not None and fin.scales.dtype == torch.float32) else 0,
            1 if single_round else 0,
            float(self.eps),
            float(self.head_eps),
            float(1.0 / self.q_scale),
            int(kv_cache.stride(0)),
            int(kv_cache.stride(1)),
            int(kv_cache.stride(2)),
            int(m.block_table.stride(0)) if m else 0,
            int(sl.stride(0)) if sl is not None else 0,
            int(sl.stride(1)) if sl is not None else 0,
            int(pt.stride(0)) if pt is not None else 0,
            int(m.block_table.size(1)) if m else 0,
            int(m.query_start_loc.size(0) - 1) if m else 0,
            int(m.num_decode_tokens) if m else 0,
            int(sl.size(0)) if sl is not None else 0,
            int(pt.size(1)) if pt is not None else 0,
        )
        if fin is not None and fin.topk > 8:
            raise ValueError("front: finalize top-k above 8")
        fn = self._compiled(rows, m is not None, args)
        import cuda.bindings.driver as cuda_driver

        fn(*args, cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))


def _max_active_clusters(fn, grid: int, smem: int):
    """CuOccupancyMaxActiveClusters of the compiled kernel, or None when its CUfunction
    is not reachable."""
    try:
        import cuda.bindings.driver as drv

        func = None
        for attr in ("kernel_info", "cuda_function", "function"):
            cand = getattr(fn, attr, None)
            if cand is not None:
                func = cand
                break
        if func is None:
            return None
        cfg = drv.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = THREADS, 1, 1
        cfg.sharedMemBytes = smem
        at = drv.CUlaunchAttribute()
        at.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = (
            CLUSTER,
            1,
            1,
        )
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, n = drv.cuOccupancyMaxActiveClusters(func, cfg)
        return int(n) if err == drv.CUresult.CUDA_SUCCESS else None
    except Exception:
        return None


__all__ = ["FinalizeIn", "QkvFrontStage", "StepMetaIO"]
