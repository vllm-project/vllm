# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the post family's CuTe DSL kernels (`post/kernel.py`).

`post_ops(world)` and `row_ops()` return objects with the calls the stage wrappers
(`allreduce_norm.py`,
`wide.py`) use: `post_route` / `kernel_info` and `mxfp8_rows`. They check
shapes, dtypes and alignment
before every launch.

Each kernel variant is compiled once per process (`cute.compile`, TVM-FFI calling
convention), always outside
CUDA-graph capture: the post stage's build (`kernel_info` + its idle launch) compiles
its post_route variants,
both logits_out flavours, and the wide front's build compiles mxfp8_rows
(`RowOps.precompile`). A compile
requested during capture raises.
"""

from __future__ import annotations

import struct
import threading

import torch

from .._comm import MAX_PEERS
from .._exchange import CLUSTER, PUSH_PARAMS_BYTES, PUSH_PARAMS_FMT

H = 6144
E = 128
TOPK = 4
LINE_U64 = 16
RCTR_STRIDE = 32
WORLDS = (4, 8)
# cluster width -> row threads (TH_A); the routing warp adds 32
TH_A = {4: 192, 8: 96, 16: 64}

_lock = threading.Lock()
_compiled: dict = {}
_infos: dict = {}


def _kernel():
    from . import kernel

    return kernel


def _stream(device=None) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _compile(key, jit_fn, sample_args, constexprs):
    """cute.compile `jit_fn` for `key` (cached). sample_args: the dynamic args' DSL
    types in order."""
    fn = _compiled.get(key)
    if fn is not None:
        return fn
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"post kernels: {key} compiles on its first call; build it before graph "
            "capture"
        )
    import cutlass.cute as cute

    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute.compile(
                jit_fn,
                *sample_args,
                *constexprs,
                cute.runtime.make_fake_stream(),
                options="--enable-tvm-ffi --keep-cubin",
            )
    return fn


def _sample(spec: str):
    """DSL sample values for a signature string of l (Int64), i (Int32), f (Float32)."""
    import cutlass

    return [
        {"l": cutlass.Int64(0), "i": cutlass.Int32(0), "f": cutlass.Float32(0.0)}[c]
        for c in spec
    ]


def _ptr(t) -> int:
    return 0 if t is None else int(t.data_ptr())


def _check(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def _cubin_info(fn, threads: int, cluster: int):
    """(static smem, regs, CTAs/SM, max active clusters) of the single kernel in fn's
    cubin."""
    from cuda.bindings import driver

    def ok(r):
        if r[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"post kernel_info: {r[0]}")
        return None if len(r) == 1 else r[1] if len(r) == 2 else r[1:]

    lib = ok(driver.cuLibraryLoadData(fn.artifacts.CUBIN, None, None, 0, None, None, 0))
    kernels = ok(driver.cuLibraryEnumerateKernels(1, lib))
    f = ok(driver.cuKernelGetFunction(kernels[0]))
    A = driver.CUfunction_attribute
    smem = ok(driver.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f))
    regs = ok(driver.cuFuncGetAttribute(A.CU_FUNC_ATTRIBUTE_NUM_REGS, f))
    occ = ok(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, threads, 0))
    ok(
        driver.cuFuncSetAttribute(
            f, A.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1
        )
    )
    cfg = driver.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cluster * 64, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = threads, 1, 1
    cfg.sharedMemBytes = 0
    at = driver.CUlaunchAttribute()
    at.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    v = driver.CUlaunchAttributeValue()
    v.clusterDim.x, v.clusterDim.y, v.clusterDim.z = cluster, 1, 1
    at.value = v
    cfg.attrs = [at]
    cfg.numAttrs = 1
    clusters = ok(driver.cuOccupancyMaxActiveClusters(f, cfg))
    return int(smem), int(regs), int(occ), int(clusters)


# --------------------------------------------------------------------------------------
# post_route
# --------------------------------------------------------------------------------------
_POST_SIG = (
    "llillllllfiillllliilillll"  # post_route's dynamic args (see kernel.post_route)
)


class PostOps:
    """The allreduce_norm stage's kernels for one world size (4 or 8)."""

    def __init__(self, world: int):
        if world not in WORLDS:
            raise ValueError(f"post_route: world {world} not in {WORLDS}")
        self.world = world

    def _post_fn(self, cl: int, rx2: int, mx: bool, lg: bool):
        if cl not in TH_A:
            raise RuntimeError("post_route: cluster width must be 4, 8 or 16")
        key = ("post_route", cl, bool(rx2), self.world, bool(mx), bool(lg))
        k = _kernel()
        return _compile(
            key,
            k.post_route,
            _sample(_POST_SIG),
            (cl, TH_A[cl], bool(rx2), self.world, bool(mx), bool(lg)),
        )

    def post_route(
        self,
        pp,
        rows,
        residual,
        gamma,
        eps,
        gemma,
        out_norm,
        out_res,
        out_q,
        out_sf,
        cl,
        rx_local,
        rxc,
        route,
        rctr,
        bias,
        repl,
        x4_q,
        x4_sf,
        x4_sfsc,
        x4_pts,
        deadline_ns,
        hang_dbg,
        rx2,
        ex_local=0,
        logits_out=None,
    ):
        _check(pp.numel() >= PUSH_PARAMS_BYTES, "post_route: push params size")
        f = struct.unpack_from(
            PUSH_PARAMS_FMT, bytes(pp[:PUSH_PARAMS_BYTES].cpu().numpy())
        )
        peers, pflags, prank, pworld, pmaxt, pcluster, pndst = (
            f[:MAX_PEERS],
            *f[MAX_PEERS:],
        )
        _check(
            pcluster == CLUSTER and (pndst == pworld or pndst == 1),
            "post_route: cluster-4 parity layout only",
        )
        _check(
            pworld == self.world,
            f"post_route: built for world {self.world} (got {pworld})",
        )
        _check(0 <= rows <= pmaxt, "post_route: rows exceed the exchange bucket")
        _check(gemma == 1, "post_route: Gemma weights only")
        _check(
            residual.dtype == torch.bfloat16
            and residual.is_contiguous()
            and residual.shape[-1] == H,
            "post_route: residual bf16 [rows, 6144]",
        )
        _check(
            gamma.dtype == torch.bfloat16 and gamma.numel() == H,
            "post_route: gamma bf16 [6144]",
        )
        _check(
            out_norm.dtype == torch.bfloat16 and out_norm.is_contiguous(),
            "post_route: out_norm bf16",
        )
        _check(
            out_res.dtype == torch.bfloat16 and out_res.is_contiguous(),
            "post_route: out_res bf16",
        )
        _check(
            (out_q is None) == (out_sf is None), "post_route: out_q and out_sf together"
        )
        _check(
            ex_local != 0 or pndst == pworld,
            "post_route: multicast push params need ex_local",
        )
        ex = ex_local if ex_local != 0 else peers[prank]
        _check(1 <= repl <= 32, "post_route: 1 <= repl <= 32")
        _check(
            route.dtype == torch.int64
            and route.is_contiguous()
            and route.numel() % (repl * LINE_U64) == 0,
            "post_route: route int64 [repl][maxt][16]",
        )
        rmaxt = route.numel() // (repl * LINE_U64)
        _check(rows <= rmaxt, "post_route: route lines have fewer rows than rows")
        _check(
            rctr.dtype == torch.int32 and rctr.numel() >= rmaxt * RCTR_STRIDE,
            "post_route: rctr int32 [maxt][32]",
        )
        _check(
            route.data_ptr() % 128 == 0 and rctr.data_ptr() % 128 == 0,
            "post_route: route lines / rctr rows must be 128-byte aligned",
        )
        _check(
            bias.dtype == torch.float32
            and bias.numel() == E
            and bias.data_ptr() % 16 == 0,
            "post_route: bias fp32 [128], 16-byte aligned",
        )
        _check(rx_local != 0 and rx_local % 16 == 0, "post_route: rx base")
        _check(
            rxc.dtype == torch.int32
            and rxc.numel() >= pmaxt * RCTR_STRIDE
            and rxc.data_ptr() % 128 == 0,
            "post_route: rxc int32 [maxt][32], 128-byte aligned",
        )
        _check(deadline_ns > 0, "post_route: deadline_ns")
        _check(
            x4_q.dtype == torch.uint8
            and x4_q.is_contiguous()
            and x4_q.numel() % (H // 2) == 0
            and x4_q.numel() // (H // 2) >= rows
            and x4_q.data_ptr() % 16 == 0,
            "post_route: x4_q uint8 [>= rows][3072], 16 B aligned",
        )
        _check(
            x4_sf.dtype == torch.uint8
            and x4_sf.is_contiguous()
            and x4_sf.numel() % (H // 16) == 0
            and x4_sf.numel() // (H // 16) >= rows
            and x4_sf.data_ptr() % 16 == 0,
            "post_route: x4_sf uint8 [>= rows][384], 16 B aligned",
        )
        _check(
            x4_sfsc.dtype == torch.float32 and x4_sfsc.numel() == 1 and x4_sfsc.is_cuda,
            "post_route: x4_sfsc",
        )
        _check(
            x4_pts.dtype == torch.float32 and x4_pts.is_cuda and x4_pts.numel() >= rows,
            "post_route: x4_pts fp32 [>= rows]",
        )
        lg = logits_out is not None
        if lg:
            _check(
                logits_out.dtype == torch.float32
                and logits_out.is_contiguous()
                and logits_out.numel() >= rows * E
                and logits_out.data_ptr() % 16 == 0,
                "post_route: logits_out fp32 [>= rows][128], 16 B aligned",
            )
        fn = self._post_fn(cl, rx2, out_q is not None, lg)
        if (
            rows == 0
            # the stage's build-time idle launch: also compile the other logits_out
            # variant (no lazy
        ):
            self._post_fn(
                cl, rx2, out_q is not None, not lg
            )  # compile can then happen inside graph capture)
        fn(
            ex,
            pflags,
            pmaxt,
            residual.data_ptr(),
            gamma.data_ptr(),
            out_res.data_ptr(),
            out_norm.data_ptr(),
            _ptr(out_q),
            _ptr(out_sf),
            float(eps),
            int(rows),
            max(int(rows), 1),
            int(rx_local),
            rxc.data_ptr(),
            bias.data_ptr(),
            route.data_ptr(),
            rctr.data_ptr(),
            int(repl),
            int(rmaxt),
            int(deadline_ns),
            int(hang_dbg),
            x4_q.data_ptr(),
            x4_sf.data_ptr(),
            x4_pts.data_ptr(),
            _ptr(logits_out),
            _stream(residual.device),
        )

    def kernel_info(self, cl: int, rx2: int, mxfp8: bool = False, logits: bool = False):
        """(static smem bytes, registers per thread, threads per CTA, CTAs per SM,
        cluster width, clusters the device
        holds at once) of the variant, for the host-side co-residency check."""
        key = (self.world, cl, bool(rx2), bool(mxfp8), bool(logits))
        if key not in _infos:
            fn = self._post_fn(cl, rx2, mxfp8, logits)
            th = TH_A[cl] + 32
            smem, regs, occ, clusters = _cubin_info(fn, th, cl)
            _infos[key] = (smem, regs, th, occ, cl, clusters)
        return _infos[key]


# --------------------------------------------------------------------------------------
# mxfp8_rows
# --------------------------------------------------------------------------------------
class RowOps:
    """The wide front's MXFP8 quantize of BF16 rows (`wide.py`)."""

    def precompile(self) -> None:
        """Compile the kernel now (outside CUDA-graph capture)."""
        _compile(("mxfp8_rows",), _kernel().mxfp8_rows, _sample("llli"), ())

    def mxfp8_rows(self, x, q, sf, rows):
        _check(1 <= rows <= 128, "mxfp8_rows: rows in 1..128")
        _check(
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == H
            and x.numel() // H >= rows
            and x.data_ptr() % 16 == 0,
            "mxfp8_rows: x bf16 [>= rows, 6144]",
        )
        _check(
            q.element_size() == 1
            and q.is_contiguous()
            and q.numel() >= rows * H
            and q.data_ptr() % 8 == 0,
            "mxfp8_rows: q 1-byte [>= rows, 6144]",
        )
        _check(
            sf.element_size() == 1 and sf.numel() >= 128 * (H // 32),
            "mxfp8_rows: sf [128 * 192] bytes",
        )
        k = _kernel()
        fn = _compile(("mxfp8_rows",), k.mxfp8_rows, _sample("llli"), ())
        fn(x.data_ptr(), q.data_ptr(), sf.data_ptr(), int(rows), _stream(x.device))


_ops: dict = {}


def post_ops(world: int) -> PostOps:
    """The post's ops for `world` (imports the CuTe DSL kernel module: raises here if
    it is unavailable)."""
    _kernel()
    if world not in _ops:
        _ops[world] = PostOps(world)
    return _ops[world]


def row_ops() -> RowOps:
    _kernel()
    if "rows" not in _ops:
        _ops["rows"] = RowOps()
    return _ops["rows"]


__all__ = ["PostOps", "RowOps", "post_ops", "row_ops"]
