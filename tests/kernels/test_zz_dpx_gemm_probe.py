# ruff: noqa
#!/usr/bin/env python3
"""dpx_gemm_probe.py - which hipBLASLt bf16 GEMM kernel runs on THIS device, and is it the fastest one offered?

Self-contained: pure torch + ctypes against the libhipblaslt that torch already loaded. No vLLM, no build step.
Run it on a real DPX partition (and, for reference, on the same GPU in SPX mode):

    python dpx_gemm_probe.py --out dpx_probe.jsonl                 # default: 5 models' linear-layer shapes x 17 row counts
    python dpx_gemm_probe.py --shapes 3072x2048,16384x2048 --rows 8,64 --topk 16
    grep DPXPROBE job.log                                          # when run inside CI

It can also be dropped into a vLLM checkout as tests/kernels/test_zz_dpx_gemm_probe.py (pytest collects test_dpx_gemm_probe).

For every (weight shape m x k, rows n) of F.linear(x[n, k], W[m, k]) it prints one DPXPROBE JSON line with
  - sel: the solution index hipBLASLt selects on this device (same call torch makes), its compact name
    (MT = macro tile M x N x depthU, SK = Stream-K mode, ...) and whether the size hit a tuned table,
  - sha: sha1 of the F.linear output bits for fixed CPU-generated inputs (compare across machines / partition modes),
  - t_sel_us and the time of every alternative: the next --topk solutions of hipBLASLt's own ranking for this device
    plus the solution a full MI355X (256 CUs, chip 0x75a3) selects for the same problem (table below, only valid for
    hipBLASLt git 8d1ae90e), each run through hipblasLtMatmul with that algo,
  - best: the fastest one, and sel/best.
Timing: one HIP graph of 2 x R launches cycling over R weight copies (R x weight bytes >= 400 MB, so weights are read
cold from HBM as in a real model), replayed 6 times per round, median of 5 rounds, HIP events. Device must be idle.

Inputs come from a CPU generator on purpose: torch's GPU RNG stream depends on the reported CU count.
"""

import argparse
import ctypes as C
import hashlib
import json
import re
import statistics
import sys
import warnings

import torch
import torch.nn.functional as F

HIP_R_16BF, HIP_R_16F, HIP_R_32F = 14, 2, 0
OP_N, OP_T = 111, 112
COMPUTE_32F = 2
DESC_TRANSA, DESC_TRANSB, DESC_SM_COUNT_TARGET = 0, 1, 33
PREF_MAX_WORKSPACE_BYTES = 1
GEMM_TYPE_GEMM = 1
TORCH_WS = 79691776  # what torch 2.13 passes as max workspace (HIPBLASLT_LOG_MASK=16)


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint8 * 16), ("max_workspace_bytes", C.c_size_t)]


class Heur(C.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspaceSize", C.c_size_t),
        ("state", C.c_int),
        ("wavesCount", C.c_float),
        ("reserved", C.c_int * 4),
    ]


class _Vec(C.Structure):  # libstdc++ std::vector<T>: begin, end, end_of_storage
    _fields_ = [("b", C.c_void_p), ("e", C.c_void_p), ("c", C.c_void_p)]


class _Str(C.Structure):  # libstdc++ cxx11 std::string
    _fields_ = [("p", C.c_void_p), ("n", C.c_size_t), ("loc", C.c_char * 16)]


def _libpath():
    for line in open("/proc/self/maps"):
        if "libhipblaslt.so" in line:
            return line.split()[-1]
    raise RuntimeError("libhipblaslt not loaded; is this a ROCm torch build?")


def _chk(st, what):
    if st != 0:
        raise RuntimeError(f"{what} failed: hipblasStatus {st}")


def short(name):
    """Compact the parameters that matter out of a Tensile solution name."""
    g = lambda p: (re.search(p, name) or [None, "?"])[1]
    return (
        f"MT{g(r'_MT(\d+x\d+x\d+)')} MI{g(r'_MI(\d+x\d+x\d+)')} SK{g(r'_SK(\d+)_')} GSU{g(r'_GSU(\d+)_')}"
        f" WGM{g(r'_WGM(n?\d+)')} SU{g(r'_SU(\d+)_')} SUS{g(r'_SUS(\d+)_')} PGR{g(r'_PGR(\d+)')} PLR{g(r'_PLR(\d+)')}"
        f" LRVW{g(r'_LRVW(\d+)')} NTA{g(r'_NTA(\d+)')} NTB{g(r'_NTB(\d+)')}"
    )


class HBL:
    def __init__(self):
        torch.cuda.init()
        self.lib = lib = C.CDLL(_libpath())
        self.path = _libpath()
        self.h = C.c_void_p()
        _chk(lib.hipblasLtCreate(C.byref(self.h)), "hipblasLtCreate")
        self.ws = torch.empty(TORCH_WS, dtype=torch.uint8, device="cuda")
        self._all = None
        ext = "_ZN13hipblaslt_ext"
        self._getAll = getattr(
            lib,
            ext
            + "11getAllAlgosEPvNS_8GemmTypeE18hipblasOperation_tS2_11hipDataTypeS3_S3_S3_20hipblasComputeType_tRSt6vectorI33_hipblasLtMatmulHeuristicResult_tSaIS6_EE",
        )
        self._isSup = getattr(
            lib,
            ext
            + "21matmulIsAlgoSupportedEPvP27hipblasLtMatmulDescOpaque_tPKvP29hipblasLtMatrixLayoutOpaque_tS6_S4_S6_S6_R22_hipblasLtMatmulAlgo_tRm",
        )
        self._idx = getattr(lib, ext + "16getIndexFromAlgoER22_hipblasLtMatmulAlgo_t")
        self._sname = getattr(
            lib, ext + "23getSolutionNameFromAlgoB5cxx11EPvR22_hipblasLtMatmulAlgo_t"
        )
        self._kname = getattr(
            lib, ext + "21getKernelNameFromAlgoB5cxx11EPvR22_hipblasLtMatmulAlgo_t"
        )
        self._fromIdx = getattr(
            lib,
            ext
            + "17getAlgosFromIndexEPvRSt6vectorIiSaIiEERS1_I33_hipblasLtMatmulHeuristicResult_tSaIS5_EE",
        )
        self._idx.restype = C.c_int
        self._tuned = getattr(
            lib,
            ext
            + "13matmulIsTunedEPvP27hipblasLtMatmulDescOpaque_tP29hipblasLtMatrixLayoutOpaque_tS4_S4_S4_",
            None,
        )
        for f in (self._sname, self._kname):
            f.restype = C.c_void_p
            f.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        lib.hipblasLtMatmul.argtypes = [C.c_void_p] * 13 + [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
        ]
        lib.hipblasLtMatrixLayoutCreate.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_uint64,
            C.c_uint64,
            C.c_int64,
        ]
        self.one = C.c_float(1.0)
        self.zero = C.c_float(0.0)

    # ---- problem description: C[m,n] (col-major) = op_T(A[k,m]) * B[k,n]  == F.linear(x[n,k], W[m,k]).T
    def problem(self, m, n, k, dtype=HIP_R_16BF, sm_count_target=0):
        lib = self.lib
        p = type("P", (), {})()
        p.m, p.n, p.k = m, n, k
        p.A, p.B, p.Cd, p.desc, p.pref = (C.c_void_p() for _ in range(5))
        _chk(lib.hipblasLtMatrixLayoutCreate(C.byref(p.A), dtype, k, m, k), "layout A")
        _chk(lib.hipblasLtMatrixLayoutCreate(C.byref(p.B), dtype, k, n, k), "layout B")
        _chk(lib.hipblasLtMatrixLayoutCreate(C.byref(p.Cd), dtype, m, n, m), "layout C")
        _chk(
            lib.hipblasLtMatmulDescCreate(C.byref(p.desc), COMPUTE_32F, HIP_R_32F),
            "desc",
        )
        for attr, v in ((DESC_TRANSA, OP_T), (DESC_TRANSB, OP_N)):
            vv = C.c_int32(v)
            _chk(
                lib.hipblasLtMatmulDescSetAttribute(p.desc, attr, C.byref(vv), 4),
                "desc attr",
            )
        if sm_count_target:
            vv = C.c_int32(sm_count_target)
            _chk(
                lib.hipblasLtMatmulDescSetAttribute(
                    p.desc, DESC_SM_COUNT_TARGET, C.byref(vv), 4
                ),
                "sm_count_target",
            )
        _chk(lib.hipblasLtMatmulPreferenceCreate(C.byref(p.pref)), "pref")
        ws = C.c_uint64(TORCH_WS)
        _chk(
            lib.hipblasLtMatmulPreferenceSetAttribute(
                p.pref, PREF_MAX_WORKSPACE_BYTES, C.byref(ws), 8
            ),
            "pref ws",
        )
        return p

    def free(self, p):
        for d in (p.A, p.B, p.Cd):
            self.lib.hipblasLtMatrixLayoutDestroy(d)
        self.lib.hipblasLtMatmulDescDestroy(p.desc)
        self.lib.hipblasLtMatmulPreferenceDestroy(p.pref)

    def heuristic(self, p, count=1):
        """Ranked list as hipBLASLt returns it; element 0 is what torch runs."""
        res = (Heur * count)()
        got = C.c_int(0)
        _chk(
            self.lib.hipblasLtMatmulAlgoGetHeuristic(
                self.h, p.desc, p.A, p.B, p.Cd, p.Cd, p.pref, count, res, C.byref(got)
            ),
            "heuristic",
        )
        return [self._copy(res[i]) for i in range(got.value) if res[i].state == 0]

    @staticmethod
    def _copy(h):
        o = Heur()
        C.memmove(C.byref(o), C.byref(h), C.sizeof(Heur))
        return o

    def all_algos(self, dtype=HIP_R_16BF):
        """Every solution of the library for (T, N, bf16 x4, 32F); not yet checked against a problem."""
        if self._all is None:
            v = _Vec()
            _chk(
                self._getAll(
                    self.h,
                    GEMM_TYPE_GEMM,
                    OP_T,
                    OP_N,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    COMPUTE_32F,
                    C.byref(v),
                ),
                "getAllAlgos",
            )
            n = ((v.e or 0) - (v.b or 0)) // C.sizeof(Heur)
            arr = C.cast(v.b, C.POINTER(Heur))
            self._all = [
                self._copy(arr[i]) for i in range(n)
            ]  # the vector storage is leaked on purpose
        return self._all

    def supported(self, p, h):
        """Workspace bytes if this solution supports problem p (and fits torch's workspace), else None."""
        ws = C.c_size_t(0)
        st = self._isSup(
            self.h,
            p.desc,
            C.byref(self.one),
            p.A,
            p.B,
            C.byref(self.zero),
            p.Cd,
            p.Cd,
            C.byref(h.algo),
            C.byref(ws),
        )
        if st != 0 or ws.value > TORCH_WS:
            return None
        return ws.value

    def is_tuned(self, p):
        """1 if this exact (m, n, k) has an entry in a tuned (exact-size) table valid for this device, else 0."""
        return (
            int(self._tuned(self.h, p.desc, p.A, p.B, p.Cd, p.Cd))
            if self._tuned
            else -1
        )

    def from_index(self, idxs):
        iv = (C.c_int * len(idxs))(*idxs)
        base = C.addressof(iv)
        vin = _Vec(base, base + 4 * len(idxs), base + 4 * len(idxs))
        vout = _Vec()
        _chk(self._fromIdx(self.h, C.byref(vin), C.byref(vout)), "getAlgosFromIndex")
        n = ((vout.e or 0) - (vout.b or 0)) // C.sizeof(Heur)
        arr = C.cast(vout.b, C.POINTER(Heur))
        return [self._copy(arr[i]) for i in range(n)]

    def index(self, h):
        return self._idx(C.byref(h.algo))

    def _s(self, f, h):
        s = _Str()
        f(C.byref(s), self.h, C.byref(h.algo))
        return C.string_at(s.p, s.n).decode() if s.p else ""

    def name(self, h):
        return self._s(self._sname, h)

    def kernel(self, h):
        return self._s(self._kname, h)

    def run(self, p, h, W, x, out):
        """out[n,m] = x[n,k] @ W[m,k].T with solution h, on torch's current stream."""
        assert (
            W.shape == (p.m, p.k) and x.shape == (p.n, p.k) and out.shape == (p.n, p.m)
        )
        assert W.is_contiguous() and x.is_contiguous() and out.is_contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        _chk(
            self.lib.hipblasLtMatmul(
                self.h,
                p.desc,
                C.addressof(self.one),
                W.data_ptr(),
                p.A,
                x.data_ptr(),
                p.B,
                C.addressof(self.zero),
                out.data_ptr(),
                p.Cd,
                out.data_ptr(),
                p.Cd,
                C.addressof(h.algo),
                self.ws.data_ptr(),
                TORCH_WS,
                stream,
            ),
            "hipblasLtMatmul",
        )
        return out


# hipDeviceAttribute_t values in HIP 7.x headers; re-read from the installed header when present.
_HIP_ATTR_FALLBACK = {
    "hipDeviceAttributeMultiprocessorCount": 63,
    "hipDeviceAttributePhysicalMultiProcessorCount": 10015,
    "hipDeviceAttributeNumberOfXccs": 10018,
    "hipDeviceAttributePciChipId": 10020,
}


def _hip_attr_ids(hip_path):
    import os

    inc = os.path.join(
        os.path.dirname(os.path.dirname(hip_path)),
        "include",
        "hip",
        "hip_runtime_api.h",
    )
    try:
        body = open(inc).read()
    except OSError:
        return dict(_HIP_ATTR_FALLBACK)
    body = body[body.index("typedef enum hipDeviceAttribute_t") :]
    body = body[body.index("{") + 1 : body.index("}")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    ids, val = {}, -1
    for ent in body.split(","):
        ent = ent.strip()
        if not ent:
            continue
        if "=" in ent:
            nm, v = [t.strip() for t in ent.split("=")]
            val = ids[v] if v in ids else int(v, 0)
        else:
            nm, val = ent, val + 1
        ids[nm] = val
    return ids


def device_info():
    """What hipBLASLt's selection sees: CU count, XCC count, PCI chip id (HIP attributes)."""
    hip = None
    for line in open("/proc/self/maps"):
        if "libamdhip64" in line:
            hip = C.CDLL(line.split()[-1])
            break
    out = {
        "name": torch.cuda.get_device_properties(0).name,
        "props.multi_processor_count": torch.cuda.get_device_properties(
            0
        ).multi_processor_count,
    }
    if hip is not None:
        ids = _hip_attr_ids(hip._name)
        dev = torch.cuda.current_device()
        for nm in (
            "MultiprocessorCount",
            "PhysicalMultiProcessorCount",
            "NumberOfXccs",
            "PciChipId",
        ):
            if "hipDeviceAttribute" + nm in ids:
                v = C.c_int(0)
                # global lookup first, so an LD_PRELOAD interposer (if any) is honoured like hipBLASLt honours it
                fn = (
                    getattr(C.CDLL(None), "hipDeviceGetAttribute", None)
                    or hip.hipDeviceGetAttribute
                )
                st = fn(C.byref(v), ids["hipDeviceAttribute" + nm], dev)
                out[nm] = (
                    (hex(v.value) if nm == "PciChipId" else v.value)
                    if st == 0
                    else f"err{st}"
                )
    return out


LLC_TARGET_BYTES = 400 * 1024**2  # above the 256 MB MI355X memory-side cache


def n_rot(weight_bytes, lo=2, hi=32):
    return max(lo, min(hi, -(-LLC_TARGET_BYTES // weight_bytes)))


def timeit_graph(calls, inner=2, replays=6, rounds=5, warmup=3, budget_us=None):
    """Median over rounds of the mean time per call (microseconds). calls: list of zero-arg callables, one
    per rotating slot. All launches are captured into one HIP graph and replayed."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.cuda.synchronize()
    per = len(calls) * inner
    if budget_us:  # keep very slow candidates from eating the run: one probe replay decides the replay count
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        g.replay()
        torch.cuda.synchronize()
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        one = s.elapsed_time(e) * 1e3
        replays = max(1, min(replays, int(budget_us / max(one, 1.0) / rounds)))
        warmup = min(warmup, replays)
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        g.replay()
    for s, e in ev:
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1e3 / (replays * per) for s, e in ev]
    del g
    return statistics.median(samples), min(samples), max(samples)


# Linear-layer weight shapes (out m, in k) vLLM issues for five models (qkv and gate_up are merged in vLLM).
MODELS = {
    "Llama-3.2-1B": {
        "qkv": (3072, 2048),
        "o": (2048, 2048),
        "gate_up": (16384, 2048),
        "down": (2048, 8192),
        "lm_head": (128256, 2048),
    },
    "Qwen3-0.6B": {
        "qkv": (4096, 1024),
        "o": (1024, 2048),
        "gate_up": (6144, 1024),
        "down": (1024, 3072),
        "lm_head": (151936, 1024),
    },
    "Llama-3.1-70B": {
        "qkv": (10240, 8192),
        "o": (8192, 8192),
        "gate_up": (57344, 8192),
        "down": (8192, 28672),
        "lm_head": (128256, 8192),
    },
    "Mixtral-8x22B": {"qkv": (8192, 6144), "o": (6144, 6144), "lm_head": (32768, 6144)},
    "gpt-oss-120b": {"qkv": (5120, 2880), "o": (2880, 4096), "lm_head": (201088, 2880)},
}
ROWS = [2, 4, 6, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
# What a full MI355X (256 CUs, 8 XCCs, chip 0x75a3) selects, hipBLASLt git 8d1ae90e (ROCm 10.0.0 wheels): "mxk" -> {n: index}
SPX_PICKS_GIT = "8d1ae90e"
SPX_PICKS = {
    "3072x2048": {
        "2": 18620,
        "4": 18620,
        "6": 18620,
        "8": 18620,
        "9": 18620,
        "12": 18620,
        "16": 18620,
        "24": 18622,
        "32": 18622,
        "48": 18634,
        "64": 18619,
        "96": 18634,
        "128": 18602,
        "192": 18601,
        "256": 19228,
        "384": 19270,
        "512": 19205,
    },
    "2048x2048": {
        "2": 18620,
        "4": 18620,
        "6": 18620,
        "8": 18620,
        "9": 18620,
        "12": 18620,
        "16": 18620,
        "24": 18622,
        "32": 18622,
        "48": 19251,
        "64": 18622,
        "96": 18634,
        "128": 19234,
        "192": 19264,
        "256": 19230,
        "384": 19268,
        "512": 19229,
    },
    "16384x2048": {
        "2": 18617,
        "4": 18617,
        "6": 18617,
        "8": 18617,
        "9": 18617,
        "12": 18617,
        "16": 18617,
        "24": 18600,
        "32": 18600,
        "48": 18638,
        "64": 18599,
        "96": 18641,
        "128": 18580,
        "192": 18578,
        "256": 18595,
        "384": 18643,
        "512": 18502,
    },
    "2048x8192": {
        "2": 18615,
        "4": 18615,
        "6": 18615,
        "8": 18615,
        "9": 18615,
        "12": 18615,
        "16": 18620,
        "24": 18618,
        "32": 18618,
        "48": 18634,
        "64": 18622,
        "96": 18601,
        "128": 18580,
        "192": 18578,
        "256": 18624,
        "384": 19273,
        "512": 19112,
    },
    "128256x2048": {
        "2": 18594,
        "4": 18594,
        "6": 18594,
        "8": 18594,
        "9": 18594,
        "12": 18594,
        "16": 18594,
        "24": 18556,
        "32": 18556,
        "48": 18593,
        "64": 18569,
        "96": 18483,
        "128": 18482,
        "192": 18480,
        "256": 18478,
        "384": 18667,
        "512": 19322,
    },
    "4096x1024": {
        "2": 18615,
        "4": 18615,
        "6": 18615,
        "8": 18615,
        "9": 18615,
        "12": 18615,
        "16": 18615,
        "24": 18618,
        "32": 18618,
        "48": 18634,
        "64": 18619,
        "96": 18634,
        "128": 18602,
        "192": 18601,
        "256": 19229,
        "384": 19271,
        "512": 20578,
    },
    "1024x2048": {
        "2": 18620,
        "4": 18620,
        "6": 18620,
        "8": 18620,
        "9": 18620,
        "12": 18620,
        "16": 18620,
        "24": 18620,
        "32": 18620,
        "48": 19250,
        "64": 19250,
        "96": 19251,
        "128": 19251,
        "192": 19234,
        "256": 19234,
        "384": 19264,
        "512": 19230,
    },
    "6144x1024": {
        "2": 18616,
        "4": 18616,
        "6": 18616,
        "8": 18616,
        "9": 18616,
        "12": 18616,
        "16": 18616,
        "24": 18604,
        "32": 18604,
        "48": 18634,
        "64": 18602,
        "96": 18601,
        "128": 19228,
        "192": 18641,
        "256": 18580,
        "384": 19204,
        "512": 19203,
    },
    "1024x3072": {
        "2": 18605,
        "4": 18620,
        "6": 18620,
        "8": 18620,
        "9": 18620,
        "12": 18620,
        "16": 18620,
        "24": 18620,
        "32": 18620,
        "48": 19250,
        "64": 19250,
        "96": 19251,
        "128": 19251,
        "192": 19234,
        "256": 19234,
        "384": 19264,
        "512": 19230,
    },
    "151936x1024": {
        "2": 18607,
        "4": 18616,
        "6": 18616,
        "8": 18616,
        "9": 18616,
        "12": 18616,
        "16": 18616,
        "24": 18558,
        "32": 18558,
        "48": 18638,
        "64": 18564,
        "96": 18563,
        "128": 18562,
        "192": 18644,
        "256": 18550,
        "384": 18678,
        "512": 19328,
    },
    "10240x8192": {
        "2": 18594,
        "4": 18594,
        "6": 18594,
        "8": 18594,
        "9": 18594,
        "12": 18594,
        "16": 18617,
        "24": 18565,
        "32": 18598,
        "48": 18593,
        "64": 18564,
        "96": 18641,
        "128": 18562,
        "192": 18644,
        "256": 18502,
        "384": 18665,
        "512": 18595,
    },
    "8192x8192": {
        "2": 18621,
        "4": 18621,
        "6": 18621,
        "8": 18621,
        "9": 18621,
        "12": 18621,
        "16": 18621,
        "24": 18604,
        "32": 18604,
        "48": 18634,
        "64": 18599,
        "96": 18641,
        "128": 18567,
        "192": 18643,
        "256": 18502,
        "384": 18667,
        "512": 18478,
    },
    "57344x8192": {
        "2": 18594,
        "4": 18594,
        "6": 18594,
        "8": 18594,
        "9": 18594,
        "12": 18594,
        "16": 18594,
        "24": 18558,
        "32": 18558,
        "48": 18593,
        "64": 18557,
        "96": 18483,
        "128": 18490,
        "192": 18480,
        "256": 18478,
        "384": 18678,
        "512": 18656,
    },
    "8192x28672": {
        "2": 18594,
        "4": 18617,
        "6": 18621,
        "8": 18621,
        "9": 18621,
        "12": 18621,
        "16": 18621,
        "24": 18556,
        "32": 18556,
        "48": 18634,
        "64": 18599,
        "96": 18641,
        "128": 18567,
        "192": 18643,
        "256": 18478,
        "384": 18667,
        "512": 18478,
    },
    "128256x8192": {
        "2": 18594,
        "4": 18594,
        "6": 18594,
        "8": 20458,
        "9": 18594,
        "12": 18594,
        "16": 20459,
        "24": 18556,
        "32": 20460,
        "48": 18593,
        "64": 20461,
        "96": 18483,
        "128": 20462,
        "192": 18480,
        "256": 18478,
        "384": 18667,
        "512": 19322,
    },
    "8192x6144": {
        "2": 18621,
        "4": 18621,
        "6": 18621,
        "8": 18621,
        "9": 18621,
        "12": 18621,
        "16": 18621,
        "24": 18604,
        "32": 18604,
        "48": 18634,
        "64": 18602,
        "96": 18601,
        "128": 18580,
        "192": 18512,
        "256": 18510,
        "384": 18665,
        "512": 19197,
    },
    "6144x6144": {
        "2": 21503,
        "4": 18621,
        "6": 18621,
        "8": 18621,
        "9": 18621,
        "12": 18621,
        "16": 18621,
        "24": 18604,
        "32": 21407,
        "48": 18634,
        "64": 21456,
        "96": 18641,
        "128": 18580,
        "192": 18520,
        "256": 18518,
        "384": 18707,
        "512": 21424,
    },
    "32768x6144": {
        "2": 18594,
        "4": 18594,
        "6": 18594,
        "8": 18617,
        "9": 18594,
        "12": 18617,
        "16": 18617,
        "24": 18556,
        "32": 18570,
        "48": 18638,
        "64": 18569,
        "96": 18568,
        "128": 18567,
        "192": 18643,
        "256": 18502,
        "384": 18661,
        "512": 18478,
    },
    "5120x2880": {
        "2": 19509,
        "4": 19509,
        "6": 18611,
        "8": 19787,
        "9": 18611,
        "12": 18611,
        "16": 19648,
        "24": 19986,
        "32": 19580,
        "48": 19788,
        "64": 19918,
        "96": 19581,
        "128": 19919,
        "192": 19920,
        "256": 19721,
        "384": 19922,
        "512": 19793,
    },
    "2880x4096": {
        "2": 19486,
        "4": 19486,
        "6": 18620,
        "8": 19894,
        "9": 18620,
        "12": 18620,
        "16": 19694,
        "24": 19764,
        "32": 19831,
        "48": 19626,
        "64": 19832,
        "96": 19833,
        "128": 19834,
        "192": 19836,
        "256": 19838,
        "384": 19840,
        "512": 19842,
    },
    "201088x2880": {
        "2": 18607,
        "4": 18607,
        "6": 19665,
        "8": 19527,
        "9": 20003,
        "12": 19734,
        "16": 19666,
        "24": 20004,
        "32": 19735,
        "48": 19873,
        "64": 19530,
        "96": 18563,
        "128": 19938,
        "192": 18696,
        "256": 18683,
        "384": 18665,
        "512": 19328,
    },
}


def make_inputs(m, k, rot=True):
    g = torch.Generator().manual_seed(m * 31 + k)
    base = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    step = max(1, (64 << 20) // k)
    for i in range(0, m, step):
        base[i : i + step] = (
            (torch.randn(min(step, m - i), k, generator=g) * 0.05)
            .to(torch.bfloat16)
            .cuda()
        )
    X = torch.randn(512, k, generator=g).to(torch.bfloat16).cuda()
    Ws = [base] + (
        [base.roll(r, 0).contiguous() for r in range(1, n_rot(m * k * 2))]
        if rot
        else []
    )
    return Ws, X


def sha(t):
    return hashlib.sha1(t.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:12]


def emit(kind, **f):
    print("DPXPROBE " + json.dumps({"kind": kind, **f}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="also write the JSON lines to this file")
    ap.add_argument(
        "--models",
        default=",".join(MODELS),
        help="comma list out of: " + ", ".join(MODELS),
    )
    ap.add_argument(
        "--shapes", help="comma list of MxK weight shapes; overrides --models"
    )
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument(
        "--topk",
        type=int,
        default=12,
        help="how many of hipBLASLt's ranked solutions to time",
    )
    ap.add_argument(
        "--no-time", action="store_true", help="selection and output hashes only"
    )
    ap.add_argument(
        "--budget-ms",
        type=float,
        default=500,
        help="rough time budget per timed candidate",
    )
    a = ap.parse_args(argv)
    rows = [int(r) for r in a.rows.split(",")]
    if a.shapes:
        layers = [
            ("custom", s, int(s.split("x")[0]), int(s.split("x")[1]))
            for s in a.shapes.split(",")
        ]
    else:
        layers = [
            (mo, la, m, k)
            for mo in a.models.split(",")
            for la, (m, k) in MODELS[mo].items()
        ]
    H = HBL()
    rev = C.create_string_buffer(128)
    H.lib.hipblasLtGetGitRevision(H.h, rev)
    rev = rev.value.decode()
    free, total = torch.cuda.mem_get_info()
    dev = dict(
        device_info(),
        torch=torch.__version__,
        hipblaslt_git=rev,
        hipblaslt_lib=H.path,
        total_gib=round(total / 2**30, 1),
        free_gib=round(free / 2**30, 1),
    )
    emit("device", **dev)
    if free < 0.9 * total:
        print(
            "WARNING: device is not idle; timings will be unreliable", file=sys.stderr
        )
    fout = open(a.out, "w") if a.out else None
    if fout:
        fout.write(json.dumps({"kind": "device", **dev}) + "\n")
    use_spx = rev == SPX_PICKS_GIT
    tot = worse = 0
    for model, layer, m, k in layers:
        Ws, X = make_inputs(m, k, rot=not a.no_time)
        for n in rows:
            x = X[:n].contiguous()
            p = H.problem(m, n, k)
            ranked = H.heuristic(p, max(a.topk, 1))
            sel = ranked[0]
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                ref = F.linear(x, Ws[0])
                torch.cuda.synchronize()
            rec = {
                "model": model,
                "layer": layer,
                "m": m,
                "n": n,
                "k": k,
                "sel": H.index(sel),
                "sel_name": short(H.name(sel)),
                "tuned": H.is_tuned(p),
                "sha": sha(ref),
                "sha_row0": sha(ref[0]),
                "torch_fallback": any(
                    "recover by calling cublas" in str(w.message) for w in wl
                ),
            }
            if not a.no_time:
                cands = [(H.index(h), h, "rank%d" % i) for i, h in enumerate(ranked)]
                have = {c[0] for c in cands}
                spx = (
                    SPX_PICKS.get("%dx%d" % (m, k), {}).get(str(n)) if use_spx else None
                )
                if spx is not None and spx not in have:
                    got = H.from_index([spx])
                    if got:
                        cands.append((spx, got[0], "spx_pick"))
                rec["spx_pick"] = spx
                outs = [
                    torch.empty(n, m, dtype=torch.bfloat16, device="cuda") for _ in Ws
                ]
                rec["cands"] = []
                for idx, h, tag in cands:
                    c = {"idx": idx, "tag": tag, "name": short(H.name(h))}
                    rec["cands"].append(c)
                    if H.supported(p, h) is None:
                        c["err"] = "unsupported here"
                        continue
                    try:
                        H.run(p, h, Ws[0], x, outs[0])
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        c["err"] = str(e)
                        continue
                    c["bits_eq_sel"] = bool((outs[0] == ref).all())
                    if (
                        (outs[0].float() - ref.float()).abs().max()
                        / ref.float().abs().max()
                    ).item() > 0.05:
                        c["err"] = "wrong result"
                        continue
                    try:
                        c["us"] = round(
                            timeit_graph(
                                [
                                    (lambda W=W, o=o, h=h: H.run(p, h, W, x, o))
                                    for W, o in zip(Ws, outs)
                                ],
                                budget_us=a.budget_ms * 1e3,
                            )[0],
                            2,
                        )
                    except RuntimeError as e:
                        c["err"] = "timing: " + str(e)[:160]
                try:
                    rec["t_flinear_us"] = round(
                        timeit_graph(
                            [(lambda W=W: F.linear(x, W)) for W in Ws],
                            budget_us=a.budget_ms * 1e3,
                        )[0],
                        2,
                    )
                except RuntimeError as e:
                    rec["t_flinear_err"] = str(e)[:160]
                ok = [c for c in rec["cands"] if "us" in c]
                if ok:
                    best = min(ok, key=lambda c: c["us"])
                    s0 = next((c for c in ok if c["idx"] == rec["sel"]), None)
                    rec.update(
                        best=best["idx"],
                        best_name=best["name"],
                        best_tag=best["tag"],
                        t_best_us=best["us"],
                        t_sel_us=s0["us"] if s0 else None,
                        sel_over_best=round(s0["us"] / best["us"], 3) if s0 else None,
                    )
                    tot += 1
                    worse += bool(s0 and s0["us"] > 1.1 * best["us"])
            emit("gemm", **rec)
            if fout:
                fout.write(json.dumps({"kind": "gemm", **rec}) + "\n")
                fout.flush()
            H.free(p)
        del Ws, X
        torch.cuda.empty_cache()
    if tot:
        emit(
            "summary", timed_cases=tot, selected_more_than_10pct_slower_than_best=worse
        )


def test_dpx_gemm_probe():  # pytest entry when copied to tests/kernels/test_zz_dpx_gemm_probe.py
    if not torch.cuda.is_available() or "rocm" not in torch.__version__:
        import pytest

        pytest.skip("ROCm only")
    main([])


if __name__ == "__main__":
    main()
