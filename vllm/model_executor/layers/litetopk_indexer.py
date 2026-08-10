#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LiteTopK fused sparse top-k indexer for vLLM's DSA prefill path.

The fixed production route pair-swaps carry HOT12288 into the ordinary
paged-cache gather prefix. DeepGEMM scores and seed prep emits that prefix
once, one fixed-threshold no-histogram kernel scans only the suffix, and the
qualified GLM K=2048 path uses an h2048 physical selector with an exact
overflow continuation. One winner-only epilogue then maps final TOPK indices
back to original token positions and accumulates carry votes.

Both fused paths avoid materializing and rereading the full ``[Q, S]`` logits
matrix.

Env knobs:
  VLLM_LITETOPK=1            enable
  VLLM_LITETOPK_SO            optional prebuilt extension whose basename and
                              module name match the current source digest
  VLLM_LITETOPK_SO_SHA256     required SHA256 when VLLM_LITETOPK_SO is set
  VLLM_LITETOPK_DENSE_SELECT
                              replace vLLM's dense prefill top-k after the
                              logits GEMM with one exact histogram+select
                              kernel (default 1; set 0 to roll back)
  VLLM_LITETOPK_DENSE_SELECT_MIN_S
                              first dense length to replace (default 40960)
  VLLM_LITETOPK_DENSE_SELECT_MAX_S
                              last dense length to replace (default 262144)
  VLLM_LITETOPK_DENSE_SELECT_BINS
                              exact coarse histogram bins (fixed at 4096)
  VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB
                              minimum dense slice size (default 0 MiB)
  VLLM_LITETOPK_PRODUCTION_MIN_S
                              FP8 fused-path crossover (default 196608); an
                              explicit value also overrides the FP4 default
  VLLM_LITETOPK_FP4_PRODUCTION_MIN_S
                              FP4 fused-path crossover (default 65536)
  VLLM_LITETOPK_CHECK=1      also run the official path and log top-k recall
  VLLM_LITETOPK_DEDUP_CARRY_WAIT
                              elide an already-satisfied carry event wait (default 1)
  VLLM_LITETOPK_MERGE_CAP    per-row candidate capacity (default 196608)
  VLLM_LITETOPK_HEADROOM     bucket-scale headroom (default 0)
  VLLM_LITETOPK_OVF_LOG=1    log new candidate-count maxima
  VLLM_LITETOPK_PROBE_EVERY  overflow telemetry cadence (default 8); probed
                              chunks synchronously validate selector status;
                              other chunks device-trap before winner mapping
  VLLM_LITETOPK_OVF_WATERMARK
                              count row-chunks whose candidate count exceeds
                              this (default 65536); accumulated on device
                              every call, so the reported running max is
                              complete even though readback is sampled
"""

import functools
import hashlib
import importlib.util
import os
import sys

import torch

# Keep the JIT source inside the Python package so editable installs and wheels
# hash and compile the exact same vendored implementation.
_DSA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litetopk_kernels")
_BUILD_DIR = os.environ.get(
    "VLLM_LITETOPK_BUILD",
    os.path.expanduser("~/.cache/vllm/litetopk_build"),
)

ENABLED = os.environ.get("VLLM_LITETOPK", "0") == "1"
_PRODUCTION_MIN_S_OVERRIDE = os.environ.get("VLLM_LITETOPK_PRODUCTION_MIN_S")
PRODUCTION_MIN_S = int(_PRODUCTION_MIN_S_OVERRIDE or "196608")
FP4_PRODUCTION_MIN_S = int(
    os.environ.get(
        "VLLM_LITETOPK_FP4_PRODUCTION_MIN_S",
        _PRODUCTION_MIN_S_OVERRIDE or "65536",
    )
)
PRODUCTION_MAX_S = 1 << 20
if not (
    16384 <= PRODUCTION_MIN_S <= PRODUCTION_MAX_S
    and 16384 <= FP4_PRODUCTION_MIN_S <= PRODUCTION_MAX_S
):
    # The exact-once prefix/suffix split needs HOT12288 plus a chunk-step of
    # certified suffix below the crossover (16384 is the compressed-coordinate
    # floor for DeepSeek-V4's ratio-4 indexer; the selector cap floor is
    # enforced K-relative at the call sites).
    raise ValueError(
        "LiteTopK FP8/FP4 production min-S values must be in [16384, 1<<20]"
    )


def production_min_s(use_fp4: bool) -> int:
    """Return the qualified crossover for the selected cache format."""
    return FP4_PRODUCTION_MIN_S if use_fp4 else PRODUCTION_MIN_S


FUSED_QUERY_LEN = 8192
FUSED_TAIL_QUERY_LEN = 8128
HOT_PREFIX = 12288
DENSE_SELECT = os.environ.get("VLLM_LITETOPK_DENSE_SELECT", "1") == "1"
DENSE_SELECT_MIN_S = int(os.environ.get("VLLM_LITETOPK_DENSE_SELECT_MIN_S", "40960"))
DENSE_SELECT_MAX_S = int(os.environ.get("VLLM_LITETOPK_DENSE_SELECT_MAX_S", "262144"))
DENSE_SELECT_BINS = int(os.environ.get("VLLM_LITETOPK_DENSE_SELECT_BINS", "4096"))
DENSE_SELECT_MIN_LOGITS_MB = int(
    os.environ.get("VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB", "0")
)
if DENSE_SELECT_BINS != 4096:
    raise ValueError("VLLM_LITETOPK_DENSE_SELECT_BINS must be 4096")
if DENSE_SELECT_MIN_S < 0 or DENSE_SELECT_MAX_S < DENSE_SELECT_MIN_S:
    raise ValueError("invalid LiteTopK dense-select length interval")
if DENSE_SELECT_MIN_LOGITS_MB < 0:
    raise ValueError("VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB must be >= 0")
NB = int(os.environ.get("VLLM_LITETOPK_NB", "256"))
CHECK = os.environ.get("VLLM_LITETOPK_CHECK", "0") == "1"
_TELEMETRY = {"calls": 0, "candidate_max": 0}
# Absolute forward headroom on the bucket scale (fraction of the sample span
# prepended ABOVE the sample max). Pair with a proportionally larger NB to
# keep bucket width unchanged (e.g. HEADROOM=1.0 + NB=512 == today's width).
HEADROOM = float(os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0"))
# recent-1536 HOT reduced the real adjacent-chunk maximum to 32,864 records.
# Use the smallest supported power-of-two slab by default; overflow telemetry
# remains enabled in production qualification runs so longer-tail layers fail
# closed instead of silently truncating candidates.
MERGE_CAP = int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "196608"))
# The K-relative floor (cap >= 32*topk, e.g. 16384 at K=512) is enforced at
# the call sites where topk is known; this import-time check only rejects
# configurations no supported K could satisfy.
if MERGE_CAP < 16384:
    raise ValueError(
        "VLLM_LITETOPK_MERGE_CAP must be at least 16384 for the "
        "fixed-HOT no-hist production path"
    )
# OVF_LOG: print the running max of sampled per-row candidate counts (from
# the existing deferred 1-in-8 probe; sync-free). Sizes MERGE_CAP.
OVF_LOG = os.environ.get("VLLM_LITETOPK_OVF_LOG", "0") == "1"
_HOT_STREAM = {}
PROBE_EVERY = int(os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8"))
if PROBE_EVERY < 1:
    raise ValueError("VLLM_LITETOPK_PROBE_EVERY must be >= 1")
OVF_WATERMARK = int(os.environ.get("VLLM_LITETOPK_OVF_WATERMARK", "65536"))
# Enriched sampling uses positions from the previous part's top-k indices.
# Any selected position set still gives a valid exact-subset bound. Zero
# disables it; a positive value is the hot-column budget.
HOTSAMPLE = HOT_PREFIX
# HOTONLY: the hot columns ARE the whole sample (no uniform probe). The
# subset bound stays provable for any chosen sample; drift discovery is
# carried by the scan->select->carry loop itself. Probe survives only as
# the cold-start fallback (hot_prev is None).
HOTONLY = True
# Real adjacent-chunk capture selected this window: all K winners from the last
# 1536 query rows predict the next chunk substantially better than the old
# rotating 1/8 sample over all rows, while adding only atomics to the mandatory
# winner-map pass.
CARRY_RECENT_ROWS = 1536
# The production selector parallelizes each 256-bin radix prefix search in
# warp 0 and selects the remaining 16 score bits in two passes.
# The carry-ready event also guards reuse of its vote slab. The main stream
# consumes both in one exact-once call, so the second wait can be elided when
# it is provably the exact same Event object.
DEDUP_CARRY_WAIT = os.environ.get("VLLM_LITETOPK_DEDUP_CARRY_WAIT", "1") == "1"
_HOT_CARRY = {}
# GATE4 writes BUCKET-SPACE high24 candidates (affine order-preserving).
# Both seed-prefix emission and the suffix producer use the same packed score
# contract, so the mapped postpass can process their concatenation directly.

_EXT = None
_FAILED = False
_AUX_CACHE = {}  # (device, head) -> (zeros[Qmax], full_head[Qmax]) int32
_DENSE_SELECT_LOGGED = False
_DENSE_DECLINE_LOGGED = False


def _dense_decline_note(stage, seq_len_hint, logits, out_indices, topk):
    """One-shot diagnostic: dense-select declined while S was in-window."""
    global _DENSE_DECLINE_LOGGED
    if (
        _DENSE_DECLINE_LOGGED
        or not ENABLED
        or not DENSE_SELECT
        or not DENSE_SELECT_MIN_S <= seq_len_hint <= DENSE_SELECT_MAX_S
    ):
        return
    _DENSE_DECLINE_LOGGED = True
    print(
        f"[litetopk] dense-select declined in-window at {stage}: "
        f"S={seq_len_hint} topk={topk} "
        f"logits={tuple(logits.shape)}/{logits.dtype}/"
        f"strides=({logits.stride(0)},{logits.stride(1)}) "
        f"bytes={logits.numel() * logits.element_size()} "
        f"out={tuple(out_indices.shape)}/{out_indices.dtype}/"
        f"contig={out_indices.is_contiguous()}",
        flush=True,
    )


_SINGLE_SCAN_LOGGED = False


def _dsa_source_id():
    digest = hashlib.sha256()
    for filename in (
        "dsa_litetopk.cu",
        "sm100_dsa_litetopk.cuh",
        "dense_topk_litetopk.cuh",
    ):
        path = os.path.join(_DSA_DIR, filename)
        digest.update(filename.encode())
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def _ks0_keh(Q, head, dev):
    """Cached zero-starts and sample-end tensors: torch.zeros/torch.full are a
    kernel launch each and measurably cost ~0.1-0.2ms/chunk on the hot path."""
    key = (str(dev), head)
    entry = _AUX_CACHE.get(key)
    if entry is None or entry[0].shape[0] < Q:
        qmax = max(Q, 1024)
        entry = (
            torch.zeros(qmax, dtype=torch.int32, device=dev),
            torch.full((qmax,), head, dtype=torch.int32, device=dev),
        )
        _AUX_CACHE[key] = entry
    return entry[0][:Q], entry[1][:Q]


def _ext():
    global _EXT, _FAILED
    if _EXT is None and not _FAILED:
        try:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
            source_id = _dsa_source_id()
            name = f"vllm_litetopk_dsa_b200_production_{source_id}"
            override_path = os.environ.get("VLLM_LITETOPK_SO", "")
            override_sha256 = os.environ.get("VLLM_LITETOPK_SO_SHA256", "")
            if bool(override_path) != bool(override_sha256):
                raise RuntimeError(
                    "VLLM_LITETOPK_SO and VLLM_LITETOPK_SO_SHA256 must be set together"
                )
            if override_path:
                resolved_path = os.path.realpath(os.path.expanduser(override_path))
                expected_basename = f"{name}.so"
                if os.path.basename(resolved_path) != expected_basename:
                    raise RuntimeError(
                        "LiteTopK override basename must be "
                        f"{expected_basename}, got "
                        f"{os.path.basename(resolved_path)}"
                    )
                if not os.path.isfile(resolved_path):
                    raise FileNotFoundError(
                        f"LiteTopK override does not exist: {resolved_path}"
                    )
                expected_sha256 = override_sha256.lower()
                if len(expected_sha256) != 64 or any(
                    c not in "0123456789abcdef" for c in expected_sha256
                ):
                    raise RuntimeError(
                        "VLLM_LITETOPK_SO_SHA256 must be 64 hexadecimal characters"
                    )
                digest = hashlib.sha256()
                with open(resolved_path, "rb") as binary:
                    for chunk in iter(lambda: binary.read(1 << 20), b""):
                        digest.update(chunk)
                actual_sha256 = digest.hexdigest()
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        "LiteTopK override SHA256 mismatch: expected "
                        f"{expected_sha256}, got {actual_sha256}"
                    )
                spec = importlib.util.spec_from_file_location(name, resolved_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot create module spec for {resolved_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
                _EXT = module
                load_kind = "prebuilt"
            else:
                from torch.utils.cpp_extension import load

                dg25 = os.environ.get("DEEPGEMM_DIR")
                src = "dsa_litetopk.cu"
                bdir = f"{_BUILD_DIR}_production_{source_id}"
                if dg25:
                    dg_inc = os.path.join(dg25, "deep_gemm/include")
                    cutlass_inc = os.path.join(dg25, "third-party/cutlass/include")
                    if not (os.path.isdir(dg_inc) and os.path.isdir(cutlass_inc)):
                        raise RuntimeError(
                            "DEEPGEMM_DIR does not contain DeepGEMM and CUTLASS headers"
                        )
                    incs = [_DSA_DIR, dg_inc, cutlass_inc]
                else:
                    # Prefer a pinned external package, then the copy installed
                    # in official vLLM wheels. Both bundle DeepGEMM and CUTLASS
                    # headers under the package include directory.
                    from vllm.utils.deep_gemm import _import_deep_gemm

                    deep_gemm = _import_deep_gemm()
                    if deep_gemm is None or not getattr(deep_gemm, "__file__", None):
                        raise RuntimeError(
                            "DeepGEMM is unavailable; install a compatible "
                            "package or set DEEPGEMM_DIR"
                        )
                    pkg_inc = os.path.join(
                        os.path.dirname(deep_gemm.__file__), "include"
                    )
                    if not os.path.isfile(
                        os.path.join(pkg_inc, "cutlass/arch/barrier.h")
                    ):
                        raise RuntimeError(
                            "DeepGEMM/CUTLASS headers not found; set "
                            "DEEPGEMM_DIR to a DeepGEMM 2.5 checkout"
                        )
                    incs = [_DSA_DIR, pkg_inc]
                cuda_flags = [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ]
                if os.environ.get("LITETOPK_LINEINFO") == "1":
                    cuda_flags.append("-lineinfo")
                os.makedirs(bdir, exist_ok=True)
                _EXT = load(
                    name=name,
                    sources=[os.path.join(_DSA_DIR, src)],
                    extra_include_paths=incs,
                    extra_cuda_cflags=cuda_flags,
                    build_directory=bdir,
                    extra_ldflags=["-lcuda"],
                    verbose=False,
                )
                load_kind = "JIT"
            reported_u16 = getattr(_EXT, "candidate_value_u16_litetopk", None)
            if reported_u16 is None or not bool(reported_u16()):
                raise RuntimeError(
                    "loaded LiteTopK extension is not the U16 candidate ABI"
                )
            reported_fp24 = getattr(_EXT, "candidate_fp24_global_litetopk", None)
            if reported_fp24 is None or not bool(reported_fp24()):
                raise RuntimeError(
                    "loaded LiteTopK extension is not the production high24 ABI"
                )
            for required_op in (
                "plan_and_permuted_paged_gather_out",
                "h2048_safe_topk_out_litetopk_",
                "dense_topk_litetopk_",
            ):
                if not hasattr(_EXT, required_op):
                    raise RuntimeError(
                        "loaded LiteTopK extension is missing required op "
                        f"{required_op}"
                    )
            print(
                f"[litetopk] using {load_kind} fixed vendored B200 "
                f"production kernel (source={source_id})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            _FAILED = True
            print(
                f"[litetopk] extension load/build failed, falling back: {e}",
                flush=True,
            )
    return _EXT


@functools.cache
def production_extension_available(*, use_fp4: bool, topk: int) -> bool:
    """Whether the loaded extension can serve an unsplit production chunk.

    Metadata planning must use this stronger check instead of treating the
    environment switch and SM version as proof that the extension can run. If
    JIT compilation, a prebuilt override, or an ABI check fails, returning
    ``False`` keeps the stock path's logits-budget chunking intact.
    """
    if (
        not ENABLED
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0)
        or topk <= 0
        or topk > 2048
        or max(16384, 32 * topk) > MERGE_CAP
    ):
        return False

    ext = _ext()
    if ext is None:
        return False

    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None or not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        return False

    common_ops = (
        "plan_and_permuted_paged_gather_out",
        "seed_prep_litetopk_",
        "map_topk_indices_and_accumulate_votes_litetopk_",
        "cand_count_stats_litetopk_",
    )
    scan_op = (
        "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_"
        if use_fp4
        else "mqa_logits_dsa_static_hot_nohist_litetopk_"
    )
    use_h2048_safe = (
        topk == 2048
        and max(16384, 32 * topk) <= MERGE_CAP <= PRODUCTION_MAX_S
        and NB == 256
    )
    selector_ops = (
        ("h2048_safe_topk_out_litetopk_",)
        if use_h2048_safe
        else (
            "finalize_static_hot_meta_litetopk_",
            "compact_topk_min_thr_inplace_idx_out_litetopk",
        )
    )
    return all(hasattr(ext, op) for op in (*common_ops, scan_op, *selector_ops))


def try_dense_topk(
    logits,
    cu_seqlen_ks,
    cu_seqlen_ke,
    out_indices,
    topk,
    *,
    seq_len_hint,
    num_init_tokens=0,
    num_local_tokens=0,
):
    """Exact replacement for vLLM's dense prefill top-k on measured lengths.

    One CTA per row builds an FP16-coarse histogram in shared memory, then
    reuses the same allocation to select the exact FP32 winners. Oversized or
    tied cutoff buckets refine through a device-only exact radix fallback; no
    global metadata, host synchronization, or ``-1`` fallback is used.
    """
    global _DENSE_SELECT_LOGGED
    if (
        not ENABLED
        or not DENSE_SELECT
        or seq_len_hint < DENSE_SELECT_MIN_S
        or seq_len_hint > DENSE_SELECT_MAX_S
        or logits.numel() * logits.element_size()
        < DENSE_SELECT_MIN_LOGITS_MB * (1 << 20)
        or logits.device.type != "cuda"
        or torch.cuda.get_device_capability(logits.device) != (10, 0)
        or logits.dtype != torch.float32
        or logits.dim() != 2
        or logits.stride(0) <= 0
        or logits.stride(1) <= 0
        or out_indices.dtype != torch.int32
        or out_indices.dim() != 2
        or not out_indices.is_contiguous()
        or out_indices.device != logits.device
        or cu_seqlen_ks.device != logits.device
        or cu_seqlen_ke.device != logits.device
        or topk <= 0
        or topk > 2048
        or num_init_tokens < 0
        or num_local_tokens < 0
        or num_init_tokens + num_local_tokens >= topk
    ):
        _dense_decline_note("gates", seq_len_hint, logits, out_indices, topk)
        return False
    rows = logits.shape[0]
    if (
        rows <= 0
        or out_indices.shape != (rows, topk)
        or cu_seqlen_ks.dtype != torch.int32
        or cu_seqlen_ke.dtype != torch.int32
        or not cu_seqlen_ks.is_contiguous()
        or not cu_seqlen_ke.is_contiguous()
        or cu_seqlen_ks.numel() < rows
        or cu_seqlen_ke.numel() < rows
    ):
        _dense_decline_note("shapes", seq_len_hint, logits, out_indices, topk)
        return False
    ext = _ext()
    if ext is None or not hasattr(ext, "dense_topk_litetopk_"):
        _dense_decline_note("ext", seq_len_hint, logits, out_indices, topk)
        return False
    ext.dense_topk_litetopk_(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out_indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        topk,
        num_init_tokens,
        num_local_tokens,
    )
    if not _DENSE_SELECT_LOGGED:
        print(
            "[litetopk] exact fused dense histogram+selector active "
            f"(S={seq_len_hint}, rows={rows}, bins={DENSE_SELECT_BINS})",
            flush=True,
        )
        _DENSE_SELECT_LOGGED = True
    return True


_HINTS_VALIDATED = False
_PENDING_TELEMETRY = None  # (cuda event, pinned int32 tensor)
# VLLM_LITETOPK_PATH_TIMING=1: CUDA-event sub-segment timing of the fused
# exact-once chain (seed | scan | select | map+vote+probe), keyed by 64K band
# of S. Lazy readback of completed pairs only — sync-free.
_SEG_ON = os.environ.get("VLLM_LITETOPK_PATH_TIMING", "0") == "1"
_SEG_STATS: dict = {}


def _seg_mark():
    if not _SEG_ON:
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _seg_commit(seq_len, evs):
    if not _SEG_ON or not evs or evs[0] is None:
        return
    key = f"fxseg_s{(seq_len + 65535) // 65536}"
    rec = _SEG_STATS.setdefault(key, {"pend": [], "tot": [0.0] * 8, "n": 0})
    rec["pend"].append(evs)
    while rec["pend"] and rec["pend"][0][-1].query():
        seq = rec["pend"].pop(0)
        for i in range(len(seq) - 1):
            rec["tot"][i] += seq[i].elapsed_time(seq[i + 1])
        rec["n"] += 1
        if rec["n"] % 128 == 0:
            segs = " ".join(f"{t / rec['n']:.3f}" for t in rec["tot"][: len(seq) - 1])
            print(
                f"[litetopk] {key}: n={rec['n']} seed|scan|select|map_ms=[{segs}]",
                flush=True,
            )


_CAND_ACC = None  # (device running max[1], device over-watermark count[1]):
# accumulated unconditionally every call so the sampled
# probe readback still reports the complete running max
_PROBE_RES = None  # cached (device stats, pinned buffer, event): allocating
# these per arm blocked the CPU ~17ms inside
# cudaHostAlloc-class calls (nsys), starving the GPU
# stream for ~3.4ms/call at 256K/Q=512 when throttled


def _check_selector_status(
    status,
    candidate_count,
    *,
    stage,
    sequence_length,
    common_end,
    cap,
    layer,
):
    # The item() is intentional: sampled calls must observe this selector's
    # result before winner mapping and attention consume its output.
    selector_status = int(status.amax().item())
    if selector_status == 0:
        return
    candidate_max = int(candidate_count.amax().item())
    if stage == "h2048-safe-select":
        status_help = (
            "1=bad count, 2=nonfinite score, "
            "4=invalid physical ID, 16=invalid certificate, "
            "32=unrecovered overflow, 64=fallback compact failure"
        )
    else:
        status_help = (
            "1=bad count, 2=underfill, 4=invalid threshold, "
            "8=invalid boundary, 16=invalid index map"
        )
    raise RuntimeError(
        "large exact-once stage="
        f"{stage} status={selector_status} "
        f"({status_help}); "
        f"S={sequence_length}, common_end={common_end}, cap={cap}, "
        f"candidate_max={candidate_max}, layer={layer}"
    )


def _poll_candidate_telemetry():
    """Non-blocking read of the previous chunk's candidate telemetry."""
    global _PENDING_TELEMETRY
    if _PENDING_TELEMETRY is not None:
        ev, pinned, kk, watermark = _PENDING_TELEMETRY
        if ev.query():  # finished long ago; no sync
            mx = int(pinned[0])
            run_max = int(pinned[2])
            over = int(pinned[3])
            if OVF_LOG and run_max > _TELEMETRY["candidate_max"]:
                # run_max/over are device-accumulated over EVERY call; only
                # the readback is sampled, so the printed max is the true
                # running max as of the probed chunk
                _TELEMETRY["candidate_max"] = run_max
                print(
                    f"[litetopk] cand max -> {run_max} "
                    f"(probed chunk max {mx}, "
                    f"mean {float(pinned[1]) / kk:.2f}xK, "
                    f"row-chunks over {watermark}: {over})",
                    flush=True,
                )
            _PENDING_TELEMETRY = None


_PREP_BUFS = {}  # (dev, NB) -> dict of caller-owned seed_prep buffers
_SLOG_SLABS = {}  # dev -> persistent seed-GEMM logits slab (out= reuse)
_OPS_VERIFIED = None  # required-ops hasattr walk, done once per ext load


def _slog_slab(Q, seq_len_kv, dev):
    """Persistent output slab for the seed GEMM: kills the ~392 MiB
    alloc/free per fused call. Sized generously for DeepGEMM's internal
    [align(Q, block_q), align(seq_len_kv + block_kv, 8)] padding."""
    need = (Q + 8) * (seq_len_kv + 512)
    slab = _SLOG_SLABS.get(str(dev))
    if slab is None or slab.numel() < need:
        _SLOG_SLABS[str(dev)] = None
        slab = torch.empty(need, dtype=torch.float32, device=dev)
        _SLOG_SLABS[str(dev)] = slab
    return slab


_CAND_BUFS = {}  # dev -> opaque U16 slab carrying delayed-high24 codes
_VOTE_BUF_HOT = {}  # dev -> persistent stash-carry vote histogram
_CARRY_VOTE_BUFS = {}  # (dev, layer) -> selector-fused vote slab + free event
_CARRY_TOPK_WORKSPACE = {}  # dev -> single-side-stream partial/state workspace
_CARRY_TOPK_MAX_BLOCKS = 128
_CARRY_TOPK_STATE_INTS = 136
# One pair-swap workspace is owned by each main CUDA stream.  Planning and the
# paged gather are submitted together through the production extension; no
# side-stream plan, prepared ticket, or per-layer permutation cache exists.
_PAIR_PLAN_BUFS = {}
_PAIR_PLAN_EPOCH = {}


def _stream_id(dev):
    return (
        int(torch.cuda.current_stream(dev).cuda_stream)
        if getattr(dev, "type", None) == "cuda"
        else 0
    )


def _pair_plan_bufs(sequence_length, dev):
    """Persistent pair-swap planner workspace with geometric growth."""
    key = (str(dev), _stream_id(dev))
    state = _PAIR_PLAN_BUFS.get(key)
    if state is None or state["cap"] < sequence_length:
        cap = max(16384, 1 << (sequence_length - 1).bit_length())
        state = {
            "cap": cap,
            "hot_epoch": torch.zeros(cap, dtype=torch.int32, device=dev),
            "permutation": torch.arange(cap, dtype=torch.int32, device=dev),
            "swap_a": torch.empty(HOT_PREFIX, dtype=torch.int32, device=dev),
            "swap_b": torch.empty(HOT_PREFIX, dtype=torch.int32, device=dev),
            # [previous pair count, current A count, current B count, status]
            "counts": torch.zeros(4, dtype=torch.int32, device=dev),
        }
        _PAIR_PLAN_BUFS[key] = state
        _PAIR_PLAN_EPOCH[key] = 0
    epoch = _PAIR_PLAN_EPOCH.get(key, 0) + 1
    if epoch >= (1 << 31):
        # This is many years of continuous prefill calls. Recreate the planner
        # state instead of making epoch wrap a correctness concern.
        del _PAIR_PLAN_BUFS[key]
        _PAIR_PLAN_EPOCH.pop(key, None)
        return _pair_plan_bufs(sequence_length, dev)
    _PAIR_PLAN_EPOCH[key] = epoch
    return state, epoch


def _retire_request_state(dev):
    """Retire all state after a confirmed per-layer carry-extent rollback."""
    device_index = dev.index if dev.index is not None else torch.cuda.current_device()
    dev_key = str(torch.device("cuda", device_index))

    request_state = (
        _HOT_CARRY,
        _CARRY_VOTE_BUFS,
        _PAIR_PLAN_BUFS,
    )
    has_request_state = any(
        any(isinstance(key, tuple) and key and key[0] == dev_key for key in cache)
        for cache in request_state
    )
    side = _HOT_STREAM.get(dev_key)
    if has_request_state and side is not None:
        side.synchronize()

    for cache in request_state:
        stale_keys = [
            key for key in cache if isinstance(key, tuple) and key and key[0] == dev_key
        ]
        for key in stale_keys:
            cache.pop(key, None)

    stale_epoch_keys = [
        key
        for key in _PAIR_PLAN_EPOCH
        if isinstance(key, tuple) and key and key[0] == dev_key
    ]
    for key in stale_epoch_keys:
        _PAIR_PLAN_EPOCH.pop(key, None)


def release_pair_swap_workspace(dev):
    """Drop the pair-swap planner workspace owned by the current stream."""
    key = (str(dev), _stream_id(dev))
    _PAIR_PLAN_BUFS.pop(key, None)
    _PAIR_PLAN_EPOCH.pop(key, None)


def _vote_hist(nv, dev):
    """Reuse one side-stream vote histogram for official-path carry seeding."""
    cache = _VOTE_BUF_HOT
    key = str(dev)
    b = cache.get(key)
    if b is None or b.numel() < nv:
        b = torch.empty(max(nv, 1024), dtype=torch.int32, device=dev)
        cache[key] = b
    buf = b[:nv]
    buf.zero_()
    return buf


def _cand_bufs(Q, cap, dev):
    key = str(dev)
    b = _CAND_BUFS.get(key)
    if b is None or b["cap"] != cap or b["q"] < Q:
        _CAND_BUFS[key] = None  # drop old slab BEFORE allocating the new one
        del b
        qm = max(Q, 1024)
        b = {
            "q": qm,
            "cap": cap,
            # float16 is opaque 16-bit storage in the packed ABI; CUDA
            # interprets its bits as uint16 rather than doing half arithmetic.
            "cv": torch.empty(qm, cap, dtype=torch.float16, device=dev),
            "ci": torch.empty(qm, cap, dtype=torch.int32, device=dev),
        }
        _CAND_BUFS[key] = b
    return b


def _carry_vote_hist(nv, dev, hot_key, waited_event=None):
    """Acquire a per-layer histogram for selector-fused carry votes."""
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS.get(key)
    if entry is not None and entry["free_event"] is not None:
        free_event = entry["free_event"]
        if not (DEDUP_CARRY_WAIT and free_event is waited_event):
            torch.cuda.current_stream(dev).wait_event(free_event)
    if entry is None or entry["buf"].numel() < nv:
        cap = max(1024, 1 << (nv - 1).bit_length())
        hot = None if entry is None else entry.get("hot")
        ready_event = None if entry is None else entry.get("ready_event")
        if ready_event is None:
            ready_event = torch.cuda.Event()
        entry = {
            # The carry top-k kernel clears every live vote. Zero the whole
            # geometric slab once so future growth inside this capacity also
            # exposes clean, never-before-used tail positions.
            "buf": torch.zeros(cap, dtype=torch.int32, device=dev),
            "free_event": None,
            "ready_event": ready_event,
            "hot": (
                hot
                if hot is not None
                else torch.empty(max(HOTSAMPLE, 1), dtype=torch.int64, device=dev)
            ),
            "needs_reset": False,
            "dirty_extent": 0,
        }
        _CARRY_VOTE_BUFS[key] = entry
    votes = entry["buf"][:nv]
    if entry.get("needs_reset", False):
        reset_extent = max(nv, entry.get("dirty_extent", 0))
        entry["buf"][:reset_extent].zero_()
        entry["needs_reset"] = False
        entry["dirty_extent"] = 0
    # The selector that follows will dirty the slab. A successfully enqueued
    # custom publisher clears this flag because its K2 owns the reset.
    entry["needs_reset"] = True
    entry["dirty_extent"] = max(entry.get("dirty_extent", 0), nv)
    return votes


def _carry_topk_workspace(max_vote, dev):
    """Caller-owned workspace serialized by the one side stream per device."""
    key = str(dev)
    bins = max_vote + 1
    entry = _CARRY_TOPK_WORKSPACE.get(key)
    if entry is None or entry["partial"].shape[1] < bins:
        entry = {
            "partial": torch.empty(
                (_CARRY_TOPK_MAX_BLOCKS, bins),
                dtype=torch.int16,
                device=dev,
            ),
            # state[0] is the reusable last-block completion ticket.
            "state": torch.zeros(
                _CARRY_TOPK_STATE_INTS,
                dtype=torch.int32,
                device=dev,
            ),
        }
        _CARRY_TOPK_WORKSPACE[key] = entry
    return entry


_CARRY_TIMING = os.environ.get("VLLM_LITETOPK_CARRY_TIMING", "0") == "1"
# Publish the voted-hot carry every Nth fused chunk. The skip decision runs
# BEFORE the vote-hist acquire and the map-kernel accumulation, so skipped
# chunks pay neither the acquire zeroing, the vote atomics, nor the publish
# select (0.35 ms/call side-stream). The published hot set is voted from the
# publish chunk's recent rows only -- the same information the every-chunk
# scheme retains, since an unpublished accumulate was discarded by the
# needs_reset safety net anyway. (An earlier hang attributed to stride > 1
# was misattributed; the lifecycle audit found no hang or corruption path.)
_CARRY_EVERY = max(1, int(os.environ.get("VLLM_LITETOPK_CARRY_EVERY", "1")))
_CARRY_SKIP_COUNTS: dict = {}
_CARRY_IO_ENV = os.environ.get("VLLM_LITETOPK_CARRY_IO", "1") == "1"
_CARRY_TIME_STATS = {"pend": [], "tot": 0.0, "n": 0}


def _carry_time_commit(ev0, ev1):
    st = _CARRY_TIME_STATS
    st["pend"].append((ev0, ev1))
    while st["pend"] and st["pend"][0][1].query():
        a, b = st["pend"].pop(0)
        st["tot"] += a.elapsed_time(b)
        st["n"] += 1
        if st["n"] % 256 == 0:
            print(
                f"[litetopk] carry-timing: n={st['n']} "
                f"side_ms={st['tot'] / st['n']:.3f}",
                flush=True,
            )


def _publish_carry(hot_key, votes, nv, min_index, max_vote):
    """Publish HOT12288 on the per-device side stream."""
    if nv - min_index < HOT_PREFIX:
        return
    dev = votes.device
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS[key]
    side = _HOT_STREAM.get(str(dev))
    if side is None:
        side = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = side
    carry_ext = _ext()
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        if _CARRY_TIMING:
            _ct0 = torch.cuda.Event(enable_timing=True)
            _ct0.record(side)
        votes.record_stream(side)
        hot_n = HOT_PREFIX
        use_custom = (
            carry_ext is not None
            and hasattr(carry_ext, "carry_votes_topk_reset_")
            and 0 < HOTSAMPLE <= HOT_PREFIX
            and nv <= 1_048_576
            and 0 < max_vote <= 8192
        )
        if use_custom:
            hot = entry["hot"][:hot_n]
            workspace = _carry_topk_workspace(max_vote, dev)
            carry_ext.carry_votes_topk_reset_(
                votes,
                hot,
                workspace["partial"],
                workspace["state"],
                hot_n,
                max_vote,
                min_index,
            )
            entry["needs_reset"] = False
            entry["dirty_extent"] = 0
        else:
            if min_index > 0:
                votes[:min_index].fill_(torch.iinfo(torch.int32).min)
            hot = votes.topk(hot_n).indices
        ready = entry["ready_event"]
        ready.record(side)
        if _CARRY_TIMING:
            _ct1 = torch.cuda.Event(enable_timing=True)
            _ct1.record(side)
            _carry_time_commit(_ct0, _ct1)
    entry["free_event"] = ready
    _HOT_CARRY[key] = (hot, nv, ready, min_index)


def _prep_bufs(Q, nb, cap, dev):
    """Caller-owned seed_prep outputs. Reusing these kills the 0.1-0.5GB
    per-call alloc churn that forced the CUDA allocator into pathological
    behavior at small Q (256K: 4.5ms without an event-paced probe)."""
    key = (str(dev), nb)
    b = _PREP_BUFS.get(key)
    if b is None or b["q"] < Q:
        qm = max(Q, 1024)
        b = {
            "q": qm,
            "o": torch.empty(qm, device=dev),
            "inv": torch.empty(qm, device=dev),
            "th": torch.empty(qm, dtype=torch.int32, device=dev),
            "bc": torch.empty(qm, nb, dtype=torch.int32, device=dev),
            "cc": torch.empty(qm, dtype=torch.int32, device=dev),
            "status": torch.empty(qm, dtype=torch.int32, device=dev),
        }
        _PREP_BUFS[key] = b
    return b


def prepare_permuted_gather(
    kv_cache,
    dst_k,
    dst_scale,
    block_table,
    *,
    sequence_length,
    query_length,
    num_reqs,
    common_end,
    window_start,
    hot_key,
):
    """Pair-swap HOT12288 into the paged-gather prefix on the main stream."""
    try:
        S = int(sequence_length)
        Q = int(query_length)
        ks = int(window_start)
        common_end = int(common_end)
        if (
            not ENABLED
            or not HOTONLY
            or HOTSAMPLE != HOT_PREFIX
            or not (
                production_min_s(dst_k.dtype == torch.uint8) <= S <= PRODUCTION_MAX_S
            )
            or Q not in (FUSED_QUERY_LEN, FUSED_TAIL_QUERY_LEN)
            or num_reqs != 1
            or hot_key is None
            or ks < 0
            or ks % 4 != 0
            or ks + HOT_PREFIX > common_end
            or common_end > S
            or dst_scale.shape != (S, 4)
            or dst_scale.dtype != torch.uint8
            # fp8 rows are 128 fp8 bytes + one packed fp32 scale; fp4 rows are
            # 64 packed e2m1 bytes + 4 ue8m0 scale bytes (same (S, 4) uint8).
            or (tuple(dst_k.shape), dst_k.dtype)
            not in (
                ((S, 128), torch.float8_e4m3fn),
                ((S, 64), torch.uint8),
            )
            or dst_k.device.type != "cuda"
            or torch.cuda.get_device_capability(dst_k.device) != (10, 0)
        ):
            return None
        carry = _HOT_CARRY.get((str(dst_k.device), hot_key))
        carry_valid = not (
            carry is None
            or len(carry) != 4
            or carry[1] > common_end
            or carry[3] < ks
            or carry[0].dim() != 1
            or carry[0].numel() < HOT_PREFIX
        )
        if (
            not carry_valid
            and os.environ.get("VLLM_LITETOPK_COLDSTART_IDENTITY", "0") != "1"
        ):
            if os.environ.get("VLLM_LITETOPK_CARRY_DEBUG", "0") == "1":
                if carry is None:
                    why = "missing"
                else:
                    carry_ext = int(carry[1]) if len(carry) > 1 else -1
                    carry_min = int(carry[3]) if len(carry) > 3 else -1
                    why = (
                        f"len={len(carry)} ext={carry_ext} "
                        f"vs common_end={common_end}, min={carry_min} "
                        f"vs ks={ks}"
                    )
                print(
                    f"[litetopk] carry invalid ({why}) S={S} Q={Q}",
                    flush=True,
                )
            return None
        ext = _ext()
        if ext is None or not hasattr(ext, "plan_and_permuted_paged_gather_out"):
            return None
        if carry_valid:
            hot = carry[0][:HOT_PREFIX]
            carry_event = carry[2]
        else:
            # Identity cold start: HOT = the physical window prefix. The seed
            # gate starts looser than a voted-hot carry, and the ring daemon
            # tightens it during the scan; recall is machinery-guaranteed.
            hot = torch.arange(
                ks, ks + HOT_PREFIX, dtype=torch.int32, device=dst_k.device
            )
            carry_event = None
        if carry_event is not None:
            carry_event.wait()
        hot.record_stream(torch.cuda.current_stream(dst_k.device))
        state, epoch = _pair_plan_bufs(S, dst_k.device)
        permutation = state["permutation"][:S]
        ext.plan_and_permuted_paged_gather_out(
            hot,
            state["hot_epoch"][:S],
            permutation,
            state["swap_a"],
            state["swap_b"],
            state["counts"],
            ks,
            common_end,
            epoch,
            kv_cache,
            dst_k.view(torch.uint8),
            dst_scale,
            block_table,
        )
        return {
            "permutation": permutation,
            "carry_event": carry_event,
            "sequence_length": S,
            "query_length": Q,
            "window_start": ks,
            "common_end": common_end,
            # Hold planner storage until the selector has consumed its map.
            "planner_state": state,
        }
    except Exception as e:  # noqa: BLE001
        detail = ""
        if os.environ.get("VLLM_LITETOPK_CARRY_DEBUG", "0") == "1":
            try:
                parts = []
                for name, t in (
                    ("kv_cache", kv_cache),
                    ("dst_k", dst_k),
                    ("dst_scale", dst_scale),
                    ("block_table", block_table),
                ):
                    parts.append(
                        f"{name}: shape={tuple(t.shape)} "
                        f"stride={tuple(t.stride())} "
                        f"contig={t.is_contiguous()} dtype={t.dtype}"
                    )
                detail = " | " + "; ".join(parts)
            except Exception:
                pass
        print(
            f"[litetopk] exact-once permuted gather declined: {e}{detail}",
            flush=True,
        )
        return None


def stash_carry(hot_key, idx, S, min_index=0, *, next_sequence_length=None):
    """Seed a layer's hot carry from the OFFICIAL path's topk output, called
    by the container on the LAST official chunk before MIN_S. The
    official->ours boundary is deterministic, so this one seed is all the
    first ours-chunk needs to run HOT (no cold start, no cold prefix). Stored
    compressed (voted hot columns, ~64KB/layer).

    The vote+topk selection and the store run on a per-device SIDE STREAM
    (async): seeding overlaps the model forward instead of stalling the
    official path. The exact-once gather consumer waits on the stored event
    before touching the carry."""
    if hot_key is None or not (HOTONLY and HOTSAMPLE > 0):
        return
    dev = idx.device
    nv = int(S)
    if nv - min_index < HOT_PREFIX:
        return
    # max_tokens=1 can finish directly from the final prefill logits, without
    # ever entering the no-prefill/decode branch that normally retires this
    # state.  Across prefill steps a layer's carry extent is strictly
    # increasing, so a *decrease* identifies a new request.  Equal extents are
    # expected when the 2-GiB logits budget splits one prefill step into several
    # internal Q chunks; treating equality as a reset drops the carry already
    # published by the other layers.  The first layer that observes a decrease
    # clears every old per-device planner/carry, and the remaining layers build
    # fresh state.
    previous = _HOT_CARRY.get((str(dev), hot_key))
    if previous is not None and len(previous) >= 2 and int(previous[1]) > nv:
        _retire_request_state(dev)
    # The caller reuses one persistent output tensor across layers.  The next
    # chunk is best predicted by every winner from the most recent query
    # window, so snapshot only that window before the async reader starts.
    # Besides matching the steady-state fused publisher, this cuts the
    # dense->fused boundary copy from Q*K to min(Q,1536)*K indices.
    recent_rows = min(int(idx.shape[0]), CARRY_RECENT_ROWS)
    idx_snapshot = idx[-recent_rows:].clone()
    ss = _HOT_STREAM.get(str(dev))
    if ss is None:
        ss = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = ss
    # Kept in the public signature while older wrappers still pass the hint;
    # planning now happens only on the consuming main stream.
    _ = next_sequence_length
    carry_ext = _ext()
    ss.wait_stream(torch.cuda.current_stream())  # see the just-written topk
    idx_snapshot.record_stream(ss)  # keep it alive for the read
    with torch.cuda.stream(ss):
        votes = _vote_hist(nv, dev)
        hpf = idx_snapshot.reshape(-1).long().clamp_(0, nv - 1)
        votes.scatter_add_(0, hpf, torch.ones_like(hpf, dtype=torch.int32))
        hot_n = HOT_PREFIX
        # Each row contains every winner at most once, so recent_rows is the
        # exact per-index vote upper bound used by the custom carry selector.
        max_vote = recent_rows
        use_custom = (
            carry_ext is not None
            and hasattr(carry_ext, "carry_votes_topk_reset_")
            and nv <= PRODUCTION_MAX_S
            and 0 < max_vote <= 8192
        )
        if use_custom:
            hot = torch.empty(hot_n, dtype=torch.int64, device=dev)
            workspace = _carry_topk_workspace(max_vote, dev)
            carry_ext.carry_votes_topk_reset_(
                votes,
                hot,
                workspace["partial"],
                workspace["state"],
                hot_n,
                max_vote,
                min_index,
            )
        else:
            if min_index > 0:
                votes[:min_index].fill_(torch.iinfo(torch.int32).min)
            hot = votes.topk(hot_n).indices
        ev = torch.cuda.Event()
        ev.record()
    _HOT_CARRY[(str(dev), hot_key)] = (hot, nv, ev, min_index)


def try_large_exact_once_chunk(
    q,
    k,
    k_scale,
    weights,
    ks,
    ke,
    out_idx,
    topk,
    *,
    permuted_plan,
    num_reqs,
    ke_min_hint,
    cap=None,
    hot_key=None,
    ks_common_hint=0,
    carry_extent_hint=None,
    headroom=None,
    q_sf=None,
    _carry_io=True,
):
    """Run the fixed-HOT producer without rescanning HOT12288.

    The paged gather has pair-swapped the carried HOT set into one physical
    prefix.  Seed prep emits that prefix from its existing score matrix, the
    single no-hist producer starts immediately after it, and one compact-only
    postpass rebuilds boundary metadata in physical space. Selection then maps
    only the final winners back to corpus order while accumulating carry votes.
    No scan-time threshold update or checkpoint remains.
    """
    global _HINTS_VALIDATED, _SINGLE_SCAN_LOGGED
    global _PENDING_TELEMETRY, _PROBE_RES, _CAND_ACC
    try:
        Q = int(q.shape[0])
        S = int(k.shape[0])
        prefix_base = int(ks_common_hint)
        common_end = int(ke_min_hint)
        cap_eff = MERGE_CAP if cap is None else int(cap)
        # fp4 operands: q/k are packed e2m1 (64 bytes per 128-dim row, uint8
        # or deep_gemm's int8 tag view) with int32 ue8m0 scale streams; the
        # presence of q_sf selects the fp4graft scan.
        use_fp4 = q_sf is not None
        min_s = production_min_s(use_fp4)
        packed_dim = 64 if use_fp4 else 128
        packed_dtypes = (torch.uint8, torch.int8) if use_fp4 else (torch.float8_e4m3fn,)
        if (
            not isinstance(permuted_plan, dict)
            or num_reqs != 1
            or Q not in (FUSED_QUERY_LEN, FUSED_TAIL_QUERY_LEN)
            or min_s > S
            or S > PRODUCTION_MAX_S
            or q.dim() != 3
            or tuple(q.shape[1:]) not in ((32, packed_dim), (64, packed_dim))
            or q.dtype not in packed_dtypes
            or tuple(k.shape) != (S, packed_dim)
            or k.dtype not in packed_dtypes
            or tuple(k_scale.shape) != (S,)
            or k_scale.dtype != (torch.int32 if use_fp4 else torch.float32)
            or weights.shape != (Q, int(q.shape[1]))
            or ks.shape != (Q,)
            or ke.shape != (Q,)
            or ks.dtype != torch.int32
            or ke.dtype != torch.int32
            or out_idx.shape != (Q, topk)
            or out_idx.dtype != torch.int32
            or topk <= 0
            or topk > 2048
            or cap_eff < max(16384, 32 * topk)
            or prefix_base < 0
            or prefix_base % 4 != 0
            or prefix_base + HOT_PREFIX > common_end
            or common_end > S
            or int(permuted_plan.get("sequence_length", -1)) != S
            or int(permuted_plan.get("query_length", -1)) != Q
            or int(permuted_plan.get("window_start", -1)) != prefix_base
            or int(permuted_plan.get("common_end", -1)) != common_end
        ):
            return False
        permutation = permuted_plan.get("permutation")
        if (
            not isinstance(permutation, torch.Tensor)
            or permutation.shape != (S,)
            or permutation.dtype != torch.int32
            or permutation.device != q.device
            or not permutation.is_contiguous()
        ):
            return False
        if not (k.is_contiguous() and k_scale.is_contiguous()):
            return False
        kv_sf_ext = None
        if use_fp4:
            if (
                not isinstance(q_sf, torch.Tensor)
                or q_sf.dtype != torch.int32
                or tuple(q_sf.shape) != (Q, int(q.shape[1]))
                or q_sf.device != q.device
            ):
                return False
            if not q_sf.is_contiguous():
                q_sf = q_sf.contiguous()
            # The fp4graft TMA descriptor declares a 4-aligned kv_sf extent;
            # widen the view inside the backing storage when S % 4 != 0.
            sf_aligned = (S + 3) & ~3
            kv_sf_ext = k_scale
            if kv_sf_ext.numel() < sf_aligned:
                storage_i32 = (
                    kv_sf_ext.untyped_storage().nbytes() // 4
                    - kv_sf_ext.storage_offset()
                )
                if storage_i32 < sf_aligned:
                    return False
                kv_sf_ext = kv_sf_ext.as_strided((sf_aligned,), (1,))
        if not q.is_contiguous():
            q = q.contiguous()
        if weights.dtype != torch.float32:
            weights = weights.float()
        if not weights.is_contiguous():
            weights = weights.contiguous()
        if not (ks.is_contiguous() and ke.is_contiguous() and out_idx.is_contiguous()):
            return False
        if not _HINTS_VALIDATED:
            real_ks_min = int(ks.min().item())
            real_ks_max = int(ks.max().item())
            real_ke_min = int(ke.min().item())
            assert real_ks_min == real_ks_max == prefix_base
            assert real_ke_min == common_end
            _HINTS_VALIDATED = True
            print(
                "[litetopk] CPU hints validated; sync-free path active",
                flush=True,
            )

        _poll_candidate_telemetry()
        ext = _ext()
        use_h2048_safe = (
            topk == 2048 and max(16384, 32 * topk) <= cap_eff <= (1 << 20) and NB == 256
        )
        scan_op = (
            "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_"
            if use_fp4
            else "mqa_logits_dsa_static_hot_nohist_litetopk_"
        )
        required_ops = (
            "seed_prep_litetopk_",
            scan_op,
            "map_topk_indices_and_accumulate_votes_litetopk_",
        )
        required_ops += (
            ("h2048_safe_topk_out_litetopk_",)
            if use_h2048_safe
            else (
                "finalize_static_hot_meta_litetopk_",
                "compact_topk_min_thr_inplace_idx_out_litetopk",
            )
        )
        global _OPS_VERIFIED
        if ext is None:
            return False
        if not isinstance(_OPS_VERIFIED, dict):
            _OPS_VERIFIED = {}
        ops_key = (scan_op, use_h2048_safe)
        if ops_key not in _OPS_VERIFIED:
            _OPS_VERIFIED[ops_key] = all(hasattr(ext, name) for name in required_ops)
        if not _OPS_VERIFIED[ops_key]:
            return False
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        if deep_gemm is None or not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
            return False

        prefix_end = prefix_base + HOT_PREFIX
        prefix_k = k[prefix_base:prefix_end]
        prefix_scale = k_scale[prefix_base:prefix_end]
        sample_start, sample_end = _ks0_keh(Q, HOT_PREFIX, q.device)
        _seg0 = _seg_mark()
        if use_fp4:
            seed_q = (q.view(torch.int8), q_sf)
            seed_k = (prefix_k.view(torch.int8), prefix_scale)
        else:
            seed_q = (q, None)
            seed_k = (prefix_k, prefix_scale)
        sample_logits = deep_gemm.fp8_fp4_mqa_logits(
            seed_q,
            seed_k,
            weights,
            sample_start,
            sample_end,
            clean_logits=False,
            out=_slog_slab(Q, HOT_PREFIX, q.device),
        )
        b = _prep_bufs(Q, NB, cap_eff, q.device)
        cb = _cand_bufs(Q, cap_eff, q.device)
        origin = b["o"][:Q]
        inv = b["inv"][:Q]
        threshold = b["th"][:Q]
        boundary_meta = b["bc"][:Q]
        candidate_count = b["cc"][:Q]
        status = b["status"][:Q]
        candidate_value = cb["cv"][:Q]
        candidate_index = cb["ci"][:Q]
        diagnostic_stages = os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
        call_number = _TELEMETRY["calls"] + 1
        probe_due = call_number == 1 or call_number % PROBE_EVERY == 0

        def _check_static_stage(stage):
            if not (diagnostic_stages or probe_due):
                return
            _check_selector_status(
                status,
                candidate_count,
                stage=stage,
                sequence_length=S,
                common_end=common_end,
                cap=cap_eff,
                layer=hot_key,
            )

        headroom_eff = HEADROOM if headroom is None else float(headroom)
        if headroom_eff < 0.0:
            raise ValueError(f"headroom must be non-negative, got {headroom_eff}")

        # In exact-once mode the historical probe_stride argument is the
        # physical prefix base used for emitted candidate indices.
        ext.seed_prep_litetopk_(
            sample_logits,
            NB,
            topk,
            cap_eff,
            HOT_PREFIX,
            headroom_eff,
            prefix_base,
            1,
            origin,
            inv,
            threshold,
            boundary_meta,
            candidate_value,
            candidate_index,
            candidate_count,
        )
        if diagnostic_stages:
            seed_min = int(candidate_count.min().item())
            if seed_min < topk:
                raise RuntimeError(
                    "large exact-once stage=seed-emission underfill; "
                    f"S={S}, common_end={common_end}, min={seed_min}, "
                    f"topk={topk}, layer={hot_key}"
                )
        del sample_logits
        _seg1 = _seg_mark()

        # All rows share the physical prefix.  Reuse the immutable cached
        # filled tensor instead of launching an add kernel in every layer.
        suffix_start = _ks0_keh(Q, prefix_end, q.device)[1]
        if use_fp4:
            ext.mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_(
                q.view(torch.uint8),
                q_sf,
                k.view(torch.uint8),
                kv_sf_ext,
                weights,
                suffix_start,
                ke,
                origin,
                inv,
                threshold,
                candidate_value,
                candidate_index,
                candidate_count,
                boundary_meta,
                NB,
                topk,
            )
        else:
            ext.mqa_logits_dsa_static_hot_nohist_litetopk_(
                q,
                k,
                k_scale,
                weights,
                suffix_start,
                ke,
                origin,
                inv,
                threshold,
                candidate_value,
                candidate_index,
                candidate_count,
                boundary_meta,
                NB,
                topk,
            )
        _seg2 = _seg_mark()
        if not use_h2048_safe:
            # Compatibility path for LongCat's K=1008, DSV4's K=512, and
            # non-default CAP/NB. It retains the existing certificate and
            # destructive selector.
            ext.finalize_static_hot_meta_litetopk_(
                candidate_value,
                candidate_index,
                candidate_count,
                threshold,
                boundary_meta,
                status,
                NB,
                topk,
                S,
            )
            _check_static_stage("physical-finalize")

        _TELEMETRY["calls"] = call_number

        carry_votes = None
        carry_nv = 0
        carry_recent_rows = min(Q, CARRY_RECENT_ROWS)
        carry_event = permuted_plan.get("carry_event")
        carry_due = True
        if _CARRY_EVERY > 1 and hot_key is not None:
            _ck = (str(q.device), hot_key)
            _cn = _CARRY_SKIP_COUNTS.get(_ck, 0) + 1
            _CARRY_SKIP_COUNTS[_ck] = _cn
            carry_due = _cn % _CARRY_EVERY == 0
        if (
            _carry_io
            and _CARRY_IO_ENV
            and carry_due
            and hot_key is not None
            and HOTONLY
            and HOTSAMPLE == HOT_PREFIX
        ):
            carry_nv = int(carry_extent_hint if carry_extent_hint is not None else S)
            carry_votes = _carry_vote_hist(carry_nv, q.device, hot_key, carry_event)
        if carry_votes is None:
            carry_votes = candidate_count[:0]
        if use_h2048_safe:
            # The h2048 fast selector and exact bit-32 continuation both emit
            # physical IDs. boundary_meta is dead after the no-hist scan, so
            # its first Q*5 int32 values serve as diagnostic scratch without
            # allocating another hot-path tensor.
            ext.h2048_safe_topk_out_litetopk_(
                candidate_value,
                candidate_index,
                candidate_count,
                out_idx,
                status,
                boundary_meta,
                S,
            )
            _check_static_stage("h2048-safe-select")
            _seg3 = _seg_mark()
        else:
            ext.compact_topk_min_thr_inplace_idx_out_litetopk(
                candidate_value,
                candidate_index,
                candidate_count,
                threshold,
                boundary_meta,
                NB,
                topk,
                out_idx,
                candidate_count[:0],
                1,
            )
        if _CAND_ACC is None or _CAND_ACC[0].device != q.device:
            _CAND_ACC = (
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.zeros(1, dtype=torch.int32, device=q.device),
            )
        run_max, over_events = _CAND_ACC
        # The CUDA map kernel checks every status row before mapping a winner.
        # This preserves fail-closed ordering between synchronous probes.
        if hasattr(ext, "map_topk_vote_stats_litetopk_"):
            ext.map_topk_vote_stats_litetopk_(
                out_idx,
                permutation,
                status,
                carry_votes,
                carry_recent_rows,
                candidate_count,
                run_max,
                over_events,
                OVF_WATERMARK,
            )
        else:
            ext.map_topk_indices_and_accumulate_votes_litetopk_(
                out_idx,
                permutation,
                status,
                carry_votes,
                carry_recent_rows,
            )
            torch.maximum(run_max, candidate_count.amax(0, keepdim=True), out=run_max)
            over_events.add_((candidate_count > OVF_WATERMARK).sum(dtype=torch.int32))
        if _PENDING_TELEMETRY is None and probe_due:
            # Candidate telemetry remains asynchronous. Selector correctness
            # was synchronously checked before the winner-map launch above.
            if _PROBE_RES is None or _PROBE_RES[0].device != q.device:
                _PROBE_RES = (
                    torch.empty(2, dtype=torch.int32, device=q.device),
                    torch.empty(4, dtype=torch.int32, pin_memory=True),
                    torch.cuda.Event(),
                )
            stats, pinned, event = _PROBE_RES
            ext.cand_count_stats_litetopk_(candidate_count, stats)
            pinned[:2].copy_(stats, non_blocking=True)
            pinned[2:3].copy_(run_max, non_blocking=True)
            pinned[3:4].copy_(over_events, non_blocking=True)
            event.record()
            _PENDING_TELEMETRY = (event, pinned, topk, OVF_WATERMARK)
        if _SEG_ON and use_h2048_safe:
            _seg_commit(S, [_seg0, _seg1, _seg2, _seg3, _seg_mark()])
        if CHECK:
            logits = deep_gemm.fp8_fp4_mqa_logits(
                (q.view(torch.int8), q_sf) if use_fp4 else (q, None),
                (k.view(torch.int8), k_scale) if use_fp4 else (k, k_scale),
                weights,
                ks,
                ke,
                clean_logits=True,
            )
            ref_physical = logits.topk(topk, dim=1).indices
            ref = permutation[ref_physical]
            refs = ref.sort(dim=1).values
            got = out_idx.long().sort(dim=1).values
            pos = torch.searchsorted(refs, got).clamp(max=topk - 1)
            recall = (torch.gather(refs, 1, pos) == got).float().mean()
            print(
                f"[litetopk] large exact-once Q={Q} S={S} "
                f"recall={100 * recall.item():.3f}%",
                flush=True,
            )

        if carry_votes.numel() > 0:
            _publish_carry(
                hot_key,
                carry_votes,
                carry_nv,
                prefix_base,
                carry_recent_rows,
            )
        if not _SINGLE_SCAN_LOGGED:
            print(
                "[litetopk] HOT12288 exact-once active: prefix emit + one "
                "fixed-threshold histogram-free suffix scan + physical "
                "select + winner-only map/vote",
                flush=True,
            )
            _SINGLE_SCAN_LOGGED = True
        return True
    except Exception as e:  # noqa: BLE001
        if os.environ.get("VLLM_LITETOPK_CARRY_DEBUG", "0") == "1":
            import traceback

            traceback.print_exc()
        print(
            f"[litetopk] large exact-once declined: {e}",
            flush=True,
        )
        return False
