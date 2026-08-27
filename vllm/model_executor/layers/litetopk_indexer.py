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
  VLLM_LITETOPK_PRODUCTION_MIN_S
                              FP8 fused-path crossover (default 196608); an
                              explicit value also overrides the FP4 default
  VLLM_LITETOPK_FP4_PRODUCTION_MIN_S
                              FP4 fused-path crossover (default 65536)
  VLLM_LITETOPK_PCP_FRONTIER_CARRY
                              opt in to broadcasting the global DualChunkSwap
                              frontier carry after each of its two fused phases
  VLLM_LITETOPK_MERGE_CAP    per-row candidate capacity (default 49152)
  VLLM_LITETOPK_HEADROOM     bucket-scale headroom (default 0)
  VLLM_LITETOPK_OVF_LOG=1    log new candidate-count maxima
  VLLM_LITETOPK_PROBE_EVERY  overflow telemetry cadence (default 8); probed
                              chunks asynchronously validate selector status;
                              every chunk device-traps before winner mapping
  VLLM_LITETOPK_OVF_WATERMARK
                              count row-chunks whose candidate count exceeds
                              this (default 40960); accumulated on device
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
TP_QUERY_SHARD_ENABLED = os.environ.get("VLLM_LITETOPK_TP_QUERY_SHARD", "0") == "1"
TP8_FP4_SHARD_FULL_QUERY_LENS = (
    FUSED_QUERY_LEN,
    FUSED_TAIL_QUERY_LEN,
    32768,
    32704,
)
TP8_FP4_SHARD_QUERY_LENS = tuple(q // 8 for q in TP8_FP4_SHARD_FULL_QUERY_LENS)
TP4_FP8_SHARD_QUERY_LENS = (FUSED_QUERY_LEN // 4, FUSED_TAIL_QUERY_LEN // 4)
TP_SHARD_QUERY_LENS = TP8_FP4_SHARD_QUERY_LENS + TP4_FP8_SHARD_QUERY_LENS


def _supported_fused_query_len(q: int) -> bool:
    return q in (FUSED_QUERY_LEN, FUSED_TAIL_QUERY_LEN) or (
        TP_QUERY_SHARD_ENABLED and q in TP_SHARD_QUERY_LENS
    )


HOT_PREFIX = 12288
NB = int(os.environ.get("VLLM_LITETOPK_NB", "256"))
_TELEMETRY = {"calls": 0, "candidate_max": 0}
# Absolute forward headroom on the bucket scale (fraction of the sample span
# prepended ABOVE the sample max). Pair with a proportionally larger NB to
# keep bucket width unchanged (e.g. HEADROOM=1.0 + NB=512 == today's width).
HEADROOM = float(os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0"))
# Target the aligned 24*K slab for GLM K=2048. It leaves 30% headroom over the
# 37,752-record maximum observed in the prior GLM-5.2 1M run; the reduced shape
# is separately qualified below, and overflow remains fail-closed.
MERGE_CAP = int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "49152"))


# GLM-5.2 K=2048 targets 24*K. Keep the historical 32*K floor for every other
# selection width rather than widening their support as an accidental side
# effect of the GLM memory reduction.
def minimum_merge_cap(topk: int) -> int:
    return 49152 if topk == 2048 else max(16384, 32 * topk)


# The K-relative floor is enforced at the call sites where topk is known; this
# import-time check only rejects configurations no supported K could satisfy.
if MERGE_CAP < 16384:
    raise ValueError(
        "VLLM_LITETOPK_MERGE_CAP must be at least 16384 for the "
        "fixed-HOT no-hist production path"
    )
# OVF_LOG: print the running max of sampled per-row candidate counts (from
# the existing deferred 1-in-8 probe; sync-free). Sizes MERGE_CAP.
OVF_LOG = os.environ.get("VLLM_LITETOPK_OVF_LOG", "0") == "1"
_HOT_STREAM: dict = {}
PROBE_EVERY = int(os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8"))
if PROBE_EVERY < 1:
    raise ValueError("VLLM_LITETOPK_PROBE_EVERY must be >= 1")
# Warn one full 8192-record chunk before the hard cap. This is telemetry only;
# candidate_count > MERGE_CAP still fails closed in the selector/map path.
OVF_WATERMARK = int(os.environ.get("VLLM_LITETOPK_OVF_WATERMARK", "40960"))
# Real adjacent-chunk capture selected this window: all K winners from the last
# 1536 query rows predict the next chunk substantially better than the old
# rotating 1/8 sample over all rows, while adding only atomics to the mandatory
# winner-map pass.
CARRY_RECENT_ROWS = 1536
# The production selector parallelizes each 256-bin radix prefix search in
# warp 0 and selects the remaining 16 score bits in two passes.
_HOT_CARRY: dict = {}
# GATE4 writes BUCKET-SPACE high24 candidates (affine order-preserving).
# Both seed-prefix emission and the suffix producer use the same packed score
# contract, so the mapped postpass can process their concatenation directly.

_EXT = None
_FAILED = False
_AUX_CACHE: dict = {}  # (device, head) -> (zeros[Qmax], full_head[Qmax]) int32
_SINGLE_SCAN_LOGGED = False


def _dsa_source_id():
    digest = hashlib.sha256()
    for filename in (
        "dsa_litetopk.cu",
        "sm100_dsa_litetopk.cuh",
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
        or minimum_merge_cap(topk) > MERGE_CAP
    ):
        return False

    ext = _ext()
    if ext is None:
        return False

    from vllm.utils.deep_gemm import is_fp8_fp4_mqa_logits_out_supported

    if not is_fp8_fp4_mqa_logits_out_supported():
        return False

    common_ops = (
        "plan_and_permuted_paged_gather_out",
        "seed_prep_litetopk_",
        "map_topk_vote_stats_litetopk_",
        "cand_count_stats_litetopk_",
    )
    scan_op = (
        "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_"
        if use_fp4
        else "mqa_logits_dsa_static_hot_nohist_litetopk_"
    )
    use_h2048_safe = (
        topk == 2048
        and minimum_merge_cap(topk) <= MERGE_CAP <= PRODUCTION_MAX_S
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


_HINTS_VALIDATED = False
# (cuda event, pinned int32 tensor, cap, K, watermark)
_PENDING_TELEMETRY = None


_CAND_ACC = None  # (device running max[1], device over-watermark count[1]):
# accumulated unconditionally every call so the sampled
# probe readback still reports the complete running max
# Cached (device stats, device status max, pinned buffer, event): allocating
_PROBE_RES = None
# these per arm blocked the CPU ~17ms inside
# cudaHostAlloc-class calls (nsys), starving the GPU
# stream for ~3.4ms/call at 256K/Q=512 when throttled


def _poll_candidate_telemetry():
    """Non-blocking read of the previous chunk's status and telemetry."""
    global _PENDING_TELEMETRY
    if _PENDING_TELEMETRY is not None:
        ev, pinned, capv, kk, watermark = _PENDING_TELEMETRY
        if ev.query():  # finished long ago; no sync
            mx = int(pinned[0])
            selector_status = int(pinned[2])
            run_max = int(pinned[3])
            over = int(pinned[4])
            if selector_status != 0:
                raise RuntimeError(
                    f"[litetopk] selector status={selector_status} on a "
                    f"probed chunk (candidate max {mx}, cap {capv}); the "
                    "emitted top-k indices are unreliable — raise "
                    "VLLM_LITETOPK_MERGE_CAP"
                )
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


_PREP_BUFS: dict = {}  # (dev, NB) -> dict of caller-owned seed_prep buffers
_SLOG_SLABS: dict = {}  # dev -> persistent seed-GEMM logits slab (out= reuse)
_OPS_VERIFIED: dict | None = None  # required-ops hasattr walk, done once per ext load


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


_CAND_BUFS: dict = {}  # dev -> opaque U16 slab carrying delayed-high24 codes
_VOTE_BUF_HOT: dict = {}  # dev -> persistent stash-carry vote histogram
_CARRY_VOTE_BUFS: dict = {}  # (dev, layer) -> selector-fused vote slab + free event
_CARRY_TOPK_WORKSPACE: dict = {}  # dev -> side-stream partial/state workspace
_CARRY_TOPK_MAX_BLOCKS = 128
_CARRY_TOPK_STATE_INTS = 136
# One pair-swap workspace is owned by each main CUDA stream.  Planning and the
# paged gather are submitted together through the production extension; no
# side-stream plan, prepared ticket, or per-layer permutation cache exists.
_PAIR_PLAN_BUFS: dict = {}
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
    device_index = (
        dev.index if dev.index is not None else torch.accelerator.current_device_index()
    )
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
        if free_event is not waited_event:
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
            "hot": hot
            if hot is not None
            else torch.empty(HOT_PREFIX, dtype=torch.int64, device=dev),
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


def _pcp_frontier_broadcast_carry(
    hot,
    local_extent,
    side,
    *,
    broadcast_src=None,
    broadcast_extent=None,
):
    """Broadcast one PCP frontier carry on its producing side stream."""
    if broadcast_src is None:
        if broadcast_extent is not None:
            raise ValueError("PCP frontier carry extent requires a source rank")
        return local_extent
    from vllm.distributed import get_pcp_group

    pcp_group = get_pcp_group()
    if (
        pcp_group.world_size <= 1
        or not 0 <= int(broadcast_src) < pcp_group.world_size
        or broadcast_extent is None
    ):
        raise ValueError("invalid PCP frontier carry broadcast plan")
    publish_extent = int(broadcast_extent)
    if pcp_group.rank_in_group == int(broadcast_src) and publish_extent != local_extent:
        raise ValueError("PCP frontier source extent does not match its local carry")
    communicator = pcp_group.device_communicator
    pynccl_comm = (
        None if communicator is None else getattr(communicator, "pynccl_comm", None)
    )
    if pynccl_comm is None or pynccl_comm.disabled:
        raise ValueError("PCP frontier carry requires the device PyNccl communicator")
    # Enqueue directly on the carry stream so its ready event covers the
    # selection and the communication.
    pynccl_comm.broadcast(hot, src=int(broadcast_src), stream=side)
    return publish_extent


def _publish_carry(
    hot_key,
    votes,
    nv,
    min_index,
    max_vote,
    *,
    broadcast_src=None,
    broadcast_extent=None,
):
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
        votes.record_stream(side)
        hot_n = HOT_PREFIX
        use_custom = (
            carry_ext is not None
            and hasattr(carry_ext, "carry_votes_topk_reset_")
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
        publish_extent = _pcp_frontier_broadcast_carry(
            hot,
            nv,
            side,
            broadcast_src=broadcast_src,
            broadcast_extent=broadcast_extent,
        )
        ready = entry["ready_event"]
        ready.record(side)
    entry["free_event"] = ready
    _HOT_CARRY[key] = (hot, publish_extent, ready, min_index)


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
            or not (
                production_min_s(dst_k.dtype == torch.uint8) <= S <= PRODUCTION_MAX_S
            )
            or not _supported_fused_query_len(Q)
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
        if not carry_valid:
            return None
        ext = _ext()
        if ext is None or not hasattr(ext, "plan_and_permuted_paged_gather_out"):
            return None
        assert carry is not None
        hot = carry[0][:HOT_PREFIX]
        carry_event = carry[2]
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
        print(
            f"[litetopk] exact-once permuted gather declined: {e}",
            flush=True,
        )
        return None


def stash_carry(
    hot_key,
    idx,
    S,
    min_index=0,
    *,
    broadcast_src=None,
    broadcast_extent=None,
):
    """Seed a layer's hot carry from the OFFICIAL path's topk output, called
    by the container on the LAST official chunk before MIN_S. The
    official->ours boundary is deterministic, so this one seed is all the
    first ours-chunk needs to run HOT (no cold start, no cold prefix). Stored
    compressed (voted hot columns, ~64KB/layer).

    The vote+topk selection and the store run on a per-device SIDE STREAM
    (async): seeding overlaps the model forward instead of stalling the
    official path. The exact-once gather consumer waits on the stored event
    before touching the carry."""
    if hot_key is None:
        return
    dev = idx.device
    nv = int(S)
    lifecycle_extent = int(broadcast_extent) if broadcast_src is not None else nv
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
    if (
        previous is not None
        and len(previous) >= 2
        and int(previous[1]) > lifecycle_extent
    ):
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
        publish_extent = _pcp_frontier_broadcast_carry(
            hot,
            nv,
            ss,
            broadcast_src=broadcast_src,
            broadcast_extent=broadcast_extent,
        )
        ev = torch.cuda.Event()
        ev.record()
    _HOT_CARRY[(str(dev), hot_key)] = (hot, publish_extent, ev, min_index)


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
    carry_broadcast_src=None,
    carry_broadcast_extent=None,
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
            or not _supported_fused_query_len(Q)
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
            or cap_eff < minimum_merge_cap(topk)
            or prefix_base < 0
            or prefix_base % 4 != 0
            or prefix_base + HOT_PREFIX > common_end
            or common_end > S
            or int(permuted_plan.get("sequence_length", -1)) != S
            or int(permuted_plan.get("query_length", -1)) != Q
            or (
                Q in TP_SHARD_QUERY_LENS
                and permuted_plan.get("tp_query_shard") is not True
            )
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
            topk == 2048
            and minimum_merge_cap(topk) <= cap_eff <= (1 << 20)
            and NB == 256
        )
        scan_op = (
            "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_"
            if use_fp4
            else "mqa_logits_dsa_static_hot_nohist_litetopk_"
        )
        required_ops: tuple[str, ...] = (
            "seed_prep_litetopk_",
            scan_op,
            "map_topk_vote_stats_litetopk_",
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
        from vllm.utils.deep_gemm import (
            fp8_fp4_mqa_logits,
            is_fp8_fp4_mqa_logits_out_supported,
        )

        if not is_fp8_fp4_mqa_logits_out_supported():
            return False

        prefix_end = prefix_base + HOT_PREFIX
        prefix_k = k[prefix_base:prefix_end]
        prefix_scale = k_scale[prefix_base:prefix_end]
        sample_start, sample_end = _ks0_keh(Q, HOT_PREFIX, q.device)
        if use_fp4:
            seed_q = (q.view(torch.int8), q_sf)
            seed_k = (prefix_k.view(torch.int8), prefix_scale)
        else:
            seed_q = (q, None)
            seed_k = (prefix_k, prefix_scale)
        sample_logits = fp8_fp4_mqa_logits(
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
        call_number = _TELEMETRY["calls"] + 1
        probe_due = call_number == 1 or call_number % PROBE_EVERY == 0

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
        del sample_logits

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

        _TELEMETRY["calls"] = call_number

        carry_votes = None
        carry_nv = 0
        carry_recent_rows = min(Q, CARRY_RECENT_ROWS)
        carry_event = permuted_plan.get("carry_event")
        if _carry_io and hot_key is not None:
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
        if _PENDING_TELEMETRY is None and probe_due:
            # Arm one deferred event+pinned probe after the selector. The map
            # kernel checks every status row device-side, so bad output cannot
            # reach attention. Healthy probes are reported without a
            # stream-wide .item() synchronization.
            if _PROBE_RES is None or _PROBE_RES[0].device != q.device:
                _PROBE_RES = (
                    torch.empty(2, dtype=torch.int32, device=q.device),
                    torch.empty(1, dtype=torch.int32, device=q.device),
                    torch.empty(5, dtype=torch.int32, pin_memory=True),
                    torch.cuda.Event(),
                )
            stats, status_max, pinned, event = _PROBE_RES
            ext.cand_count_stats_litetopk_(candidate_count, stats)
            torch.amax(status, dim=0, keepdim=True, out=status_max)
            pinned[:2].copy_(stats, non_blocking=True)
            pinned[2:3].copy_(status_max, non_blocking=True)
            pinned[3:4].copy_(run_max, non_blocking=True)
            pinned[4:5].copy_(over_events, non_blocking=True)
            event.record()
            _PENDING_TELEMETRY = (
                event,
                pinned,
                cap_eff,
                topk,
                OVF_WATERMARK,
            )
        if carry_votes.numel() > 0:
            _publish_carry(
                hot_key,
                carry_votes,
                carry_nv,
                prefix_base,
                carry_recent_rows,
                broadcast_src=carry_broadcast_src,
                broadcast_extent=carry_broadcast_extent,
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
        print(
            f"[litetopk] large exact-once declined: {e}",
            flush=True,
        )
        return False
