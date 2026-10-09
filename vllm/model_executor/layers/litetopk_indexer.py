#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LiteTopK fused sparse top-k indexer for GLM-5.2 FP8 TP8 prefill.

A fixed 32K random page sample is gathered and scored by DeepGEMM. Seed prep
emits its candidates once, and the scoring kernel reads the remaining paged
cache directly through TMA, skipping sampled pages. The h2048 selector finds
winners and a compact page map restores their original token positions.

Enable with VLLM_LITETOPK=1. Only K=2048, NB=256 is supported; invalid
selector output traps on the device, independently of optional logging.
"""

import functools
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

# Keep the JIT source inside the Python package so editable installs and wheels
# hash and compile the exact same vendored implementation.
_DSA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litetopk_kernels")
_BUILD_DIR = os.environ.get(
    "VLLM_LITETOPK_BUILD",
    os.path.expanduser("~/.cache/vllm/litetopk_build"),
)

ENABLED = os.environ.get("VLLM_LITETOPK", "0") == "1"
PRODUCTION_MIN_S = int(os.environ.get("VLLM_LITETOPK_PRODUCTION_MIN_S", "196608"))
PRODUCTION_MAX_S = 1 << 20
if not 16384 <= PRODUCTION_MIN_S <= PRODUCTION_MAX_S:
    # Keep a complete historical sample and a suffix chunk.
    raise ValueError("LiteTopK production min-S must be in [16384, 1<<20]")


FUSED_QUERY_LEN = 8192
FUSED_TAIL_QUERY_LEN = 8128
TP_SHARD_QUERY_LENS = (FUSED_QUERY_LEN // 8, FUSED_TAIL_QUERY_LEN // 8)


def _supported_fused_query_len(q: int) -> bool:
    return q in TP_SHARD_QUERY_LENS


def supports_model(config, tp_size: int) -> bool:
    return (
        tp_size == 8
        and getattr(config, "model_type", None) == "glm_moe_dsa"
        and tuple(
            getattr(config, name, None)
            for name in ("index_n_heads", "index_head_dim", "index_topk")
        )
        == (32, 128, 2048)
    )


RANDOM_SAMPLE_SIZE = 32768
_RANDOM_PAGE_ORDER: dict = {}  # one immutable logical page order per device
NB = int(os.environ.get("VLLM_LITETOPK_NB", "256"))
_TELEMETRY = {"calls": 0, "candidate_max": 0}
# Headroom is a fraction of the sample span.
HEADROOM = float(os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0"))
MERGE_CAP = int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "49152"))


MIN_MERGE_CAP = 49152


def _supports_h2048(topk: int, cap: int) -> bool:
    return topk == 2048 and NB == 256 and MIN_MERGE_CAP <= cap <= PRODUCTION_MAX_S


if MERGE_CAP < MIN_MERGE_CAP:
    raise ValueError(
        "VLLM_LITETOPK_MERGE_CAP must be at least 49152 for the h2048 path"
    )
# Logging reads counters asynchronously; device statistics include every chunk.
OVF_LOG = os.environ.get("VLLM_LITETOPK_OVF_LOG", "0") == "1"
PROBE_EVERY = int(os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8"))
if PROBE_EVERY < 1:
    raise ValueError("VLLM_LITETOPK_PROBE_EVERY must be >= 1")
# Warn one full 8192-record chunk before the hard cap. This is telemetry only;
# candidate_count > MERGE_CAP still fails closed in the selector/map path.
OVF_WATERMARK = int(os.environ.get("VLLM_LITETOPK_OVF_WATERMARK", "40960"))
# h2048 caches up to 4096 boundary candidates; larger boundaries are streamed.
# Seed and suffix share a bucket-space high24 score key (FP32 low 8 bits dropped).

_EXT = None
_FAILED = False
_REQUIRED_OPS = (
    "gather_paged_sample_out",
    "seed_prep_litetopk_",
    "mqa_logits_dsa_paged_litetopk_",
    "h2048_safe_topk_out_litetopk_",
    "map_topk_stats_litetopk_",
)
_SINGLE_SCAN_LOGGED = False


def _dsa_source_id():
    digest = hashlib.sha256()
    for name in ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh"):
        digest.update(name.encode())
        digest.update((Path(_DSA_DIR) / name).read_bytes())
    return digest.hexdigest()[:12]


def _load_extension(name, source_id):
    path = os.environ.get("VLLM_LITETOPK_SO", "")
    sha256 = os.environ.get("VLLM_LITETOPK_SO_SHA256", "")
    if bool(path) != bool(sha256):
        raise RuntimeError("VLLM_LITETOPK_SO and its SHA256 must be set together")
    if path:
        path = Path(path).expanduser().resolve()
        if path.name != f"{name}.so":
            raise RuntimeError(f"LiteTopK override must be named {name}.so")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256.lower():
            raise RuntimeError("LiteTopK override SHA256 mismatch")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load extension from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    from torch.utils.cpp_extension import load

    dg = os.environ.get("DEEPGEMM_DIR")
    if dg:
        includes = [
            Path(dg) / "deep_gemm/include",
            Path(dg) / "third-party/cutlass/include",
        ]
    else:
        from vllm.utils.deep_gemm import _import_deep_gemm

        module = _import_deep_gemm()
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError("DeepGEMM unavailable; set DEEPGEMM_DIR")
        includes = [Path(module.__file__).parent / "include"]
    if not any((d / "cutlass/arch/barrier.h").is_file() for d in includes):
        raise RuntimeError("DeepGEMM/CUTLASS headers missing; set DEEPGEMM_DIR")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    build = f"{_BUILD_DIR}_production_{source_id}"
    os.makedirs(build, exist_ok=True)
    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_100a,code=sm_100a",
    ]
    if os.environ.get("LITETOPK_LINEINFO") == "1":
        flags.append("-lineinfo")
    return load(
        name=name,
        sources=[str(Path(_DSA_DIR) / "dsa_litetopk.cu")],
        extra_include_paths=[_DSA_DIR, *map(str, includes)],
        extra_cuda_cflags=flags,
        build_directory=build,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _ext():
    global _EXT, _FAILED
    if _EXT is None and not _FAILED:
        try:
            source_id = _dsa_source_id()
            module = _load_extension(
                f"vllm_litetopk_dsa_b200_production_{source_id}", source_id
            )
            for name in (
                "candidate_value_u16_litetopk",
                "candidate_fp24_global_litetopk",
            ):
                if not getattr(module, name, lambda: False)():
                    raise RuntimeError(f"LiteTopK candidate ABI mismatch: {name}")
            if not all(hasattr(module, op) for op in _REQUIRED_OPS):
                raise RuntimeError("LiteTopK extension is missing a required operation")
            _EXT = module
            print(f"[litetopk] loaded B200 production kernel ({source_id})", flush=True)
        except Exception as error:  # noqa: BLE001
            _FAILED = True
            print(f"[litetopk] extension unavailable: {error}", flush=True)
    return _EXT


@functools.cache
def production_extension_available(*, use_fp4: bool, topk: int) -> bool:
    """Check the kernel and out= ABI before exempting a chunk from logits budgets."""
    if (
        not ENABLED
        or use_fp4
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0)
        or not _supports_h2048(topk, MERGE_CAP)
    ):
        return False

    ext = _ext()
    if ext is None:
        return False

    from vllm.utils.deep_gemm import is_fp8_fp4_mqa_logits_out_supported

    if not is_fp8_fp4_mqa_logits_out_supported():
        return False

    return all(hasattr(ext, op) for op in _REQUIRED_OPS)


_HINTS_VALIDATED = False
_CAND_ACC = None  # Persistent device max and over-watermark count.
_PROBE_RES = None
_WORKSPACES: dict = {}


def _probe_candidate_telemetry(run_max, over_events):
    """Read running counters only when logging; the GPU always validates winners."""
    global _PROBE_RES
    if not OVF_LOG:
        return
    _TELEMETRY["calls"] += 1
    if _PROBE_RES is None or _PROBE_RES["device"] != run_max.device:
        _PROBE_RES = dict(
            device=run_max.device,
            pending=False,
            event=torch.cuda.Event(),
            host=torch.empty(2, dtype=torch.int32, pin_memory=True),
        )
    probe = _PROBE_RES
    if probe["pending"] and probe["event"].query():
        maximum, over = probe["host"].tolist()
        if maximum > _TELEMETRY["candidate_max"]:
            _TELEMETRY["candidate_max"] = maximum
            print(
                f"[litetopk] cand max -> {maximum}; "
                f"row-chunks over {OVF_WATERMARK}: {over}",
                flush=True,
            )
        probe["pending"] = False
    due = _TELEMETRY["calls"] == 1 or _TELEMETRY["calls"] % PROBE_EVERY == 0
    if due and not probe["pending"]:
        probe["host"][:1].copy_(run_max, non_blocking=True)
        probe["host"][1:].copy_(over_events, non_blocking=True)
        probe["event"].record()
        probe["pending"] = True


def _workspace(Q, cap, sample_size, dev):
    """Reuse one growing workspace per device, including the padded seed logits."""
    key = str(dev)
    entry = _WORKSPACES.get(key)
    if entry is None or entry["q"] < Q or entry["shape"] != (cap, sample_size, NB):
        # Release old large slabs before allocating replacements.
        _WORKSPACES.pop(key, None)
        del entry
        qm = max(Q, 1024)
        specs = {
            "o": ((qm,), torch.float32),
            "inv": ((qm,), torch.float32),
            "th": ((qm,), torch.int32),
            "cc": ((qm,), torch.int32),
            "status": ((qm,), torch.int32),
            "bc": ((qm, NB), torch.int32),
            "cv": ((qm, cap), torch.float16),
            "ci": ((qm, cap), torch.int32),
            "slog": (((qm + 8) * (sample_size + 512),), torch.float32),
        }
        buffers = {
            name: torch.empty(shape, dtype=dtype, device=dev)
            for name, (shape, dtype) in specs.items()
        }
        buffers["start"] = torch.zeros(qm, dtype=torch.int32, device=dev)
        buffers["end"] = torch.full((qm,), sample_size, dtype=torch.int32, device=dev)
        entry = dict(q=qm, shape=(cap, sample_size, NB), buffers=buffers)
        _WORKSPACES[key] = entry
    views = entry.setdefault("views", {})
    if Q not in views:
        views[Q] = {
            name: value if name == "slog" else value[:Q]
            for name, value in entry["buffers"].items()
        }
    return views[Q]


def _random_page_order(sequence_length, common_end, dev):
    """Fixed uniform page sample shared by layers, independent of cache addresses.

    Permute only complete pages visible to every query. Keep the causal tail
    unchanged, so virtual query end offsets remain valid. Rebuild once when
    the context shape changes, without advancing the model's random state.
    """
    historical_pages = common_end // 64
    sample_size = min(RANDOM_SAMPLE_SIZE, historical_pages * 64 // 256 * 256)
    if 8192 < sample_size < 12288:
        sample_size = 8192
    if sample_size < 2048:
        raise ValueError("not enough common history for the fixed page sample")
    pages = (sequence_length + 63) // 64
    key = (pages, historical_pages, sample_size)
    cached = _RANDOM_PAGE_ORDER.get(str(dev))
    if cached is None or cached[0] != key:
        generator = torch.Generator(device="cpu").manual_seed(17)
        selected = (
            torch.randperm(historical_pages, generator=generator)[: sample_size // 64]
            .sort()
            .values
        )
        remaining = torch.ones(pages, dtype=torch.bool)
        remaining[selected] = False
        order = torch.cat((selected, torch.arange(pages)[remaining])).to(
            device=dev, dtype=torch.int32
        )
        cached = (key, order)
        _RANDOM_PAGE_ORDER[str(dev)] = cached
    return cached[1], sample_size


def prepare_paged_sample(
    kv_cache,
    dst_k,
    dst_scale,
    block_table,
    *,
    sequence_length,
    query_length,
    num_reqs,
    common_end,
):
    """Gather only the fixed random sample; leave the suffix in paged cache."""
    try:
        S = int(sequence_length)
        Q = int(query_length)
        common_end = int(common_end)
        if (
            not ENABLED
            or not PRODUCTION_MIN_S <= S <= PRODUCTION_MAX_S
            or not _supported_fused_query_len(Q)
            or num_reqs != 1
            or not 12288 <= common_end <= S
            or dst_scale.shape != (S, 4)
            or dst_scale.dtype != torch.uint8
            or tuple(dst_k.shape) != (S, 128)
            or dst_k.dtype != torch.float8_e4m3fn
            or dst_k.device.type != "cuda"
            or torch.cuda.get_device_capability(dst_k.device) != (10, 0)
            or kv_cache.dim() != 3
            or kv_cache.shape[1:] != (64, 132)
        ):
            return None
        ext = _ext()
        if ext is None:
            return None
        page_order, sample_size = _random_page_order(S, common_end, dst_k.device)
        # Reuse only the sampled prefix of the ordinary workspace.
        ext.gather_paged_sample_out(
            kv_cache,
            block_table,
            page_order,
            dst_k[:sample_size],
            dst_scale[:sample_size].view(torch.float32).view(-1),
        )
        return {
            "page_order": page_order,
            "paged_cache": kv_cache,
            "block_table": block_table,
            "sample_size": sample_size,
            "sequence_length": S,
            "query_length": Q,
            "common_end": common_end,
        }
    except Exception as e:  # noqa: BLE001
        print(f"[litetopk] paged sample preparation declined: {e}", flush=True)
        return None


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
    sample_plan,
    num_reqs,
    ke_min_hint,
    cap=None,
    headroom=None,
):
    """Score each position once, select top-k, and restore logical token IDs."""
    global _HINTS_VALIDATED, _SINGLE_SCAN_LOGGED
    global _CAND_ACC
    try:
        Q = int(q.shape[0])
        S = int(k.shape[0])
        common_end = int(ke_min_hint)
        if not isinstance(sample_plan, dict):
            return False
        sample_size = int(sample_plan.get("sample_size", 0))
        cap_eff = MERGE_CAP if cap is None else int(cap)
        if (
            num_reqs != 1
            or not _supported_fused_query_len(Q)
            or PRODUCTION_MIN_S > S
            or S > PRODUCTION_MAX_S
            or q.dim() != 3
            or tuple(q.shape[1:]) != (32, 128)
            or q.dtype != torch.float8_e4m3fn
            or tuple(k.shape) != (S, 128)
            or k.dtype != torch.float8_e4m3fn
            or tuple(k_scale.shape) != (S,)
            or k_scale.dtype != torch.float32
            or weights.shape != (Q, int(q.shape[1]))
            or ks.shape != (Q,)
            or ke.shape != (Q,)
            or ks.dtype != torch.int32
            or ke.dtype != torch.int32
            or out_idx.shape != (Q, topk)
            or out_idx.dtype != torch.int32
            or not _supports_h2048(topk, cap_eff)
            or not 2048 <= sample_size <= RANDOM_SAMPLE_SIZE
            or sample_size % 256 != 0
            or sample_size > common_end
            or common_end > S
            or int(sample_plan.get("sequence_length", -1)) != S
            or int(sample_plan.get("query_length", -1)) != Q
            or int(sample_plan.get("common_end", -1)) != common_end
        ):
            return False
        page_order = sample_plan.get("page_order")
        if (
            not isinstance(page_order, torch.Tensor)
            or page_order.shape != ((S + 63) // 64,)
            or page_order.dtype != torch.int32
            or page_order.device != q.device
            or not page_order.is_contiguous()
        ):
            return False
        if not (k.is_contiguous() and k_scale.is_contiguous()):
            return False
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
            assert real_ks_min == real_ks_max == 0
            assert real_ke_min == common_end
            _HINTS_VALIDATED = True
            print(
                "[litetopk] CPU hints validated; sync-free path active",
                flush=True,
            )

        ext = _ext()
        if ext is None:
            return False
        from vllm.utils.deep_gemm import (
            fp8_fp4_mqa_logits,
            is_fp8_fp4_mqa_logits_out_supported,
        )

        if not is_fp8_fp4_mqa_logits_out_supported():
            return False

        prefix_k = k[:sample_size]
        prefix_scale = k_scale[:sample_size]
        b = _workspace(Q, cap_eff, sample_size, q.device)
        sample_start, sample_end = b["start"], b["end"]
        sample_logits = fp8_fp4_mqa_logits(
            (q, None),
            (prefix_k, prefix_scale),
            weights,
            sample_start,
            sample_end,
            clean_logits=False,
            out=b["slog"],
        )
        origin = b["o"]
        inv = b["inv"]
        threshold = b["th"]
        boundary_meta = b["bc"]
        candidate_count = b["cc"]
        status = b["status"]
        candidate_value = b["cv"]
        candidate_index = b["ci"]

        headroom_eff = HEADROOM if headroom is None else float(headroom)
        if headroom_eff < 0.0:
            raise ValueError(f"headroom must be non-negative, got {headroom_eff}")

        ext.seed_prep_litetopk_(
            sample_logits,
            NB,
            topk,
            cap_eff,
            headroom_eff,
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
        suffix_start = sample_end
        ext.mqa_logits_dsa_paged_litetopk_(
            q,
            sample_plan["paged_cache"],
            sample_plan["block_table"],
            page_order,
            S,
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

        # The scan has consumed the seed histogram; reuse it as Q*5 scratch.
        ext.h2048_safe_topk_out_litetopk_(
            candidate_value,
            candidate_index,
            candidate_count,
            out_idx,
            status,
            boundary_meta,
            S,
        )
        if _CAND_ACC is None or _CAND_ACC[0].device != q.device:
            _CAND_ACC = (
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.zeros(1, dtype=torch.int32, device=q.device),
            )
        run_max, over_events = _CAND_ACC
        # The CUDA map kernel checks every status row before mapping a winner.
        ext.map_topk_stats_litetopk_(
            out_idx,
            page_order,
            status,
            candidate_count,
            run_max,
            over_events,
            OVF_WATERMARK,
            S,
        )
        _probe_candidate_telemetry(run_max, over_events)
        if not _SINGLE_SCAN_LOGGED:
            print(
                "[litetopk] fixed random pages active: sample emit + direct "
                "paged suffix scan + h2048 + page map",
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
