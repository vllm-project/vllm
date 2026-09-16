# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM decode hook for VLLM_INDEXER_BACKEND=fuse_score_remap.

Uses the fused SM90 score kernel. For seq capacity <= 8192 the C++
merge is used; above that (up to 64k) the compact pack is viewed as logits
and vLLM cooperative/persistent topk + gather_physical emit physical ids.

Constraints (else fall back to DeepGEMM):
  SM90, next_n=1, H=32, D=128, page=64, FP8 cache, no DCP/PCP/prefill mix,
  topk in {512, 1024, 2048}, capacity <= 65536.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NUM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
HEAD_DIM_WITH_SCALE = 132
SPLIT_KV = 256
MERGE_MAX_SEQ = 65536
LARGE_SEQ_PARTS = 32  # cub BlockRadixSort covers 32 * 256 = 8192


def _deep_gemm_include() -> Path:
    env = os.environ.get("DSA_FUSE_OPT_DG_INCLUDE")
    candidates = []
    if env:
        candidates.append(Path(env))
    try:
        import vllm

        candidates.append(
            Path(vllm.__file__).resolve().parent / "third_party" / "deep_gemm" / "include"
        )
    except Exception:
        pass
    marker = Path("deep_gemm") / "scheduler" / "sm90_paged_mqa_logits.cuh"
    for path in candidates:
        if path.is_dir() and (path / marker).is_file():
            return path
    raise FileNotFoundError(
        "DeepGEMM include dir with sm90_paged_mqa_logits.cuh not found"
    )


def _digest() -> str:
    parts = []
    for path in (
        HERE / "launch_fused.cu",
        HERE / "sm90_fp8_paged_mqa_logits_fused.cuh",
        HERE / "cta_bitonic_select.cuh",
        HERE / "online_topk_heap.cuh",
        HERE / "merge_cta_topk.cuh",
    ):
        parts.append(path.read_bytes())
    return hashlib.sha1(b"".join(parts)).hexdigest()[:12]


@lru_cache(maxsize=1)
def load_fused():
    from torch.utils.cpp_extension import load

    if not (HERE / "launch_fused.cu").is_file():
        raise FileNotFoundError(HERE / "launch_fused.cu")
    dg_include = _deep_gemm_include()

    cache_root = os.environ.get("VLLM_CACHE_ROOT")
    if not cache_root:
        try:
            import vllm.envs as envs

            cache_root = envs.VLLM_CACHE_ROOT
        except Exception:
            cache_root = os.path.expanduser("~/.cache/vllm")
    ext_dir = Path(cache_root) / "torch_extensions" / "fuse_score_remap"
    ext_dir.mkdir(parents=True, exist_ok=True)
    prev_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    prev_ext = os.environ.get("TORCH_EXTENSIONS_DIR")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    extra_includes = [str(HERE), str(dg_include)]
    cuda_inc = Path("/usr/local/cuda/include")
    if cuda_inc.is_dir():
        extra_includes.append(str(cuda_inc))
    cccl_inc = Path("/usr/local/cuda/include/cccl")
    if cccl_inc.is_dir():
        extra_includes.append(str(cccl_inc))
    try:
        return load(
            name=f"dsa_fused_paged_mqa_{_digest()}",
            sources=[str(HERE / "launch_fused.cu")],
            extra_include_paths=extra_includes,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--extended-lambda",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-lineinfo",
                "-gencode=arch=compute_90a,code=sm_90a",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
            verbose=False,
        )
    finally:
        if prev_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev_arch
        if prev_ext is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = prev_ext


def fuse_score_remap_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return False, f"need SM90, got sm_{major}x"
    try:
        ext = load_fused()
        ext.warmup_fused()
        return True, "sm90_fp8_paged_mqa_logits_fused+topk+gather (64k)"
    except Exception as exc:
        return False, str(exc)


def can_use_fuse_score_remap(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    next_n: int,
    head_dim: int,
    topk_tokens: int,
    use_fp4_cache: bool,
    use_pcp: bool = False,
    dcp_world_size: int = 1,
    has_prefill: bool = False,
    block_table: torch.Tensor | None = None,
    max_model_len: int | None = None,
) -> bool:
    if use_fp4_cache or use_pcp or dcp_world_size > 1 or has_prefill:
        return False
    if next_n != 1 or head_dim != HEAD_DIM or topk_tokens not in (512, 1024, 2048):
        return False
    if q.dim() == 4 and int(q.size(1)) != 1:
        return False
    heads = int(q.size(-2)) if q.dim() >= 2 else -1
    if heads != NUM_HEADS:
        return False
    if int(kv_cache.size(1)) != PAGE_SIZE:
        return False
    if int(kv_cache.size(-1)) != HEAD_DIM_WITH_SCALE:
        return False
    if max_model_len is not None and int(max_model_len) > MERGE_MAX_SEQ:
        return False
    if block_table is not None and block_table.dim() >= 2:
        table_cap = int(block_table.size(1)) * PAGE_SIZE
        parts = max(1, (table_cap + SPLIT_KV - 1) // SPLIT_KV)
        if parts * SPLIT_KV > 262144:
            return False
    return True


def pack_parts(block_table: torch.Tensor, max_model_len: int | None = None) -> int:
    cap = int(block_table.size(1)) * PAGE_SIZE
    return max(1, (cap + SPLIT_KV - 1) // SPLIT_KV)


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _must_contig(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.is_contiguous():
        return t
    if _capturing():
        raise RuntimeError(
            f"fuse_score_remap {name} must be contiguous during CUDA graph capture"
        )
    return t.contiguous()


def _as_q_b1hd(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        q = q.unsqueeze(1)
    if q.dim() != 4:
        raise ValueError(
            f"fuse_score_remap Q must be [B,1,H,D] or [B,H,D], got {tuple(q.shape)}"
        )
    return _must_contig(q, "q")


def _as_seq_lens(seq_lens: torch.Tensor, batch: int) -> torch.Tensor:
    seq = seq_lens
    if seq.dim() == 2:
        if seq.size(1) == 1:
            seq = seq.reshape(batch)
        else:
            seq = seq[:, -1]
    seq = seq.reshape(batch)
    if seq.dtype != torch.int32:
        if _capturing():
            raise RuntimeError(
                "fuse_score_remap seq_lens must be int32 during CUDA graph capture"
            )
        seq = seq.to(dtype=torch.int32)
    return _must_contig(seq, "seq_lens")


_GRAPH: dict[tuple, torch.cuda.CUDAGraph] = {}
_GRAPH_WARM: set[tuple] = set()
_GRAPH_MAX = 256


def _graph_key(
    q: torch.Tensor,
    kv: torch.Tensor,
    w: torch.Tensor,
    seq: torch.Tensor,
    table: torch.Tensor,
    sched: torch.Tensor,
    pack_scores: torch.Tensor | None,
    pack_indices: torch.Tensor | None,
    logical: torch.Tensor | None,
    topk_workspace: torch.Tensor | None,
    dst: torch.Tensor,
    topk_tokens: int,
    max_parts: int,
) -> tuple:
    def _p(t: torch.Tensor | None) -> int:
        return 0 if t is None else int(t.data_ptr())

    return (
        int(q.size(0)),
        topk_tokens,
        max_parts,
        _p(q),
        _p(kv),
        _p(w),
        _p(seq),
        _p(table),
        _p(sched),
        _p(pack_scores),
        _p(pack_indices),
        _p(logical),
        _p(topk_workspace),
        _p(dst),
    )


def fuse_score_remap_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    out: torch.Tensor,
    topk_tokens: int,
    pack_scores: torch.Tensor,
    pack_indices: torch.Tensor,
    logical: torch.Tensor | None = None,
    topk_workspace: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Write physical KV slot ids into ``out`` [B, K].

    Capture-safe: no new allocations. Outer FULL graphs record the launches;
    piecewise / eager uses a small CUDAGraph keyed by buffer pointers.
    """
    capturing = _capturing()
    q = _as_q_b1hd(q_index_fp8)
    if q.dtype != torch.float8_e4m3fn:
        q = q.view(torch.float8_e4m3fn)
    batch = int(q.size(0))
    w = weights.reshape(batch, NUM_HEADS)
    w = _must_contig(w, "weights")
    seq = _as_seq_lens(seq_lens, batch)
    table = _must_contig(block_table, "block_table")
    sched = _must_contig(schedule_metadata, "schedule_metadata")
    kv = k_index_cache_fp8
    if kv.dtype != torch.uint8:
        kv = kv.view(torch.uint8)
    if not out.is_contiguous():
        raise RuntimeError("fuse_score_remap out must be contiguous")
    dst = out
    ext = load_fused()
    max_parts = int(pack_scores.size(1))
    use_large = max_parts > LARGE_SEQ_PARTS
    if use_large:
        if logical is None or topk_workspace is None:
            if capturing:
                raise RuntimeError(
                    "fuse_score_remap large path needs pack/logical/topk workspace "
                    "during CUDA graph capture"
                )
            if logical is None:
                logical = torch.empty(
                    (batch, topk_tokens), dtype=torch.int32, device=q.device
                )
            if topk_workspace is None:
                topk_workspace = torch.empty(
                    1024 * 1024, dtype=torch.uint8, device=q.device
                )
        pack_scores = _must_contig(pack_scores, "pack_scores")
        pack_indices = _must_contig(pack_indices, "pack_indices")
        logical = _must_contig(logical, "logical")
        topk_workspace = _must_contig(topk_workspace, "topk_workspace")

    table_cap = int(table.size(1)) * PAGE_SIZE
    pack_cols = max_parts * SPLIT_KV
    # Graph replay freezes this scalar; use the table/pack upper bound so later
    # tokens can grow. seq_lens still gates the valid prefix.
    scan_graph = min(table_cap, pack_cols)
    scan_eager = (
        scan_graph if max_seq_len is None else min(int(max_seq_len), scan_graph)
    )

    def _invoke(scan_n: int) -> None:
        if not use_large:
            ext.fused_paged_mqa_topk(q, kv, w, seq, table, sched, dst)
            return
        ext.fused_paged_mqa_score(
            q, kv, w, seq, table, sched, pack_scores, pack_indices
        )
        logits = pack_scores.reshape(batch, -1)
        seq2d = seq.reshape(batch, 1)
        if batch <= 32 and logits.stride(0) % 4 == 0:
            torch.ops._C.cooperative_topk(
                logits,
                seq2d,
                logical,
                topk_workspace,
                topk_tokens,
                scan_n,
            )
        else:
            torch.ops._C.persistent_topk(
                logits,
                seq2d,
                logical,
                topk_workspace,
                topk_tokens,
                int(logits.shape[1]),
            )
        ext.gather_physical(pack_indices.reshape(batch, -1), logical, dst)

    if capturing:
        _invoke(scan_graph)
        return out

    gkey = _graph_key(
        q,
        kv,
        w,
        seq,
        table,
        sched,
        pack_scores if use_large else None,
        pack_indices if use_large else None,
        logical if use_large else None,
        topk_workspace if use_large else None,
        dst,
        topk_tokens,
        max_parts,
    )
    graph = _GRAPH.get(gkey)
    if graph is not None:
        graph.replay()
        return out

    _invoke(scan_eager)
    if gkey not in _GRAPH_WARM:
        _GRAPH_WARM.add(gkey)
        return out
    if len(_GRAPH) >= _GRAPH_MAX:
        return out
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _invoke(scan_graph)
        _GRAPH[gkey] = graph
    except RuntimeError:
        pass
    return out
