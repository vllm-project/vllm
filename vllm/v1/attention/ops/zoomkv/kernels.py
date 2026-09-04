# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV production kernel dispatch (chunk mean / Top-K / KIVI).

Development keeps PyTorch / Triton reference paths.  Strict mode raises when
a production CUDA extension is required but unavailable.
"""

from __future__ import annotations

import os
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ZOOMKV_C: Any | None = None
_ZOOMKV_C_TRIED = False


def _want_strict() -> bool:
    return os.environ.get("VLLM_ZOOMKV_STRICT_KERNELS", "0") == "1"


def _preload_torch_python_symbols() -> None:
    """Make libtorch_python symbols globally visible for the prebuilt module.

    The CMake-built ``vllm._zoomkv_C`` is a pybind11 module that references
    ``pybind11::detail::type_caster<at::Tensor>`` symbols defined in
    ``libtorch_python``. Loading it with RTLD_GLOBAL lets the extension resolve
    those symbols; otherwise import fails and we fall back to JIT.
    """
    import ctypes
    import os

    import torch

    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_python.so")
    if os.path.exists(lib):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


def try_load_zoomkv_c() -> Any | None:
    """Load ``vllm._zoomkv_C`` if present (CMake / JIT)."""
    global _ZOOMKV_C, _ZOOMKV_C_TRIED
    if _ZOOMKV_C_TRIED:
        return _ZOOMKV_C
    _ZOOMKV_C_TRIED = True
    try:
        with suppress(Exception):
            _preload_torch_python_symbols()
        import vllm._zoomkv_C as mod  # type: ignore

        _ZOOMKV_C = mod
        logger.info("Loaded vllm._zoomkv_C production ZoomKV kernels")
    except Exception as e:  # noqa: BLE001
        _ZOOMKV_C = None
        logger.debug("vllm._zoomkv_C not available: %s", e)
    return _ZOOMKV_C


def _has_cuda_rerank(mod: Any | None) -> bool:
    return mod is not None and all(
        hasattr(mod, name)
        for name in ("partial_chunk_density_scores", "mask_from_topk")
    )


_DIRECT_PHYSICAL_SYMBOLS = (
    "centroid_score_physical",
    "density_score_physical",
    "kivi_physical",
    "float_topk_values_3d",
    "float_topk_3d_varlen",
    "float_topk_values_3d_varlen",
)


def direct_physical_retrieval_available() -> bool:
    """Whether the unified extension has the no-materialization fast path."""
    mod = try_load_zoomkv_c()
    return mod is not None and all(
        hasattr(mod, name) for name in _DIRECT_PHYSICAL_SYMBOLS
    )


def density_score_physical(
    chunk_ids: torch.Tensor,
    physical_ids: torch.Tensor,
    centroid: torch.Tensor,
    valid: torch.Tensor,
    raw_q: torch.Tensor,
    scores: torch.Tensor,
    n_chunks: int,
    actual_num_chunks: torch.Tensor | None = None,
) -> None:
    mod = try_load_zoomkv_c()
    if mod is None or not hasattr(mod, "density_score_physical"):
        raise RuntimeError("ZoomKV direct physical density kernel unavailable")
    mod.density_score_physical(
        chunk_ids,
        physical_ids,
        centroid,
        valid,
        raw_q,
        scores,
        int(n_chunks),
        actual_num_chunks,
    )


def centroid_score_physical(
    physical_ids: torch.Tensor,
    centroid: torch.Tensor,
    valid: torch.Tensor,
    raw_q: torch.Tensor,
    scores: torch.Tensor,
    n_chunks: int,
    actual_num_chunks: torch.Tensor | None = None,
) -> None:
    """Score every retrieval-zone child chunk with ``q · centroid``."""
    mod = try_load_zoomkv_c()
    if mod is None or not hasattr(mod, "centroid_score_physical"):
        raise RuntimeError("ZoomKV centroid score kernel unavailable")
    mod.centroid_score_physical(
        physical_ids,
        centroid,
        valid,
        raw_q,
        scores,
        int(n_chunks),
        actual_num_chunks,
    )


def kivi_physical(
    chunk_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    physical_ids: torch.Tensor,
    packed: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    valid: torch.Tensor,
    raw_q: torch.Tensor,
    dense_topk: int,
    sparse_topk: int,
    token_offset: int,
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
    actual_num_chunks: torch.Tensor | None = None,
    *,
    compact: bool = False,
    n_dense: int = 0,
) -> None:
    mod = try_load_zoomkv_c()
    if mod is None or not hasattr(mod, "kivi_physical"):
        raise RuntimeError("ZoomKV direct physical KIVI kernel unavailable")
    kwargs = {}
    # Older wheels omit compact layout args; pack in Python instead.
    text_sig = getattr(mod.kivi_physical, "__text_signature__", None) or ""
    supports_compact = "compact" in text_sig or hasattr(mod, "centroid_score_physical")
    if compact and supports_compact:
        kwargs["compact"] = True
        kwargs["n_dense"] = int(n_dense)
    mod.kivi_physical(
        chunk_ids,
        dense_mask,
        physical_ids,
        packed,
        chunk_min,
        chunk_max,
        valid,
        raw_q,
        int(dense_topk),
        int(sparse_topk),
        int(token_offset),
        out_scores,
        out_indices,
        actual_num_chunks,
        **kwargs,
    )
    if compact and not supports_compact:
        _pack_padded_kivi_to_compact(
            out_scores,
            out_indices,
            nk=int(chunk_ids.shape[2]),
            n_dense=int(n_dense),
            dense_topk=int(dense_topk),
            sparse_topk=int(sparse_topk),
        )


def _pack_padded_kivi_to_compact(
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
    *,
    nk: int,
    n_dense: int,
    dense_topk: int,
    sparse_topk: int,
) -> None:
    """Fallback: rewrite ``nk * max(d,s)`` KIVI output into compact 1040 layout."""
    output_slots = max(dense_topk, sparse_topk)
    compact_w = n_dense * dense_topk + (nk - n_dense) * sparse_topk
    padded = out_scores[..., : nk * output_slots].reshape(
        *out_scores.shape[:2], nk, output_slots
    )
    padded_idx = out_indices[..., : nk * output_slots].reshape(
        *out_indices.shape[:2], nk, output_slots
    )
    compact_scores = out_scores.new_full((*out_scores.shape[:2], compact_w), float("-inf"))
    compact_idx = out_indices.new_full((*out_indices.shape[:2], compact_w), -1)
    if n_dense > 0:
        dense = padded[..., :n_dense, :dense_topk].reshape(
            *out_scores.shape[:2], n_dense * dense_topk
        )
        dense_i = padded_idx[..., :n_dense, :dense_topk].reshape(
            *out_indices.shape[:2], n_dense * dense_topk
        )
        compact_scores[..., : n_dense * dense_topk] = dense
        compact_idx[..., : n_dense * dense_topk] = dense_i
    if nk > n_dense:
        sparse = padded[..., n_dense:, :sparse_topk].reshape(
            *out_scores.shape[:2], (nk - n_dense) * sparse_topk
        )
        sparse_i = padded_idx[..., n_dense:, :sparse_topk].reshape(
            *out_indices.shape[:2], (nk - n_dense) * sparse_topk
        )
        compact_scores[..., n_dense * dense_topk :] = sparse
        compact_idx[..., n_dense * dense_topk :] = sparse_i
    out_scores[..., :compact_w].copy_(compact_scores)
    out_indices[..., :compact_w].copy_(compact_idx)
    if out_scores.shape[-1] > compact_w:
        out_scores[..., compact_w:].fill_(float("-inf"))
        out_indices[..., compact_w:].fill_(-1)


@lru_cache
def _try_load_float_topk_cuda() -> Any | None:
    """JIT-load the standalone radix Top-K kernel when the wheel lacks it."""
    source = Path(__file__).with_name("cuda") / "float_topk.cu"
    if not source.exists():
        return None
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="vllm_zoomkv_float_topk",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ZoomKV radix Top-K CUDA load failed: %s", e)
        return None


@lru_cache
def _try_load_rerank_cuda() -> Any | None:
    """JIT-load fused CDS density/mask kernels."""
    mod = try_load_zoomkv_c()
    if _has_cuda_rerank(mod):
        return mod
    source = Path(__file__).with_name("cuda") / "rerank_topk.cu"
    if not source.exists():
        return None
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="vllm_zoomkv_rerank_topk",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ZoomKV CDS CUDA load failed: %s", e)
        return None


def float_topk_3d(
    scores: torch.Tensor, k: int, strict: bool | None = None
) -> torch.Tensor:
    strict = _want_strict() if strict is None else strict
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "float_topk_3d"):
        return mod.float_topk_3d(scores, k)
    if scores.is_cuda:
        topk_mod = _try_load_float_topk_cuda()
        if topk_mod is not None:
            return topk_mod.float_topk_3d(scores, k)
    if strict:
        raise RuntimeError("ZoomKV strict mode: float_topk_3d CUDA required")
    k = max(1, min(int(k), scores.shape[-1]))
    return scores.topk(k, dim=-1, largest=True).indices


def float_topk_values_3d(
    scores: torch.Tensor,
    values: torch.Tensor,
    k: int,
    strict: bool | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select Top-K scores and return their associated int64 values."""
    strict = _want_strict() if strict is None else strict
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "float_topk_values_3d"):
        result = mod.float_topk_values_3d(scores, values, int(k), out)
        return out if out is not None else result
    if scores.is_cuda:
        topk_mod = _try_load_float_topk_cuda()
        if topk_mod is not None and hasattr(topk_mod, "float_topk_values_3d"):
            result = topk_mod.float_topk_values_3d(scores, values, int(k), out)
            return out if out is not None else result
    if strict:
        raise RuntimeError("ZoomKV strict mode: value-returning CUDA Top-K required")
    positions = scores.topk(k, dim=-1, largest=True).indices
    if out is not None:
        torch.gather(values, -1, positions, out=out)
        return out
    return torch.gather(values, -1, positions)


def float_topk_3d_varlen(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    ks: torch.Tensor,
    max_k: int,
    strict: bool | None = None,
) -> torch.Tensor:
    """Per-row Top-K over ``scores[..., :lengths]`` with fixed output width."""
    strict = _want_strict() if strict is None else strict
    max_k = max(1, min(int(max_k), scores.shape[-1]))
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "float_topk_3d_varlen"):
        return mod.float_topk_3d_varlen(scores, lengths, ks, max_k)
    if scores.is_cuda:
        topk_mod = _try_load_float_topk_cuda()
        if topk_mod is not None and hasattr(topk_mod, "float_topk_3d_varlen"):
            return topk_mod.float_topk_3d_varlen(scores, lengths, ks, max_k)
    if strict:
        raise RuntimeError("ZoomKV strict mode: ragged CUDA Top-K required")
    scan = torch.arange(scores.shape[-1], device=scores.device)
    masked = scores.masked_fill(
        scan.view(1, 1, -1) >= lengths.unsqueeze(-1), float("-inf")
    )
    positions = masked.topk(max_k, dim=-1, largest=True).indices
    slots = torch.arange(max_k, device=scores.device).view(1, 1, -1)
    return positions.masked_fill(slots >= ks.unsqueeze(-1), -1)


def float_topk_values_3d_varlen(
    scores: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    ks: torch.Tensor,
    max_k: int,
    strict: bool | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged Top-K returning associated values in a fixed-width output."""
    strict = _want_strict() if strict is None else strict
    max_k = max(1, min(int(max_k), scores.shape[-1]))
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "float_topk_values_3d_varlen"):
        result = mod.float_topk_values_3d_varlen(
            scores, values, lengths, ks, max_k, out
        )
        return out if out is not None else result
    if scores.is_cuda:
        topk_mod = _try_load_float_topk_cuda()
        if topk_mod is not None and hasattr(
            topk_mod, "float_topk_values_3d_varlen"
        ):
            result = topk_mod.float_topk_values_3d_varlen(
                scores, values, lengths, ks, max_k, out
            )
            return out if out is not None else result
    if strict:
        raise RuntimeError("ZoomKV strict mode: ragged value Top-K required")
    positions = float_topk_3d_varlen(
        scores, lengths, ks, max_k, strict=False
    )
    selected = torch.gather(values, -1, positions.clamp_min(0))
    selected.masked_fill_(positions < 0, -1)
    if out is not None:
        out.copy_(selected)
        return out
    return selected


def chunk_density_scores(
    chunk_ids: torch.Tensor,
    centroids: torch.Tensor,
    raw_q: torch.Tensor,
    out: torch.Tensor | None = None,
    strict: bool | None = None,
) -> torch.Tensor:
    """Score selected chunk centroids without materializing a gather."""
    strict = _want_strict() if strict is None else strict
    if chunk_ids.is_cuda:
        mod = _try_load_rerank_cuda()
        if mod is not None:
            if out is None:
                out = torch.empty_like(chunk_ids, dtype=torch.float32)
            mod.partial_chunk_density_scores(
                chunk_ids,
                centroids,
                raw_q.to(torch.bfloat16),
                out,
            )
            return out
    if strict:
        raise RuntimeError("ZoomKV strict mode: CDS density CUDA required")
    idx = chunk_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, centroids.shape[-1])
    selected = torch.gather(centroids, 2, idx)
    scores = (
        selected.to(torch.float32) * raw_q.unsqueeze(2).to(torch.float32)
    ).sum(-1)
    if out is not None:
        out.copy_(scores)
        return out
    return scores


def dense_mask_from_topk(
    positions: torch.Tensor,
    num_chunks: int,
    out: torch.Tensor | None = None,
    strict: bool | None = None,
) -> torch.Tensor:
    """Build the CDS dense mask with one fused CUDA launch."""
    strict = _want_strict() if strict is None else strict
    mask_shape = (*positions.shape[:2], int(num_chunks))
    if out is not None and tuple(out.shape) == mask_shape:
        mask = out
    else:
        mask = torch.empty(mask_shape, dtype=torch.bool, device=positions.device)
    if positions.is_cuda:
        mod = _try_load_rerank_cuda()
        if mod is not None:
            mod.mask_from_topk(positions.contiguous(), mask)
            return mask
    if strict:
        raise RuntimeError("ZoomKV strict mode: CDS mask CUDA required")
    counts = torch.zeros(mask_shape, dtype=torch.int32, device=positions.device)
    valid = positions >= 0
    counts.scatter_add_(2, positions.clamp_min(0), valid.to(torch.int32))
    mask.copy_(counts > 0)
    return mask


@lru_cache
def _try_load_h2d_cuda() -> Any | None:
    """JIT-load the K-only H2D gather kernels."""
    source = Path(__file__).with_name("cuda") / "h2d_gather_tokens.cu"
    if not source.exists():
        return None
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="vllm_zoomkv_h2d_keys",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ZoomKV H2D CUDA load failed: %s", e)
        return None


def h2d_gather_keys(
    src_k: torch.Tensor,
    cpu_slots: torch.Tensor,
    token_offsets: torch.Tensor,
    out_k: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
    strict: bool | None = None,
) -> None:
    """Gather [slot, offset] tokens from pinned CPU Key into GPU out_k.

    src_k: [num_slots, block_size, H, D] pinned CPU
    cpu_slots / token_offsets: [N]
    out_k: [N, H, D] GPU
    """
    strict = _want_strict() if strict is None else strict
    slots = cpu_slots
    offs = token_offsets
    if not slots.is_cuda:
        slots = slots.to(device=out_k.device, dtype=torch.int64)
    if not offs.is_cuda:
        offs = offs.to(device=out_k.device, dtype=torch.int64)

    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "h2d_gather_keys"):
        if stream is not None:
            with torch.cuda.stream(stream):
                mod.h2d_gather_keys(src_k, slots, offs, out_k)
        else:
            mod.h2d_gather_keys(src_k, slots, offs, out_k)
        return
    h2d_mod = _try_load_h2d_cuda()
    if h2d_mod is not None and hasattr(h2d_mod, "h2d_gather_keys"):
        if stream is not None:
            with torch.cuda.stream(stream):
                h2d_mod.h2d_gather_keys(src_k, slots, offs, out_k)
        else:
            h2d_mod.h2d_gather_keys(src_k, slots, offs, out_k)
        return

    # Reference path: index on CPU then non_blocking H2D.
    slots_cpu = slots.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    offs_cpu = offs.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    n = slots_cpu.numel()
    if n == 0:
        return
    if strict:
        raise RuntimeError("ZoomKV strict mode: h2d_gather_keys CUDA required")
    block_size = src_k.shape[1]
    H, D = src_k.shape[2], src_k.shape[3]
    k_host = torch.empty(n, H, D, dtype=src_k.dtype, pin_memory=True)
    for i in range(n):
        s = int(slots_cpu[i].item())
        o = int(offs_cpu[i].item())
        if s < 0 or o < 0 or o >= block_size:
            k_host[i].zero_()
            continue
        k_host[i].copy_(src_k[s, o])
    if stream is not None:
        with torch.cuda.stream(stream):
            out_k.copy_(k_host, non_blocking=True)
    else:
        out_k.copy_(k_host, non_blocking=True)


def h2d_fill_keys_hybrid(
    src_k: torch.Tensor,
    logical_ids: torch.Tensor,
    block_table: torch.Tensor,
    cpu_slots: torch.Tensor,
    offloaded_mask: torch.Tensor,
    start_block: int,
    out_k: torch.Tensor,
    strict: bool | None = None,
) -> None:
    """Overwrite out_k entries whose physical blocks are Key-offloaded."""
    strict = _want_strict() if strict is None else strict
    bt = block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "h2d_gather_keys_hybrid"):
        mod.h2d_gather_keys_hybrid(
            src_k, logical_ids, bt, cpu_slots, offloaded_mask, int(start_block), out_k
        )
        return
    h2d_mod = _try_load_h2d_cuda()
    if h2d_mod is not None and hasattr(h2d_mod, "h2d_gather_keys_hybrid"):
        h2d_mod.h2d_gather_keys_hybrid(
            src_k, logical_ids, bt, cpu_slots, offloaded_mask, int(start_block), out_k
        )
        return

    if strict:
        raise RuntimeError("ZoomKV strict mode: h2d_gather_keys_hybrid CUDA required")

    # Reference: for each token, if physical block is offloaded, copy Key from CPU.
    kv_heads, n_tok = logical_ids.shape
    block_size = src_k.shape[1]
    for h in range(kv_heads):
        for t in range(n_tok):
            logical = int(logical_ids[h, t].item())
            if logical < 0:
                continue
            lb = logical // block_size
            phys = int(bt[lb].item())
            if phys < 0 or phys >= offloaded_mask.numel():
                continue
            if not bool(offloaded_mask[phys].item()):
                continue
            rel = lb - int(start_block)
            if rel < 0 or rel >= cpu_slots.numel():
                continue
            slot = int(cpu_slots[rel].item())
            if slot < 0:
                continue
            offset = logical - lb * block_size
            out_k[h, t].copy_(src_k[slot, offset].to(device=out_k.device))


def h2d_fill_kv_hybrid(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    gpu_k: torch.Tensor,
    gpu_v: torch.Tensor,
    logical_ids: torch.Tensor,
    block_table: torch.Tensor,
    physical_to_slot: torch.Tensor,
    offloaded_mask: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    strict: bool | None = None,
) -> None:
    """Overwrite cold-token K/V from pinned host memory in one CUDA launch."""
    strict = _want_strict() if strict is None else strict
    bt = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    mod = try_load_zoomkv_c()
    if mod is not None and hasattr(mod, "h2d_gather_kv_hybrid"):
        mod.h2d_gather_kv_hybrid(
            src_k,
            src_v,
            gpu_k,
            gpu_v,
            logical_ids,
            bt,
            physical_to_slot,
            offloaded_mask,
            out_k,
            out_v,
        )
        return
    h2d_mod = _try_load_h2d_cuda()
    if h2d_mod is not None and hasattr(h2d_mod, "h2d_gather_kv_hybrid"):
        h2d_mod.h2d_gather_kv_hybrid(
            src_k,
            src_v,
            gpu_k,
            gpu_v,
            logical_ids,
            bt,
            physical_to_slot,
            offloaded_mask,
            out_k,
            out_v,
        )
        return
    if strict:
        raise RuntimeError("ZoomKV strict mode: h2d_gather_kv_hybrid CUDA required")

    # Slow reference path used only when the extension is optional.
    block_size = src_k.shape[1]
    for h in range(logical_ids.shape[0]):
        for t in range(logical_ids.shape[1]):
            logical = int(logical_ids[h, t].item())
            if logical < 0:
                continue
            lb, offset = divmod(logical, block_size)
            phys = int(bt[lb].item())
            if phys < 0:
                continue
            if bool(offloaded_mask[phys].item()):
                slot = int(physical_to_slot[phys].item())
                if slot < 0:
                    continue
                out_k[h, t].copy_(src_k[slot, offset, h], non_blocking=True)
                out_v[h, t].copy_(src_v[slot, offset, h], non_blocking=True)
            else:
                out_k[h, t].copy_(gpu_k[phys, offset, h])
                out_v[h, t].copy_(gpu_v[phys, offset, h])
