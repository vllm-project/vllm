# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV production kernel dispatch (Quest / TopK / KIVI).

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
from vllm.v1.attention.ops.zoomkv.quest import QuestTorchOps, quest_bound_scores

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


def _has_cuda_quest(mod: Any | None) -> bool:
    return mod is not None and all(
        hasattr(mod, name)
        for name in ("quest_chunk_score", "quest_sub_chunk_score", "quest_map_back")
    )


def _make_quest_fallback(prefer_triton: bool, strict: bool):
    if prefer_triton:
        try:
            from vllm.v1.attention.ops.zoomkv.quest_triton import QuestTritonOps

            return QuestTritonOps()
        except Exception as e:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"ZoomKV Quest Triton unavailable: {e}") from e
            logger.warning("Quest Triton unavailable (%s); using PyTorch", e)
    if strict:
        raise RuntimeError("ZoomKV strict mode: Quest CUDA/Triton required")
    return QuestTorchOps()


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


def get_quest_ops(prefer_triton: bool = True, strict: bool | None = None):
    strict = _want_strict() if strict is None else strict
    mod = try_load_zoomkv_c()
    if _has_cuda_quest(mod):
        return _CudaQuestOps(
            mod,
            lambda: _make_quest_fallback(prefer_triton=prefer_triton, strict=strict),
        )
    return _make_quest_fallback(prefer_triton=prefer_triton, strict=strict)


class _CudaQuestOps:
    def __init__(self, mod: Any, fallback_factory) -> None:
        self._mod = mod
        self._fallback_factory = fallback_factory
        self._fallback: Any | None = None

    def _fallback_ops(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    @staticmethod
    def _normalize_valid(
        chunk_valid: torch.Tensor | None,
        raw_q: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor | None:
        if chunk_valid is None:
            return None
        valid = chunk_valid
        if valid.dtype != torch.bool or valid.device != raw_q.device:
            valid = valid.to(device=raw_q.device, dtype=torch.bool)
        n_chunks = int(n_chunks)
        if valid.dim() == 1:
            valid = (
                valid[:n_chunks]
                .view(1, 1, n_chunks)
                .expand(raw_q.shape[0], raw_q.shape[1], n_chunks)
            )
        elif valid.dim() == 2:
            valid = (
                valid[:, :n_chunks]
                .unsqueeze(0)
                .expand(raw_q.shape[0], raw_q.shape[1], n_chunks)
            )
        else:
            valid = valid[:, :, :n_chunks]
        return valid.contiguous()

    @staticmethod
    def _can_use_cuda_scores(
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        scores_out: torch.Tensor,
    ) -> bool:
        return (
            raw_q.is_cuda
            and chunk_min.is_cuda
            and chunk_max.is_cuda
            and scores_out.is_cuda
            and raw_q.is_floating_point()
            and chunk_min.dtype == raw_q.dtype
            and chunk_max.dtype == raw_q.dtype
            and scores_out.dtype == torch.float32
        )

    def quest_chunk_score(
        self,
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        scores_out: torch.Tensor,
        n_chunks: int,
        chunk_valid: torch.Tensor | None = None,
    ) -> None:
        if not self._can_use_cuda_scores(raw_q, chunk_min, chunk_max, scores_out):
            self._fallback_ops().quest_chunk_score(
                raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid
            )
            return

        n_chunks = int(n_chunks)
        q = raw_q.contiguous()
        cmin = chunk_min.contiguous()
        cmax = chunk_max.contiguous()
        valid = self._normalize_valid(chunk_valid, raw_q, n_chunks)
        out = scores_out if scores_out.is_contiguous() else torch.empty_like(scores_out)
        self._mod.quest_chunk_score(q, cmin, cmax, out, n_chunks, valid)
        if out is not scores_out:
            scores_out.copy_(out)
        if scores_out.shape[-1] > n_chunks:
            scores_out[..., n_chunks:].fill_(float("-inf"))

    def quest_sub_chunk_score(
        self,
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        large_idx: torch.Tensor,
        sub_scores: torch.Tensor,
        nk_large: int,
        factor: int,
    ) -> None:
        if (
            not self._can_use_cuda_scores(raw_q, chunk_min, chunk_max, sub_scores)
            or not large_idx.is_cuda
            or large_idx.dtype != torch.int64
        ):
            self._fallback_ops().quest_sub_chunk_score(
                raw_q,
                chunk_min,
                chunk_max,
                large_idx,
                sub_scores,
                nk_large,
                factor,
            )
            return

        nk_large = int(nk_large)
        factor = int(factor)
        q = raw_q.contiguous()
        cmin = chunk_min.contiguous()
        cmax = chunk_max.contiguous()
        idx = large_idx.contiguous()
        out = sub_scores if sub_scores.is_contiguous() else torch.empty_like(sub_scores)
        self._mod.quest_sub_chunk_score(q, cmin, cmax, idx, out, nk_large, factor)
        if out is not sub_scores:
            sub_scores.copy_(out)
        n_written = nk_large * factor
        if sub_scores.shape[-1] > n_written:
            sub_scores[..., n_written:].fill_(float("-inf"))

    def quest_map_back(
        self,
        large_idx: torch.Tensor,
        sub_topk_pos: torch.Tensor,
        chunk_idx: torch.Tensor,
        factor: int,
        n_chunks: int,
    ) -> None:
        if (
            not large_idx.is_cuda
            or not sub_topk_pos.is_cuda
            or not chunk_idx.is_cuda
            or large_idx.dtype != torch.int64
            or sub_topk_pos.dtype != torch.int64
            or chunk_idx.dtype != torch.int64
        ):
            self._fallback_ops().quest_map_back(
                large_idx, sub_topk_pos, chunk_idx, factor, n_chunks
            )
            return

        out = chunk_idx if chunk_idx.is_contiguous() else torch.empty_like(chunk_idx)
        self._mod.quest_map_back(
            large_idx.contiguous(),
            sub_topk_pos.contiguous(),
            out,
            int(factor),
            int(n_chunks),
        )
        if out is not chunk_idx:
            chunk_idx.copy_(out)


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


def chunk_density_scores(
    chunk_ids: torch.Tensor,
    centroids: torch.Tensor,
    raw_q: torch.Tensor,
    strict: bool | None = None,
) -> torch.Tensor:
    """Score selected chunk centroids without materializing a gather."""
    strict = _want_strict() if strict is None else strict
    if chunk_ids.is_cuda:
        mod = _try_load_rerank_cuda()
        if mod is not None:
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
    return (selected.to(torch.float32) * raw_q.unsqueeze(2).to(torch.float32)).sum(-1)


def dense_mask_from_topk(
    positions: torch.Tensor,
    num_chunks: int,
    strict: bool | None = None,
) -> torch.Tensor:
    """Build the CDS dense mask with one fused CUDA launch."""
    strict = _want_strict() if strict is None else strict
    mask = torch.empty(
        *positions.shape[:2],
        int(num_chunks),
        dtype=torch.bool,
        device=positions.device,
    )
    if positions.is_cuda:
        mod = _try_load_rerank_cuda()
        if mod is not None:
            mod.mask_from_topk(positions.contiguous(), mask)
            return mask
    if strict:
        raise RuntimeError("ZoomKV strict mode: CDS mask CUDA required")
    mask.zero_()
    mask.scatter_(2, positions, True)
    return mask


def quest_score_reference(
    raw_q: torch.Tensor, chunk_min: torch.Tensor, chunk_max: torch.Tensor
) -> torch.Tensor:
    return quest_bound_scores(raw_q, chunk_min, chunk_max)


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
