# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import functools
import os
import time

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.dsa_litetopk import (
    dsa_litetopk_latest_available,
    dsa_litetopk_latest_dense_topk,
    dsa_litetopk_latest_prepare_permuted_gather,
    dsa_litetopk_latest_release_pair_swap_workspace,
    dsa_litetopk_latest_stash_dense,
    dsa_litetopk_latest_streaming_indexer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

# Must agree with the fused-indexer planner and litetopk_indexer's format-
# specific crossover. An explicit legacy min-S remains a universal override.
_LITETOPK_MIN_S_OVERRIDE = os.environ.get("VLLM_LITETOPK_PRODUCTION_MIN_S")
_LITETOPK_FUSED_MIN_SEQ_LEN = int(_LITETOPK_MIN_S_OVERRIDE or "196608")
_LITETOPK_FP4_FUSED_MIN_SEQ_LEN = int(
    os.environ.get(
        "VLLM_LITETOPK_FP4_PRODUCTION_MIN_S",
        _LITETOPK_MIN_S_OVERRIDE or "65536",
    )
)


def _litetopk_fused_min_seq_len(use_fp4: bool) -> int:
    return _LITETOPK_FP4_FUSED_MIN_SEQ_LEN if use_fp4 else _LITETOPK_FUSED_MIN_SEQ_LEN


# VLLM_LITETOPK_PATH_TIMING=1: per-call CUDA-event timing of the prefill
# indexer, split by (path, 64K band of S) and (gather, topk) segments.
# Events are read back lazily (only pairs whose end already completed), so
# the instrumented run stays sync-free.
_LITETOPK_PATH_TIMING = os.environ.get("VLLM_LITETOPK_PATH_TIMING", "0") == "1"
_LITETOPK_PT_STATS: dict = {}
_LITETOPK_HOST_TIMING = os.environ.get("VLLM_LITETOPK_HOST_TIMING", "0") == "1"
_LITETOPK_HOST_STATS: dict = {}


def _litetopk_host_commit(kind, dt):
    rec = _LITETOPK_HOST_STATS.setdefault(kind, [0.0, 0])
    rec[0] += dt
    rec[1] += 1
    if rec[1] % 256 == 0:
        print(
            f"[litetopk] host-timing {kind}: n={rec[1]} "
            f"host_ms={rec[0] / rec[1] * 1e3:.3f}",
            flush=True,
        )


def _litetopk_pt_mark():
    if not _LITETOPK_PATH_TIMING:
        return None
    import torch as _torch

    ev = _torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _litetopk_pt_commit(kind, seq_len, ev_seq):
    if not _LITETOPK_PATH_TIMING or not ev_seq or ev_seq[0] is None:
        return
    key = f"{kind}_s{(seq_len + 65535) // 65536}"
    rec = _LITETOPK_PT_STATS.setdefault(key, {"pend": [], "tot": [0.0, 0.0], "n": 0})
    rec["pend"].append(ev_seq)
    while rec["pend"] and rec["pend"][0][-1].query():
        seq = rec["pend"].pop(0)
        for i in range(len(seq) - 1):
            rec["tot"][i] += seq[i].elapsed_time(seq[i + 1])
        rec["n"] += 1
        if rec["n"] % 256 == 0:
            print(
                f"[litetopk] path-timing {key}: n={rec['n']} "
                f"gather_ms={rec['tot'][0] / rec['n']:.3f} "
                f"topk_ms={rec['tot'][1] / rec['n']:.3f}",
                flush=True,
            )


logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _mask_init_and_local_tokens(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    num_init_tokens: int,
    num_local_tokens: int,
) -> None:
    """Force streaming tokens into the top-k set (streaming-aware indexing):
    scatter +inf into the first ``num_init_tokens`` and last
    ``num_local_tokens`` columns of each row's valid range
    ``[row_starts, row_ends)``. Out-of-range writes are clamped into
    ``[row_starts, row_ends)`` so they never leak into other rows' ranges.
    """
    device = logits.device
    ends = row_ends.to(device=device, dtype=torch.int64).reshape(-1)
    if row_starts is None:
        starts = torch.zeros_like(ends)
    else:
        starts = row_starts.to(device=device, dtype=torch.int64).reshape(-1)
    last = (ends - 1).clamp_max_(logits.shape[1] - 1).clamp_min_(starts)
    if num_init_tokens > 0:
        init_idx = starts[:, None] + torch.arange(
            num_init_tokens, dtype=torch.int64, device=device
        )
        init_idx = torch.minimum(init_idx, last[:, None])
        logits.scatter_(1, init_idx, float("inf"))
    if num_local_tokens > 0:
        local_idx = last[:, None] - torch.arange(
            num_local_tokens, dtype=torch.int64, device=device
        )
        local_idx = torch.maximum(local_idx, starts[:, None])
        logits.scatter_(1, local_idx, float("inf"))


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


# Bumped whenever topk_indices_buffer is (re)written (see the note at the
# bump site inside sparse_attn_indexer).
_SPARSE_IDX_VERSION = [0]


@functools.cache
def _cooperative_topk_available() -> bool:
    """The cooperative_topk op is only compiled on CUDA 12.9+ (its cuda::ptx
    sem_relaxed TMA path); fall back to persistent_topk when it is absent."""
    return hasattr(torch.ops._C, "cooperative_topk")


_DSA_DUMP_STATE = {"seen": set(), "count": 0}


def _maybe_dsa_dump(use_fp4, layer_key, q, q_sf, kv, kv_sf, w, ks, ke, topk, total_s):
    """Env-gated real-cache dump on the official dense fp4 path.

    VLLM_DSA_DUMP_DIR=<dir> enables; dumps the first call at each new
    total_seq_lens >= VLLM_DSA_DUMP_MIN_S (default 245760), up to
    VLLM_DSA_DUMP_MAX (default 2) events, rank 0 only. The first call
    per forward is the first DSA layer, so consecutive dumps are the
    same layer's consecutive chunks.
    """
    d = os.environ.get("VLLM_DSA_DUMP_DIR")
    if d and os.environ.get("VLLM_DSA_DUMP_DEBUG") == "1":
        st0 = _DSA_DUMP_STATE
        st0["dbg"] = st0.get("dbg", 0) + 1
        if st0["dbg"] <= 40:
            print(
                f"[dsa-dump-dbg] call#{st0['dbg']} fp4={use_fp4} "
                f"S={total_s} q={tuple(q.shape)} layer={layer_key}",
                flush=True,
            )
    if not d or not use_fp4:
        return
    st = _DSA_DUMP_STATE
    thresh = int(os.environ.get("VLLM_DSA_DUMP_MIN_S", "245760"))
    maxn = int(os.environ.get("VLLM_DSA_DUMP_MAX", "2"))
    if total_s < thresh or total_s in st["seen"] or st["count"] >= maxn:
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    st["seen"].add(total_s)
    st["count"] += 1
    from safetensors.torch import save_file

    tag = str(layer_key).replace("/", "_").replace(".", "_")
    save_file(
        {
            "q": q.view(torch.uint8).contiguous().cpu(),
            "q_sf": q_sf.contiguous().cpu(),
            "kv": kv.view(torch.uint8).contiguous().cpu(),
            "kv_sf": kv_sf.contiguous().cpu(),
            "weights": w.float().contiguous().cpu(),
            "ks": ks.int().cpu(),
            "ke": ke.int().cpu(),
            "topk": topk.int().cpu(),
        },
        os.path.join(d, f"dsv4_dump_S{total_s}_{tag}.safetensors"),
    )
    print(
        f"[dsa-dump] wrote S={total_s} layer={tag} "
        f"q={tuple(q.shape)} kv={tuple(kv.shape)}",
        flush=True,
    )


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    if not has_prefill:
        dsa_litetopk_latest_release_pair_swap_workspace(hidden_states.device)

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # Monotonic version of topk_indices_buffer contents: 'shared'-indexer
    # layers (e.g. GLM indexer_types) reuse the previous full layer's indices
    # without re-entering this function, so downstream consumers (the grouped
    # sparse attention in litedsa.py) can cache per-version work.
    _SPARSE_IDX_VERSION[0] += 1
    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            _pt0 = _litetopk_pt_mark()
            _host_t0 = time.perf_counter() if _LITETOPK_HOST_TIMING else 0.0
            permuted_plan = None
            if (
                chunk.fused_indexer_planned
                and os.environ.get("VLLM_LITETOPK_CARRY_DEBUG") == "1"
            ):
                print(
                    f"[litetopk] chunk dbg skip_gather={chunk.skip_kv_gather} "
                    f"lts={chunk.local_total_seq_lens} "
                    f"mlts={chunk.max_local_total_seq_lens} "
                    f"tts={chunk.total_seq_lens} "
                    f"q={chunk.token_end - chunk.token_start} "
                    f"nreq={chunk.num_reqs}",
                    flush=True,
                )
            noop_fused = (
                chunk.fused_indexer_planned
                and os.environ.get("VLLM_LITETOPK_NOOP_INDEXER", "0") == "1"
            )
            if (
                not chunk.skip_kv_gather
                and chunk.local_total_seq_lens > 0
                and not noop_fused
            ):
                query_length = chunk.token_end - chunk.token_start
                if (
                    chunk.fused_indexer_planned
                    and envs.VLLM_LITETOPK
                    and envs.VLLM_DSA_MODE in ("litetopk", "litedsa")
                    and dcp_world_size == 1
                    and not current_platform.is_xpu()
                    # The pair-swap plan is a cooperative launch, which CUDA
                    # graph capture cannot record; capture warmups take the
                    # ordinary gather + materialized path below.
                    and not torch.cuda.is_current_stream_capturing()
                    and dsa_litetopk_latest_available(
                        use_fp4=use_fp4_cache,
                        topk=topk_tokens,
                    )
                ):
                    # The carry is consumed before the ordinary gather.  The
                    # planner pair-swaps HOT12288 into the physical prefix and
                    # its gather writes the same shared workspace, so there is
                    # no later index_select or second cache read.
                    permuted_plan = dsa_litetopk_latest_prepare_permuted_gather(
                        kv_cache,
                        k_quant,
                        k_scale,
                        chunk.block_table,
                        sequence_length=chunk.max_local_total_seq_lens,
                        query_length=query_length,
                        num_reqs=chunk.num_reqs,
                        common_end=(
                            (
                                chunk.common_ke_min
                                if chunk.common_ke_min > 0
                                else chunk.total_seq_lens - query_length + 1
                            )
                            - num_local_tokens
                        ),
                        window_start=num_init_tokens,
                        topk=topk_tokens,
                        hot_key=k_cache_prefix,
                    )
                if permuted_plan is None:
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        k_quant,
                        k_scale,
                        chunk.block_table,
                        chunk.local_cu_seq_lens,
                    )
            _pt1 = _litetopk_pt_mark()

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                # The metadata flag is the source of truth: it records that
                # the planner deliberately left this whole [Q, S] chunk
                # unsplit because LiteTopK was expected to materialize no
                # logits. Runtime requirements are checked separately so a
                # planner/runtime mismatch fails clearly instead of trying an
                # unsafe multi-GiB dense fallback.
                if noop_fused:
                    # Ceiling probe: charge the fused chunks zero indexer
                    # cost. Output indices are valid positions but wrong,
                    # so tokens are meaningless -- timing only. Modes bracket
                    # the downstream sparse-attention cost: arange (one hot
                    # L2-resident block, optimistic), window (per-row local
                    # tail), random (uniform scatter, pessimistic).
                    mode = os.environ.get("VLLM_LITETOPK_NOOP_MODE", "arange")
                    rows = topk_indices.shape[0]
                    dev = topk_indices.device
                    if mode == "random":
                        hi = max(1, chunk.common_ke_min)
                        topk_indices[:] = torch.randint(
                            0, hi, (rows, topk_tokens), dtype=torch.int32, device=dev
                        )
                    elif mode == "window":
                        base = cu_seqlen_ke[:rows].to(torch.int32)
                        offs = torch.arange(topk_tokens, dtype=torch.int32, device=dev)
                        topk_indices[:] = (
                            base.unsqueeze(1) - topk_tokens + offs.unsqueeze(0)
                        ).clamp_(min=0)
                    else:
                        topk_indices[:] = torch.arange(
                            topk_tokens, dtype=torch.int32, device=dev
                        ).unsqueeze(0)
                    logits = None
                    _pt1b = _litetopk_pt_mark()
                    _litetopk_pt_commit(
                        "fused", chunk.max_local_total_seq_lens, (_pt0, _pt1, _pt1b)
                    )
                    continue
                fused_indexer_planned = chunk.fused_indexer_planned
                fused_indexer_runtime_eligible = (
                    fused_indexer_planned
                    and not torch.cuda.is_current_stream_capturing()
                    and envs.VLLM_LITETOPK
                    and envs.VLLM_DSA_MODE in ("litetopk", "litedsa")
                    and dcp_world_size == 1
                    and not current_platform.is_xpu()
                    and q_slice_cast.dim() == 3
                    and q_slice_cast.shape[1] in (32, 64)
                    and q_slice_cast.shape[2] == (64 if use_fp4_cache else 128)
                    and (not use_fp4_cache or q_scale_slice is not None)
                    and dsa_litetopk_latest_available(
                        use_fp4=use_fp4_cache,
                        topk=topk_tokens,
                    )
                )
                if fused_indexer_planned and not fused_indexer_runtime_eligible:
                    if torch.cuda.is_current_stream_capturing():
                        # Capture warmup shapes are small enough for the
                        # materialized path even though the planner left the
                        # chunk unsplit.
                        fused_indexer_planned = False
                    else:
                        raise RuntimeError(
                            "LiteTopK was planned for an unsplit prefill "
                            "chunk, but its runtime requirements are not "
                            "satisfied; dense fallback is unsafe"
                        )
                permuted_prefix_expected = (
                    fused_indexer_planned
                    and _litetopk_fused_min_seq_len(use_fp4_cache)
                    <= chunk.max_local_total_seq_lens
                    <= 1 << 20
                )
                if (
                    fused_indexer_runtime_eligible
                    and permuted_prefix_expected
                    and permuted_plan is None
                ):
                    raise RuntimeError(
                        "an exact-once HOT-prefix path was planned for an "
                        "unsplit prefill chunk, but no valid HOT12288 carry "
                        "was available before the paged-cache gather"
                    )
                if fused_indexer_runtime_eligible and (
                    dsa_litetopk_latest_streaming_indexer(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_tokens,
                        topk_indices,
                        num_init_tokens,
                        num_local_tokens,
                        num_reqs=chunk.num_reqs,
                        ke_min_hint=(
                            chunk.common_ke_min
                            if chunk.common_ke_min > 0
                            else chunk.total_seq_lens
                            - (chunk.token_end - chunk.token_start)
                            + 1
                        ),
                        hot_key=k_cache_prefix,
                        permuted_plan=permuted_plan,
                        q_sf=q_scale_slice if use_fp4_cache else None,
                    )
                ):
                    logits = None
                elif fused_indexer_runtime_eligible:
                    # The metadata builder deliberately skipped dense-logits
                    # sub-chunking for this fused-eligible single-request
                    # chunk. Falling back here would materialize the whole
                    # [Q, S] matrix (about 32 GiB at Q=8192, S=1M), so fail
                    # closed with the original eligibility problem instead
                    # of turning it into an unrelated OOM.
                    raise RuntimeError(
                        "LiteTopK was selected for an unsplit prefill chunk "
                        "but declined at runtime; dense fallback is unsafe"
                    )
                elif current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                # The fused indexer already wrote topk_indices and set
                # logits=None; only the dense paths need the top-k selection.
                _pt_kind = "fused" if logits is None else "dense"
                if logits is not None:
                    num_rows = logits.shape[0]
                    if not dsa_litetopk_latest_dense_topk(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        topk_tokens,
                        seq_len_hint=chunk.max_local_total_seq_lens,
                        num_init_tokens=num_init_tokens,
                        num_local_tokens=num_local_tokens,
                    ):
                        _pt_kind = "official"
                        # Streaming-aware indexing (LongCat): the LiteTopK
                        # selector appends sink/local indices itself. Native
                        # top-k still receives the historical +inf mask.
                        if num_init_tokens > 0 or num_local_tokens > 0:
                            _mask_init_and_local_tokens(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                num_init_tokens,
                                num_local_tokens,
                            )
                        ops.top_k_per_row_prefill(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            topk_indices,
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            topk_tokens,
                        )
                    dsa_litetopk_latest_stash_dense(
                        topk_indices,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        seq_len=chunk.total_seq_lens,
                        num_init_tokens=num_init_tokens,
                        num_local_tokens=num_local_tokens,
                        hot_key=k_cache_prefix,
                        use_fp4=use_fp4_cache,
                    )
                    _maybe_dsa_dump(
                        use_fp4_cache,
                        k_cache_prefix,
                        q_slice_cast,
                        q_scale_slice,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        chunk.total_seq_lens,
                    )

            # The fused indexer path writes topk_indices directly and produces no
            # logits tensor; it is gated to dcp_world_size == 1, so the DCP merge
            # (a no-op without context parallelism) is skipped.
            if logits is not None:
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )
            if _LITETOPK_HOST_TIMING:
                _litetopk_host_commit(_pt_kind, time.perf_counter() - _host_t0)
            if _pt0 is not None and chunk.local_total_seq_lens > 0:
                _pt2 = _litetopk_pt_mark()
                _litetopk_pt_commit(
                    _pt_kind,
                    chunk.max_local_total_seq_lens,
                    [_pt0, _pt1, _pt2],
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if num_init_tokens > 0 or num_local_tokens > 0:
            # seq_lens is (B, next_n) whenever next_n > 1, so flattening
            # yields the per-row context length.
            _mask_init_and_local_tokens(
                logits,
                None,
                seq_lens.reshape(-1)[:num_rows],
                num_init_tokens,
                num_local_tokens,
            )

        use_cooperative_topk = (
            current_platform.is_cuda()
            and _cooperative_topk_available()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        num_init_tokens: int = 0,
        num_local_tokens: int = 0,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""
        self.num_init_tokens = num_init_tokens
        self.num_local_tokens = num_local_tokens
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            num_init_tokens=self.num_init_tokens,
            num_local_tokens=self.num_local_tokens,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        assert self.num_init_tokens == 0 and self.num_local_tokens == 0, (
            "Streaming-aware indexing (index_init_tokens/index_local_tokens) is "
            "not supported on the ROCm sparse_attn_indexer path yet"
        )
        if rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
