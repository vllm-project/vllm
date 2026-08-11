# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped (packed) sparse prefill attention for DSA models on Blackwell.

G adjacent query tokens' heads are packed into one 128-row attention call
over the UNION of their top-k lists (6-7x KV dedup from causal locality),
and a per-query query-major membership bitmask restores each query's exact
attend-set inside the fp8 kernel. Floating-point outputs can differ slightly
from stock sparse attention because grouping changes the execution order.

Orchestrates two C++ primitives (``litedsa_union_qm`` /
``litedsa_masked_mla_fp8``, SM100 only) with two host-side optimizations from
the reference research stack:

* **version cache**: 'shared'-indexer layers (e.g. GLM ``indexer_types``)
  reuse the leader layer's ``topk_indices_buffer`` without re-entering the
  indexer, and the whole union/membership build is layer-invariant — it is
  built once per buffer version and reused by every consumer layer;
* **overwrite-write buffers**: one persistent buffer set per shape, refilled
  in place on a version miss (no allocation, no fill/zero pre-pass; padding
  is bounded by ``counts`` inside the attention kernel).

Enabled by ``VLLM_DSA_MODE=litedsa`` (G auto = 128 // num_local_heads,
e.g. 16 at TP8). Requires the fp8 KV cache path (query
arrives quantized e4m3). Falls back to the stock per-token path otherwise.

Reproduction (research harness, GLM-5.2 78L TP8 util0.95 @1M, chunk 8192,
REPEATS=2 best): dense-indexer official @2GB logits budget 147.97s; fused
LiteTopK indexer alone 123.27s (1.20x); + this grouped attention
(PACK=16) 116.01s (1.28x, the same official attend-sets).
"""

import functools
import os
import threading
from collections import OrderedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_SEQ_SPACE = 1 << 20
_DYNAMIC_SEQ_SPACE = os.environ.get("VLLM_LITEDSA_DYNAMIC_SPAN", "1") == "1"
_UNION_SO = os.environ.get("VLLM_LITEDSA_UNION_SO")
_UNION_MOD = None
# VLLM_LITEDSA_SO: load BOTH litedsa ops (union + masked attention) from a
# TVM-FFI module instead of torch.ops._C — lets a freshly built kernel ship
# without rebuilding the whole vLLM extension.
_LITEDSA_SO = os.environ.get("VLLM_LITEDSA_SO")
_LITEDSA_MOD = None
_EXECUTION_LOGGED = False


def _litedsa_mod():
    global _LITEDSA_MOD
    if _LITEDSA_MOD is None:
        import tvm_ffi

        _LITEDSA_MOD = tvm_ffi.load_module(_LITEDSA_SO)
    return _LITEDSA_MOD


def _union_seq_space(max_seq_len: int) -> int:
    """Return a 1024-position-aligned bitmap span, bounded by the 1M ABI."""
    if max_seq_len <= 0 or max_seq_len > _MAX_SEQ_SPACE:
        raise ValueError(
            f"LiteDSA max_seq_len must be in (0, {_MAX_SEQ_SPACE}], got {max_seq_len}"
        )
    if not _DYNAMIC_SEQ_SPACE:
        return _MAX_SEQ_SPACE
    return (max_seq_len + 1023) & ~1023


def _run_union(*args) -> None:
    """Run the native union op, or an explicit TVM-FFI A/B override."""
    global _UNION_MOD
    if _LITEDSA_SO:
        _litedsa_mod().union_qm(*args)
    elif _UNION_SO:
        if _UNION_MOD is None:
            import tvm_ffi

            _UNION_MOD = tvm_ffi.load_module(_UNION_SO)
        _UNION_MOD.union_qm(*args)
    else:
        torch.ops._C.litedsa_union_qm(*args)


@functools.cache
def litedsa_available() -> bool:
    """True iff native or explicitly loaded SM100 kernels are available."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (10, 0):
        return False
    if _LITEDSA_SO:
        try:
            mod = _litedsa_mod()
            _ = mod.union_qm, mod.masked_mla_fp8
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LiteDSA override %s is unavailable (%s); falling back to "
                "the stock sparse-attention path",
                _LITEDSA_SO,
                exc,
            )
            return False
    return hasattr(torch.ops._C, "litedsa_union_qm") and hasattr(
        torch.ops._C, "litedsa_masked_mla_fp8"
    )


_UNION_STATS = os.environ.get("VLLM_LITEDSA_UNION_STATS", "0") == "1"
_REUSE_OUTPUT_BUFS = os.environ.get("VLLM_LITEDSA_REUSE_OUTPUT_BUFS", "0") == "1"

# single-slot version cache: [key, payload, hits, misses]
_CACHE: list = [None, None, 0, 0]
# persistent overwrite-write buffers per (ng, G, cap): (u_phys, counts, qm)
_BUFS: dict = {}

# ``out`` is returned to the MLA wrapper, so reuse is deliberately opt-in.
# The wrapper immediately enqueues o_proj on the same CUDA stream before the
# next layer can enter this function. Reusing the storage on that stream is
# therefore ordered after the only production consumer. max_logits/lse never
# escape this function. A host-thread id is included as well as the CUDA stream
# to avoid aliasing two concurrent Python dispatchers that happen to target the
# same stream.
#
# Capacity grows geometrically in ng. Bound the number of resident
# device/stream/thread pools so transient streams cannot make this cache grow
# without limit.
_OUTPUT_BUF_CACHE_MAX_POOLS = 4
_OUTPUT_BUFS: OrderedDict = OrderedDict()
_OUTPUT_BUFS_LOCK = threading.Lock()


def _geometric_capacity(required: int) -> int:
    """Smallest power-of-two capacity covering ``required``."""
    if required <= 0:
        raise ValueError("LiteDSA output buffer capacity must be positive")
    return 1 << (required - 1).bit_length()


def _output_buffer_key(
    device: torch.device,
    stream_handle: int,
    owner_thread: int,
) -> tuple:
    """Key one overwrite pool by ownership, shape signature, and dtype."""
    return (
        device.type,
        device.index,
        stream_handle,
        owner_thread,
        (128, 512),
        torch.bfloat16,
        (128,),
        torch.float32,
    )


def _get_or_create_output_buffers(
    device: torch.device,
    stream_handle: int,
    owner_thread: int,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous ``ng`` views from a bounded geometric-capacity pool.

    This helper is parameterized by the stream handle so its cache semantics
    can be tested without a CUDA device. Production callers must pass the
    handle of ``torch.cuda.current_stream(device)``.
    """
    key = _output_buffer_key(device, stream_handle, owner_thread)
    with _OUTPUT_BUFS_LOCK:
        entry = _OUTPUT_BUFS.get(key)
        if entry is None or entry[0] < ng:
            capacity = _geometric_capacity(ng)
            entry = (
                capacity,
                torch.empty(
                    capacity,
                    128,
                    512,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.empty(
                    capacity,
                    128,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    capacity,
                    128,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            _OUTPUT_BUFS[key] = entry
        _OUTPUT_BUFS.move_to_end(key)
        while len(_OUTPUT_BUFS) > _OUTPUT_BUF_CACHE_MAX_POOLS:
            _OUTPUT_BUFS.popitem(last=False)
    _, out, max_logits, lse = entry
    return out[:ng], max_logits[:ng], lse[:ng]


def _reuse_output_buffers_allowed() -> bool:
    if not _REUSE_OUTPUT_BUFS:
        return False
    # The backend already excludes this path from CUDA Graph capture. Keep the
    # lower-level helper fail-closed in case a future/direct caller bypasses it.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "VLLM_LITEDSA_REUSE_OUTPUT_BUFS is not capture-safe; "
            "LiteDSA must fall back before CUDA Graph capture"
        )
    # Reusing a returned tensor would violate autograd's saved-tensor lifetime.
    # vLLM eager inference runs with gradients disabled.
    return not torch.is_grad_enabled()


def _build_or_reuse(
    topk_indices: torch.Tensor,
    num_tokens: int,
    group_size: int,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    version: int,
    seq_space: int,
):
    ng = num_tokens // group_size
    cap = group_size * topk_indices.shape[1]
    key = (version, num_tokens, group_size, cap, seq_space)
    if _CACHE[0] == key:
        _CACHE[2] += 1
        return _CACHE[1]
    buf_key = (ng, group_size, cap)
    bufs = _BUFS.get(buf_key)
    if bufs is None:
        dev = topk_indices.device
        bufs = _BUFS[buf_key] = (
            torch.empty(ng, cap, dtype=torch.int32, device=dev),
            torch.empty(ng, dtype=torch.int32, device=dev),
            torch.empty(ng, group_size, cap // 32, dtype=torch.int32, device=dev),
        )
    u_phys, counts, memb_qm = bufs
    _run_union(
        topk_indices.contiguous(),
        u_phys,
        counts,
        memb_qm,
        req_id_per_token[:num_tokens:group_size].contiguous(),
        block_table.contiguous(),
        block_size,
        seq_space,
    )
    _CACHE[0], _CACHE[1] = key, bufs
    _CACHE[3] += 1
    if _UNION_STATS and _CACHE[3] % 256 == 0:
        print(
            f"[litedsa] union mean={counts.float().mean().item():.0f} "
            f"max={counts.max().item()} cap={cap} "
            f"cache hits={_CACHE[2]} misses={_CACHE[3]}",
            flush=True,
        )
    return bufs


def litedsa_masked_mqa(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_heads: int,
    bmm1_scale: float,
    bmm2_scale: float,
    group_size: int,
    version: int,
    max_seq_len: int = _MAX_SEQ_SPACE,
) -> torch.Tensor:
    """Grouped masked sparse attention over the group union.

    ``q``: [T, num_heads, 576] fp8_e4m3 (T % group_size == 0,
    num_heads * group_size == 128). Returns bf16 [T, num_heads, 512].
    """
    # Evaluate the capture guard before _build_or_reuse can launch the union
    # kernel, so a direct caller that bypasses the backend fails atomically.
    reuse_output_buffers = _reuse_output_buffers_allowed()
    num_tokens = q.shape[0]
    ng = num_tokens // group_size
    seq_space = _union_seq_space(max_seq_len)
    u_phys, counts, memb_qm = _build_or_reuse(
        topk_indices,
        num_tokens,
        group_size,
        req_id_per_token,
        block_table,
        block_size,
        version,
        seq_space,
    )
    # ``view`` itself requires compatible strides, so make the input
    # contiguous before reshaping (rather than after it).
    qp = q.contiguous().view(ng, 128, q.shape[2])
    if reuse_output_buffers:
        stream = torch.cuda.current_stream(q.device)
        out, max_logits, lse = _get_or_create_output_buffers(
            q.device,
            int(stream.cuda_stream),
            threading.get_ident(),
            ng,
        )
    else:
        out = torch.empty(ng, 128, 512, dtype=torch.bfloat16, device=q.device)
        max_logits = torch.empty(ng, 128, dtype=torch.float32, device=q.device)
        lse = torch.empty(ng, 128, dtype=torch.float32, device=q.device)
    attention_op = (
        _litedsa_mod().masked_mla_fp8
        if _LITEDSA_SO
        else torch.ops._C.litedsa_masked_mla_fp8
    )
    attention_op(
        qp,
        kv_cache.view(-1, 1, kv_cache.shape[-1]),
        u_phys.view(ng, 1, -1),
        bmm1_scale,
        bmm2_scale,
        counts,
        memb_qm,
        num_heads,
        out,
        max_logits,
        lse,
    )
    global _EXECUTION_LOGGED
    if not _EXECUTION_LOGGED:
        logger.info("LITEDSA_KERNEL_EXECUTED grouped masked MLA kernel dispatched")
        _EXECUTION_LOGGED = True
    return out.view(num_tokens, num_heads, 512)
