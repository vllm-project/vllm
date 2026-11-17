# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the FlyDSL sliding-window GQA attention (gfx950).

``flydsl_swa_gqa_fwd`` runs the split kernel of ``flydsl_swa_gqa_gfx950``
and a Triton combine over its partials, for the Kimi-K3.1 DSpark draft
layers: head_dim 64, a GQA group of 6, a causal sliding window, at most
``MAX_ROWS`` query rows per request, over the TRITON_ATTN cache (bf16, or
fp8 e4m3 with per-tensor scales). ``flydsl_swa_gqa_config_reason`` /
``flydsl_swa_gqa_call_reason`` say why a layer / call is outside that slice.

CUDA graphs: the grid depends only on (requests, max query rows), the
partials live in one buffer per device that is allocated before capture
(``flydsl_swa_gqa_reserve``), no host value is read back, and a config that
has not run before a capture is refused inside it (``_COMPILED``).
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

logger = init_logger(__name__)

HEAD_DIM = 64
QUERY_GROUP = 6
MAX_ROWS = 16
# The partials buffer is sized for this many requests at MAX_ROWS rows (51 MiB
# for 2 kv heads and a 2048 window); fewer rows fit more requests.
RESERVE_REQS = 16
LOG2E = 1.4426950408889634

__all__ = [
    "flydsl_swa_gqa_call_reason",
    "flydsl_swa_gqa_config_reason",
    "flydsl_swa_gqa_fwd",
    "flydsl_swa_gqa_reserve",
    "flydsl_swa_gqa_warmup",
]


@functools.cache
def _is_gfx950() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx950

        return bool(on_gfx950())
    except Exception:  # noqa: BLE001
        return False


@functools.cache
def _flydsl_import_error() -> str | None:
    try:
        import flydsl.compiler  # noqa: F401
        import flydsl.expr  # noqa: F401

        from vllm.models.kimi_k3.amd.ops import flydsl_swa_gqa_gfx950  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def _config(max_rows: int) -> tuple[int, int]:
    """(query tiles, tiles per workgroup) for ``max_rows`` rows per request."""
    nt = (max_rows * QUERY_GROUP + 15) // 16
    tpc = 4 if nt <= 3 else 2
    return nt, tpc


@functools.cache
def _launcher(num_kv_heads, n_qtiles, tiles_per_cta, window, kv_fp8):
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa_gfx950 import (
        build_swa_gqa_module,
    )

    return build_swa_gqa_module(
        qg=QUERY_GROUP,
        num_kv_heads=num_kv_heads,
        n_qtiles=n_qtiles,
        tiles_per_cta=tiles_per_cta,
        window=window,
        kv_fp8=kv_fp8,
    )


# Launcher keys that have run once (compiled and loaded): only these may run
# inside a CUDA-graph capture.
_COMPILED: set[tuple[int, int, int, int, bool]] = set()


def _num_splits(window: int, max_rows: int, tiles_per_cta: int) -> int:
    max_tiles = (window - 1 + max_rows + 15) // 16 + 1
    return (max_tiles + tiles_per_cta - 1) // tiles_per_cta


def _partial_numel(num_reqs, num_kv_heads, window, max_rows):
    nt, tpc = _config(max_rows)
    ctas = num_reqs * num_kv_heads * _num_splits(window, max_rows, tpc)
    return ctas * nt * 16 * HEAD_DIM, ctas * 2 * nt * 16


# One partial buffer per device, shared by every layer (they run in order on
# one stream). Sized at reserve time; a call that does not fit falls back.
_PARTIALS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _dev(device) -> torch.device:
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.accelerator.current_device_index())
    return device


def flydsl_swa_gqa_reserve(device, num_kv_heads: int, window: int) -> None:
    device = _dev(device)
    po_n, pml_n = _partial_numel(RESERVE_REQS, num_kv_heads, window, MAX_ROWS)
    cur = _PARTIALS.get(device)
    if cur is not None and cur[0].numel() >= po_n and cur[1].numel() >= pml_n:
        return
    _PARTIALS[device] = (
        torch.empty(po_n, dtype=torch.float32, device=device),
        torch.empty(pml_n, dtype=torch.float32, device=device),
    )


def flydsl_swa_gqa_config_reason(
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    window: int,
    kv_dtype: torch.dtype,
) -> str | None:
    """Why a layer cannot use the kernel at all (None: it can)."""
    if not _is_gfx950():
        return "not a gfx950 GPU"
    err = _flydsl_import_error()
    if err is not None:
        return f"FlyDSL unavailable ({err})"
    if head_size != HEAD_DIM:
        return f"head size {head_size} != {HEAD_DIM}"
    if num_heads != num_kv_heads * QUERY_GROUP:
        return f"{num_heads} q heads over {num_kv_heads} kv heads, not a group of 6"
    if window <= 0:
        return "no sliding window"
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return f"KV cache dtype {kv_dtype} is neither bf16 nor fp8 e4m3fn"
    return None


def flydsl_swa_gqa_call_reason(
    *,
    num_reqs: int,
    max_query_len: int,
    window: int,
    kv_cache: torch.Tensor,
) -> str | None:
    """Why this call cannot use the kernel (None: it can). Host values only."""
    if not 1 <= max_query_len <= MAX_ROWS:
        return f"{max_query_len} query rows a request (max {MAX_ROWS})"
    if kv_cache.stride(-1) != 1 or kv_cache.shape[-1] != 2 * HEAD_DIM:
        return "KV cache content is not [K | V] contiguous"
    parts = _PARTIALS.get(kv_cache.device)
    if parts is None:
        return "no partials reserved on this device"
    po_n, pml_n = _partial_numel(num_reqs, kv_cache.shape[1], window, max_query_len)
    if po_n > parts[0].numel() or pml_n > parts[1].numel():
        return f"{num_reqs} requests x {max_query_len} rows exceed the partials"
    if torch.cuda.is_current_stream_capturing():
        nt, tpc = _config(max_query_len)
        key = (kv_cache.shape[1], nt, tpc, window, kv_cache.element_size() == 1)
        if key not in _COMPILED:
            return f"config {key} not compiled before CUDA-graph capture"
    return None


@triton.jit
def _swa_gqa_combine(
    out_ptr,
    po_ptr,
    pml_ptr,
    qsl_ptr,
    sl_ptr,
    vsc_ptr,
    stride_ot,
    stride_oh,
    n_split,
    max_rows,
    QG: tl.constexpr,
    NP: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    TPC: tl.constexpr,
    W: tl.constexpr,
    S_PAD: tl.constexpr,
    HAS_VSC: tl.constexpr,
):
    # One program per (request row, q head): all of its splits in one load.
    pid = tl.program_id(0)
    r = pid // max_rows
    row = pid - r * max_rows
    h = tl.program_id(1)
    q_start = tl.load(qsl_ptr + r)
    q_len = tl.load(qsl_ptr + r + 1) - q_start
    if row >= q_len:
        return
    seq_len = tl.load(sl_ptr + r)
    p0 = seq_len - q_len
    t_lo = tl.maximum(p0 - (W - 1), 0) // 16
    t_hi = (seq_len - 1) // 16
    n_act = tl.maximum((t_hi - t_lo + TPC) // TPC, 0)

    kvh = h // QG
    n = row * QG + (h - kvh * QG)
    cta0 = (r * KVH + kvh) * n_split
    s = tl.arange(0, S_PAD)
    smask = s < n_act
    m_off = (cta0 + s) * 2 * NP + n
    m = tl.load(pml_ptr + m_off, mask=smask, other=float("-inf"))
    m_max = tl.max(m, axis=0)
    m_max = tl.where(m_max > float("-inf"), m_max, 0.0)
    w = tl.exp2(m - m_max)
    l_sum = tl.sum(tl.load(pml_ptr + m_off + NP, mask=smask, other=0.0) * w, axis=0)
    d = tl.arange(0, D)
    o = tl.load(
        po_ptr + ((cta0 + s) * NP + n)[:, None] * D + d[None, :],
        mask=smask[:, None],
        other=0.0,
    )
    acc = tl.sum(o * w[:, None], axis=0)
    scale = tl.where(l_sum > 0.0, 1.0 / l_sum, 0.0)
    if HAS_VSC:
        scale = scale * tl.load(vsc_ptr)
    tl.store(
        out_ptr + (q_start + row) * stride_ot + h * stride_oh + d,
        (acc * scale).to(out_ptr.dtype.element_ty),
    )


def flydsl_swa_gqa_fwd(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    out: torch.Tensor,
    scale: float,
    window: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal sliding-window GQA attention of each request's query rows.

    query [T, Hq, 64]; kv_cache logical [blocks, Hkv, block_size, 128] (K | V)
    at any strides, bf16 or fp8 (k_scale / v_scale fp32 per-tensor then);
    block_table [R, max_blocks]; seq_lens [R] (keys incl. the query rows);
    query_start_loc [R + 1]; out [T, Hq, 64]. Row i of request r sits at
    position seq_lens[r] - q_len + i and attends keys in (pos - window, pos].
    """
    num_reqs = query_start_loc.shape[0] - 1
    num_kv_heads = kv_cache.shape[1]
    kv_fp8 = kv_cache.element_size() == 1
    nt, tpc = _config(max_query_len)
    n_split = _num_splits(window, max_query_len, tpc)
    po, pml = _PARTIALS[kv_cache.device]
    es = kv_cache.element_size()
    key = (num_kv_heads, nt, tpc, window, kv_fp8)
    launch = _launcher(*key)
    ksc = k_scale if kv_fp8 else seq_lens
    _run_compiled(
        launch,
        query,
        kv_cache.view(torch.uint8) if kv_fp8 else kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        ksc,
        po,
        pml,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0) * es,
        kv_cache.stride(1) * es,
        kv_cache.stride(2) * es,
        HEAD_DIM * es,
        block_table.stride(0),
        kv_cache.shape[2],
        float(scale) * LOG2E,
        n_split,
        num_reqs,
        torch.cuda.current_stream(query.device),
    )
    _swa_gqa_combine[(num_reqs * max_query_len, num_kv_heads * QUERY_GROUP)](
        out,
        po,
        pml,
        query_start_loc,
        seq_lens,
        v_scale if kv_fp8 else po,
        out.stride(0),
        out.stride(1),
        n_split,
        max_query_len,
        QG=QUERY_GROUP,
        NP=nt * 16,
        KVH=num_kv_heads,
        D=HEAD_DIM,
        TPC=tpc,
        W=window,
        S_PAD=triton.next_power_of_2(n_split),
        HAS_VSC=kv_fp8,
        num_warps=2,
    )
    _COMPILED.add(key)
    return out


def flydsl_swa_gqa_warmup(
    device,
    num_kv_heads: int,
    window: int,
    kv_dtype: torch.dtype,
    rows: tuple[int, ...] = tuple(range(1, MAX_ROWS + 1)),
) -> None:
    """Reserve the partials and compile + run every query-tile config on a
    one-key problem, so nothing compiles inside a CUDA-graph capture."""
    dev = _dev(device)
    flydsl_swa_gqa_reserve(dev, num_kv_heads, window)
    kv = torch.zeros(1, num_kv_heads, 16, 2 * HEAD_DIM, dtype=kv_dtype, device=dev)
    bt = torch.zeros(1, 1, dtype=torch.int32, device=dev)
    sl = torch.ones(1, dtype=torch.int32, device=dev)
    sc = torch.ones(1, dtype=torch.float32, device=dev)
    # Every row count, not one per tile config: the combine's Triton
    # specializations also depend on it.
    for q_rows in rows:
        nq = num_kv_heads * QUERY_GROUP
        q = torch.zeros(q_rows, nq, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(q)
        qsl = torch.tensor([0, q_rows], dtype=torch.int32, device=dev)
        flydsl_swa_gqa_fwd(
            q, kv, bt, sl, qsl, q_rows, out, 0.125, window, sc, sc
        )
