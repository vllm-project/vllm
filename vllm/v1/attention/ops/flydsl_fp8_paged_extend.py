# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the FlyDSL fp8 paged extend attention kernel (gfx950).

Same contract as `triton_fp8_paged_extend.fp8_paged_extend_attention`, plus an
optional sliding window. The KV split is chosen per request from CPU-side lengths:
a request whose own workgroups cannot fill the GPU splits its context scan, and a
Triton kernel merges the partials. Batches without a split request write the
output directly.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

logger = init_logger(__name__)

HEAD_SIZE = 64
GQA = 8
PAGE_SIZE = 16
MAX_REQS = 64
_BQ = 16
_TILE = 128
_MAX_SPLITS = 32
_MIN_TILES_PER_SPLIT = 64


@functools.cache
def is_flydsl_fp8_paged_extend_available() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False
        from vllm.v1.attention.ops.flydsl_kernels import (  # noqa: F401
            fp8_paged_extend,
        )
    except Exception as ex:  # noqa: BLE001
        logger.warning_once("FlyDSL fp8 paged extend is unavailable: %s", ex)
        return False
    return True


def flydsl_fp8_paged_extend_supported(
    num_heads: int, num_kv_heads: int, head_size: int
) -> bool:
    return (
        head_size == HEAD_SIZE
        and num_heads == GQA * num_kv_heads
        and is_flydsl_fp8_paged_extend_available()
    )


@triton.jit
def _merge_kernel(
    Out,
    PartO,
    PartL,
    Plan,
    CuQ,
    VDesc,
    o_s0,
    o_s1,
    nreq,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_REQS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, MAX_REQS)
    starts = tl.load(CuQ + 1 + i, mask=i < (nreq - 1), other=1 << 30)
    r = tl.sum((starts <= t).to(tl.int32))
    tsb = tl.load(Plan + r * 4 + 1)
    nsp = tl.load(Plan + r * 4 + 3)
    base = tsb + (t - tl.load(CuQ + r)) * nsp
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    m = tl.full([H], float("-inf"), tl.float32)
    for sp in range(nsp):
        m = tl.maximum(m, tl.load(PartL + (base + sp) * H + h))
    m = tl.where(m == float("-inf"), 0.0, m)
    w_sum = tl.zeros([H], tl.float32)
    o = tl.zeros([H, D], tl.float32)
    for sp in range(nsp):
        row = (base + sp) * H + h
        w = tl.math.exp2(tl.load(PartL + row) - m)
        w_sum += w
        o += w[:, None] * tl.load(PartO + row[:, None] * D + d[None, :])
    o = o * (tl.load(VDesc) / tl.where(w_sum > 0, w_sum, 1.0))[:, None]
    tl.store(
        Out + t * o_s0 + h[:, None] * o_s1 + d[None, :], o.to(Out.dtype.element_ty)
    )


def _split_plan(
    cu_seqlens_q_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_kv_heads: int,
    window: int,
) -> tuple[list[int], int, int]:
    """Per-request rows (unit base, partial-slot base, q blocks, splits).

    A request with fewer than 4 workgroups per CU splits its unmasked context
    tiles into up to 32 parts of at least 64 tiles each. Windowed scans are short
    and never split. Returns the rows, the total units and the partial slots.
    """
    target = 4 * num_compute_units()
    cu = cu_seqlens_q_cpu.tolist()
    lens = seq_lens_cpu.tolist()
    rows: list[int] = []
    units = slots = 0
    for r, seq_len in enumerate(lens):
        q_len = cu[r + 1] - cu[r]
        q_blocks = cdiv(q_len, _BQ)
        nsp = 1
        if window == 0 and q_blocks * num_kv_heads < target:
            nsp = min(
                cdiv(target, q_blocks * num_kv_heads),
                _MAX_SPLITS,
                cdiv(seq_len, _TILE) // _MIN_TILES_PER_SPLIT,
            )
            nsp = max(nsp, 1)
        rows += [units, slots, q_blocks, nsp]
        units += q_blocks * nsp
        slots += q_len * nsp
    return rows, units, slots


def flydsl_fp8_paged_extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_q_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    window: int = 0,
) -> None:
    """Causal extend attention of new tokens over a paged fp8 e4m3 KV cache.

    Args:
        q: `[T, H, 64]` query of the extend requests, fp8 e4m3 or bf16.
        k_cache: `[num_blocks, 16, H // 8, 64]` fp8 e4m3 keys.
        v_cache: Values, same shape and strides as `k_cache`.
        out: `[T, H, 64]` bf16 output.
        cu_seqlens_q: `[R + 1]` query offsets of the requests, starting at 0.
        seq_lens: `[R]` context + new tokens per request.
        block_table: `[R, max_blocks]` page ids per request.
        max_query_len: Longest query among the requests.
        q_descale: `[T]` fp32 per-token descale of an fp8 `q`; None for a bf16
            `q`, which the kernel quantizes with one scale per token and KV head.
        k_descale: Per-tensor K descale.
        v_descale: Per-tensor V descale.
        softmax_scale: Scale of QK^T.
        cu_seqlens_q_cpu: CPU copy of `cu_seqlens_q`.
        seq_lens_cpu: CPU copy of `seq_lens`.
        window: Sliding-window size in tokens, 0 for full attention.

    """
    import flydsl.expr as fx

    from vllm.v1.attention.ops.flydsl_kernels.fp8_paged_extend import (
        LOG2E,
        build_launcher,
    )

    num_tokens, num_heads, head_size = q.shape
    num_kv_heads = k_cache.shape[2]
    nreq = seq_lens.numel()
    assert head_size == HEAD_SIZE and num_heads == GQA * num_kv_heads
    assert k_cache.shape[1] == PAGE_SIZE and nreq <= MAX_REQS
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(3) == 1
    assert k_cache.stride(1) % HEAD_SIZE == 0
    assert q.stride(1) == HEAD_SIZE and q.stride(2) == 1
    assert out.dtype == torch.bfloat16
    assert out.stride(1) == HEAD_SIZE and out.stride(2) == 1
    q_fused = q_descale is None
    if q_descale is None:
        assert q.dtype == torch.bfloat16
    else:
        assert q.dtype == torch.float8_e4m3fn
        assert q_descale.dtype == torch.float32 and q_descale.shape == (num_tokens,)

    rows, units, slots = _split_plan(
        cu_seqlens_q_cpu, seq_lens_cpu, num_kv_heads, window
    )
    split = max(rows[3::4], default=1) > 1
    k_descale = k_descale.reshape(1)
    v_descale = v_descale.reshape(1)
    cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    block_table = block_table.to(torch.int32)
    q_blocks = cdiv(max_query_len, _BQ)
    if split:
        plan = torch.tensor(rows, dtype=torch.int32, pin_memory=True).to(
            q.device, non_blocking=True
        )
        part_o = torch.empty(
            slots, num_heads, HEAD_SIZE, dtype=torch.float32, device=q.device
        )
        part_l = torch.empty(slots, num_heads, dtype=torch.float32, device=q.device)
        grid = units * num_kv_heads
    else:
        plan = cu_seqlens_q
        part_o = part_l = out
        grid = q_blocks * nreq * num_kv_heads

    launcher = build_launcher(
        num_kv_heads,
        k_cache.stride(0),
        k_cache.stride(2),
        k_cache.stride(1),
        float(softmax_scale) * LOG2E,
        split,
        window,
        q_fused,
    )
    _run_compiled(
        launcher,
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        out.data_ptr(),
        cu_seqlens_q,
        seq_lens,
        block_table,
        k_descale if q_fused else q_descale,
        k_descale,
        v_descale,
        part_o.data_ptr(),
        part_l.data_ptr(),
        plan,
        nreq,
        q_blocks,
        block_table.stride(0),
        q.stride(0),
        out.stride(0),
        grid,
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    if split:
        _merge_kernel[(int(cu_seqlens_q_cpu[-1]),)](
            out,
            part_o,
            part_l,
            plan,
            cu_seqlens_q,
            v_descale,
            out.stride(0),
            out.stride(1),
            nreq,
            H=num_heads,
            D=HEAD_SIZE,
            MAX_REQS=MAX_REQS,
            num_warps=4,
        )
