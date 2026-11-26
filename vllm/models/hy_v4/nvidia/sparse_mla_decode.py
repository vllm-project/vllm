# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K sparse MLA decode for HY V4 on SM90 with 8 or 16 local query heads.

FlashMLA's BF16 sparse kernel runs one CTA per query row. With the few rows of a
decode step, each CTA walks 512 or 2048 gathered KV rows serially, far from the
HBM floor. This kernel splits each row's top-k columns across CTAs (split
roles) and combines the partials (merge roles) in one launch.

The contract is `flash_mla_sparse_fwd`'s for `out=None`, without the LSE:

- `q`: `[T, h, 576]` BF16, unit last stride (a head slice of a wider buffer is
  fine).
- `kv`: `[rows, 1, 576]` BF16 flat KV rows; the value is `[..., :512]`.
- `indices`: `[T, 1, width]` int32. -1 and any index >= rows are unused.
- `attn_sink`: `[h]` FP32, added to the softmax denominator as `exp(sink)`.
- `topk_length`: `[T]` int32, the valid prefix of each index row.
- Returns `[T, h, 512]` BF16. Rows without a valid key are exactly 0, and so
  is a head whose sink is `+inf`.

`sparse_mla_decode` returns None for every call it does not take, and the
caller then runs FlashMLA: other shapes (it takes `(h, width)` = `(8, 2048)`
with T <= 40 and `(16, 512)` with T <= 16, the HY4-Flash DSA and SWA layers at
TP4, where it measured faster), other dtypes or layouts, a missing sink or
`topk_length`, and every call for a head count whose preparation failed (not
SM90, Triton < 3.3, a compile or self-test failure).

Every CTA first takes a ticket from a counter in the workspace. Tickets below
`T * splits` are split roles, the rest merge roles. A split writes its FP32
partial output and (max, sum), then increments one arrival counter per (row,
head, merge chunk) with release semantics. A merge waits (acquire) for
`splits` arrivals, re-arms its counter and combines the partials in a fixed
order, so the result is deterministic. A merge only waits on lower tickets,
held by CTAs that are already running, so the launch cannot deadlock under any
block scheduling. Counters only grow within a launch: a ticket or count past
its bound, or a wait of more than `_SPIN_LIMIT` polls (seconds), means two
launches share a workspace or something else wrote it, and the kernel executes
`trap` instead of hanging. A trap is a sticky CUDA error that ends the process
(fail-loud); `VLLM_DISABLE_HY4_SM90_SPLITK_MLA_DECODE=1` turns the path off.

The workspace (counters, then partials) is one zero-initialised FP32 buffer per
workspace-manager slot (ubatch, lane), reserved for every ubatch in the
memory-profiling run. Each launch returns its counters to 0, so CUDA graphs
that captured the buffer replay in any order; launches on one slot run in
stream order. Without a workspace (no manager, or a slot first used under
capture) a call runs the split and merge roles as two launches on scratch,
with no counters and no wait.

Tested with Triton 3.8.0 (the version vLLM's torch pin brings). The loop's
dots are mma.sync because the heads pad to M = 16; `prepare()` rejects a
variant that compiles to async wgmma, which the Hopper software pipeliner can
race (triton-lang/triton#6691).
"""

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from packaging.version import Version

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupChoices,
    WarmupIntRange,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_D_QK = 576
_D_V = 512
_DV = tl.constexpr(_D_V)
_DR = tl.constexpr(_D_QK - _D_V)
_LOG2E = math.log2(math.e)
_HEADS_PAD = 16  # MMA M
_MAX_SPLITS = 64
_SMAX = tl.constexpr(_MAX_SPLITS)
_NUM_WARPS = 4
_NUM_STAGES = 3
_NT = tl.constexpr(_NUM_WARPS * 32)
# int32 words at the head of the workspace: arrival counters, then the ticket
# on its own 128-byte line.
_CNT_ELEMS = 4096
_TICKET_SLOT = _CNT_ELEMS - 32
_CNT = tl.constexpr(_CNT_ELEMS)
_TK = tl.constexpr(_TICKET_SLOT)
_SPIN_LIMIT = tl.constexpr(1 << 22)
_MODE_SPLIT, _MODE_MERGE, _MODE_FUSED = 1, 2, 3
_MERGE_CHUNKS = 4  # two-launch merge CTAs per (row, head)
_WS_ROWS_PER_SM = 64  # T * splits * h partial rows at most

# heads -> (index width, largest T taken). Above these T, FlashMLA's one CTA
# per row is as fast (measured on H200).
_SHAPES = {8: (2048, 40), 16: (512, 16)}


@triton.jit
def _trap(x):
    tl.inline_asm_elementwise(
        "trap;", "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _lane0_add(ptr, val, SEM: tl.constexpr):
    """`atomic_add(*ptr, val)` by one thread; the old value is broadcast to the
    CTA."""
    lane = tl.arange(0, _NT)
    first = lane == 0
    old = tl.atomic_add(
        ptr + lane * 0,
        tl.zeros([_NT], tl.int32) + val,
        mask=first,
        sem=SEM,
        scope="gpu",
    )
    return tl.sum(tl.where(first, old, 0), 0)


@triton.jit
def _attend(q_nope, q_pe, kv_ptr, rows, valid, stride_kv, scale_log2, m_i, l_i, acc):
    dv = tl.arange(0, _DV)
    dr = tl.arange(0, _DR)
    kv_row = kv_ptr + rows.to(tl.int64) * stride_kv
    k_nope = tl.load(kv_row[:, None] + dv[None, :], mask=valid[:, None], other=0.0)
    k_pe = tl.load(kv_row[:, None] + _DV + dr[None, :], mask=valid[:, None], other=0.0)
    s = tl.dot(q_nope, tl.trans(k_nope))
    s = tl.dot(q_pe, tl.trans(k_pe), s)
    s = s * scale_log2
    s = tl.where(valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_safe)
    p = tl.exp2(s - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
    return m_new, l_i, acc


@triton.jit
def _split_role(
    pid,
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    ws_ptr,
    T,
    width,
    cols_per_split,
    num_splits,
    kv_rows,
    stride_qt,
    stride_qh,
    stride_kv,
    stride_it,
    scale_log2,
    H: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
    NCH: tl.constexpr,
):
    t = pid // num_splits
    sid = pid - t * num_splits
    t64 = t.to(tl.int64)
    heads = tl.arange(0, HP)
    hmask = heads < H
    dv = tl.arange(0, _DV)
    dr = tl.arange(0, _DR)

    start = sid * cols_per_split
    end = tl.minimum(tl.minimum(start + cols_per_split, width), tl.load(len_ptr + t))

    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, _DV], tl.float32)
    if start < end:
        q_row = q_ptr + t64 * stride_qt + heads[:, None] * stride_qh
        q_nope = tl.load(q_row + dv[None, :], mask=hmask[:, None], other=0.0)
        q_pe = tl.load(q_row + _DV + dr[None, :], mask=hmask[:, None], other=0.0)
        idx_row = idx_ptr + t64 * stride_it
        for c0 in range(start, end, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            rows = tl.load(idx_row + cols, mask=cols < end, other=-1)
            valid = (rows >= 0) & (rows < kv_rows)
            m_i, l_i, acc = _attend(
                q_nope, q_pe, kv_ptr, rows, valid, stride_kv, scale_log2, m_i, l_i, acc
            )

    part_ptr = ws_ptr + _CNT
    ml_ptr = part_ptr + T.to(tl.int64) * num_splits * (H * _DV)
    r = (t64 * num_splits + sid) * H + heads
    tl.store(ml_ptr + r * 2, m_i, mask=hmask)
    tl.store(ml_ptr + r * 2 + 1, l_i, mask=hmask)
    has = m_i > float("-inf")
    tl.store(
        part_ptr + r[:, None] * _DV + dv[None, :], acc, mask=(hmask & has)[:, None]
    )
    if MODE == 3:
        # The barrier orders every thread's stores before the release
        # increments; lane e < H * NCH bumps counter (t, e // NCH, e % NCH).
        tl.debug_barrier()
        lane = tl.arange(0, _NT)
        c_ptr = ws_ptr.to(tl.pointer_type(tl.int32)) + t * (H * NCH)
        tl.atomic_add(
            c_ptr + lane,
            tl.full([_NT], 1, tl.int32),
            mask=lane < H * NCH,
            sem="release",
            scope="gpu",
        )


@triton.jit
def _merge_role(
    j,
    sink_ptr,
    out_ptr,
    ws_ptr,
    T,
    num_splits,
    H: tl.constexpr,
    WAIT: tl.constexpr,
    NCH: tl.constexpr,
):
    """Merge role `j = (t * H + head) * NCH + chunk`: `512 / NCH` output dims of
    one (row, head)."""
    DIMS: tl.constexpr = _DV // NCH
    SCH: tl.constexpr = 16 if NCH > 1 else 8
    th = j // NCH
    ch = j - th * NCH
    t = th // H
    h = th - t * H
    t64 = t.to(tl.int64)
    if WAIT:
        cp = ws_ptr.to(tl.pointer_type(tl.int32)) + j
        c = _lane0_add(cp, 0, "acquire")
        spins = 0
        while c < num_splits:
            spins += 1
            if spins > _SPIN_LIMIT:
                _trap(spins)
            c = _lane0_add(cp, 0, "acquire")
        if c > num_splits:
            _trap(c)
        lane = tl.arange(0, _NT)
        tl.atomic_xchg(
            cp + lane * 0,
            tl.zeros([_NT], tl.int32),
            mask=lane == 0,
            sem="relaxed",
            scope="gpu",
        )
        tl.debug_barrier()
    part_ptr = ws_ptr + _CNT
    ml_ptr = part_ptr + T.to(tl.int64) * num_splits * (H * _DV)
    r0 = t64 * num_splits * H + h
    sp = tl.arange(0, _SMAX)
    spm = sp < num_splits
    m = tl.load(
        ml_ptr + (r0 + sp * H) * 2,
        mask=spm,
        other=float("-inf"),
        cache_modifier=".cg",
    )
    l = tl.load(  # noqa: E741
        ml_ptr + (r0 + sp * H) * 2 + 1, mask=spm, other=0.0, cache_modifier=".cg"
    )
    m_max = tl.max(m, 0)
    sink = tl.load(sink_ptr + h) * 1.4426950408889634
    m_fin = tl.where(m_max == float("-inf"), 0.0, tl.maximum(m_max, sink))
    den = tl.sum(l * tl.exp2(m - m_fin), 0) + tl.exp2(sink - m_fin)
    d = ch * DIMS + tl.arange(0, DIMS)
    o = tl.zeros([DIMS], tl.float32)
    for s0 in range(0, num_splits, SCH):
        s = s0 + tl.arange(0, SCH)
        sm = s < num_splits
        ms = tl.load(
            ml_ptr + (r0 + s * H) * 2,
            mask=sm,
            other=float("-inf"),
            cache_modifier=".cg",
        )
        w = tl.exp2(ms - m_fin)
        ok = sm & (ms > float("-inf"))
        x = tl.load(
            part_ptr + (r0 + s * H)[:, None] * _DV + d[None, :],
            mask=ok[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        o += tl.sum(x * w[:, None], 0)
    o = o / den
    o = tl.where((m_max == float("-inf")) | (sink == float("inf")), 0.0, o)
    tl.store(out_ptr + (t64 * H + h) * _DV + d, o.to(tl.bfloat16))


# len_ptr and sink_ptr feed scalar loads: no alignment specialization, so row
# slices of larger tensors take the warmed binaries.
@triton.jit(
    do_not_specialize=["T", "width", "cols_per_split", "num_splits", "kv_rows"],
    do_not_specialize_on_alignment=["len_ptr", "sink_ptr"],
)
def _decode_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    sink_ptr,
    out_ptr,
    ws_ptr,
    T,
    width,
    cols_per_split,
    num_splits,
    kv_rows,
    stride_qt,
    stride_qh,
    stride_kv,
    stride_it,
    scale_log2,
    H: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
    NCH: tl.constexpr,
):
    role = tl.program_id(0)
    if MODE == 2:
        _merge_role(role, sink_ptr, out_ptr, ws_ptr, T, num_splits, H, False, NCH)
    elif MODE == 1:
        _split_role(
            role,
            q_ptr,
            kv_ptr,
            idx_ptr,
            len_ptr,
            ws_ptr,
            T,
            width,
            cols_per_split,
            num_splits,
            kv_rows,
            stride_qt,
            stride_qh,
            stride_kv,
            stride_it,
            scale_log2,
            H,
            HP,
            BLOCK_N,
            MODE,
            NCH,
        )
    else:
        n_split = T * num_splits
        n_cta = n_split + T * (H * NCH)
        tk_ptr = ws_ptr.to(tl.pointer_type(tl.int32)) + _TK
        ticket = _lane0_add(tk_ptr, 1, "relaxed")
        if ticket >= n_cta:
            _trap(ticket)
        if ticket == n_cta - 1:
            # Every CTA has its role: return the ticket to 0.
            lane = tl.arange(0, _NT)
            tl.atomic_add(
                tk_ptr + lane * 0,
                tl.zeros([_NT], tl.int32) - n_cta,
                mask=lane == 0,
                sem="relaxed",
                scope="gpu",
            )
        if ticket < n_split:
            _split_role(
                ticket,
                q_ptr,
                kv_ptr,
                idx_ptr,
                len_ptr,
                ws_ptr,
                T,
                width,
                cols_per_split,
                num_splits,
                kv_rows,
                stride_qt,
                stride_qh,
                stride_kv,
                stride_it,
                scale_log2,
                H,
                HP,
                BLOCK_N,
                MODE,
                NCH,
            )
        else:
            _merge_role(
                ticket - n_split, sink_ptr, out_ptr, ws_ptr, T, num_splits, H, True, NCH
            )


def _plan(T: int, h: int, width: int, sms: int) -> tuple[int, int]:
    """`(splits, cols_per_split)` from shapes only, so a captured graph is valid
    for any inputs of its shapes. Few rows: one block of columns per split;
    more rows: about 4 (wide rows) or 2 CTAs per SM."""
    bn = 32 if T <= 8 else 64
    target = (4 if width >= 1024 else 2) * sms
    max_splits = max(
        1,
        min(_MAX_SPLITS, triton.cdiv(width, bn), _WS_ROWS_PER_SM * sms // (T * h)),
    )
    splits = min(max_splits, max(1, target // T))
    cols = triton.cdiv(triton.cdiv(width, splits), bn) * bn
    return triton.cdiv(width, cols), cols


def _grid(T: int, h: int, splits: int, key: "SparseMLADecodeKernel.CompileKey"):
    if key.mode == _MODE_SPLIT:
        return (T * splits, 1, 1)
    if key.mode == _MODE_MERGE:
        return (T * h * key.merge_chunks, 1, 1)
    return (T * splits + T * h * key.merge_chunks, 1, 1)


def _workspace_numel(rows: int) -> int:
    return _CNT_ELEMS + rows * (_D_V + 2)


class _NoWorkspace(RuntimeError):
    pass


def _new_workspace() -> torch.Tensor:
    # A zero fill recorded only into one CUDA graph would not hold for eager
    # calls.
    if torch.cuda.is_current_stream_capturing():
        raise _NoWorkspace
    device = torch.accelerator.current_device_index()
    sms = current_platform.num_compute_units(device)
    return torch.zeros(
        _workspace_numel(_WS_ROWS_PER_SM * sms),
        dtype=torch.float32,
        device=torch.device(current_platform.device_type, device),
    )


_WORKSPACE_KEY = ("hy_v4_sparse_mla_decode", "workspace")


def _workspace() -> torch.Tensor | None:
    try:
        return current_workspace_manager().get_persistent_resource(
            _WORKSPACE_KEY, _new_workspace
        )
    except (AssertionError, _NoWorkspace):
        return None


def reserve_workspace() -> torch.Tensor | None:
    """Allocate the zero-initialised workspace in every ubatch slot of the
    current lane (call it from the memory-profiling run) and return the current
    slot's; None without a workspace manager, under capture or once locked."""
    try:
        return current_workspace_manager().reserve_persistent_resource_all_ubatches(
            _WORKSPACE_KEY, _new_workspace
        )
    except (AssertionError, _NoWorkspace):
        return None


def splitk_decode_supported(num_heads: int) -> bool:
    """Whether layers with `num_heads` local heads may use the split-K decode on
    this device (the per-call shape and layout checks come on top)."""
    return (
        HAS_TRITON
        and num_heads in _SHAPES
        and current_platform.is_device_capability_family(90)
        and not envs.VLLM_DISABLE_HY4_SM90_SPLITK_MLA_DECODE
        and not envs.VLLM_BATCH_INVARIANT
        and Version(triton.__version__).release[:2] >= (3, 3)
    )


class SparseMLADecodeKernel(VllmJitKernel["SparseMLADecodeKernel.CompileKey"]):
    """Compiled variants of `_decode_kernel` and the launch plan per row count.

    The launch uses the `CompiledKernel` that `JITFunction.warmup` returns
    (`compiled[grid](*args)`, as `CPTritonContext` does), which skips Triton's
    per-call dispatch. That calling convention is not a stable Triton API, so
    the first preparation for a head count checks it bit for bit against the
    JIT entry point (and against FP32 math) and disables the path on a
    mismatch or on any error.
    """

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        heads: int
        block_n: int
        mode: int
        merge_chunks: int

    def __init__(self) -> None:
        super().__init__()
        # (device, T, heads) -> (scalars, fused launch, two-launch launches).
        self._calls: dict[tuple[int, int, int], tuple] = {}
        # (device, heads) -> whether the split path is on.
        self._ready: dict[tuple[int, int], bool] = {}

    def dispatch(  # type: ignore[override]
        self, *, heads: int, num_tokens: int, mode: int
    ) -> CompileKey:
        rows = num_tokens * heads
        return self.CompileKey(
            heads=heads,
            block_n=32 if num_tokens <= 8 else 64,
            mode=mode,
            merge_chunks=(
                (4 if rows <= 64 else 2 if rows <= 128 else 1)
                if mode == _MODE_FUSED
                else _MERGE_CHUNKS
                if mode == _MODE_MERGE
                else 1
            ),
        )

    def get_warmup_keys(self, *, heads: int) -> list[CompileKey]:
        if heads not in _SHAPES:
            return []
        return self._trace_dispatch(self.dispatch)(
            heads=heads,
            num_tokens=WarmupIntRange(1, _SHAPES[heads][1] + 1),
            mode=WarmupChoices(_MODE_FUSED, _MODE_SPLIT, _MODE_MERGE),
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        width = _SHAPES[compile_key.heads][0]
        stride = triton_scalar_specialization_rep(16)
        self._compiled_cache[compile_key] = _decode_kernel.warmup(
            TritonWarmupTensor(torch.bfloat16),
            TritonWarmupTensor(torch.bfloat16),
            TritonWarmupTensor(torch.int32),
            TritonWarmupTensor(torch.int32),
            TritonWarmupTensor(torch.float32),
            TritonWarmupTensor(torch.bfloat16),
            TritonWarmupTensor(torch.float32),
            1,  # T, width, cols_per_split, num_splits, kv_rows: unspecialized
            width,
            width,
            1,
            1,
            stride,
            stride,
            stride,
            stride,
            1.0,
            H=compile_key.heads,
            HP=_HEADS_PAD,
            BLOCK_N=compile_key.block_n,
            MODE=compile_key.mode,
            NCH=compile_key.merge_chunks,
            grid=(1,),
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

    def compile_many(self, compile_keys: Iterable[CompileKey]) -> None:
        for heads in sorted({key.heads for key in compile_keys}):
            self.prepare(heads)

    def prepare(self, heads: int) -> bool:
        """Compile, plan and self-test the split path for `heads` on the current
        device, once. Never raises; False routes every such call to FlashMLA."""
        device = torch.accelerator.current_device_index()
        if (device, heads) in self._ready:
            return self._ready[(device, heads)]
        reason = None
        if not splitk_decode_supported(heads):
            reason = "unsupported device, Triton version or head count"
        else:
            start = time.perf_counter()
            try:
                self._build(device, heads)
                self._self_test(device, heads)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
        if reason is None:
            logger.info(
                "HY V4 SM90 split-K sparse MLA decode for %d heads ready in %.1f s "
                "(compile, plan and self-test).",
                heads,
                time.perf_counter() - start,
            )
        else:
            for key in [k for k in self._calls if k[0] == device and k[2] == heads]:
                del self._calls[key]
            logger.warning(
                "HY V4 SM90 split-K sparse MLA decode disabled for %d heads, "
                "using FlashMLA: %s",
                heads,
                reason,
            )
        self._ready[(device, heads)] = reason is None
        return reason is None

    def _build(self, device: int, heads: int) -> None:
        width, max_tokens = _SHAPES[heads]
        sms = current_platform.num_compute_units(device)
        for T in range(1, max_tokens + 1):
            splits, cols = _plan(T, heads, width, sms)
            launches = []
            for modes in ((_MODE_FUSED,), (_MODE_SPLIT, _MODE_MERGE)):
                runs = []
                for mode in modes:
                    key = self.dispatch(heads=heads, num_tokens=T, mode=mode)
                    compiled = self._get_or_compile(key)
                    # The heads pad to M = 16, so the loop's dots are synchronous
                    # mma.sync. With async wgmma it would be exposed to the
                    # software pipeliner's WAR race (triton-lang/triton#6691).
                    if "wgmma" in compiled.asm["ptx"]:
                        raise RuntimeError(f"{key} compiled to async wgmma")
                    grid = _grid(T, heads, splits, key)
                    consts = (heads, _HEADS_PAD, key.block_n, mode, key.merge_chunks)
                    runs.append((compiled[grid], consts, grid))
                launches.append(tuple(runs))
            scalars = (T, width, cols, splits)
            self._calls[(device, T, heads)] = (scalars, *launches, T * splits * heads)

    def _self_test(self, device: int, heads: int) -> None:
        width, max_tokens = _SHAPES[heads]
        gen = torch.Generator(device=f"cuda:{device}").manual_seed(0)
        for T in (min(4, max_tokens), max_tokens):
            q, kv, idx, length, sink = _synthetic(T, heads, width, gen)
            scalars, fused, two, rows = self._calls[(device, T, heads)]
            ref = _reference(q, kv, idx, length, sink, 0.0625)
            for runs in (fused, two):
                outs = []
                for use_compiled in (True, False):
                    ws = torch.zeros(
                        _workspace_numel(rows), dtype=torch.float32, device=q.device
                    )
                    out = torch.empty((T, heads, _D_V), dtype=q.dtype, device=q.device)
                    args = (q, kv, idx, length, sink, out, ws, *scalars, kv.shape[0])
                    args += (q.stride(0), q.stride(1), kv.stride(0), idx.stride(0))
                    args += (0.0625 * _LOG2E,)
                    for run, consts, grid in runs:
                        if use_compiled:
                            run(*args, *consts, stream=_current_stream())
                        else:
                            _decode_kernel[grid](
                                *args,
                                *consts,
                                num_warps=_NUM_WARPS,
                                num_stages=_NUM_STAGES,
                            )
                    outs.append(out)
                if not torch.equal(
                    outs[0].view(torch.int16), outs[1].view(torch.int16)
                ):
                    raise RuntimeError(f"compiled launch != JIT launch at T={T}")
                err = (outs[0].float() - ref).abs().max().item()
                if not err <= 0.03:
                    raise RuntimeError(f"max abs error {err} at T={T}")

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        attn_sink: torch.Tensor | None,
        topk_length: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Sparse MLA with sinks, or None when the caller should run FlashMLA.

        See the module docstring for the shapes and the calls taken.
        """
        T, h, d = q.shape
        device = q.get_device()
        call = self._calls.get((device, T, h))
        if call is None:
            # Without the JIT warmup, prepare on the first eager call.
            if (
                (device, h) in self._ready
                or h not in _SHAPES
                or not 0 < T <= _SHAPES[h][1]
                or torch.cuda.is_current_stream_capturing()
                or not self.prepare(h)
            ):
                return None
            call = self._calls[(device, T, h)]
        if attn_sink is None or topk_length is None:
            return None
        # The compiled variants assume these dtypes, unit last strides,
        # 16-element row and head strides and 16-byte aligned q, kv, indices.
        sq0, sq1, sq2 = q.stride()
        if not (
            q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and indices.dtype == torch.int32
            and attn_sink.dtype == torch.float32
            and topk_length.dtype == torch.int32
            and d == _D_QK
            and kv.dim() == 3
            and kv.shape[1] == 1
            and kv.shape[2] == _D_QK
            and indices.shape == (T, 1, _SHAPES[h][0])
            and topk_length.shape == (T,)
            and attn_sink.shape == (h,)
            and sq2 == 1
            and kv.stride(2) == 1
            and indices.stride(2) == 1
            and topk_length.stride(0) == 1
            and attn_sink.stride(0) == 1
            and (sq0 | sq1 | kv.stride(0) | indices.stride(0)) % 16 == 0
            and (q.data_ptr() | kv.data_ptr() | indices.data_ptr()) % 16 == 0
        ):
            return None
        scalars, runs, two_launch, rows = call
        ws = _workspace()
        if ws is None:
            # No workspace for this slot (none reserved, first use under
            # capture): split and merge launches on scratch, which needs no zeros.
            ws = q.new_empty((_workspace_numel(rows),), dtype=torch.float32)
            runs = two_launch
        out = q.new_empty((T, h, _D_V))
        args = (q, kv, indices, topk_length, attn_sink, out, ws, *scalars)
        args += (kv.shape[0], sq0, sq1, kv.stride(0), indices.stride(0))
        args += (sm_scale * _LOG2E,)
        stream = _current_stream()
        for run, consts, _ in runs:
            run(*args, *consts, stream=stream)
        return out


def _current_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


# _synthetic and _reference feed the self-test in `prepare()`: they guard
# against a miscompile or a changed launch convention at startup.
def _synthetic(T: int, h: int, width: int, gen: torch.Generator, rows: int = 4096):
    """Aligned inputs of one call: `q` is a head slice of a 64-head buffer, with
    holes, an empty row and cut index rows."""
    dev = gen.device
    q = (torch.randn((T, 64, _D_QK), generator=gen, device=dev) * 0.5).bfloat16()
    kv = torch.randn((rows, 1, _D_QK), generator=gen, device=dev).bfloat16()
    idx = torch.randint(
        0, rows, (T, 1, width), generator=gen, device=dev, dtype=torch.int32
    )
    idx[:, :, ::7] = -1
    length = torch.randint(
        1, width + 1, (T,), generator=gen, device=dev, dtype=torch.int32
    )
    if T > 2:
        length[1] = 0
    sink = torch.randn((h,), generator=gen, device=dev, dtype=torch.float32)
    return q[:, :h], kv, idx, length, sink


def _reference(q, kv, idx, length, sink, sm_scale: float) -> torch.Tensor:
    """FP32 sparse MLA with sinks, one row at a time."""
    T, h, _ = q.shape
    out = torch.zeros((T, h, _D_V), dtype=torch.float32, device=q.device)
    kv2 = kv.view(-1, _D_QK).float()
    for t in range(T):
        r = idx[t, 0, : int(length[t])].long()
        r = r[(r >= 0) & (r < kv2.shape[0])]
        if r.numel() == 0:
            continue
        k = kv2[r]
        s = (q[t].float() @ k.T) * sm_scale
        m = torch.maximum(s.amax(-1), sink)
        p = torch.exp(s - m[:, None])
        out[t] = (p @ k[:, :_D_V]) / (p.sum(-1) + torch.exp(sink - m))[:, None]
    return out


sparse_mla_decode = SparseMLADecodeKernel()
