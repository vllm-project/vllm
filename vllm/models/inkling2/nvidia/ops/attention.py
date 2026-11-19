# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention kernels: FA4 relative attention, gate/sink, log-scaling."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    _num_warps_bucket,
    bucket_max_seqlen_q,
)
from vllm.triton_utils import gl, gluon, tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

# Full-layer FA4 split estimate, fitted on GB200 over uniform and mixed
# prefill batches (re-fit for other devices): one CTA per 256-row query tile
# and head runs at ~10 TFLOPS per SM, and the split merge moves fp32 partials
# at ~4 TB/s.
_FA4_TILE_M = 256
_FA4_SM_FLOPS_PER_US = 10e6
_FA4_MERGE_BYTES_PER_US = 4e6


def plan_attention(
    is_local: bool,
    query_lens: np.ndarray,
    seq_lens: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[int, bool]:
    """Return `(num_splits, use_scheduler)` for one FA4 launch over requests
    of `query_lens` new tokens and `seq_lens` keys.

    Window layers never split KV. FA4's split scheduler only pays off for the
    smallest launches. Larger ones split the keys of every query tile `s`
    ways, minimizing the estimated time: the larger of the wave-quantized
    time of the longest CTAs and the total CTA time spread over the SMs plus
    half the longest CTA, plus the merge of `s` fp32 partials (within
    256 MiB). Split counts within 2% of the best estimate resolve to the
    smallest.
    """
    if is_local:
        return 1, True
    num_tokens = int(query_lens.sum())
    if num_tokens * num_kv_heads < 512:
        return max(1, min(128, 8192 // num_tokens)), True
    num_sms = num_compute_units()
    order = np.argsort(-seq_lens)
    ctas = (-(-query_lens[order] // _FA4_TILE_M) * num_heads).astype(np.float64)
    cta_us = seq_lens[order] * (_FA4_TILE_M * 512 / _FA4_SM_FLOPS_PER_US)
    partial_bytes = num_tokens * num_heads * 512
    max_splits = max(1, min(32, 2**28 // partial_bytes))
    splits = np.arange(1, max_splits + 1, dtype=np.float64)[:, None]
    waves = np.ceil(np.cumsum(ctas * splits, axis=1) / num_sms)
    span = np.maximum(
        (waves * cta_us / splits).max(axis=1),
        (ctas * cta_us).sum() / num_sms + cta_us[0] / (2 * splits[:, 0]),
    )
    cost = span + np.where(
        splits[:, 0] > 1, splits[:, 0] * partial_bytes / _FA4_MERGE_BYTES_PER_US, 0.0
    )
    return int(np.argmax(cost <= 1.02 * cost.min())) + 1, False


class Inkling2FA4RelAttentionKernel(
    VllmJitKernel["Inkling2FA4RelAttentionKernel.CompileKey"]
):
    """tml_fa4 paged relative attention with fp32 output and the natural-log LSE.

    The relative bias is added before scaling: `(qk + bias / scale) * scale`.
    It comes sheared (`(T + 128, H, rel_extent + 256)`, as written by
    `attention_prologue`), which skips tml_fa4's shearing prepass.
    """

    @dataclass(frozen=True)
    class CompileKey:
        is_local: bool
        num_heads: int
        num_kv_heads: int
        head_dim: int
        rel_extent: int
        dtype: torch.dtype
        kv_dtype: torch.dtype
        block_size: int
        max_seqlen_q: int
        split_kv: bool
        # Split-KV scheduler warps bucket; None without the scheduler.
        num_warps_bucket: int | None
        large_num_reqs: bool

    @staticmethod
    def kernel(
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        window_size: tuple[int, int],
        sheared_bias: torch.Tensor,
        num_splits: int,
        use_scheduler: bool,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        """Write `out` and the `(num_heads, num_tokens)` LSE `lse`."""
        from vllm.third_party.tml_fa4.interface import _flash_attn_fwd

        # The varlen wrapper does not take a preallocated LSE.
        left, right = (None, None) if window_size == (-1, -1) else window_size
        _flash_attn_fwd(
            q,
            key_cache,
            value_cache,
            bias=sheared_bias,
            bias_ex2_emu=True,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            page_table=block_table,
            softmax_scale=softmax_scale,
            causal=True,
            window_size_left=left,
            window_size_right=right,
            pack_gqa=False,
            num_splits=num_splits,
            disable_scheduler_metadata=not use_scheduler,
            return_lse=True,
            out=out,
            lse=lse,
        )

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        config = vllm_config.model_config.hf_config.get_text_config()
        tp_size = get_tensor_model_parallel_world_size()
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        kv_dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        )
        assert config.swa_num_attention_heads == config.num_attention_heads
        assert config.swa_num_key_value_heads == config.num_key_value_heads
        assert config.swa_head_dim == config.head_dim
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = config.num_key_value_heads // tp_size
        warps_buckets = sorted(
            {_num_warps_bucket(n) for n in range(1, max_num_reqs + 1)}
        )
        keys = []
        for is_local in (False, True):
            max_seqlen_q = 1
            while max_seqlen_q <= bucket_max_seqlen_q(max_tokens):
                variants: list[tuple[bool, int | None]] = [(False, None)]
                if not is_local:
                    variants += [(True, bucket) for bucket in warps_buckets]
                    variants.append((True, None))
                for split_kv, bucket in variants:
                    keys.append(
                        self.CompileKey(
                            is_local=is_local,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            head_dim=config.head_dim,
                            rel_extent=(
                                config.sliding_window_size
                                if is_local
                                else config.rel_extent
                            ),
                            dtype=dtype,
                            kv_dtype=kv_dtype,
                            block_size=vllm_config.cache_config.block_size,
                            max_seqlen_q=max_seqlen_q,
                            split_kv=split_kv,
                            num_warps_bucket=bucket,
                            large_num_reqs=max_num_reqs > 1024,
                        )
                    )
                max_seqlen_q *= 2
        return keys

    def compile(self, compile_key: CompileKey) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        key = compile_key
        num_reqs = 1025 if key.large_num_reqs else 1
        if key.num_warps_bucket is not None and key.num_warps_bucket > 1:
            num_reqs = max(num_reqs, (key.num_warps_bucket // 2) * 31 + 1)
        num_heads = key.num_heads
        window_size = (key.rel_extent - 1, 0) if key.is_local else (-1, -1)
        with FakeTensorMode():
            device = torch.accelerator.current_accelerator()
            total_q = key.max_seqlen_q + num_reqs - 1
            q = torch.empty(
                total_q, num_heads, key.head_dim, dtype=key.dtype, device=device
            )
            kv = torch.empty(
                1,
                2,
                key.block_size,
                key.num_kv_heads,
                key.head_dim,
                dtype=key.kv_dtype,
                device=device,
            )
            key_cache, value_cache = kv.unbind(1)
            int_tensor = partial(torch.empty, dtype=torch.int32, device=device)
            self.kernel(
                q,
                key_cache,
                value_cache,
                block_table=int_tensor(num_reqs, 1),
                cache_seqlens=int_tensor(num_reqs),
                cu_seqlens_q=int_tensor(num_reqs + 1),
                max_seqlen_q=key.max_seqlen_q,
                softmax_scale=key.head_dim**-0.5,
                window_size=window_size,
                sheared_bias=torch.empty(
                    total_q + 128,
                    num_heads,
                    key.rel_extent + 256,
                    dtype=key.dtype,
                    device=device,
                ),
                num_splits=2 if key.split_kv else 1,
                use_scheduler=key.num_warps_bucket is not None,
                out=torch.empty_like(q, dtype=torch.float32),
                lse=torch.empty(num_heads, total_q, dtype=torch.float32, device=device),
            )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.kernel(*args, **kwargs)


_INKLING2_FA4_REL_ATTENTION_KERNEL = Inkling2FA4RelAttentionKernel()


@triton.jit
def _sigmoid(x):
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[-x * 1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[exp + 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0).to(tl.float32)
    result = x.to(tl.float32) * _sigmoid(gate)
    result = result * _sigmoid(lse.to(tl.float32) - sink)
    tl.store(Y + row * H * D + col, result, mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Write bf16 `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result`.

    `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are
    `(T, H)` and `output` is `(T, H, D)`.
    """
    num_tokens, num_heads = sinks.shape
    assert lse.shape == (num_tokens, num_heads)
    assert output.shape[1] == num_heads and output.is_contiguous()
    assert gate.stride(1) == sinks.stride(1) == 1
    assert result.shape == gate.shape and result.is_contiguous()
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
        launch_pdl=True,
    )


@triton.jit
def _log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
        positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
    )
    return output


# FA4's key block; the ex2 emulation pattern repeats every block.
_FA4_BLOCK_N = 128
# Query rows (tokens x GQA group) of a CTA.
_VERIFY_BLOCK_M = 64
_VERIFY_STAGES = 3


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
) -> int:
    """Return the split count of `rel_decode_attention` (or
    `block_decode_attention`), or 0 to use FA4.

    Each CTA holds all queries of one request for one KV head. Local layers
    (`rel_extent`: the window) give each split one key block of the
    window, or a few blocks for larger batches (a single split writes the
    gated output without the combine kernel); decode keeps one block per
    split up to twice the work of verify. Full layers give each request and
    KV head `num_sms / 8` splits, fewer once that exceeds 6 CTAs per SM,
    but never less than one CTA per SM. The count depends only on the batch,
    not the KV lengths, so it holds for every replay of a CUDA graph.
    """
    if max_query_len > _VERIFY_BLOCK_M // (num_heads // num_kv_heads):
        return 0
    work = num_reqs * num_kv_heads
    num_sms = num_compute_units()
    if is_local:
        if work <= (16 if max_query_len <= 2 else 8):
            return triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1
        return 5 if work <= 16 else 3 if work <= 32 else 2 if work <= 64 else 1
    return max(num_sms // work, min(num_sms // 8, 6 * num_sms // work))


_EX2_ASM = tl.constexpr("ex2.approx.ftz.f32 $0, $1;")
# tml_fa4 utils.ex2_emulation_2: degree-3 polynomial on the fraction.
_EX2_EMU_ASM = tl.constexpr("""{
    .reg .f32 c, r, rb, f, p;
    .reg .b32 e;
    max.ftz.f32 c, $1, 0fC2FE0000;
    add.rm.ftz.f32 r, c, 0f4B400000;
    sub.rn.ftz.f32 rb, r, 0f4B400000;
    sub.rn.ftz.f32 f, c, rb;
    fma.rn.ftz.f32 p, f, 0f3D9DF09D, 0f3E6906A4;
    fma.rn.ftz.f32 p, p, f, 0f3F31F519;
    fma.rn.ftz.f32 p, p, f, 0f3F800000;
    mov.b32 e, r;
    shl.b32 e, e, 23;
    mov.b32 $0, p;
    add.s32 $0, $0, e;
    }""")
_RCP_ASM = tl.constexpr("rcp.approx.ftz.f32 $0, $1;")
_LG2_ASM = tl.constexpr("lg2.approx.ftz.f32 $0, $1;")


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise(
        _EX2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _lg2(x):
    return tl.inline_asm_elementwise(
        _LG2_ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rel_decode_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    num_tokens,
    num_actual_tokens,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GATED: tl.constexpr,
):
    """Merge the split partials, then apply the output gate and attention sink
    if `GATED`."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    if GATED:
        # The split kernel launches this one after its own PDL wait, so the
        # gate and sink are ready.
        gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
        sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    y_ptrs = Y + row * H * D + head * D + offs_d
    if row >= num_actual_tokens:
        tl.store(y_ptrs, tl.zeros([BLOCK_D], tl.float32))
        return
    splits = tl.arange(0, BLOCK_S)
    part = (splits * num_tokens + row) * H + head
    lse_s = tl.load(LsePart + part, mask=splits < NUM_SPLITS, other=float("-inf"))
    lse_max = tl.max(lse_s)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    w = _ex2(lse_s - lse_max)
    w_sum = tl.sum(w)
    o_s = tl.load(
        OPart + part[:, None] * D + offs_d[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    # Requests without keys (padding) come out zero.
    o = tl.sum(o_s * (w / tl.where(w_sum == 0.0, 1.0, w_sum))[:, None], 0)
    if GATED:
        lse = (_lg2(w_sum) + lse_max) * 0.6931471805599453
        o = o * _sigmoid(gate)
        o = o * _sigmoid(lse - sink)
    tl.store(y_ptrs, o)


_bw = gl.nvidia.blackwell


@gluon.jit
def _gl_ex2(x):
    return gl.inline_asm_elementwise(
        _EX2_ASM, "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _gl_ex2_emu(x):
    return gl.inline_asm_elementwise(
        _EX2_EMU_ASM, "=r,f", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _gl_sigmoid(x):
    """`_sigmoid` (non-ftz ex2/rcp approximations)."""
    e = gl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        "=f,f",
        [-x * 1.4426950408889634],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return gl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        "=f,f",
        [e + 1.0],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _gl_rcp(x):
    return gl.inline_asm_elementwise(
        _RCP_ASM, "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _gl_lg2(x):
    return gl.inline_asm_elementwise(
        _LG2_ASM, "=f,f", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _verify_load_block(
    kv_desc,
    k_smem,
    v_smem,
    kv_ready,
    BlockTable,
    req,
    kv_head,
    stride_bt,
    n_first,
    num_splits,
    i,
    page_rows,
    head_rows,
    head_cols,
    PAGE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D: gl.constexpr,
    STAGES: gl.constexpr,
):
    """TMA-load K and V of the split's block `i`, one page-sized slab at a time."""
    SUB: gl.constexpr = min(PAGE, BLOCK_N)
    block_table = BlockTable + req * stride_bt
    key = (n_first + i * num_splits) * BLOCK_N
    stage = i % STAGES
    _bw.mbarrier.expect(kv_ready.index(stage), 2 * BLOCK_N * D * 2)
    for t in gl.static_range(BLOCK_N // SUB):
        page = gl.load(block_table + (key + t * SUB) // PAGE)
        row = page * page_rows + kv_head * head_rows + (key + t * SUB) % PAGE
        k_dst = k_smem.index(stage).slice(t * SUB, SUB)
        v_dst = v_smem.index(stage).slice(t * SUB, SUB)
        col = kv_head * head_cols
        _bw.tma.async_load(kv_desc, [row, col], kv_ready.index(stage), k_dst)
        _bw.tma.async_load(kv_desc, [row, col + D], kv_ready.index(stage), v_dst)


@gluon.jit
def _verify_load(
    kv_desc,
    k_smem,
    v_smem,
    kv_ready,
    kv_empty,
    BlockTable,
    req,
    kv_head,
    stride_bt,
    n_first,
    num_splits,
    num_pre,
    num_iters,
    page_rows,
    head_rows,
    head_cols,
    PAGE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D: gl.constexpr,
    STAGES: gl.constexpr,
):
    """Stream the split's K/V blocks after the first `num_pre`."""
    for i in range(num_pre, num_iters):
        _bw.mbarrier.wait(kv_empty.index(i % STAGES), (i // STAGES & 1) ^ 1)
        _verify_load_block(
            kv_desc,
            k_smem,
            v_smem,
            kv_ready,
            BlockTable,
            req,
            kv_head,
            stride_bt,
            n_first,
            num_splits,
            i,
            page_rows,
            head_rows,
            head_cols,
            PAGE,
            BLOCK_N,
            D,
            STAGES,
        )


@gluon.jit
def _verify_p(s_tmem, b, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    """Alias bf16 P over the first half of S buffer `b`."""
    layout: gl.constexpr = _bw.TensorMemoryLayout((BLOCK_M, 2 * BLOCK_N), col_stride=1)
    p = s_tmem._reinterpret(gl.bfloat16, [2, BLOCK_M, 2 * BLOCK_N], layout)
    return p.index(b).slice(0, BLOCK_N)


@gluon.jit
def _verify_mma(
    q_smem,
    k_smem,
    v_smem,
    s_tmem,
    o_tmem,
    kv_ready,
    kv_empty,
    s_ready,
    p_ready,
    o_done,
    o_final,
    num_iters,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    STAGES: gl.constexpr,
):
    """Issue S = Q K^T for block i, then O += P V for block i - 1."""
    for i in range(num_iters + 1):
        if i < num_iters:
            stage = i % STAGES
            _bw.mbarrier.wait(kv_ready.index(stage), i // STAGES & 1)
            _bw.tcgen05_mma(
                q_smem,
                k_smem.index(stage).permute((1, 0)),
                s_tmem.index(i % 2),
                use_acc=False,
                mbarriers=[s_ready.index(i % 2)],
            )
        if i > 0:
            j = i - 1
            _bw.mbarrier.wait(p_ready.index(j % 2), j // 2 & 1)
            _bw.tcgen05_mma(
                _verify_p(s_tmem, j % 2, BLOCK_M, BLOCK_N),
                v_smem.index(j % STAGES),
                o_tmem,
                use_acc=j > 0,
                mbarriers=[kv_empty.index(j % STAGES), o_done],
            )
    _bw.tcgen05_commit(o_final)


@gluon.jit
def _verify_softmax(
    s_tmem,
    o_tmem,
    s_ready,
    p_ready,
    o_done,
    o_final,
    Rel,
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    stride_gate,
    stride_sink,
    q_start,
    q_len,
    seqlen_k,
    pos0,
    kv_head,
    split,
    num_splits,
    num_tokens,
    n_first,
    n_bias,
    num_iters,
    fa_mask_left,
    fa_mask_right,
    softmax_scale,
    stride_rel_t,
    rel_base,
    H: gl.constexpr,
    G: gl.constexpr,
    D: gl.constexpr,
    REL: gl.constexpr,
    REL_PAD: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FA_BLOCK: gl.constexpr,
    FUSE_GATE: gl.constexpr,
    DRAFT: gl.constexpr,
    WINDOW: gl.constexpr,
):
    """Online softmax following FA4 per element.

    `qk + bias / scale` as an FMA, lazy rescaling (threshold 8),
    `ex2(qk * scale_log2 - max * scale_log2)` with FA4's ex2 emulation
    columns on blocks without causal or window masking, bf16 P, fp32 row sums.
    O is rescaled in tensor memory only when some row's max moved by more
    than the lazy-rescale threshold, after the previous P @ V completed.
    """
    num_warps: gl.constexpr = gl.num_warps()
    s_layout: gl.constexpr = s_tmem.index(0).get_reg_layout(num_warps)
    o_layout: gl.constexpr = o_tmem.get_reg_layout(num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, s_layout)
    o_row_layout: gl.constexpr = gl.SliceLayout(1, o_layout)

    rows = gl.arange(0, BLOCK_M, row_layout)
    tok = rows // G
    row_valid = (rows < BLOCK_M // G * G) & (tok < q_len)
    pos = gl.full([BLOCK_M], 0, gl.int32, row_layout) + pos0 if DRAFT else pos0 + tok
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, s_layout))
    gl.static_assert(BLOCK_N == FA_BLOCK)
    emu_col = (offs_n // 32 % 3 != 0) & (offs_n % 16 >= 12)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(gl.int64)

    inv_scale = 1.0 / softmax_scale
    scale_log2 = softmax_scale * 1.4426950408889634
    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, row_layout)
    l_i = gl.full([BLOCK_M], 0.0, gl.float32, row_layout)
    if not DRAFT:
        # The bias of the next block is loaded ahead.
        b_row = Rel + token * stride_rel_t + head * REL_PAD - rel_base
        b_ptrs = b_row[:, None] + offs_n[None, :]
        bias_next = gl.load(
            b_ptrs + n_first * BLOCK_N,
            mask=row_valid[:, None] & (n_first >= n_bias) & (num_iters > 0),
            other=0.0,
        )
    for i in range(num_iters):
        n = n_first + i * num_splits
        if not DRAFT:
            bias = bias_next
            bias_next = gl.load(
                b_ptrs + (n + num_splits) * BLOCK_N,
                mask=row_valid[:, None]
                & (n + num_splits >= n_bias)
                & (i + 1 < num_iters),
                other=0.0,
            )
        _bw.mbarrier.wait(s_ready.index(i % 2), i // 2 & 1)
        s = s_tmem.index(i % 2).load(s_layout)
        emu = emu_col
        if n >= n_bias:
            if not DRAFT:
                s = gl.fma(bias.to(gl.float32), inv_scale, s)
            # Only blocks with future keys or keys left of a query's window.
            key0 = n * BLOCK_N
            if key0 + BLOCK_N - 1 > pos0 or (IS_LOCAL and key0 < seqlen_k - WINDOW):
                dist = pos[:, None] - (key0 + offs_n)[None, :]
                visible = dist >= 0
                if IS_LOCAL:
                    visible &= dist < WINDOW
                s = gl.where(visible, s, float("-inf"))
            fa_block = key0 // FA_BLOCK
            emu &= (fa_block < fa_mask_right) & (fa_block >= fa_mask_left)

        m_new = gl.maximum(m_i, gl.max(s, 1))
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha_log2 = (m_i - m_safe) * scale_log2
        keep = alpha_log2 >= -8.0
        alpha = gl.where(keep, 1.0, _gl_ex2(alpha_log2))
        m_use = gl.where(keep, m_i, m_safe)
        m_i = gl.where(keep, m_i, m_new)
        x = gl.fma(s, scale_log2, -(m_use * scale_log2)[:, None])
        p = gl.where(emu[None, :], _gl_ex2_emu(x), _gl_ex2(x))
        l_i = l_i * alpha + gl.sum(p, 1)
        if i > 0 and gl.min(alpha, 0) < 1.0:
            _bw.mbarrier.wait(o_done, i - 1 & 1)
            o = o_tmem.load(o_layout)
            o_tmem.store(o * gl.convert_layout(alpha, o_row_layout)[:, None])
        _verify_p(s_tmem, i % 2, BLOCK_M, BLOCK_N).store(p.to(gl.bfloat16))
        _bw.mbarrier.arrive(p_ready.index(i % 2), count=1)

    # The combine kernel (PDL) may start loading the gate and sink.
    gl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // $0",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    empty = (l_i == 0.0) | (l_i != l_i)
    lse = gl.where(empty, float("-inf"), m_i * scale_log2 + _gl_lg2(l_i))
    o_valid = gl.convert_layout(row_valid, o_row_layout)
    offs_d = gl.arange(0, D, gl.SliceLayout(0, o_layout))
    if num_iters > 0:
        _bw.mbarrier.wait(o_final, 0)
        inv_l = _gl_rcp(gl.where(empty, 1.0, l_i))
        o = o_tmem.load(o_layout) * gl.convert_layout(inv_l, o_row_layout)[:, None]
    else:
        o = gl.full([BLOCK_M, D], 0.0, gl.float32, o_layout)
    if FUSE_GATE and DRAFT:
        col = gl.convert_layout(head * D, o_row_layout)[:, None] + offs_d[None, :]
        token_o = gl.convert_layout(token, o_row_layout)[:, None]
        gl.store(Y + token_o * (H * D) + col, o, mask=o_valid[:, None])
    elif FUSE_GATE:
        # A single split: apply the combine kernel's gate and sink here.
        lse_e = gl.convert_layout(lse * 0.6931471805599453, o_row_layout)
        sink = gl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
        sink = gl.convert_layout(sink.to(gl.float32), o_row_layout)
        col = gl.convert_layout(head * D, o_row_layout)[:, None] + offs_d[None, :]
        token_o = gl.convert_layout(token, o_row_layout)[:, None]
        gate = gl.load(
            Gate + token_o * stride_gate + col, mask=o_valid[:, None], other=0.0
        )
        y = o * _gl_sigmoid(gate.to(gl.float32))
        y = y * _gl_sigmoid(lse_e - sink)[:, None]
        gl.store(Y + token_o * (H * D) + col, y, mask=o_valid[:, None])
    else:
        out = (split * num_tokens + q_start + tok) * H + head
        gl.store(LsePart + out, lse, mask=row_valid)
        o_rows = gl.convert_layout(out, o_row_layout)
        gl.store(
            OPart + o_rows[:, None] * D + offs_d[None, :], o, mask=o_valid[:, None]
        )


@gluon.jit
def _rel_verify_kernel(
    Q,
    kv_desc,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    softmax_scale,
    stride_q_t,
    stride_rel_t,
    stride_bt,
    stride_gate,
    stride_sink,
    page_rows,
    head_rows,
    head_cols,
    num_splits,
    num_tokens,
    H: gl.constexpr,
    HKV: gl.constexpr,
    D: gl.constexpr,
    REL: gl.constexpr,
    REL_PAD: gl.constexpr,
    PAGE: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FA_BLOCK: gl.constexpr,
    STAGES: gl.constexpr,
    FUSE_GATE: gl.constexpr,
    DRAFT: gl.constexpr,
    WINDOW: gl.constexpr,
):
    """One split of one request and KV head: fp32 partial output, log2 LSE.

    One CTA holds all of a request's queries for one KV head (M = 64 rows of
    tokens x GQA group): a TMA warp streams K/V, an MMA warp runs tcgen05
    S = Q K^T into double-buffered tensor memory and O += P V, and the
    default warps run the softmax and write the partials. Split `s` takes
    key blocks `s, s + num_splits, ...` (of the window, for local layers).
    `Rel` holds the sheared relative logits of `attention_prologue`: key
    `k` of a request is column `k - (cdiv(seqlen_k, 128) * 128 - REL_PAD)`.
    `DRAFT` is plain block attention (the DSpark drafter): no relative
    logits, gate or sink, and every query of a request sits at its last
    position. Local layers see `WINDOW` keys up to a query's position.
    """
    G: gl.constexpr = H // HKV
    split = gl.program_id(0)
    kv_head = gl.program_id(1)
    req = gl.program_id(2)
    q_start = gl.load(CuSeqlensQ + req)
    q_len = gl.load(CuSeqlensQ + req + 1) - q_start
    seqlen_k = gl.load(SeqLens + req)
    past = seqlen_k - q_len
    # The first query's position, and the first block with relative logits.
    if DRAFT:
        pos0 = seqlen_k - 1
        n_bias = 0
    else:
        pos0 = past
        n_bias = gl.maximum(past - (REL - 1), 0) // BLOCK_N
    n_first = split
    fa_mask_left = 0
    if IS_LOCAL:
        n_first += gl.maximum(pos0 - (WINDOW - 1), 0) // BLOCK_N
        fa_mask_left = gl.cdiv(pos0 + FA_BLOCK - (WINDOW - 1), FA_BLOCK)
    fa_mask_right = pos0 // FA_BLOCK
    # Key of column 0 of the sheared relative logits.
    rel_base = gl.cdiv(seqlen_k, FA_BLOCK) * FA_BLOCK - REL_PAD
    num_iters = gl.maximum(gl.cdiv(gl.cdiv(seqlen_k, BLOCK_N) - n_first, num_splits), 0)

    smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16
    )
    q_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, q_layout))
    offs_d = gl.arange(0, D, gl.SliceLayout(0, q_layout))
    tok = rows // G
    row_valid = (rows < BLOCK_M // G * G) & (tok < q_len)
    token = (q_start + tok).to(gl.int64)
    q_ptrs = Q + token[:, None] * stride_q_t + (kv_head * G + rows % G)[:, None] * D

    s_layout: gl.constexpr = _bw.TensorMemoryLayout((BLOCK_M, BLOCK_N), col_stride=1)
    o_layout: gl.constexpr = _bw.TensorMemoryLayout((BLOCK_M, D), col_stride=1)
    k_smem = gl.allocate_shared_memory(
        gl.bfloat16, [STAGES, BLOCK_N, D], kv_desc.layout
    )
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [STAGES, BLOCK_N, D], kv_desc.layout
    )
    s_tmem = _bw.allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], s_layout)
    o_tmem = _bw.allocate_tensor_memory(gl.float32, [BLOCK_M, D], o_layout)
    bar_layout: gl.constexpr = _bw.mbarrier.MBarrierLayout()
    kv_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    kv_empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], bar_layout)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    p_ready = gl.allocate_shared_memory(gl.int64, [2, 1], bar_layout)
    o_done = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    o_final = gl.allocate_shared_memory(gl.int64, [1], bar_layout)
    for k in gl.static_range(STAGES):
        _bw.mbarrier.init(kv_ready.index(k), count=1)
        _bw.mbarrier.init(kv_empty.index(k), count=1)
    for k in gl.static_range(2):
        _bw.mbarrier.init(s_ready.index(k), count=1)
        _bw.mbarrier.init(p_ready.index(k), count=1)
    _bw.mbarrier.init(o_done, count=1)
    _bw.mbarrier.init(o_final, count=1)

    _bw.fence_async_shared()
    gl.barrier()
    # Key blocks before the new tokens were cached by earlier steps: load the
    # first of them while the producer of Q and the new keys (PDL) finishes.
    num_old = gl.maximum(gl.cdiv(past // BLOCK_N - n_first, num_splits), 0)
    num_pre = gl.minimum(gl.minimum(num_old, num_iters), STAGES)
    for i in gl.static_range(STAGES):
        if i < num_pre:
            _verify_load_block(
                kv_desc,
                k_smem,
                v_smem,
                kv_ready,
                BlockTable,
                req,
                kv_head,
                stride_bt,
                n_first,
                num_splits,
                i,
                page_rows,
                head_rows,
                head_cols,
                PAGE,
                BLOCK_N,
                D,
                STAGES,
            )
    gl.inline_asm_elementwise(
        "griddepcontrol.wait; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )
    q = gl.load(q_ptrs + offs_d[None, :], mask=row_valid[:, None], other=0.0)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, D], smem_layout, q)
    _bw.fence_async_shared()

    gl.warp_specialize(
        [
            (
                _verify_softmax,
                (
                    s_tmem,
                    o_tmem,
                    s_ready,
                    p_ready,
                    o_done,
                    o_final,
                    Rel,
                    OPart,
                    LsePart,
                    Gate,
                    Sink,
                    Y,
                    stride_gate,
                    stride_sink,
                    q_start,
                    q_len,
                    seqlen_k,
                    pos0,
                    kv_head,
                    split,
                    num_splits,
                    num_tokens,
                    n_first,
                    n_bias,
                    num_iters,
                    fa_mask_left,
                    fa_mask_right,
                    softmax_scale,
                    stride_rel_t,
                    rel_base,
                    H,
                    G,
                    D,
                    REL,
                    REL_PAD,
                    IS_LOCAL,
                    BLOCK_M,
                    BLOCK_N,
                    FA_BLOCK,
                    FUSE_GATE,
                    DRAFT,
                    WINDOW,
                ),
            ),
            (
                _verify_load,
                (
                    kv_desc,
                    k_smem,
                    v_smem,
                    kv_ready,
                    kv_empty,
                    BlockTable,
                    req,
                    kv_head,
                    stride_bt,
                    n_first,
                    num_splits,
                    num_pre,
                    num_iters,
                    page_rows,
                    head_rows,
                    head_cols,
                    PAGE,
                    BLOCK_N,
                    D,
                    STAGES,
                ),
            ),
            (
                _verify_mma,
                (
                    q_smem,
                    k_smem,
                    v_smem,
                    s_tmem,
                    o_tmem,
                    kv_ready,
                    kv_empty,
                    s_ready,
                    p_ready,
                    o_done,
                    o_final,
                    num_iters,
                    BLOCK_M,
                    BLOCK_N,
                    STAGES,
                ),
            ),
        ],
        [1, 1],
        [24, 24],
    )


def rel_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    is_local: bool,
    num_splits: int,
) -> None:
    """Write the gated bf16 `(T, H * D)` attention output for small batches.

    Split-KV relative attention on `_rel_verify_kernel` (see
    `plan_decode_attention`) reading the sheared relative logits. With more
    than one split, a second launch merges the fp32 partials and applies the
    output gate and attention sink. Tokens from `num_actual_tokens` on are
    zeroed.
    """
    num_tokens, num_heads = q.shape[:2]
    extent_pad = rel_logits.shape[2]
    extent = extent_pad - 2 * _FA4_BLOCK_N
    assert rel_logits.shape[0] >= num_tokens and rel_logits.shape[1] == num_heads
    assert rel_logits.stride(1) == extent_pad and rel_logits.stride(2) == 1
    assert extent % _FA4_BLOCK_N == 0
    assert gate.stride(1) == sinks.stride(1) == 1
    _decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        (rel_logits, gate, sinks),
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_query_len,
        num_actual_tokens,
        softmax_scale,
        extent if is_local else 0,
        num_splits,
    )


def block_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block: int,
    window: int,
    num_splits: int,
) -> None:
    """Write the bf16 `(T, H * D)` attention of each `block`-token request
    to `out`: every query sees the request's keys up to its last one, or the
    last `window` of them if `window` (the DSpark draft block attention).

    Runs `_rel_verify_kernel` like `rel_decode_attention`, without the
    relative logits, gate and sink.
    """
    _decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        None,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        block,
        q.shape[0],
        q.shape[2] ** -0.5,
        window,
        num_splits,
    )


def _decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    rel_gate_sinks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    """Launch `_rel_verify_kernel` (and the combine for several splits);
    `rel_gate_sinks` None is the draft mode, `window` 0 a full layer."""
    num_tokens, num_heads, head_dim = q.shape
    num_blocks, page, num_kv_heads = key_cache.shape[:3]
    group = num_heads // num_kv_heads
    draft = rel_gate_sinks is None
    if rel_gate_sinks is None:
        # Unused placeholders keep the launches typed alike.
        rel_logits = gate = sinks = out
        extent = 0
    else:
        rel_logits, gate, sinks = rel_gate_sinks
        extent = rel_logits.shape[2] - 2 * _FA4_BLOCK_N
    assert max_query_len <= _VERIFY_BLOCK_M // group
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(1) == head_dim and q.stride(2) == 1
    assert out.is_contiguous()
    grid = (num_splits, num_kv_heads, cache_seqlens.shape[0])
    # A single split writes the (gated) output itself.
    fuse_gate = num_splits == 1
    o_part = q.new_empty(
        (0 if fuse_gate else num_splits, *q.shape), dtype=torch.float32
    )
    lse_part = q.new_empty(
        (0 if fuse_gate else num_splits, num_tokens, num_heads), dtype=torch.float32
    )
    # The paged cache as 2D rows of (K | V) per KV head: a page's slots
    # are rows, KV heads sit either side by side in a row or in row blocks.
    page_stride, slot_stride, head_stride = key_cache.stride()[:3]
    assert value_cache.data_ptr() == key_cache.data_ptr() + head_dim * 2
    assert page_stride % slot_stride == 0 and slot_stride >= 2 * head_dim
    if head_stride < slot_stride:
        head_rows, head_cols = 0, head_stride
    else:
        assert head_stride % slot_stride == 0
        head_rows, head_cols = head_stride // slot_stride, 0
    kv_rows = key_cache.as_strided(
        (num_blocks * page_stride // slot_stride, slot_stride), (slot_stride, 1)
    )
    block = [min(page, _FA4_BLOCK_N), head_dim]
    kv_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        kv_rows, block, gl.NVMMASharedLayout.get_default_for(block, gl.bfloat16)
    )
    _rel_verify_kernel[grid](
        q,
        kv_desc,
        rel_logits,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        softmax_scale,
        q.stride(0),
        rel_logits.stride(0),
        block_table.stride(0),
        gate.stride(0),
        sinks.stride(0),
        page_stride // slot_stride,
        head_rows,
        head_cols,
        num_splits,
        num_tokens,
        H=num_heads,
        HKV=num_kv_heads,
        D=head_dim,
        REL=extent,
        REL_PAD=extent + 2 * _FA4_BLOCK_N,
        PAGE=page,
        IS_LOCAL=window > 0,
        BLOCK_M=_VERIFY_BLOCK_M,
        BLOCK_N=_FA4_BLOCK_N,
        FA_BLOCK=_FA4_BLOCK_N,
        STAGES=_VERIFY_STAGES,
        FUSE_GATE=fuse_gate,
        DRAFT=draft,
        WINDOW=window,
        num_warps=4,
        launch_pdl=True,
    )
    if fuse_gate:
        out[num_actual_tokens:].zero_()
        return
    # Small batches spread the merge over more programs; larger ones merge a
    # whole head per (narrow) program.
    if num_tokens * num_heads >= 512:
        block_d, num_warps = head_dim, 2 if num_splits > 32 else 1
    else:
        block_d, num_warps = 32, 4
    _rel_decode_combine_kernel[(num_tokens, num_heads, head_dim // block_d)](
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        num_tokens,
        num_actual_tokens,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=block_d,
        GATED=not draft,
        num_warps=num_warps,
        launch_pdl=True,
    )
