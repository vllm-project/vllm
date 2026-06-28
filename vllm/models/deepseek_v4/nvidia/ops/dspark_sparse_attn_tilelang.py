# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse attention used by DeepSeek V4 DSpark draft blocks.

This is intentionally separate from the normal DeepSeek V4 sparse-MLA path:
DSpark attends a block of draft queries over a small rolling target-KV window
plus the draft block KV, indexed by a dense ``topk_idxs`` tensor.  It does not
consume vLLM paged MLA metadata.
"""

from functools import cache
from typing import Any

import torch


@cache
def _build_dspark_sparse_attn_kernel(num_heads: int, head_dim: int, scale: float):
    import tilelang
    import tilelang.language as T

    pass_configs: dict[tilelang.PassConfigKey, Any] = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    bf16 = "bfloat16"
    fp32 = "float32"
    int32 = "int32"

    @tilelang.jit(pass_configs=pass_configs)
    def kernel(h: int, d: int, softmax_scale: float):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        topk = T.symbolic("topk")

        num_stages = 2
        threads = 256
        # The upstream DSpark reference uses 64, which asks for ~104 KiB
        # dynamic shared memory on DeepSeek V4 head dims and is rejected on
        # this sm120 stack. 32 keeps the same online-softmax algorithm while
        # fitting under the per-block shared-memory limit.
        block = 32
        num_blocks = tilelang.cdiv(topk, block)

        @T.prim_func
        def sparse_attn_kernel_(
            q: T.Tensor[(b, m, h, d), bf16],
            kv: T.Tensor[(b, n, d), bf16],
            out: T.Tensor[(b, m, h, d), bf16],
            attn_sink: T.Tensor[(h,), fp32],
            topk_idxs: T.Tensor[(b, m, topk), int32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), bf16)
                kv_shared = T.alloc_shared((block, d), bf16)
                out_shared = T.alloc_shared((h, d), bf16)
                acc_s_cast = T.alloc_shared((h, block), bf16)

                idxs = T.alloc_fragment(block, int32)
                acc_s = T.alloc_fragment((h, block), fp32)
                acc_o = T.alloc_fragment((h, d), fp32)
                scores_max = T.alloc_fragment(h, fp32)
                scores_max_prev = T.alloc_fragment(h, fp32)
                scores_scale = T.alloc_fragment(h, fp32)
                scores_sum = T.alloc_fragment(h, fp32)
                sum_exp = T.alloc_fragment(h, fp32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(fp32))
                T.copy(q[by, bx, :, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(
                            t * block + i < topk,
                            topk_idxs[by, bx, t * block + i],
                            -1,
                        )
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(
                            idxs[i] != -1,
                            kv[by, idxs[i], j],
                            0,
                        )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(
                            idxs[j] != -1,
                            0,
                            -T.infinity(fp32),
                        )
                    T.gemm(
                        q_shared,
                        kv_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= softmax_scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(
                        acc_s_cast,
                        kv_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, out_shared)
                T.copy(out_shared, out[by, bx, :, :])

        return sparse_attn_kernel_

    return kernel(num_heads, head_dim, scale)


def dspark_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run DSpark sparse attention.

    Args:
        q: ``[batch, draft_tokens, heads, head_dim]`` bf16.
        kv: ``[batch, rolling_window + draft_tokens, head_dim]`` bf16.
        attn_sink: ``[heads]`` fp32.
        topk_idxs: ``[batch, draft_tokens, topk]`` int32 indices into ``kv``.
    """
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("DSpark sparse attention currently expects bf16 q/kv")
    if topk_idxs.dtype != torch.int32:
        topk_idxs = topk_idxs.to(torch.int32)

    batch, draft_tokens, heads, head_dim = q.shape
    padded_heads = heads
    if heads < 16:
        padded_heads = 16
        q = torch.cat(
            [q, q.new_zeros(batch, draft_tokens, padded_heads - heads, head_dim)],
            dim=2,
        )
        attn_sink = torch.cat(
            [attn_sink, attn_sink.new_zeros(padded_heads - heads)],
        )

    out = torch.empty_like(q)
    kernel = _build_dspark_sparse_attn_kernel(padded_heads, head_dim, softmax_scale)
    kernel(q.contiguous(), kv.contiguous(), out, attn_sink.contiguous(), topk_idxs)
    if heads < 16:
        return out.narrow(2, 0, heads).contiguous()
    return out


def dspark_sparse_attn_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Slow PyTorch reference for DSpark sparse attention debugging.

    This mirrors the public HF DSpark sparse-attention contract and the
    TileLang kernel above: attend over gathered KV rows plus a learned
    denominator-only sink term.
    """
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("DSpark sparse attention reference expects bf16 q/kv")

    valid = topk_idxs >= 0
    gather_idxs = topk_idxs.clamp_min(0).long()
    batch = torch.arange(kv.shape[0], device=kv.device).view(-1, 1, 1)
    gathered_kv = kv[batch, gather_idxs]

    scores = torch.einsum(
        "bmhd,bmkd->bmhk",
        q.float(),
        gathered_kv.float(),
    )
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid.unsqueeze(2), -torch.inf)

    scores_max = scores.amax(dim=-1)
    weights = torch.exp(scores - scores_max.unsqueeze(-1))
    weights.masked_fill_(~valid.unsqueeze(2), 0.0)
    denom = weights.sum(dim=-1)
    denom.add_(torch.exp(attn_sink.float().view(1, 1, -1) - scores_max))

    out = torch.einsum("bmhk,bmkd->bmhd", weights, gathered_kv.float())
    out.div_(denom.unsqueeze(-1))
    return out.to(q.dtype)
