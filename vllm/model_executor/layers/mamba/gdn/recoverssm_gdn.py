# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM for Gated DeltaNet (Qwen-style GDN): speculative verify from one
checkpoint without per-draft state snapshots, and accepted-state recovery after
sampling.

GDN port of the Kimi-K3 KDA RecoverSSM path
(vllm/models/kimi_k3/nvidia/ops/recoverssm.py). Differences from KDA: the gate is a
scalar per value head (g = -exp(A_log) * softplus(a + dt_bias)), q/k heads are grouped
over value heads (HV // H), and q/k are L2-normalised inside the kernel when requested.
The verify step caches, per token, the delta-rule correction c_t = beta_t * (v_t - S'_t
k_t) (V floats), the normalised key (K floats) and the log-decay g_t (1 float) in one
fp32 "replay" record. Recovery folds the accepted prefix: S_n = exp(sum_t g_t) S_0 +
sum_t exp(sum_{s>t} g_s) c_t k_t^T and writes the checkpoint once (plus the
block-boundary state in align mode)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.ops.recoverssm import (
    compact_conv_state_kernel,
    prepare_commit_plan_kernel,
)
from vllm.model_executor.layers.mamba.recoverssm_utils import recoverssm_require
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _require(cond: bool, msg: str) -> None:
    recoverssm_require(cond, msg, "GDN RecoverSSM")


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    replay_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    softplus_beta,
    softplus_threshold,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_a_token,
    stride_b_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_block,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_out_token,
    stride_qsl,
    stride_si,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hv = tl.program_id(2)
    pid_h = pid_hv // (HV // H)
    bos = tl.load(query_start_loc_ptr + pid_b * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_qsl).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    if state_idx <= null_block_id:
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptr + (bos + t) * stride_out_token + pid_hv * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=(t < query_len) & mask_v,
            )
        return
    state = tl.load(
        state_ptr
        + state_idx * stride_state_block
        + pid_hv * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_hv).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    rec_base = (
        replay_ptr + state_idx * stride_replay_block + pid_hv * stride_replay_head
    )
    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        tok = bos + t
        q = tl.load(
            q_ptr + tok * stride_q_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + tok * stride_k_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + tok * stride_v_token + pid_hv * V + offs_v,
            mask=valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        x = (
            tl.load(a_ptr + tok * stride_a_token + pid_hv, mask=valid, other=0.0).to(
                tl.float32
            )
            + dt_bias
        )
        sp = tl.where(
            softplus_beta * x <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x)),
            x,
        )
        g = neg_A * sp
        beta = tl.sigmoid(
            tl.load(b_ptr + tok * stride_b_token + pid_hv, mask=valid, other=0.0).to(
                tl.float32
            )
        )
        if USE_QK_L2NORM:
            q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        q = q * scale
        new_state = state * tl.exp(g)
        corr = (v - tl.sum(new_state * k[None, :], 1)) * beta
        new_state = new_state + corr[:, None] * k[None, :]
        state = tl.where(valid, new_state, state)
        o = tl.sum(state * q[None, :], 1)
        tl.store(
            out_ptr + tok * stride_out_token + pid_hv * V + offs_v,
            o.to(out_ptr.dtype.element_ty),
            mask=valid & mask_v,
        )
        rec = rec_base + t * stride_replay_pos
        tl.store(rec + offs_v * stride_replay_dim, corr, mask=valid & mask_v)
        if pid_v == 0:
            tl.store(rec + (V + offs_k) * stride_replay_dim, k, mask=valid & mask_k)
            tl.store(rec + (V + K) * stride_replay_dim, g, mask=valid)


def gdn_recoverssm_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a GDN speculative window against its checkpoint without modifying the
    checkpoint. q, k: [1, T, H, K]; v: [1, T, HV, V]; a, b: [T, HV] (or [1, T, HV]);
    checkpoint_state: [blocks, HV, V, K]; replay_cache: [blocks, HV, spec_query_len, V +
    K + 1] fp32. Returns out [1, T, HV, V]."""
    _require(
        q.ndim == 4 and q.shape[0] == 1, "q must have shape [1, tokens, heads, dim]"
    )
    _, total, H, K = q.shape
    _require(k.shape == q.shape, "q and k shapes differ")
    _require(
        v.ndim == 4 and tuple(v.shape[:2]) == (1, total),
        "v must have shape [1, tokens, value heads, dim]",
    )
    HV, V = v.shape[-2], v.shape[-1]
    _require(
        H > 0 and HV >= H and HV % H == 0,
        "value heads must be a positive multiple of the q/k heads",
    )
    _require(
        a.numel() == total * HV and b.numel() == total * HV,
        "gate or beta shape is incompatible",
    )
    a2 = a.reshape(total, HV)
    b2 = b.reshape(total, HV)
    _require(
        a2.stride(1) == 1 and b2.stride(1) == 1,
        "gate and beta heads must be contiguous",
    )
    _require(
        tuple(A_log.shape) == (HV,) and tuple(dt_bias.shape) == (HV,),
        "A_log or dt_bias shape is incompatible",
    )
    _require(
        all(t_.stride(-1) == 1 and t_.stride(-2) == K for t_ in (q, k)),
        "q and k heads must be contiguous",
    )
    _require(v.stride(-1) == 1 and v.stride(-2) == V, "v heads must be contiguous")
    _require(checkpoint_state.ndim == 4, "checkpoint must be four-dimensional")
    nb = checkpoint_state.shape[0]
    _require(
        tuple(checkpoint_state.shape[1:]) == (HV, V, K),
        "checkpoint shape is incompatible",
    )
    _rec = (nb, HV, spec_query_len, V + K + 1)
    _require(tuple(replay_cache.shape) == _rec, f"replay buffer needs shape {_rec}")
    _require(replay_cache.dtype == torch.float32, "replay buffer must use float32")
    _require(state_indices.ndim == 1, "state indices must be one-dimensional")
    _require(
        query_start_loc.ndim == 1
        and query_start_loc.shape[0] == state_indices.shape[0] + 1,
        "query metadata is incompatible",
    )
    _require(
        total <= state_indices.shape[0] * spec_query_len,
        "speculative decode input exceeds its activation capacity",
    )
    _dev = q.device
    _require(
        all(
            t_.device == _dev
            for t_ in (
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                checkpoint_state,
                replay_cache,
                query_start_loc,
                state_indices,
            )
            if t_ is not None
        )
        and (out is None or out.device == _dev),
        "inputs must be on the same device",
    )
    if scale is None:
        scale = K**-0.5
    if out is None:
        out = torch.empty(1, total, HV, V, dtype=v.dtype, device=v.device)
    _require(
        tuple(out.shape) == (1, total, HV, V) and out.stride()[2:] == (V, 1),
        "output shape or layout is incompatible",
    )
    batch = state_indices.shape[0]
    if total == 0 or batch == 0:
        return out
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), batch, HV)
    _gdn_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        a2,
        b2,
        A_log,
        dt_bias,
        checkpoint_state,
        replay_cache,
        out,
        query_start_loc,
        state_indices,
        scale,
        softplus_beta,
        softplus_threshold,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a2.stride(0),
        b2.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        replay_cache.stride(0),
        replay_cache.stride(1),
        replay_cache.stride(2),
        replay_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SPEC_QUERY_LEN=spec_query_len,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _commit_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    replay_ref_ptr,
    replay_base_addrs_ptr,
    replay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_si,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    # In align mode the second half of axis 0 folds the block-boundary state
    # (the window's first boundary_len tokens) instead of the final state, so every
    # program runs one fold and keeps one tile live.
    NUM_V_TILES: tl.constexpr = (V + BV - 1) // BV
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    src = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + pid_b)
    if n == 0:
        return
    final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_idx <= null_block_id:
        return
    dst_idx = final_idx
    # ALIGN_MODE is a constexpr: kept apart so non-align builds drop the branch.
    if ALIGN_MODE:  # noqa: SIM102
        if pid_v >= NUM_V_TILES:
            pid_v -= NUM_V_TILES
            dst_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
            if dst_idx <= null_block_id:
                return
            n = tl.load(boundary_recovery_lens_ptr + pid_b)
    state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
        tl.pointer_type(state_ref_ptr.dtype.element_ty)
    )
    sbs = tl.load(state_block_strides_ptr + pid_l)
    replay_ptr = tl.load(replay_base_addrs_ptr + pid_l).to(
        tl.pointer_type(replay_ref_ptr.dtype.element_ty)
    )
    rbs = tl.load(replay_block_strides_ptr + pid_l)
    rec_base = replay_ptr + src * rbs + pid_h * stride_replay_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    sp = offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k
    s0 = tl.load(
        state_ptr + src * sbs + pid_h * stride_state_head + sp,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    decay = 1.0
    corr = tl.zeros([BV, BK], tl.float32)
    for r in range(n):
        t = n - r - 1
        rec = rec_base + t * stride_replay_pos
        c = tl.load(rec + offs_v * stride_replay_dim, mask=mask_v, other=0.0)
        kk = tl.load(rec + (V + offs_k) * stride_replay_dim, mask=mask_k, other=0.0)
        g = tl.load(rec + (V + K) * stride_replay_dim)
        corr += (c[:, None] * kk[None, :]) * decay
        decay *= tl.exp(g)
    tl.store(
        state_ptr + dst_idx * sbs + pid_h * stride_state_head + sp,
        (s0 * decay + corr).to(state_ref_ptr.dtype.element_ty),
        mask=mask_state,
    )


@dataclass
class GDNRecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    replays: tuple[torch.Tensor, ...]
    replay_base_addrs: torch.Tensor
    replay_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def from_tensors(
        cls,
        conv_states: Sequence[torch.Tensor],
        checkpoints: Sequence[torch.Tensor],
        replays: Sequence[torch.Tensor],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        conv_dim_first: bool = True,
    ) -> "GDNRecoverSSMCommitContext":
        _require(len(checkpoints) > 0, "commit requires at least one layer")
        _require(
            len(conv_states) == len(checkpoints) == len(replays),
            "conv, state and replay lists differ",
        )
        if not conv_dim_first:
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        _require(ref.ndim == 4, "checkpoint must be four-dimensional")
        nb, HV, V, K = ref.shape
        _dev = ref.device
        for s in checkpoints:
            _require(
                s.shape == ref.shape
                and s.dtype == ref.dtype
                and s.stride()[1:] == ref.stride()[1:]
                and s.device == _dev,
                "layers need matching checkpoints",
            )
        _rec = (nb, HV, spec_query_len, V + K + 1)
        for r in replays:
            _require(
                tuple(r.shape) == _rec
                and r.dtype == torch.float32
                and r.stride()[1:] == replays[0].stride()[1:]
                and r.device == _dev,
                f"layers need matching float32 replay buffers of shape {_rec}",
            )
        conv_ref = conv_states[0]
        _require(
            conv_ref.ndim == 3 and conv_ref.shape[0] == nb,
            "conv state must be [blocks, dim, window] with the checkpoint's "
            "block count",
        )
        for c in conv_states:
            _require(
                c.shape == conv_ref.shape
                and c.dtype == conv_ref.dtype
                and c.stride()[1:] == conv_ref.stride()[1:]
                and c.device == _dev,
                "layers need matching conv states",
            )
        conv_dim, conv_len = conv_ref.shape[1:]
        hist = conv_len - spec_query_len + 1
        _require(hist > 0, "conv state is shorter than its window")
        dev = ref.device
        addr = lambda ts: torch.tensor(
            [t.data_ptr() for t in ts], dtype=torch.int64, device=dev
        )
        bstr = lambda ts: torch.tensor(
            [t.stride(0) for t in ts], dtype=torch.int64, device=dev
        )
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        return cls(
            tuple(conv_states),
            addr(conv_states),
            bstr(conv_states),
            torch.tensor(
                [s.stride(1) for s in conv_states], dtype=torch.int64, device=dev
            ),
            torch.tensor(
                [s.stride(2) for s in conv_states], dtype=torch.int64, device=dev
            ),
            hist,
            tuple(checkpoints),
            addr(checkpoints),
            bstr(checkpoints),
            tuple(replays),
            addr(replays),
            bstr(replays),
            z(),
            z(),
            z(),
            z(),
            spec_query_len,
        )

    @classmethod
    def create(
        cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> "GDNRecoverSSMCommitContext":
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        if any(len(layer.kv_cache) != 3 for layer in layers):
            raise ValueError("GDN RecoverSSM pages must contain conv, state and replay")
        return cls.from_tensors(
            [layer.kv_cache[0] for layer in layers],
            [layer.kv_cache[1] for layer in layers],
            [layer.kv_cache[2] for layer in layers],
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            conv_dim_first=is_conv_state_dim_first(),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        commit_conv: bool = True,
    ) -> None:
        """Fold accepted GDN and convolution inputs into every layer's checkpoint."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        _require(state_indices.ndim == 1, "state indices must be one-dimensional")
        _require(
            batch <= self.commit_lens.shape[0], "commit batch exceeds its plan capacity"
        )
        _require(
            query_start_loc.ndim == 1 and query_start_loc.shape[0] == batch + 1,
            "commit metadata is incompatible",
        )
        _require(
            request_indices is None or request_indices.shape[0] >= batch,
            "request mapping is too short",
        )
        _require(
            num_accepted_tokens.ndim == 1
            and (request_indices is not None or num_accepted_tokens.shape[0] >= batch),
            "accepted-token counts are too short",
        )
        _align = (block_table, num_computed_tokens, mamba_block_size)
        _require(
            all(x is None for x in _align) or all(x is not None for x in _align),
            "align metadata is incomplete",
        )
        _require(
            mamba_block_size is None or mamba_block_size >= self.spec_query_len,
            "align block size must cover one speculative window",
        )
        _require(
            block_table is None or block_table.ndim == 2,
            "block table must be two-dimensional",
        )
        _dev = self.checkpoints[0].device
        _require(
            all(
                t_ is None or t_.device == _dev
                for t_ in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                    request_indices,
                    block_table,
                    num_computed_tokens,
                )
            ),
            "commit inputs must be on the same device",
        )
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            bt_stride[0],
            bt_stride[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.checkpoints)
        if commit_conv:
            conv_ref = self.conv_states[0]
            conv_dim = conv_ref.shape[1]
            compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
                conv_ref,
                self.conv_state_base_addrs,
                self.conv_state_block_strides,
                self.conv_state_dim_strides,
                self.conv_state_token_strides,
                state_indices,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_recovery_lens,
                NULL_BLOCK_ID,
                conv_dim,
                self.conv_history_len,
                state_indices.stride(0),
                BLOCK_D=256,
                BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
                ALIGN_MODE=block_table is not None,
                num_warps=4,
            )
        ref = self.checkpoints[0]
        _, HV, V, K = ref.shape
        BK = triton.next_power_of_2(K)
        # 16-row tiles: the commit streams each state once, and smaller tiles
        # keep more of them in flight (about 13% more bandwidth than 32 on H200).
        BV = min(triton.next_power_of_2(V), 16)
        rr = self.replays[0]
        align = block_table is not None
        num_v_tiles = triton.cdiv(V, BV) * (2 if align else 1)
        _commit_gdn_state_kernel[(num_v_tiles, batch, num_layers * HV)](
            ref,
            self.state_base_addrs,
            self.state_block_strides,
            rr,
            self.replay_base_addrs,
            self.replay_block_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            ref.stride(1),
            ref.stride(2),
            ref.stride(3),
            rr.stride(1),
            rr.stride(2),
            rr.stride(3),
            state_indices.stride(0),
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            NUM_HEADS=HV,
            ALIGN_MODE=block_table is not None,
            num_warps=4,
            num_stages=2,
        )


__all__ = ["GDNRecoverSSMCommitContext", "gdn_recoverssm_verify"]
