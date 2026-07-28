# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 recurrent state operators.

The runtime stores state as ``[batch, heads, value_dim, key_dim]``. This is the
transpose of the equation's key-by-value matrix and lets each Triton program
own one value row without cross-program reductions.
"""

import os
from dataclasses import dataclass
from typing import Literal, cast

import torch

import vllm.envs as envs
from vllm.triton_utils import HAS_TRITON, tl, triton

RWKV7KernelBackend = Literal["auto", "torch", "triton"]

# Do not promote Triton through ``auto`` until long-horizon greedy generation
# is exact. Users can still opt in explicitly for benchmarking and validation.
_TRITON_AUTO_ENABLED = False

# A multi-warp reduction is faster in isolation but changes fp32 accumulation
# enough to flip near-tied greedy logits after recurrent state accumulation.
_TRITON_NUM_WARPS = 1


@dataclass(frozen=True)
class RWKV7KernelParityReport:
    """Numerical comparison between the Torch and Triton recurrent paths."""

    output_close: bool
    state_close: bool
    first_output_mismatch_token: int | None
    first_state_mismatch_sequence: int | None
    output_max_abs_error: float
    output_max_rel_error: float
    output_cosine_similarity: float
    state_max_abs_error: float
    state_max_rel_error: float
    state_cosine_similarity: float
    atol: float
    rtol: float


def resolve_rwkv7_kernel_backend(
    requested: RWKV7KernelBackend | None = None,
    *,
    input_tensor: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> Literal["torch", "triton"]:
    """Resolve and validate the recurrent backend.

    ``auto`` deliberately resolves to ``torch`` while the fused kernel is
    experimental. Explicit ``triton`` requests fail instead of silently
    falling back, which keeps benchmarks and correctness gates honest.
    """
    fallback = cast(
        RWKV7KernelBackend,
        os.getenv("VLLM_RWKV7_KERNEL", "auto"),
    )
    backend: str = (
        requested
        if requested is not None
        else getattr(envs, "VLLM_RWKV7_KERNEL", fallback)
    )
    backend = backend.lower()
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(
            f"Invalid RWKV7 kernel backend {backend!r}; expected auto, torch, or triton"
        )
    if backend == "auto":
        backend = "triton" if _TRITON_AUTO_ENABLED else "torch"
    if backend == "triton":
        if not HAS_TRITON:
            raise RuntimeError(
                "RWKV7 Triton kernel requested, but Triton is unavailable"
            )
        if input_tensor is not None and not input_tensor.is_cuda:
            raise RuntimeError("RWKV7 Triton kernel requires CUDA or ROCm tensors")
        if state is not None and state.dtype != torch.float32:
            raise RuntimeError("RWKV7 Triton kernel requires fp32 recurrent state")
    return cast(Literal["torch", "triton"], backend)


def _rwkv7_step_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_dot_kk = (state * kk.unsqueeze(-2)).sum(dim=-1)
    new_state = (
        state * torch.exp(w).unsqueeze(-2)
        + v.unsqueeze(-1) * k.unsqueeze(-2)
        - state_dot_kk.unsqueeze(-1) * (kk * a).unsqueeze(-2)
    )
    output = (new_state * r.unsqueeze(-2)).sum(dim=-1)
    return output, new_state


def _rwkv7_scan_packed_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(v)
    final_state = torch.empty_like(state)
    for seq_idx in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        seq_state = state[seq_idx : seq_idx + 1]
        for token_idx in range(start, end):
            token_output, seq_state = _rwkv7_step_reference(
                r[token_idx : token_idx + 1],
                w[token_idx : token_idx + 1],
                k[token_idx : token_idx + 1],
                v[token_idx : token_idx + 1],
                kk[token_idx : token_idx + 1],
                a[token_idx : token_idx + 1],
                seq_state,
            )
            output[token_idx].copy_(token_output[0])
        final_state[seq_idx].copy_(seq_state[0])
    return output, final_state


if HAS_TRITON:

    @triton.jit
    def _rwkv7_recurrent_step_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        output_ptr,
        final_state_ptr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        value_idx = row_id % V
        batch_head_idx = row_id // V

        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K
        vector_base = batch_head_idx * K
        state_base = row_id * K

        state = tl.load(state_ptr + state_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        r = tl.load(r_ptr + vector_base + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + vector_base + offsets, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + vector_base + offsets, mask=mask, other=0.0).to(tl.float32)
        kk = tl.load(kk_ptr + vector_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        a = tl.load(a_ptr + vector_base + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(v_ptr + batch_head_idx * V + value_idx).to(tl.float32)

        state_dot_kk = tl.sum(state * kk, axis=0)
        state = state * tl.exp(w) + value * k - state_dot_kk * kk * a
        output = tl.sum(state * r, axis=0)

        tl.store(final_state_ptr + state_base + offsets, state, mask=mask)
        tl.store(output_ptr + batch_head_idx * V + value_idx, output)

    @triton.jit
    def _rwkv7_recurrent_scan_packed_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        query_start_loc_ptr,
        output_ptr,
        final_state_ptr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        value_idx = row_id % V
        seq_head_idx = row_id // V
        head_idx = seq_head_idx % H
        seq_idx = seq_head_idx // H

        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K
        state_base = row_id * K
        state = tl.load(state_ptr + state_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        token_idx = tl.load(query_start_loc_ptr + seq_idx).to(tl.int64)
        end = tl.load(query_start_loc_ptr + seq_idx + 1).to(tl.int64)
        while token_idx < end:
            vector_base = (token_idx * H + head_idx) * K
            value_base = (token_idx * H + head_idx) * V
            r = tl.load(r_ptr + vector_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_ptr + vector_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            k = tl.load(k_ptr + vector_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            kk = tl.load(kk_ptr + vector_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            a = tl.load(a_ptr + vector_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            value = tl.load(v_ptr + value_base + value_idx).to(tl.float32)

            state_dot_kk = tl.sum(state * kk, axis=0)
            state = state * tl.exp(w) + value * k - state_dot_kk * kk * a
            output = tl.sum(state * r, axis=0)
            tl.store(output_ptr + value_base + value_idx, output)
            token_idx += 1

        tl.store(final_state_ptr + state_base + offsets, state, mask=mask)


def _validate_inputs(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> tuple[int, int, int]:
    if r.ndim != 3:
        raise ValueError("r/w/k/kk/a must be shaped [batch, heads, key_dim]")
    if any(t.shape != r.shape for t in (w, k, kk, a)):
        raise ValueError("r/w/k/kk/a must have identical shapes")
    if v.ndim != 3 or v.shape[:2] != r.shape[:2]:
        raise ValueError("v must be shaped [batch, heads, value_dim]")
    batch, heads, key_dim = r.shape
    value_dim = v.shape[-1]
    if state.shape != (batch, heads, value_dim, key_dim):
        raise ValueError("state must be shaped [batch, heads, value_dim, key_dim]")
    if any(t.device != r.device for t in (w, k, v, kk, a, state)):
        raise ValueError("all RWKV7 recurrent inputs must be on the same device")
    return heads, key_dim, value_dim


def rwkv7_recurrent_step(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    *,
    backend: RWKV7KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batched RWKV7 recurrent update without modifying ``state``."""
    heads, key_dim, value_dim = _validate_inputs(r, w, k, v, kk, a, state)
    selected_backend = resolve_rwkv7_kernel_backend(
        backend, input_tensor=r, state=state
    )
    if selected_backend == "torch":
        return _rwkv7_step_reference(r, w, k, v, kk, a, state)

    output = torch.empty_like(v)
    final_state = torch.empty_like(state)
    block_key = triton.next_power_of_2(key_dim)
    _rwkv7_recurrent_step_kernel[(r.shape[0] * heads * value_dim,)](
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        kk.contiguous(),
        a.contiguous(),
        state.contiguous(),
        output,
        final_state,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_key,
        num_warps=_TRITON_NUM_WARPS,
    )
    return output, final_state


def rwkv7_recurrent_scan_packed(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    backend: RWKV7KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an RWKV7 scan over sequences delimited by ``query_start_loc``."""
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must contain one start per sequence plus end")
    num_sequences = query_start_loc.numel() - 1
    if r.ndim != 3:
        raise ValueError("packed r/w/k/kk/a must be shaped [tokens, heads, key_dim]")
    if state.shape[0] != num_sequences:
        raise ValueError("state batch size must match query_start_loc")
    if any(t.shape != r.shape for t in (w, k, kk, a)):
        raise ValueError("packed r/w/k/kk/a must have identical shapes")
    if v.ndim != 3 or v.shape[:2] != r.shape[:2]:
        raise ValueError("packed v must match the token and head dimensions")
    _, heads, key_dim = r.shape
    value_dim = v.shape[-1]
    if state.shape != (num_sequences, heads, value_dim, key_dim):
        raise ValueError("state must be shaped [sequences, heads, value_dim, key_dim]")
    if any(t.device != r.device for t in (w, k, v, kk, a, state)):
        raise ValueError("all RWKV7 recurrent inputs must be on the same device")
    if query_start_loc.device != r.device:
        raise ValueError("query_start_loc must be on the recurrent input device")

    selected_backend = resolve_rwkv7_kernel_backend(
        backend, input_tensor=r, state=state
    )
    if selected_backend == "torch":
        return _rwkv7_scan_packed_reference(r, w, k, v, kk, a, state, query_start_loc)

    output = torch.empty_like(v)
    final_state = torch.empty_like(state)
    block_key = triton.next_power_of_2(key_dim)
    _rwkv7_recurrent_scan_packed_kernel[(num_sequences * heads * value_dim,)](
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        kk.contiguous(),
        a.contiguous(),
        state.contiguous(),
        query_start_loc.contiguous(),
        output,
        final_state,
        H=heads,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_key,
        num_warps=_TRITON_NUM_WARPS,
    )
    return output, final_state


def _error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float]:
    if actual.numel() == 0:
        return 0.0, 0.0, 1.0
    actual_fp32 = actual.to(torch.float32).flatten()
    expected_fp32 = expected.to(torch.float32).flatten()
    error = (actual_fp32 - expected_fp32).abs()
    max_abs_error = error.max().item()
    max_rel_error = (error / expected_fp32.abs().clamp_min(1e-12)).max().item()
    denominator = torch.linalg.vector_norm(actual_fp32) * torch.linalg.vector_norm(
        expected_fp32
    )
    if denominator.item() == 0.0:
        cosine_similarity = float(torch.equal(actual_fp32, expected_fp32))
    else:
        cosine_similarity = (
            torch.dot(actual_fp32, expected_fp32).div(denominator).item()
        )
    return max_abs_error, max_rel_error, cosine_similarity


def _first_mismatch(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> int | None:
    if actual.shape[0] == 0:
        return None
    mismatch = (actual - expected).abs() > atol + rtol * expected.abs()
    mismatch_by_item = mismatch.reshape(actual.shape[0], -1).any(dim=1)
    indices = mismatch_by_item.nonzero()
    return None if indices.numel() == 0 else int(indices[0].item())


@torch.no_grad()
def diagnose_rwkv7_recurrent_scan_packed(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> RWKV7KernelParityReport:
    """Compare packed Torch and Triton scans and locate the first mismatch.

    This diagnostic intentionally runs both implementations and synchronizes
    scalar metrics back to the host. It is for correctness qualification, not
    serving. ``first_output_mismatch_token`` indexes the packed token axis;
    ``first_state_mismatch_sequence`` indexes the final-state batch axis.
    """
    expected_output, expected_state = rwkv7_recurrent_scan_packed(
        r,
        w,
        k,
        v,
        kk,
        a,
        state,
        query_start_loc,
        backend="torch",
    )
    actual_output, actual_state = rwkv7_recurrent_scan_packed(
        r,
        w,
        k,
        v,
        kk,
        a,
        state,
        query_start_loc,
        backend="triton",
    )
    output_metrics = _error_metrics(actual_output, expected_output)
    state_metrics = _error_metrics(actual_state, expected_state)
    first_output_mismatch = _first_mismatch(
        actual_output, expected_output, atol=atol, rtol=rtol
    )
    first_state_mismatch = _first_mismatch(
        actual_state, expected_state, atol=atol, rtol=rtol
    )
    return RWKV7KernelParityReport(
        output_close=first_output_mismatch is None,
        state_close=first_state_mismatch is None,
        first_output_mismatch_token=first_output_mismatch,
        first_state_mismatch_sequence=first_state_mismatch,
        output_max_abs_error=output_metrics[0],
        output_max_rel_error=output_metrics[1],
        output_cosine_similarity=output_metrics[2],
        state_max_abs_error=state_metrics[0],
        state_max_rel_error=state_metrics[1],
        state_cosine_similarity=state_metrics[2],
        atol=atol,
        rtol=rtol,
    )
