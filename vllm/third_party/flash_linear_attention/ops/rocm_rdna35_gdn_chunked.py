# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated delta net prefill on RDNA3.5, as a single HIP kernel.

Replaces the cumsum, WY transform, state recurrence and output kernels with
one launch of ``torch.ops._rocm_C.gdn_chunked``.
"""

from __future__ import annotations

import functools

import torch

import vllm.envs as envs


@functools.cache
def _available() -> bool:
    """Whether the op is in this build and the device can run it.

    The device test is the RDNA3.5 family rather than a broader question such
    as "is this Navi": gfx1100, gfx1103 and gfx12xx would answer yes, and the
    kernel's wave32 block layout does not hold on them.
    """
    if not envs.VLLM_GDN_HIP:
        return False
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "gdn_chunked"
    ):
        return False
    # Imported here rather than at module scope: this module is reached from
    # chunk.py on every platform, and importing vllm.platforms.rocm syncs the
    # HIP/CUDA visibility env vars and resolves the GCN arch at import time,
    # which on a non-ROCm build falls back to torch.cuda and initialises CUDA.
    from vllm.platforms.rocm import on_gfx115x

    return on_gfx115x()


def is_hip_gdn_supported(
    query: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Whether :func:`chunk_gdn_hip_fwd` can serve this call.

    Mirrors every shape and dtype the kernel asserts on, so an unsupported
    call falls back to the Triton kernels instead of raising.
    """
    if cu_seqlens is None or query.shape[0] != 1:
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if value.dtype != query.dtype:
        return False
    if query.shape[-1] != 128 or value.shape[-1] != 128:
        return False
    num_key_heads, num_value_heads = query.shape[-2], value.shape[-2]
    if num_value_heads % num_key_heads != 0:
        return False
    return _available()


def chunk_gdn_hip_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the gated delta rule and return ``(output, final_state)``.

    ``g`` is the raw per-token log decay; the kernel takes the cumsum itself,
    so this must be called before any chunk-local cumsum.
    """
    head_key_dim = k.shape[-1]
    num_value_heads, head_value_dim = v.shape[-2], v.shape[-1]
    num_seqs = len(cu_seqlens) - 1

    if core_attn_out is not None:
        output = core_attn_out[: v.numel()].view(*v.shape)
    else:
        output = torch.empty_like(v)
    final_state = q.new_empty(
        num_seqs, num_value_heads, head_value_dim, head_key_dim, dtype=torch.float32
    )

    torch.ops._rocm_C.gdn_chunked(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        g.squeeze(0).float(),
        beta.squeeze(0).float(),
        None if initial_state is None else initial_state.to(torch.float32).contiguous(),
        cu_seqlens.to(torch.int32),
        output.squeeze(0),
        final_state,
        float(scale),
    )
    return output, final_state


if hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "gdn_chunked"):
    from torch.library import register_fake

    @register_fake("_rocm_C::gdn_chunked")
    def _gdn_chunked_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        final_state: torch.Tensor,
        scale: float,
    ) -> None:
        return
