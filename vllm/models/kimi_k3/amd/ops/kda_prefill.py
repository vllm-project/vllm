# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill backends for ROCm.

The Kimi-K3 KDA layer calls :func:`chunk_kda_prefill`, which either runs the
fused HIP kernels in ``kda_chunk`` or falls back to the vendored Triton chunk
path. The AITER entry points provide an additional fused Conv1D and FlashKDA
path.
"""

from collections.abc import Callable
from typing import Literal

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.models.kimi_k3.amd.ops.kda_chunk import (
    can_use_fused_kda_chunk,
    fused_kda_chunk,
    fused_kda_prologue,
)
from vllm.models.kimi_k3.amd.ops.kda_decode import (
    make_decode_conv1d_weight_loader,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk_kda_with_fused_gate
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

logger = init_logger(__name__)

KDAPrefillBackend = Literal["auto", "triton", "flashkda", "fused"]

# Match vLLM's Kimi-K3 causal-Conv metadata, which is prepared for BLOCK_M=8.
_CAUSAL_CONV1D_BLOCK_M = 8


def chunk_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_fused_chunk: bool = False,
    out: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunk KDA from raw gate and beta projections.

    Args:
        use_fused_chunk: request the two-kernel ROCm path. It is used only when
            every one of its preconditions holds; otherwise the Triton path
            runs unchanged.
        out: buffer the result must land in. Honoured by both backends, so the
            caller can hand in a slice of its own output and skip a copy.
        checkpoint_state: destination for mid-prefill recurrent state
            snapshots, letting a later prefix-cache hit resume from a mamba
            block boundary. See :func:`fused_kda_chunk`.
        checkpoint_offsets: per-sequence token offset to snapshot at, ``0``
            for none.
        checkpoint_state_indices: optional per-sequence destination row.
        state_cache: the paged recurrent state. When given, the fused backend
            reads and writes it in place and neither a gather nor a scatter is
            needed around this call; the returned final state is ``None``.
        state_indices: per-sequence cache row.
        has_initial_state: per-sequence flag; false starts from a zero state.

    Returns:
        The output and, when requested, the final recurrent state.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # The fused prologue folds the q/k L2 norm and the gate activation in, so it
    # needs the raw projections and a bounded gate rather than the pre-normalized
    # tensors the Triton path takes.
    fused = (
        use_fused_chunk
        and use_qk_l2norm_in_kernel
        and cu_seqlens is not None
        and lower_bound is not None
        and g_bias is not None
        and can_use_fused_kda_chunk(k.shape[-1], v.shape[-1], k.dtype, FLA_CHUNK_SIZE)
    )

    if checkpoint_offsets is not None and not fused:
        raise NotImplementedError(
            "The KDA prefill checkpoint export needs kda_prefill_backend=fused for ROCm"
        )
    # Checked here rather than in the backend so both paths reject the same
    # arguments.
    if state_cache is not None:
        if initial_state is not None or output_final_state:
            raise ValueError("state_cache replaces initial_state/output_final_state")
        if state_indices is None or has_initial_state is None:
            raise ValueError("state_cache needs state_indices and has_initial_state")

    # Only the fused walk addresses the paged rows directly. The Triton path
    # gathers the rows it needs and scatters the results back here, so callers
    # pass the cache the same way for either backend.
    scatter_to: torch.Tensor | None = None
    if state_cache is not None and not fused:
        initial_state = gather_initial_states(
            state_cache, state_indices, has_initial_state
        )
        output_final_state = True
        scatter_to, state_cache = state_cache, None

    if fused:
        # Restated for the type checker; `fused` already implies all three.
        assert cu_seqlens is not None and lower_bound is not None
        assert g_bias is not None
        logger.info_once(
            "Kimi-K3 KDA prefill: dispatching the fused ROCm chunk kernel."
        )
        ws = fused_kda_prologue(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=g_bias,
            scale=scale,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return fused_kda_chunk(
            qg=ws["qg"],
            w=ws["w"],
            u=ws["u"],
            kg_t=ws["kg_t"],
            aqk=ws["aqk"],
            decay=ws["decay"],
            # v is dead by this point, so it doubles as the output buffer.
            out=out if out is not None else v,
            scale=scale,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_offsets=chunk_offsets,
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint_offsets,
            checkpoint_state_indices=checkpoint_state_indices,
            state_cache=state_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
        )

    o, final_state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    if out is not None and o.data_ptr() != out.data_ptr():
        out.copy_(o)
        o = out
    if scatter_to is not None:
        assert state_indices is not None
        scatter_to[state_indices.long()] = final_state
        return o, None
    return o, final_state


def resolve_kda_prefill_backend(backend: str) -> KDAPrefillBackend:
    """Resolve the Kimi-K3 ROCm prefill backend."""
    if backend not in ("auto", "triton", "flashkda", "fused"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    if backend == "auto" and bool(rocm_aiter_ops.is_enabled()):
        return "flashkda"
    return backend


def make_kda_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
    decode_conv1d_weight: torch.Tensor | None,
    prefill_conv1d_weight: torch.Tensor,
) -> Callable[..., None]:
    """Load Conv1D weights and stage the BF16 AITER prefill layout."""
    base_loader = make_decode_conv1d_weight_loader(
        dims,
        tp_size,
        tp_rank,
        decode_conv1d_weight,
    )
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        base_loader(param, loaded_weight, loaded_shard_id)
        if param.is_meta:
            return
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        prefill_conv1d_weight[target_start : target_start + shard_size].copy_(
            loaded_shard.squeeze(1)
        )

    return weight_loader


def aiter_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    projection_size: int,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    metadata: object | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run AITER's fused QKV causal Conv1D prefill kernel."""
    from aiter.ops.causal_conv1d_fwd_split_qkv import (
        causal_conv1d_split_qkv_hip_fn,
    )

    return causal_conv1d_split_qkv_hip_fn(
        x=x.transpose(0, 1),
        weight=weight,
        bias=bias,
        conv_states=conv_state,
        query_start_loc=query_start_loc,
        k_dim=projection_size,
        v_dim=projection_size,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        block_m=_CAUSAL_CONV1D_BLOCK_M,
        metadata=metadata,
    )


def aiter_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run AITER FlashKDA with the Kimi-K3 state and gate conventions."""
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    output, final_state = chunk_kimi_delta_attn(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
        chunk_size=None,
        cu_seqlens=cu_seqlens,
    )
    assert final_state is not None
    return output, final_state
