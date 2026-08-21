# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill backend selection for ROCm.

The Kimi-K3 KDA layer calls :func:`chunk_kda_prefill`, which either runs the
fused HIP kernels in ``kda_chunk`` or falls back to the vendored Triton chunk
path.
"""

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.ops.kda_chunk import (
    can_use_fused_kda_chunk,
    fused_kda_chunk,
    fused_kda_prologue,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk_kda_with_fused_gate
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

logger = init_logger(__name__)


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
            "The KDA prefill checkpoint export needs kda_prefill_backend=fused for "
            "ROCm"
        )

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
    return o, final_state
