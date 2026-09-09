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


def _fused_prefill_unavailable_reason(
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
    lower_bound: float | None,
    g_bias: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
) -> str | None:
    """Why the fused kernel cannot run, or None if it can."""
    if not use_qk_l2norm_in_kernel:
        return "use_qk_l2norm_in_kernel=False"
    if cu_seqlens is None:
        return "cu_seqlens is None"
    if lower_bound is None:
        return "lower_bound is None (fused prologue needs a bounded gate)"
    if g_bias is None:
        return "g_bias is None"
    if not can_use_fused_kda_chunk(k.shape[-1], v.shape[-1], k.dtype, FLA_CHUNK_SIZE):
        return (
            "fused KDA chunk requires gfx950, bf16, head_dim=128 and "
            f"chunk_size={FLA_CHUNK_SIZE} (got dtype={k.dtype}, "
            f"k_dim={k.shape[-1]}, v_dim={v.shape[-1]})"
        )
    return None


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
    require_fused_chunk: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunk KDA from raw gate and beta projections.

    Args:
        use_fused_chunk: request the two-kernel ROCm path when its
            preconditions hold.
        require_fused_chunk: if True (explicit ``--kda-prefill-backend=hipkda``),
            raise when the HIP path cannot run rather than falling back.
        out: buffer the result must land in. Honoured by both backends, so the
            caller can hand in a slice of its own output and skip a copy.

    Returns:
        The output and, when requested, the final recurrent state.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # The fused prologue requires at least one chunk. A token-less batch has
    # nothing to compute; keep the initial recurrent state (or zeros).
    if q.shape[1] == 0:
        o = out if out is not None else v
        if output_final_state:
            n = 0 if cu_seqlens is None else max(cu_seqlens.numel() - 1, 0)
            if initial_state is not None:
                final_state = initial_state.clone()
            else:
                final_state = torch.zeros(
                    n,
                    k.shape[2],
                    v.shape[-1],
                    k.shape[-1],
                    dtype=torch.float32,
                    device=k.device,
                )
            return o, final_state
        return o, None

    # The fused prologue folds the q/k L2 norm and the gate activation in, so it
    # needs the raw projections and a bounded gate rather than the pre-normalized
    # tensors the Triton path takes.
    fused_reason = None
    fused = False
    if use_fused_chunk:
        fused_reason = _fused_prefill_unavailable_reason(
            use_qk_l2norm_in_kernel, cu_seqlens, lower_bound, g_bias, k, v
        )
        fused = fused_reason is None
        if not fused:
            msg = (
                f"Fused KDA chunk prefill was requested but cannot run: {fused_reason}."
            )
            if require_fused_chunk:
                raise RuntimeError(msg)
            logger.warning_once("%s Falling back to Triton.", msg)

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
