# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.l2norm import USE_DEFAULT_FLA_NORM
from vllm.utils.flashinfer import flashinfer_recurrent_kda

logger = init_logger(__name__)


def _validate_gate(lower_bound: float | None, backend: str) -> float:
    if lower_bound is None:
        raise ValueError("FlashInfer KDA prefill requires a bounded gate.")
    if backend == "cudnn" and not -5.0 <= lower_bound < 0:
        raise ValueError(
            "FlashInfer cuDNN KDA prefill requires a bounded gate with "
            "-5 <= lower_bound < 0. Unbounded gates are unsupported."
        )
    return lower_bound


def validate_flashinfer_kda_prefill(
    head_dim: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float | None,
    backend: str = "auto",
) -> None:
    if backend not in ("auto", "cudnn"):
        raise ValueError(f"Unsupported FlashInfer KDA backend: {backend}")
    _validate_gate(lower_bound, backend)
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    if not (
        capability is not None
        and (capability.major, capability.minor) in ((10, 0), (10, 3))
        and head_dim == 128
        and input_dtype == torch.bfloat16
        and (
            state_dtype in (torch.bfloat16, torch.float32)
            if backend == "cudnn"
            else state_dtype == torch.bfloat16
        )
    ):
        raise RuntimeError(
            f"FlashInfer KDA prefill (backend={backend}) requires SM100 or SM103, "
            "bfloat16 input, and head_dim=128. Backend auto requires bfloat16 "
            "state; cudnn also supports float32 state."
        )
    logger.info_once("Using FlashInfer KDA prefill with backend=%s.", backend)


def flashinfer_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    backend: str = "auto",
    seq_order: torch.Tensor | None = None,
    prefill_workspace: object | None = None,
    *,
    inplace_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allow cuDNN to overwrite disposable gathered state when requested."""
    if backend not in ("auto", "cudnn"):
        raise ValueError(f"Unsupported FlashInfer KDA backend: {backend}")
    lower_bound = _validate_gate(lower_bound, backend)
    if backend == "cudnn" and envs.VLLM_BATCH_INVARIANT:
        raise NotImplementedError(
            "FlashInfer's KDA dispatcher does not expose batch_invariant."
        )
    normalize_in_kernel = backend != "cudnn" or not USE_DEFAULT_FLA_NORM
    if backend == "cudnn":
        if not normalize_in_kernel:
            q, k = l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
        final_state = initial_state if inplace_state else initial_state.clone()
    else:
        final_state = initial_state.contiguous()
        # FI's prefill packing does not cover one-token dispatch.
        q, k = q.contiguous(), k.contiguous()
        v, raw_g, raw_beta = v.contiguous(), raw_g.contiguous(), raw_beta.contiguous()
        if (
            cu_seqlens.is_inference()
            or cu_seqlens.dtype != torch.int64
            or not cu_seqlens.is_contiguous()
        ):
            with torch.inference_mode(False):
                cu_seqlens = cu_seqlens.to(
                    torch.int64, copy=cu_seqlens.is_inference()
                ).contiguous()
    A_log = A_log.reshape(-1).contiguous()
    dt_bias = (
        dt_bias.reshape(-1, q.shape[-1]) if backend == "cudnn" else dt_bias.reshape(-1)
    ).contiguous()
    output, _ = flashinfer_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=final_state,
        output_final_state=backend == "cudnn",
        use_qk_l2norm_in_kernel=normalize_in_kernel,
        qk_l2norm_additive_epsilon=1e-6
        if backend == "cudnn" and normalize_in_kernel
        else None,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens.contiguous(),
        beta_is_logit=True,
        output=out,
        backend=backend,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
    )
    return output, final_state
