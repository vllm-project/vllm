# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enablement gate for the HPC fused gated-MLA GEMM.

``hpc.gated_mla_gemm`` fuses the MLA output gating -- the gate projection, the
sigmoid and the elementwise product -- into a single kernel launch, replacing
the eager ``attn_out * sigmoid(hidden_states @ weight.T)`` sequence.

Kernel constraints:
  - Requires VLLM_ENABLE_HPC_OPS=1
  - Requires the hpc package (.so) built for the current arch
  - Only sm100 / sm103 (compute capability 100, 103)
  - All three operands must be bfloat16 and contiguous
  - The gate width must be a multiple of 256
  - Only elementwise gating (headwise has no fused form)
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.hpc import has_hpc

logger = init_logger(__name__)

# hpc.gated_mla_gemm fuses `attn_out * sigmoid(x @ weight.T)` into one launch.
# It is built for sm100/sm103 and needs all three operands bf16 + contiguous,
# with the gate width a multiple of 256.
_HPC_GATED_GEMM_N_ALIGN = 256
_HPC_GATED_GEMM_CAPABILITIES: frozenset[int] = frozenset({100, 103})


def hpc_gated_mla_supported(
    gating_type: str | None, attn_output_gate: torch.nn.Module | None
) -> bool:
    """Gate for running the MLA output gating through hpc.gated_mla_gemm."""
    if not envs.VLLM_ENABLE_HPC_OPS:
        return False

    if attn_output_gate is None:
        return False

    if not has_hpc():
        logger.warning_once(
            "HPC gated MLA disabled: 'hpc' package is not installed. "
            "Please install the HPC library (.so) to enable the fused kernel."
        )
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        logger.warning_once("HPC gated MLA disabled: only CUDA is supported.")
        return False

    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() not in _HPC_GATED_GEMM_CAPABILITIES:
        logger.warning_once(
            "HPC gated MLA disabled: compute capability %s not in %s.",
            capability,
            _HPC_GATED_GEMM_CAPABILITIES,
        )
        return False

    # Headwise gating broadcasts one scalar per head over v_head_dim; the
    # kernel only implements the elementwise product.
    if gating_type != "elementwise":
        logger.warning_once(
            "HPC gated MLA disabled: gating_type '%s' has no fused form.",
            gating_type,
        )
        return False

    weight = getattr(attn_output_gate, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        logger.warning_once(
            "HPC gated MLA disabled: gate weight dtype is %s, the kernel needs "
            "bfloat16 (keep the layer out of quantization).",
            None if weight is None else weight.dtype,
        )
        return False

    if weight.shape[0] % _HPC_GATED_GEMM_N_ALIGN != 0:
        logger.warning_once(
            "HPC gated MLA disabled: gate width %d is not a multiple of %d.",
            weight.shape[0],
            _HPC_GATED_GEMM_N_ALIGN,
        )
        return False

    logger.info_once("HPC gated MLA enabled by set VLLM_ENABLE_HPC_OPS.")
    return True


def hpc_gated_mla_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    atten_output: torch.Tensor,
) -> torch.Tensor:
    """Fused ``atten_output * sigmoid(x @ weight.T)``.

    Args:
        x: ``[m, k]`` bfloat16, contiguous.
        weight: ``[n, k]`` bfloat16, contiguous and row-major (the kernel
            transposes it). ``n`` must be a multiple of 256.
        atten_output: ``[m, n]`` bfloat16, contiguous.

    Returns:
        ``[m, n]`` bfloat16.
    """
    from hpc.gemm import gated_mla_gemm

    return gated_mla_gemm(x, weight, atten_output)
