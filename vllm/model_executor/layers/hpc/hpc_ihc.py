# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPC fused iHC (independent Hyper-Connections) kernels for HY V4.

Replaces the eager HYV4HCPreLayer / HYV4HCPostLayer / HYV4HCHeadLayer bodies
with single-kernel HPC implementations:

  pre  : rms square sum + w projection + sigmoid gates + hc-dim weighted sum
  post : y[n, i, :] = H_post[n, i] * x[n, :] + residual[n, i, :]
  head : same as pre but hc_mult projection rows and no H_post output

The eager path issues 20 / 5 / 15 kernels for pre / post / head; each HPC op is
a single kernel.

Constraints:
  - Requires VLLM_ENABLE_HPC_OPS=1
  - Requires the hpc package (.so) built for the current arch
  - Only sm100 / sm103 (compute capability 100, 103)
  - Only hc_mult == 4 and hidden_size in {4096, 6144}
  - x / residual must be bfloat16; hc_fn weights stay float32 (the checkpoint
    keeps them out of fp8 quantization via modules_to_not_convert)

The fused modules own no parameters: their forwards read the weights of the
eager layer they replace directly via the captured owner reference.

NOTE: The reference implementation also offers a cross-layer post+pre fusion
(``HpcIHCPostPre``) that folds one segment's post into the next segment's pre.
It requires reworking the decoder-layer forward scheduling and is not ported
here. TODO: add it once the decoder dataflow is restructured for it.
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.hpc.hpc_module import HpcModule
from vllm.utils.hpc import has_hpc

logger = init_logger(__name__)

# Shapes the HPC kernels are instantiated for. Other shapes launch nothing,
# so gate on this rather than relying on a silent no-op.
_SUPPORTED_HIDDEN_SIZES: frozenset[int] = frozenset({4096, 6144})
_SUPPORTED_HC_MULTS: frozenset[int] = frozenset({4})
_SUPPORTED_CAPABILITIES: frozenset[int] = frozenset({100, 103})


def _ihc_supported(hc_mult: int, hidden_size: int) -> bool:
    """Shared gate for all three iHC ops."""
    if not envs.VLLM_ENABLE_HPC_OPS:
        return False

    if not has_hpc():
        logger.warning_once(
            "HPC iHC disabled: 'hpc' package is not installed. "
            "Please install the HPC library (.so) to enable fused kernels."
        )
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        logger.warning_once("HPC iHC disabled: only CUDA is supported.")
        return False

    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() not in _SUPPORTED_CAPABILITIES:
        logger.warning_once(
            "HPC iHC disabled: compute capability %s not in %s.",
            capability,
            _SUPPORTED_CAPABILITIES,
        )
        return False

    if hc_mult not in _SUPPORTED_HC_MULTS:
        logger.warning_once(
            "HPC iHC disabled: hc_mult=%d not in %s.", hc_mult, _SUPPORTED_HC_MULTS
        )
        return False

    if hidden_size not in _SUPPORTED_HIDDEN_SIZES:
        logger.warning_once(
            "HPC iHC disabled: hidden_size=%d not in %s.",
            hidden_size,
            _SUPPORTED_HIDDEN_SIZES,
        )
        return False

    if envs.VLLM_BATCH_INVARIANT:
        # Batch-invariant mode relies on torch's deterministic reductions; the
        # HPC kernel reduces in a different order, so keep the two regimes
        # apart.
        logger.warning_once(
            "HPC iHC disabled: not supported under VLLM_BATCH_INVARIANT."
        )
        return False

    logger.info_once("HPC iHC enabled by set VLLM_ENABLE_HPC_OPS.")
    return True


class HpcIHCPre(HpcModule):
    """Fused iHC pre block.

    Computes, in one kernel:
        x_flat = x.flatten(1)
        r      = rsqrt(x_flat.square().mean(-1) + rms_norm_eps)
        mixes  = (x_flat @ w.T) * r
        H_pre  = sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
        H_post = magnitude * sigmoid(
                     mixes[:, hc:] * hc_scale[1] + hc_base[hc:]) + hc_eps
        y      = sum_i H_pre[:, i] * x[:, i, :]

    Args:
        hc_mult: HC expand ratio.
        hidden_size: Model hidden dimension.
        magnitude: H_post multiplier (config.hc_magnitude).
        hc_eps: Epsilon added to both gates (config.hc_eps).
        norm_eps: Epsilon inside the rsqrt (config.rms_norm_eps).
        fallback_op: The eager HYV4HCPreLayer to source weights from.
        norm_owner: Optional RMSNorm that immediately follows this pre block.
            When given, its weight/eps are folded into the kernel so a single
            launch covers pre + RMSNorm (the caller then skips the separate
            layernorm). None keeps the original pre-only behaviour.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        magnitude: float,
        hc_eps: float,
        norm_eps: float,
        fallback_op: torch.nn.Module,
        norm_owner: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.magnitude = magnitude
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps
        # Stash the owner outside nn.Module's attribute machinery: callers pass
        # their own `self` here, and registering that as a submodule would make
        # the module tree cyclic (state_dict() then hits RecursionError).
        object.__setattr__(self, "_fallback_op", fallback_op)
        # Same anti-cycle idiom for the fused RMSNorm owner (a sibling module).
        object.__setattr__(self, "_norm_owner", norm_owner)

    @classmethod
    def support(cls, hc_mult: int, hidden_size: int) -> bool:
        return _ihc_supported(hc_mult, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import hpc

        owner = self._fallback_op
        norm_owner = self._norm_owner
        return hpc.fuse_ihc_pre(
            x,
            owner.hc_fn.weight,
            owner.hc_scale,
            owner.hc_base,
            self.norm_eps,
            self.hc_eps,
            self.magnitude,
            norm_owner.weight if norm_owner is not None else None,
            norm_owner.variance_epsilon if norm_owner is not None else 0.0,
        )


class HpcIHCPost(HpcModule):
    """Fused iHC post block: H_post gating plus multi-channel residual add.

    Unlike mHC there is no comb matrix, so each output channel only needs its
    own residual channel and the whole thing is one fused multiply-add per
    element.
    """

    def __init__(self, hc_mult: int, hidden_size: int) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size

    @classmethod
    def support(cls, hc_mult: int, hidden_size: int) -> bool:
        return _ihc_supported(hc_mult, hidden_size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, H_post: torch.Tensor
    ) -> torch.Tensor:
        import hpc

        return hpc.fuse_ihc_post(x, residual, H_post)


class HpcIHCHead(HpcModule):
    """Fused iHC head block: merge the hc channels back into one hidden state.

    Same structure as HpcIHCPre but the projection emits only hc_mult gate
    logits and there is no H_post output. Called once per forward, after the
    last decoder layer.
    """

    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_eps: float,
        norm_eps: float,
        fallback_op: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps
        # See HpcIHCPre: keep the owner out of the module tree.
        object.__setattr__(self, "_fallback_op", fallback_op)

    @classmethod
    def support(cls, hc_mult: int, hidden_size: int) -> bool:
        return _ihc_supported(hc_mult, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import hpc

        owner = self._fallback_op
        return hpc.fuse_ihc_head(
            x,
            owner.hc_head_fn.weight,
            owner.hc_head_scale,
            owner.hc_head_base,
            self.norm_eps,
            self.hc_eps,
        )
