# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dense layers' MLP (the first `first_k_dense_replace` layers) of up to 8
decode tokens on SM90, on the GEMV kernels of `..dense_mlp`.

Those kernels (FP8 `mma.sync`, `cp.async` and PDL) run on SM90 as they are, on
the stock Hopper weights: the FP8 gate_up `[6144, 4096]` and down `[4096, 3072]`
of a TP4 shard with FP32 128 x 128 block scales. A launch takes up to 8 tokens
(the MMA's N). Above that the stock MLP stays faster: chunks of 8 tokens each
stream the weights again (51 us at 24 tokens on H200, stock 25 us). The output
is TP-partial, as `DenseMlp`'s.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HIDDEN, INTER, LIMIT = 4096, 3072, 10.0
MAX_TOKENS = 8


class HopperDenseMlp:
    """FP8 block-scaled gate_up with the clamped SwiGLU, then the down
    projection of one TP shard of `Glm5NextMLP` (output still TP-partial)."""

    MAX_TOKENS = MAX_TOKENS

    def __init__(self, w13, w13_scale, w2, w2_scale, limit: float, ue8m0: bool):
        from ..dense_mlp import DenseMlp

        self.mlp = DenseMlp(w13, w13_scale, w2, w2_scale, limit, ue8m0)

    @classmethod
    def of(cls, mlp) -> "HopperDenseMlp | None":
        """The kernels of a stock `Glm5NextMLP` after its weights are processed,
        or None (logging why) for any other layout."""
        from vllm.platforms import current_platform
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        gate_up, down = mlp.gate_up_proj, mlp.down_proj
        reasons = []
        if not current_platform.is_device_capability(90):
            reasons.append("a GPU other than SM90")
        if mlp.swiglu_limit != LIMIT:
            reasons.append(f"swiglu_limit={mlp.swiglu_limit} (expects {LIMIT})")
        shapes = ((gate_up, (2 * INTER, HIDDEN)), (down, (HIDDEN, INTER)))
        for linear, shape in shapes:
            weight = linear.weight
            if (
                tuple(weight.shape) != shape
                or weight.dtype != torch.float8_e4m3fn
                or not weight.is_contiguous()
                or getattr(linear, "weight_scale_inv", None) is None
            ):
                reasons.append(f"{linear.prefix} is not a contiguous FP8 {shape}")
        if reasons:
            logger.warning_once(
                "GLM-5.3-Flash Hopper dense MLP disabled: %s", "; ".join(reasons)
            )
            return None
        return cls(
            gate_up.weight,
            gate_up.weight_scale_inv,
            down.weight,
            down.weight_scale_inv,
            mlp.swiglu_limit,
            is_deep_gemm_e8m0_used(),
        )

    def supports(self, tokens: int) -> bool:
        return 0 < tokens <= MAX_TOKENS

    def prepare(self, tokens: list[int]) -> None:
        """Compiles for these token counts before graph capture."""
        self.mlp.prepare([m for m in sorted(set(tokens)) if self.supports(m)])
        torch.accelerator.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """BF16 `[M, 4096]` TP-partial MLP output of the BF16 tokens `x`."""
        assert self.supports(x.shape[0]), x.shape[0]
        return self.mlp(x)
