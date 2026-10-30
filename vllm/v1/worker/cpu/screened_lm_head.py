# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy logits from an INT8 screen of a BF16 lm_head on AMD Zen CPUs.

zentorch_dynamic_qlinear scores every row j as s_j from a per-row INT8 copy W8
of the lm_head W. It quantizes each token x as x8 = round(x / t) * bf16(t) with
t = absmax(x) / 127. With e_j = ||W_j - W8_j||, Rh_j = ||W8_j||,
R_j = max(||W_j||, ||W8_j||), r = ||x - x8|| / ||x|| and gamma_K the fp32
accumulation error over K terms, the BF16 head's logit F_j satisfies
|F_j - s_j| <= beta_j with

    G_j = Rh_j ||x - x8|| + e_j ||x|| + gamma_K R_j ||x|| (2 + r) (1 + 2^-7)
    beta_j = G_j (1 + 2^-7) + 2^-7 (1 + 2^-5) |s_j|

where the |s_j| term covers the BF16 rounding of both outputs. A row with
s_j + beta_j below max_i (s_i - beta_i) is beaten by that row i, so only the
rows above it are recomputed in BF16 and every other row gets -inf. The argmax
matches the full lm_head except on ties within one output ulp, which depend on
summation order. The screen reads half the bytes of the BF16 head.
"""

import numpy as np
import torch

from vllm.model_executor.kernels.linear import (
    ZentorchInt8ScaledMMLinearKernel,
    init_int8_linear_kernel,
)
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.screened_head import ScreenedLMHead

_QUANT_CHUNK = 16384
_OPS = ["zentorch_dynamic_qlinear", "zentorch_bmm", "zentorch_clear_weight_cache"]


def _up(x: torch.Tensor) -> torch.Tensor:
    """Round a non-negative fp32 bound outward."""
    return x * (1 + 2.0**-20)


def _quantize_rows(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row INT8 copy of w, its BF16 scale, and the stacked row norms
    e, Rh and R of the module docstring."""
    w = w.float()
    scale = (w.abs().amax(1, keepdim=True) / 127).bfloat16().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale).clamp_(-127, 127)
    w_hat, w = (q * scale).double(), w.double()
    w_norm, w_hat_norm = w.norm(dim=1), w_hat.norm(dim=1)
    stats = torch.stack(
        [(w - w_hat).norm(dim=1), w_hat_norm, torch.maximum(w_norm, w_hat_norm)]
    )
    return q.to(torch.int8), scale.view(-1).bfloat16(), _up(stats.float())


def _bound(
    x: torch.Tensor, s: torch.Tensor, stats: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, bool]:
    """The module docstring's beta for screen logits s of x, and whether
    ||x|| is finite."""
    xf, xd = x.float(), x.double()
    t = (xf.abs().amax(1, keepdim=True).clamp(min=1e-10) / 127).clamp(min=1e-10)
    x_hat = torch.round(xf / t) * t.bfloat16().float()
    xn = _up(xd.norm(dim=1, keepdim=True).float())
    eta = _up((xd - x_hat.double()).norm(dim=1, keepdim=True).float())
    r = torch.where(xn > 0, eta / xn, torch.zeros_like(xn))
    err, w_hat_norm, w_max_norm = stats.unsqueeze(1)
    beta = w_hat_norm * eta
    beta.addcmul_(err, xn)
    beta.addcmul_(w_max_norm, gamma * (2 + r) * (1 + 2.0**-7) * xn)
    beta.mul_(1 + 2.0**-7).add_(s.abs(), alpha=2.0**-7 * (1 + 2.0**-5))
    beta.mul_(1 + 2.0**-20).add_(1e-30)
    return beta, bool(torch.isfinite(xn).all())


class ZenScreenedLMHead(ScreenedLMHead):
    """ScreenedLMHead for greedy batches on AMD Zen CPUs: an INT8 screen through
    zentorch_dynamic_qlinear and a BF16 recheck of the surviving rows."""

    MAX_TOKENS = 256
    # zentorch_bmm is 30-100x slower on fewer rows from 4 tokens up.
    MIN_RECHECK_ROWS = 2048

    @classmethod
    def unsupported_platform_reason(cls) -> str | None:
        if current_platform.is_zen_cpu() and has_zentorch_op(_OPS):
            return None
        return "an AMD Zen CPU with zentorch"

    def __init__(
        self,
        weight: torch.Tensor,
        vocab_size: int,
        sampler: Sampler,
        max_num_tokens: int,
    ):
        if (
            weight.dim() != 2
            or weight.dtype != torch.bfloat16
            or weight.shape[0] < vocab_size
        ):
            raise ValueError("screened_lm_head requires a BF16 lm_head on Zen CPUs.")
        self.kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=type(self).__name__,
        )
        if not isinstance(self.kernel, ZentorchInt8ScaledMMLinearKernel):
            raise ValueError(
                "screened_lm_head requires the zentorch int8 linear kernel."
            )
        self.weight = weight[:vocab_size]
        self.sampler = sampler
        self.vocab_size, self.hidden_size = self.weight.shape
        self.max_num_tokens = min(max_num_tokens, self.MAX_TOKENS)
        # Past this many surviving rows the recheck costs more than it saves.
        self.max_union = self.vocab_size // 16
        k_u = self.hidden_size * 2.0**-24
        self.gamma = k_u / (1 - k_u)
        self.layer = torch.nn.Module()
        self.layer.weight = torch.nn.Parameter(
            torch.empty_like(self.weight, dtype=torch.int8), requires_grad=False
        )
        self.layer.weight_scale = torch.nn.Parameter(
            torch.empty(self.vocab_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.stats = torch.empty(3, self.vocab_size)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the INT8 copy and its row norms from the lm_head weight."""
        for start in range(0, self.vocab_size, _QUANT_CHUNK):
            rows = slice(start, start + _QUANT_CHUNK)
            q, scale, stats = _quantize_rows(self.weight[rows])
            self.layer.weight.data[rows] = q
            self.layer.weight_scale.data[rows] = scale
            self.stats[:, rows] = stats
        # ZenDNN caches reordered constant weights by address, so an in-place
        # update is invisible to it until the cache is cleared.
        torch.ops.zentorch.zentorch_clear_weight_cache()

    def required_top_k(self, idx_mapping_np: np.ndarray, num_tokens: int) -> int | None:
        temperature = self.sampler.sampling_states.temperature.np[idx_mapping_np]
        if np.any(temperature != 0.0):
            return None
        return super().required_top_k(idx_mapping_np, num_tokens)

    def __call__(self, x: torch.Tensor, top_k: int = 1) -> torch.Tensor | None:
        if top_k != 1 or not 0 < x.shape[0] <= self.max_num_tokens:
            return None
        x = x.contiguous()
        s = self.kernel.apply_weights(self.layer, x).float()
        beta, finite = _bound(x, s, self.stats, self.gamma)
        keep = (s + beta) >= (s - beta).amax(1, keepdim=True)
        cols = keep.any(0).nonzero().flatten()
        n = cols.numel()
        if not finite or not 0 < n <= self.max_union:
            return None
        idx = cols
        if n < self.MIN_RECHECK_ROWS:
            idx = torch.cat([cols, cols[-1:].expand(self.MIN_RECHECK_ROWS - n)])
        # zentorch_bmm does not cache its weight; zentorch_mm and
        # zentorch_linear_unary would, by the address of this temporary.
        exact = torch.ops.zentorch.zentorch_bmm(
            x.unsqueeze(0), self.weight.index_select(0, idx).unsqueeze(0).mT
        )[0, :, :n]
        logits = x.new_full((x.shape[0], self.vocab_size), float("-inf"))
        logits[:, cols] = exact
        return logits
