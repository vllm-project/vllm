# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities for the AMD model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This AMD variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import direct_register_custom_op

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    _hc_combine_norm,
    _hc_gate_mix,
    _hc_silu,
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

# AITER PR #5913's FlyDSL two-stage combine_and_mix, off by default. It only
# runs at or below _TWO_STAGE_MAX_M tokens, where its skinny decode path beats
# the Triton chain; above that the op runs the same kernels as the default
# path. The choice is made inside the op because the compiled graph has a
# symbolic token count.
logger = init_logger(__name__)

_USE_TWO_STAGE = os.getenv("VLLM_ROCM_HC_FLYDSL_TWO_STAGE", "0") == "1"
_TWO_STAGE_MAX_M = int(os.getenv("VLLM_ROCM_HC_FLYDSL_MAX_M", "8"))

_two_stage_folded: dict[tuple[int, int], torch.Tensor] = {}


def _two_stage_folded_weight(
    w_down_inject: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_count: int,
    lora_rank: int,
) -> torch.Tensor:
    # Built on the first call, which is the eager profiling run, so the fold
    # never happens inside a CUDA graph capture.
    key = (w_down_inject.data_ptr(), norm_weight.data_ptr())
    folded = _two_stage_folded.get(key)
    if folded is None:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
            fold_norm_weight,
        )

        # Folded at the module's own 16-row padding rather than AITER's 64: only
        # the skinny decode path runs here, and it then writes the injection in
        # the same layout as the down-GEMM output of the fallback.
        folded = fold_norm_weight(w_down_inject, norm_weight, hc_count)
        _two_stage_folded[key] = folded
        logger.info_once(
            "Using AITER FlyDSL two-stage HC combine_and_mix for <= %d tokens.",
            _TWO_STAGE_MAX_M,
        )
    return folded


_two_stage_norm_f32: dict[int, torch.Tensor] = {}


def _two_stage_norm_weight(norm_weight: torch.Tensor) -> torch.Tensor:
    # Passed as f32 so the kernel skips a per-call cast launch.
    w = _two_stage_norm_f32.get(norm_weight.data_ptr())
    if w is None:
        w = norm_weight.reshape(-1).float().contiguous()
        _two_stage_norm_f32[norm_weight.data_ptr()] = w
    return w


def _hc_two_stage_combine_and_mix(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down_inject: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    folded = _two_stage_folded_weight(w_down_inject, norm_weight, hc_count, lora_rank)
    norm_f32 = _two_stage_norm_weight(norm_weight)
    if 1 <= residual.shape[0] <= max_m:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
            flydsl_gr_two_stage_combine_and_mix,
        )

        # Inductor checks custom-op output strides against the fake. With the
        # 16-row folded weight the skinny path returns the injection as a column
        # slice of a [tokens, 336] buffer, the same layout as the fallback's.
        return flydsl_gr_two_stage_combine_and_mix(
            residual,
            block_output,
            injection_logits,
            norm_f32,
            w_down_inject[:lora_rank],
            w_up,
            w_down_inject[lora_rank : lora_rank + hc_count],
            hc_count,
            eps,
            w_down_merged=folded,
            fold_w=True,
        )

    out, xn = _hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, eps, hc_count
    )
    down = rocm_unquantized_gemm_impl(xn, w_down_inject)
    lora = _hc_silu(down[:, :lora_rank], hc_count)
    gate = rocm_unquantized_gemm_impl(lora, w_up)
    block_input = _hc_gate_mix(xn, gate, hc_count)
    return out, block_input, down[:, lora_rank : lora_rank + hc_count]


def _hc_two_stage_combine_and_mix_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down_inject: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = residual.shape
    down = residual.new_empty((n, w_down_inject.shape[0]))
    return (
        residual.new_empty(residual.shape),
        residual.new_empty((n, dim // hc_count)),
        down[:, lora_rank : lora_rank + hc_count],
    )


direct_register_custom_op(
    op_name="qwen4_exp_hc_two_stage_combine_and_mix",
    op_func=_hc_two_stage_combine_and_mix,
    fake_impl=_hc_two_stage_combine_and_mix_fake,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows for
        # alignment and efficient backend dispatch.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        if _USE_TWO_STAGE and self.use_combine:
            return torch.ops.vllm.qwen4_exp_hc_two_stage_combine_and_mix(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.input_mix_weight_down_block_inject.weight,
                self.input_mix_weight_up.weight,
                self.config.rms_norm_eps,
                self.hc_count,
                self.lora_rank,
                _TWO_STAGE_MAX_M,
            )

        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
