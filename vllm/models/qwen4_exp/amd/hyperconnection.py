# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities for the AMD model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This AMD variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in `ops/hc.py`; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape `[..., HC*HS]` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
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

logger = init_logger(__name__)

# Decode graphs pad to 1, 2, 4 or 8 tokens. On MI355X and MI325X the two-stage
# kernels beat the Triton chain up to 4 tokens and lose to it at 8.
_HC_TWO_STAGE_MAX_TOKENS = 4


def _hc_two_stage_combine_and_mix(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down_inject: torch.Tensor,
    w_down_inject_folded: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc_count: int,
    hc_lowrank: int,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The graph is compiled for a symbolic token count, so the choice between
    # the skinny decode kernels and the Triton chain has to be made here.
    if 1 <= residual.shape[0] <= max_tokens:
        return rocm_aiter_ops.hc_two_stage_combine_and_mix(
            residual=residual,
            block_output=block_output,
            injection=injection,
            norm_weight=norm_weight,
            w_down_inject_folded=w_down_inject_folded,
            w_up=w_up,
            hc_count=hc_count,
            hc_lowrank=hc_lowrank,
            eps=eps,
        )

    out, xn = _hc_combine_norm(
        residual, block_output, injection, norm_weight, eps, hc_count
    )
    down = rocm_unquantized_gemm_impl(xn, w_down_inject)
    lowrank = _hc_silu(down[:, :hc_lowrank], hc_count)
    gate = rocm_unquantized_gemm_impl(lowrank, w_up)
    block_input = _hc_gate_mix(xn, gate, hc_count)
    return out, block_input, down[:, hc_lowrank : hc_lowrank + hc_count]


def _hc_two_stage_combine_and_mix_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down_inject: torch.Tensor,
    w_down_inject_folded: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc_count: int,
    hc_lowrank: int,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Both paths return the injection as a column slice of the down output,
    # and Inductor checks the real strides against these.
    n, dim = residual.shape
    down = residual.new_empty((n, w_down_inject.shape[0]))
    return (
        residual.new_empty(residual.shape),
        residual.new_empty((n, dim // hc_count)),
        down[:, hc_lowrank : hc_lowrank + hc_count],
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

    `combine_and_mix()` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    `use_combine=False` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    w_down_inject_folded: torch.Tensor

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_lowrank = config.hc_lowrank
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
        self.pad_size = (-(self.hc_lowrank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.hc_lowrank, self.hc_count]
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
                self.hc_lowrank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.hc_lowrank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

        # The two-stage skinny decode kernels need the norm affine folded into
        # the down+inject weight; process_weights_after_loading fills it.
        # Dispatching on token count would make outputs depend on batch size.
        self.hc_two_stage_max_tokens = 0
        if (
            use_combine
            and not envs.VLLM_BATCH_INVARIANT
            and rocm_aiter_ops.is_hc_two_stage_enabled()
        ):
            down_rows = self.hc_lowrank + self.hc_count + self.pad_size
            self.hc_two_stage_max_tokens = min(
                _HC_TWO_STAGE_MAX_TOKENS,
                rocm_aiter_ops.hc_two_stage_max_tokens(
                    self.hyper_hidden_size, down_rows, self.hc_count, self.hc_lowrank
                ),
            )
            if self.hc_two_stage_max_tokens:
                self.register_buffer(
                    "w_down_inject_folded",
                    torch.empty(
                        down_rows, self.hyper_hidden_size, dtype=config.params_dtype
                    ),
                    persistent=False,
                )
                logger.info_once(
                    "Using AITER FlyDSL two-stage HC combine_and_mix for <= %d tokens.",
                    self.hc_two_stage_max_tokens,
                )
            else:
                logger.warning_once(
                    "AITER FlyDSL two-stage HC combine_and_mix does not support "
                    "hidden size %d with hc_count %d and hc_lowrank %d; using "
                    "the Triton kernels.",
                    self.hyper_hidden_size,
                    self.hc_count,
                    self.hc_lowrank,
                )

    def process_weights_after_loading(self) -> None:
        if self.hc_two_stage_max_tokens:
            w_down_inject = self.input_mix_weight_down_block_inject.weight
            assert isinstance(w_down_inject, torch.Tensor)
            self.w_down_inject_folded.copy_(
                rocm_aiter_ops.hc_fold_norm_weight(
                    w_down_inject, self.hc_norm.weight, self.hc_count
                )
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
            split_sizes = [self.hc_lowrank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lowrank, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lowrank = self.input_mix_weight_down(xn)
            injection = None

        lowrank = hc_silu(lowrank, self.hc_count)
        gate = self.input_mix_weight_up(lowrank)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        `hidden_states` is the multi-stream state from before the pending
        block's mix. Its combine with `block_output` is fused with this
        module's input RMSNorm.
        """
        if self.hc_two_stage_max_tokens:
            return torch.ops.vllm.qwen4_exp_hc_two_stage_combine_and_mix(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.input_mix_weight_down_block_inject.weight,
                self.w_down_inject_folded,
                self.input_mix_weight_up.weight,
                self.config.rms_norm_eps,
                self.hc_count,
                self.hc_lowrank,
                self.hc_two_stage_max_tokens,
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
            split_sizes = [self.hc_lowrank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lowrank, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lowrank = self.input_mix_weight_down(xn)
            injection = None

        lowrank = hc_silu(lowrank, self.hc_count)
        gate = self.input_mix_weight_up(lowrank)  # [M, D]
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
