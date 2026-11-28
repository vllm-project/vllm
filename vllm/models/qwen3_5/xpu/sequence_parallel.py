# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from typing import Any

from torch import nn

import vllm.envs as envs
from vllm.config import CompilationMode, VllmConfig
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
)
from vllm.models.common.ops.sp_linear import (
    BF16SPLinearKernel,
    SPLinearKernel,
    wrap_sp_linear,
)


class XPUBF16SPLinearKernel(BF16SPLinearKernel):
    # B60 TP4, Qwen3.5-27B: the fused all-gather wins from 512 tokens per
    # rank, while the fused GEMM + reduce-scatter is slower at every size.
    all_gather_min_tokens = 512
    reduce_scatter_min_tokens = sys.maxsize


_SP_LINEAR_KERNELS: tuple[type[SPLinearKernel], ...] = (
    XPUBF16SPLinearKernel,
    SPLinearKernel,
)


def use_residual_sequence_parallel(vllm_config: VllmConfig) -> bool:
    """Whether dense Qwen3.5 may shard its residual stream; other setups run
    plain TP."""
    parallel_config = vllm_config.parallel_config
    speculative_config = vllm_config.speculative_config
    return (
        envs.VLLM_RESIDUAL_SP_MIN_TOKENS > 0
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and vllm_config.model_config.hf_text_config.model_type == "qwen3_5_text"
        and vllm_config.compilation_config.mode == CompilationMode.NONE
        # LoRA wrappers bypass the base linear's patched `forward`.
        and vllm_config.lora_config is None
        and (
            speculative_config is None or not speculative_config.uses_aux_hidden_states
        )
        and not envs.VLLM_BATCH_INVARIANT
    )


class ResidualSPModelMixin:
    """Residual SP state for a dense Qwen3.5 decoder stack. Call
    `_init_residual_sp` after `make_layers`, and `_update_residual_sp` at the
    start of every forward."""

    layers: nn.ModuleList
    start_layer: int
    end_layer: int

    def _init_residual_sp(self, vllm_config: VllmConfig) -> None:
        self.use_residual_sequence_parallel = use_residual_sequence_parallel(
            vllm_config
        )
        self.residual_sp_min_tokens = envs.VLLM_RESIDUAL_SP_MIN_TOKENS
        # Token count of the in-flight forward while it runs residual SP.
        self.residual_sp_num_tokens: int | None = None
        if self.use_residual_sequence_parallel:
            self._wrap_sp_linears()

    def _wrap_sp_linears(self) -> None:
        """Wrap each layer's TP linears so the norms and residual adds between
        them run on a token shard. A GDN layer's input feeds both
        `in_proj_qkvz` and `in_proj_ba`, so a pre-hook on `linear_attn`
        gathers it once instead of a wrapped linear."""
        sp_num_tokens = lambda: self.residual_sp_num_tokens  # noqa: E731
        for layer in self.layers[self.start_layer : self.end_layer]:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.register_forward_pre_hook(
                    self._gather_gdn_input, with_kwargs=True
                )
                attn_linears = [layer.linear_attn.out_proj]
            else:
                attn_linears = [layer.self_attn.qkv_proj, layer.self_attn.o_proj]
            for linear in (*attn_linears, layer.mlp.gate_up_proj, layer.mlp.down_proj):
                wrap_sp_linear(linear, sp_num_tokens, _SP_LINEAR_KERNELS)

    def _gather_gdn_input(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        num_tokens = self.residual_sp_num_tokens
        if num_tokens is None:
            return None
        assert not args, "GDN input must be passed as `hidden_states=`"
        kwargs["hidden_states"] = sp_all_gather(kwargs["hidden_states"])[:num_tokens]
        return args, kwargs

    def _update_residual_sp(self, num_tokens: int) -> None:
        """Decide whether this forward runs residual SP."""
        active = is_sequence_parallel_active(
            self.use_residual_sequence_parallel,
            num_tokens,
            self.residual_sp_min_tokens,
        )
        self.residual_sp_num_tokens = num_tokens if active else None
