# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager sequence parallelism for the dense Qwen3.5 decoder stack.

Between the TP linears, the residual stream (norms and residual adds) runs on
a per-rank token shard. `mark_residual_sp_regions` wraps the boundary linears
so they all-gather / reduce-scatter while SP is active; the decoder layers
then run their plain forward on that shard.
"""

from collections.abc import Callable, Iterable
from typing import Any

from torch import nn

import vllm.envs as envs
from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.common.ops.sp_linear import mark_sp_region

logger = init_logger(__name__)


def use_residual_sequence_parallel(vllm_config: VllmConfig) -> bool:
    """Whether `--enable-residual-sp` applies; unsupported setups fall back to
    plain TP with a warning."""
    parallel_config = vllm_config.parallel_config
    if (
        not parallel_config.enable_residual_sp
        or parallel_config.tensor_parallel_size == 1
        or envs.VLLM_RESIDUAL_SP_MIN_TOKENS <= 0
    ):
        return False
    if vllm_config.model_config.hf_text_config.model_type != "qwen3_5_text":
        reason = "only dense Qwen3.5 is supported"
    elif parallel_config.pipeline_parallel_size > 1:
        reason = "pipeline parallelism is not supported"
    elif vllm_config.compilation_config.mode != CompilationMode.NONE:
        reason = "it requires eager mode (--enforce-eager)"
    elif vllm_config.lora_config is not None:
        # LoRA wrappers bypass the base linear's patched `forward`.
        reason = "LoRA is not supported"
    else:
        return True
    logger.warning_once("Ignoring --enable-residual-sp: %s.", reason)
    return False


def _register_gdn_input_gather(
    linear_attn: nn.Module, sp_num_tokens: Callable[[], int | None]
) -> None:
    """Gather the GDN input once: `in_proj_qkvz` and `in_proj_ba` both read
    it, so no single wrapped linear can end the region."""

    def hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        num_tokens = sp_num_tokens()
        if num_tokens is None:
            return None
        assert not args, "GDN input must be passed as `hidden_states=`"
        kwargs["hidden_states"] = sp_all_gather(kwargs["hidden_states"])[:num_tokens]
        return args, kwargs

    linear_attn.register_forward_pre_hook(hook, with_kwargs=True)


def mark_residual_sp_regions(
    layers: Iterable[nn.Module],
    sp_num_tokens: Callable[[], int | None],
) -> None:
    """Wire the SP regions of a dense Qwen3.5 layer stack.

    Each layer has two regions: [prev down_proj -> attention input] and
    [attention output -> gate_up_proj]. A GDN layer's input feeds both
    `in_proj_qkvz` and `in_proj_ba`, so its region ends at no single linear;
    a pre-hook on `linear_attn` gathers that input once instead.
    """
    prev_down_proj = None
    for layer in layers:
        if layer.layer_type == "linear_attention":
            mark_sp_region(prev_down_proj, None, sp_num_tokens)
            _register_gdn_input_gather(layer.linear_attn, sp_num_tokens)
            attn_out_proj = layer.linear_attn.out_proj
        else:
            mark_sp_region(prev_down_proj, layer.self_attn.qkv_proj, sp_num_tokens)
            attn_out_proj = layer.self_attn.o_proj
        mark_sp_region(attn_out_proj, layer.mlp.gate_up_proj, sp_num_tokens)
        prev_down_proj = layer.mlp.down_proj
    mark_sp_region(prev_down_proj, None, sp_num_tokens)
