# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x text model for CPU."""

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xAttention as _BaseMuseSpark1xAttention,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xDecoderLayer as _BaseMuseSpark1xDecoderLayer,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xForCausalLM as _BaseMuseSpark1xForCausalLM,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xModel as _BaseMuseSpark1xModel,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xMoE as _BaseMuseSpark1xMoE,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xRMSNorm as _BaseMuseSpark1xRMSNorm,
)

from .muse_moe_runner import CPUMuseSparkRoutedExperts, MuseMoERunnerCPU


def _cpu_supports_fused_kernel(self: _BaseMuseSpark1xRMSNorm, x: torch.Tensor) -> bool:
    """MuseSpark1xRMSNorm's fused kernel is a GPU-only Triton kernel; CPU
    must always fall back to forward_native. Every call site in
    nvidia/model.py constructs MuseSpark1xRMSNorm/WithInputScale directly
    (no class-attribute hook to override), so this patches the shared base
    class's method instead of duplicating every construction site.
    """
    return False


_BaseMuseSpark1xRMSNorm.supports_fused_kernel = _cpu_supports_fused_kernel


_base_attention_init = _BaseMuseSpark1xAttention.__init__


def _cpu_attention_init(self: _BaseMuseSpark1xAttention, *args, **kwargs) -> None:
    """`fuse_o_norm_gate` also reaches a GPU-only Triton kernel
    (`rms_norm_gated`), independent of MuseSpark1xRMSNorm's own fused path
    and with no device gate of its own, so it needs the same CPU override.
    Forcing it off falls back to the mathematically equivalent o_norm +
    sigmoid-gate elementwise path.
    """
    _base_attention_init(self, *args, **kwargs)
    self.fuse_o_norm_gate = False
    self.cute_o_norm_gate = False


_BaseMuseSpark1xAttention.__init__ = _cpu_attention_init


def make_cpu_weights_mapper(base: WeightsMapper) -> WeightsMapper:
    """Keep per-expert gate/up projections independent for the GPTQ loader."""
    # LlamaModel's nested loader normally packs dense gate/up projections
    # into a single stacked ".gate_up_proj" name. The GPTQ-int4 checkpoint
    # stores each expert projection independently, and RoutedExperts must
    # see those original names to place every expert/shard.
    stacked_mapping = {
        source: target
        for source, target in base.orig_to_new_stacked.items()
        if source not in (".gate_proj", ".up_proj")
    }
    return WeightsMapper(
        orig_to_new_renaming=base.orig_to_new_renaming,
        orig_to_new_regex=base.orig_to_new_regex,
        orig_to_new_substr=base.orig_to_new_substr,
        orig_to_new_stacked=stacked_mapping,
        orig_to_new_prefix=base.orig_to_new_prefix,
        orig_to_new_suffix=base.orig_to_new_suffix,
    )


class MuseSpark1xMoE(_BaseMuseSpark1xMoE):
    """MuseSpark MoE wired to the CPU post-expert-norm implementation."""

    post_norm_routed_experts_cls = CPUMuseSparkRoutedExperts
    post_norm_runner_cls = MuseMoERunnerCPU

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config, layer_idx, prefix)
        # Unlike the NVFP4/Quark checkpoints, this GPTQ-int4 AutoRound
        # checkpoint quantizes the pooler/unpooler like any other
        # `model.language_model.layers.*` linear, so they must be rebuilt
        # with the real quant_config instead of the base class's hardcoded
        # `quant_config=None`.
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        hidden_size = config.moe_hidden_size
        if self.pooler is not None:
            self.pooler = ReplicatedLinear(
                config.hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.pooler",
            )
        if self.unpooler is not None:
            self.unpooler = ReplicatedLinear(
                hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.unpooler",
            )


class MuseSpark1xDecoderLayer(_BaseMuseSpark1xDecoderLayer):
    """MuseSpark decoder layer using the CPU MoE implementation."""

    moe_cls = MuseSpark1xMoE


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class MuseSpark1xModel(_BaseMuseSpark1xModel):
    """MuseSpark decoder whose nested loader preserves expert projections.

    Unlike the nvidia/amd base class, the CPU backend doesn't run under
    breakable cudagraphs, so it re-enables torch.compile for the decoder
    backbone here instead of inheriting @ignore_torch_compile.
    """

    hf_to_vllm_mapper = make_cpu_weights_mapper(_BaseMuseSpark1xModel.hf_to_vllm_mapper)


class MuseSpark1xForCausalLM(_BaseMuseSpark1xForCausalLM):
    """MuseSpark1x causal language model for CPU."""

    model_cls = MuseSpark1xModel
    decoder_layer_cls = MuseSpark1xDecoderLayer
    supported_quantization_configs = (INCConfig,)
    supported_quantization_description = "INC/AutoRound GPTQ-int4"
    hf_to_vllm_mapper = make_cpu_weights_mapper(
        _BaseMuseSpark1xForCausalLM.hf_to_vllm_mapper
    )


__all__ = ["MuseSpark1xForCausalLM", "MuseSpark1xModel"]
