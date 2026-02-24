# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic AnyModel for NAS-optimized heterogeneous architectures.

AnyModel reuses existing decoder layer classes (LlamaDecoderLayer,
Qwen2DecoderLayer, etc.) directly, feeding them a per-layer config
derived from block_configs and patching no-op submodules post-init.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .interfaces import HasNoOps, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# ---------------------------------------------------------------------------
# Block config access helpers
# ---------------------------------------------------------------------------


def _get_block_section(block_config, section: str):
    """Get a section (e.g. 'attention', 'ffn') from a block_config entry.

    Handles both dict and namespace-object representations.
    """
    if isinstance(block_config, dict):
        return block_config.get(section, {})
    return getattr(block_config, section, {})


def _get_attr(obj, key: str, default=None):
    """Get an attribute from either a dict or namespace object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_block_attr(block_config, section: str, key: str, default=None):
    """Shortcut: get a nested attribute from block_config[section][key]."""
    section_data = _get_block_section(block_config, section)
    return _get_attr(section_data, key, default)


# ---------------------------------------------------------------------------
# No-op modules
# ---------------------------------------------------------------------------


class NoOpAttention(nn.Module):
    """No-op replacement for attention. Returns zeros so residual is
    preserved when added back (zeros + residual = residual)."""

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class NoOpMLP(nn.Module):
    """No-op replacement for MLP / MoE block. Returns zeros so residual
    is preserved when added back."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class Same(nn.Module):
    """Identity replacement for layer norms. Must handle vLLM's fused
    RMSNorm calling convention: (hidden_states) or
    (hidden_states, residual) -> (hidden_states, residual)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return hidden_states, residual
        return hidden_states


# ---------------------------------------------------------------------------
# Architecture descriptors
# ---------------------------------------------------------------------------


class AnyModelArchDescriptor(ABC):
    """Thin adapter per base architecture.

    Only ``create_decoder_layer`` must be overridden. The default
    ``create_layer_config`` and ``apply_no_ops`` work for any
    architecture that uses standard config field names and module names.
    """

    @abstractmethod
    def create_decoder_layer(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        per_layer_config,
    ) -> nn.Module: ...

    def create_layer_config(self, global_config, block_config):
        """Return a shallow copy of *global_config* with per-layer
        overrides from *block_config* applied."""
        config = copy.copy(global_config)

        attn = _get_block_section(block_config, "attention")
        ffn = _get_block_section(block_config, "ffn")

        if not _get_attr(attn, "no_op", False):
            kv_heads = _get_attr(attn, "num_key_value_heads")
            if kv_heads is not None:
                config.num_key_value_heads = kv_heads

        intermediate = _get_attr(ffn, "intermediate_size")
        if intermediate is not None:
            config.intermediate_size = intermediate

        hidden_act = _get_attr(ffn, "hidden_act")
        if hidden_act is not None:
            config.hidden_act = hidden_act

        return config

    # Module names to replace for no-ops.  Override in descriptors whose
    # architecture uses different attribute names (e.g. Mixtral).
    _attn_module: str = "self_attn"
    _attn_norm_module: str = "input_layernorm"
    _ffn_module: str = "mlp"
    _ffn_norm_module: str = "post_attention_layernorm"

    def apply_no_ops(self, layer: nn.Module, block_config) -> None:
        """Replace sub-modules with no-ops according to block_config."""
        if _get_block_attr(block_config, "attention", "no_op", False):
            setattr(layer, self._attn_module, NoOpAttention())
            setattr(layer, self._attn_norm_module, Same())
        if _get_block_attr(block_config, "ffn", "no_op", False):
            setattr(layer, self._ffn_module, NoOpMLP())
            setattr(layer, self._ffn_norm_module, Same())


# -- Dense descriptors -------------------------------------------------------


class LlamaArchDescriptor(AnyModelArchDescriptor):
    def create_decoder_layer(self, vllm_config, prefix, per_layer_config):
        from .llama import LlamaDecoderLayer

        return LlamaDecoderLayer(
            vllm_config=vllm_config, prefix=prefix, config=per_layer_config
        )


class Qwen2ArchDescriptor(AnyModelArchDescriptor):
    def create_decoder_layer(self, vllm_config, prefix, per_layer_config):
        from .qwen2 import Qwen2DecoderLayer

        return Qwen2DecoderLayer(
            config=per_layer_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )


# -- MoE descriptors ---------------------------------------------------------


class Qwen2MoeArchDescriptor(AnyModelArchDescriptor):
    def create_decoder_layer(self, vllm_config, prefix, per_layer_config):
        from .qwen2_moe import Qwen2MoeDecoderLayer

        return Qwen2MoeDecoderLayer(
            config=per_layer_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

    def create_layer_config(self, global_config, block_config):
        config = super().create_layer_config(global_config, block_config)
        moe = _get_block_attr(block_config, "ffn", "moe")
        if moe is not None:
            num_experts = _get_attr(moe, "num_local_experts")
            if num_experts is not None:
                config.num_experts = num_experts
            expert_intermediate = _get_attr(moe, "expert_intermediate_size")
            if expert_intermediate is not None:
                config.moe_intermediate_size = expert_intermediate
        return config


class MixtralArchDescriptor(AnyModelArchDescriptor):
    _ffn_module = "block_sparse_moe"

    def create_decoder_layer(self, vllm_config, prefix, per_layer_config):
        from .mixtral import MixtralDecoderLayer

        return MixtralDecoderLayer(
            config=per_layer_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

    def create_layer_config(self, global_config, block_config):
        config = super().create_layer_config(global_config, block_config)
        moe = _get_block_attr(block_config, "ffn", "moe")
        if moe is not None:
            num_experts = _get_attr(moe, "num_local_experts")
            if num_experts is not None:
                config.num_local_experts = num_experts
            expert_intermediate = _get_attr(moe, "expert_intermediate_size")
            if expert_intermediate is not None:
                config.intermediate_size = expert_intermediate
        return config


# -- Registry ----------------------------------------------------------------

DESCRIPTOR_REGISTRY: dict[str, type[AnyModelArchDescriptor]] = {
    "llama": LlamaArchDescriptor,
    "qwen2": Qwen2ArchDescriptor,
    "qwen2_moe": Qwen2MoeArchDescriptor,
    "mixtral": MixtralArchDescriptor,
}


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class AnyModel(nn.Module):
    """Generic transformer container that creates heterogeneous decoder
    layers from ``block_configs`` using the appropriate architecture
    descriptor."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        descriptor: AnyModelArchDescriptor,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self._create_layer(prefix, vllm_config, descriptor),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    @staticmethod
    def _create_layer(
        prefix: str,
        vllm_config: VllmConfig,
        descriptor: AnyModelArchDescriptor,
    ) -> nn.Module:
        layer_idx = extract_layer_index(prefix)
        config = vllm_config.model_config.hf_config
        block_config = config.block_configs[layer_idx]
        per_layer_config = descriptor.create_layer_config(config, block_config)
        layer = descriptor.create_decoder_layer(vllm_config, prefix, per_layer_config)
        descriptor.apply_no_ops(layer, block_config)
        return layer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class AnyModelForCausalLM(nn.Module, SupportsPP, HasNoOps):
    """Top-level causal LM wrapper for NAS-optimized models.

    Auto-detected when ``block_configs`` is present in the HF config
    and ``model_type`` maps to a known architecture descriptor.
    """

    has_noops = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        model_type = getattr(config, "model_type", None)
        descriptor_cls = DESCRIPTOR_REGISTRY.get(model_type)
        if descriptor_cls is None:
            raise ValueError(
                f"No AnyModel descriptor registered for model_type={model_type!r}. "
                f"Supported: {list(DESCRIPTOR_REGISTRY)}"
            )
        descriptor = descriptor_cls()

        self.model = AnyModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            descriptor=descriptor,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = None
        if self.config.tie_word_embeddings:
            skip_prefixes = ["lm_head."]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
