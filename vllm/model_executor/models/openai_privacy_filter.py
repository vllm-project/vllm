# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only OpenAI Privacy Filter model.

gpt-oss reused as a bidirectional encoder for token classification: every
layer runs non-causal attention with a banded ±sliding_window mask, and
the LM head is replaced with a 33-class BIOES score head.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors

from .gpt_oss import GptOssForCausalLM, GptOssModel, OAIAttention, TransformerBlock
from .interfaces_base import attn_type, default_pooling_type
from .utils import AutoWeightsLoader, maybe_prefix


class OpenAIPrivacyFilterAttention(OAIAttention):
    # Privacy-filter uses GPT-J style RoPE (interleaved pairs), not NeoX.
    rope_is_neox_style = False

    def _build_attention(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> EncoderOnlyAttention:
        # HF stores sliding_window+1 so each token attends to ±W neighbors;
        # the encoder-only path applies this as a symmetric (W-1, W-1) mask.
        return EncoderOnlyAttention(
            num_heads=self.num_local_attention_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window + 1,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )


class OpenAIPrivacyFilterDecoderLayer(TransformerBlock):
    attention_cls = OpenAIPrivacyFilterAttention


class OpenAIPrivacyFilterModel(GptOssModel):
    block_cls = OpenAIPrivacyFilterDecoderLayer


def _interleave_gate_up_concat_to_pairs(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    # HF gate_up_proj is concat [gate | up]; swigluoai_and_mul wants
    # gate/up interleaved. MXFP4/quark suffixes are already interleaved.
    for name, weight in weights:
        if name.endswith(".gate_up_proj") or name.endswith(".gate_up_proj_bias"):
            *lead, two_i = weight.shape
            i = two_i // 2
            weight = (
                torch.stack([weight[..., :i], weight[..., i:]], dim=-1)
                .reshape(*lead, two_i)
                .contiguous()
            )
        yield name, weight


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class OpenAIPrivacyFilterForTokenClassification(nn.Module):
    is_pooling_model = True
    hf_to_vllm_mapper = GptOssForCausalLM.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.head_dtype = vllm_config.model_config.head_dtype
        self.num_labels = config.num_labels

        self.model = OpenAIPrivacyFilterModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.score = nn.Linear(
            config.hidden_size, config.num_labels, dtype=self.head_dtype
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_token_classify(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = hidden_states.to(self.head_dtype)
        return self.score(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            _interleave_gate_up_concat_to_pairs(weights),
            mapper=self.hf_to_vllm_mapper,
        )
