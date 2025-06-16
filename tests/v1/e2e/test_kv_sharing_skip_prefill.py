# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from collections.abc import Iterable
from typing import Optional, Union

import pytest
import torch
from torch import nn
from transformers import Qwen2Config

from vllm import LLM, SamplingParams
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2 import (Qwen2Attention, Qwen2MLP,
                                              Qwen2Model)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              extract_layer_index,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from ...utils import fork_new_process_for_each_test

START_KV_SHARING_LAYER = 10


class Qwen2DecoderLayerWithKVSharing(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        attn_prefix = f"{prefix}.self_attn"
        layer_idx = extract_layer_index(prefix)
        kv_sharing_target_layer_name = None

        if layer_idx >= START_KV_SHARING_LAYER:
            # re-use KV cache from first 5 layers
            target_layer_idx = layer_idx % 5
            kv_sharing_target_layer_name = f"{attn_prefix}.attn".replace(
                str(layer_idx), str(target_layer_idx))
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=attn_prefix,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )

        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2ModelWithKVSharing(Qwen2Model):

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        decode_indices = get_forward_context().decode_indices
        if decode_indices is None:
            decode_indices = torch.arange(positions.size(0),
                                          device=positions.device)

        # Forward with full inputs up to the first layer that shares KV cache
        for layer in self.layers[self.start_layer:START_KV_SHARING_LAYER]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if decode_indices is not None:
            decode_hidden_states = hidden_states[decode_indices]
            decode_positions = positions[decode_indices]
            decode_residual = (residual[decode_indices]
                               if residual is not None else None)
        else:
            decode_hidden_states = hidden_states
            decode_positions = positions
            decode_residual = residual

        # Optimization: forward with partial inputs only for last N layers
        for layer in self.layers[START_KV_SHARING_LAYER:self.end_layer]:
            decode_hidden_states, decode_residual = layer(
                decode_positions,
                decode_hidden_states,
                decode_residual,
            )

        # Merge results back
        if decode_hidden_states is not None:
            hidden_states[decode_indices] = decode_hidden_states
            if residual is not None:
                residual[decode_indices] = decode_residual

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TestQwen2ForCausalLM(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2ModelWithKVSharing(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            decoder_layer_type=Qwen2DecoderLayerWithKVSharing)
        self.lm_head = self.model.embed_tokens
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


# TODO: make it work with torch.compile
@fork_new_process_for_each_test
@pytest.mark.parametrize("enforce_eager", [True])
def test_kv_sharing_skip_prefill(monkeypatch, enforce_eager):
    prompt = "What is the capital of France?"
    ModelRegistry.register_model("Qwen2ForCausalLM", TestQwen2ForCausalLM)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=40)
    single_prompt = [prompt]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  enforce_eager=enforce_eager)
        responses = llm.generate(single_prompt, sampling_params)
        ref_output = responses[0].outputs[0].text

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        m.setenv("VLLM_V1_KV_SHARING_SKIP_PREFILL", "1")

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  enforce_eager=enforce_eager)
        responses = llm.generate(single_prompt, sampling_params)
        output = responses[0].outputs[0].text
        assert output == ref_output
