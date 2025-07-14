# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random
from collections.abc import Iterable
from typing import Optional, Union

import pytest
import torch
from torch import nn
from transformers import Qwen2Config

from vllm import LLM, SamplingParams
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import (ignore_torch_compile,
                                         support_torch_compile)
from vllm.config import (CacheConfig, CompilationConfig, CompilationLevel,
                         VllmConfig)
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
            target_layer_idx = START_KV_SHARING_LAYER - 1
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


@support_torch_compile
class FirstLayerGroup(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layers: list[nn.Module],
    ):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        return hidden_states, residual


@support_torch_compile
class SecondLayerGroup(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layers: list[nn.Module],
    ):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ):
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        return hidden_states, residual


@ignore_torch_compile
class Qwen2ModelWithKVSharing(Qwen2Model):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[
                     nn.Module] = Qwen2DecoderLayerWithKVSharing):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )

        self.vllm_config = vllm_config

        with set_model_tag("first_layer_group"):
            self.first_layer_group = FirstLayerGroup(
                vllm_config=vllm_config,
                prefix=f"{prefix}.first_layer_group",
                layers=self.layers[self.start_layer:START_KV_SHARING_LAYER],
            )

        with set_model_tag("second_layer_group"):
            self.second_layer_group = SecondLayerGroup(
                vllm_config=vllm_config,
                prefix=f"{prefix}.second_layer_group",
                layers=self.layers[START_KV_SHARING_LAYER:self.end_layer],
            )

        # Pre-allocate static buffers for CUDA graph
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        device = next(self.parameters()).device
        hidden_size = vllm_config.model_config.get_hidden_size()
        self.residual = torch.zeros((max_num_tokens, hidden_size),
                                    dtype=dtype,
                                    device=device)
        self.hidden_states = torch.zeros((max_num_tokens, hidden_size),
                                         dtype=dtype,
                                         device=device)

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

        num_input_tokens = input_ids.size(0)
        self.hidden_states[:num_input_tokens].copy_(hidden_states)

        hidden_states, residual = self.first_layer_group(
            positions,
            self.hidden_states[:num_input_tokens],
        )

        truncated_prefill_metadata = \
            get_forward_context().truncated_prefill_metadata
        if truncated_prefill_metadata is not None:
            gen_indices_padded = \
                truncated_prefill_metadata.generation_indices_padded
            num_tokens = gen_indices_padded.shape[0]
            # CUDA graph expects static tensor addresses
            # Copy output of first layer group to second layer group
            # TODO(sarckk): Move logic to @support_torch_compile
            self.residual[:num_tokens].copy_(residual[gen_indices_padded])
            self.hidden_states[:num_tokens].copy_(
                hidden_states[gen_indices_padded])
            positions[:num_tokens].copy_(positions[gen_indices_padded])
        else:
            num_tokens = num_input_tokens
            self.residual[:num_tokens].copy_(residual)
            self.hidden_states[:num_tokens].copy_(hidden_states)

        second_hidden_states, second_residual = self.second_layer_group(
            positions[:num_tokens],
            self.hidden_states[:num_tokens],
            self.residual[:num_tokens],
        )

        if truncated_prefill_metadata is not None:
            gen_indices_padded =\
                truncated_prefill_metadata.generation_indices_padded
            # NOTE: we need to pad generation indices for CUDA graph
            # but only the first num_gen_indices positions are actually valid.
            num_gen_indices = truncated_prefill_metadata.num_generation_indices
            gen_indices = gen_indices_padded[:num_gen_indices]
            hidden_states[gen_indices] = second_hidden_states[:num_gen_indices]
            residual[gen_indices] = second_residual[:num_gen_indices]
        else:
            hidden_states = second_hidden_states
            residual = second_residual

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


@pytest.fixture
def test_prompts():
    """
    Adapted from tests/v1/e2e/test_spec_decode.py
    """
    prompt_types = ["repeat", "sentence"]
    # Setting higher num prompts increases the chance of numerics mismatch
    # due to matrix multiplication numerics depending on batch dimension
    num_prompts = 10
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""please repeat the word '{word}' 10 times."""
        elif kind == "sentence":
            prompt = f"""please give a ten-word sentence that
            uses the word {word} at least once."""
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append(prompt)

    return prompts


@fork_new_process_for_each_test
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_kv_sharing_truncated_prefill(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
    test_prompts: list[str],
):
    ModelRegistry.register_model("Qwen2ForCausalLM", TestQwen2ForCausalLM)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        level=CompilationLevel.
        PIECEWISE if not enforce_eager else CompilationLevel.NO_COMPILATION)

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(
            model="Qwen/Qwen2-1.5B-Instruct",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
        )
        ref_responses = llm.generate(test_prompts, sampling_params)

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  enforce_eager=enforce_eager,
                  compilation_config=compilation_config,
                  enable_kv_sharing_truncated_prefill=True)
        optimized_responses = llm.generate(test_prompts, sampling_params)

        misses = 0

        for ref_response, optimized_response in zip(ref_responses,
                                                    optimized_responses):
            if ref_response.outputs[0].text != optimized_response.outputs[
                    0].text:
                misses += 1

        assert misses == 0
