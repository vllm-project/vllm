# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random
from typing import Optional, Union

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.gemma3n import Gemma3nForConditionalGeneration
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors

from ...utils import fork_new_process_for_each_test


class TestGemma3nForConditionalGeneration(Gemma3nForConditionalGeneration):

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, **kwargs)
        attn_metadata = get_forward_context().attn_metadata
        # attn_metadata is None during dummy runs
        if (attn_metadata is not None
                and self.cache_config.kv_sharing_fast_prefill):
            assert isinstance(attn_metadata, dict)  # true in V1
            # Gemma3n-E2B has 30 layers, with last 20 layers being
            # cross-decoder layers. Check attention metadata is correct
            for layer_name, metadata in attn_metadata.items():
                layer_idx = extract_layer_index(layer_name)
                if layer_idx >= 20:
                    assert hasattr(metadata, 'logits_indices_padded')
                    assert hasattr(metadata, 'num_logits_indices')
                else:
                    assert not hasattr(metadata, 'logits_indices_padded')
                    assert not hasattr(metadata, 'num_logits_indices')

            # Last layer will be a KV sharing layer
            layer_attn_metadata = attn_metadata[
                self.model.language_model.layers[-1].self_attn.attn.layer_name]
            logits_indices_padded = (layer_attn_metadata.logits_indices_padded)
            assert logits_indices_padded is not None
            num_logits_indices = layer_attn_metadata.num_logits_indices
            assert num_logits_indices > 0
            # Reset hidden states to random values and
            # only set logits at logits_indices to valid values
            # Because logits_indices are the only positions that are used
            # for output token sampling, this still produces same outputs
            logits_hs = hidden_states[logits_indices_padded]
            hidden_states = torch.randn_like(hidden_states)
            gen_indices = logits_indices_padded[:num_logits_indices]
            hidden_states[gen_indices] = logits_hs[:num_logits_indices]

        return hidden_states


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
def test_kv_sharing_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
    test_prompts: list[str],
):
    ModelRegistry.register_model("Gemma3nForConditionalGeneration",
                                 TestGemma3nForConditionalGeneration)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        # This allows vLLM compilation backend to handle allocating and
        # managing buffers for cudagraph
        cudagraph_copy_inputs=True,
        level=CompilationLevel.PIECEWISE
        if not enforce_eager else CompilationLevel.NO_COMPILATION)

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(
            model="google/gemma-3n-E2B-it",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
        )
        ref_responses = llm.generate(test_prompts, sampling_params)

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        llm = LLM(model="google/gemma-3n-E2B-it",
                  enforce_eager=enforce_eager,
                  compilation_config=compilation_config,
                  kv_sharing_fast_prefill=True)
        optimized_responses = llm.generate(test_prompts, sampling_params)

        misses = 0

        for ref_response, optimized_response in zip(ref_responses,
                                                    optimized_responses):
            if ref_response.outputs[0].text != optimized_response.outputs[
                    0].text:
                misses += 1

        assert misses == 0
