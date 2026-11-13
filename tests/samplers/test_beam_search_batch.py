# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise empty and bounded beam-search batches with a local model."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, TokenizersBackend

from vllm.sampling_params import BeamSearchParams, StructuredOutputsParams


@pytest.fixture(scope="module")
def beam_batch_llm(tmp_path_factory, vllm_runner):
    model_path = tmp_path_factory.mktemp("beam-batch-model")
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"word{i}": i + 4 for i in range(28)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_path)
    torch.manual_seed(74)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=128,
            n_layer=1,
            n_head=2,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    model.save_pretrained(model_path)
    with vllm_runner(
        str(model_path),
        dtype="float32",
        max_model_len=32,
        max_num_seqs=1,
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.1,
    ) as runner:
        yield runner.llm


@pytest.mark.parametrize("concurrency_limit", [None, 1, 8])
@pytest.mark.parametrize("use_tqdm", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_beam_batch_empty(beam_batch_llm, concurrency_limit, use_tqdm, structured):
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=3,
        structured_outputs=StructuredOutputsParams(regex="word[0-9]+")
        if structured
        else None,
    )
    assert (
        beam_batch_llm.beam_search(
            [],
            params,
            concurrency_limit=concurrency_limit,
            use_tqdm=use_tqdm,
        )
        == []
    )


@pytest.mark.parametrize("concurrency_limit", [0, -1, -8])
@pytest.mark.parametrize("empty", [False, True])
def test_beam_batch_reject_nonpositive_limit(beam_batch_llm, concurrency_limit, empty):
    prompts = [] if empty else [{"prompt_token_ids": [1, 5, 6]}]
    with pytest.raises(ValueError, match="concurrency_limit must be greater than zero"):
        beam_batch_llm.beam_search(
            prompts,
            BeamSearchParams(beam_width=2, max_tokens=3),
            concurrency_limit=concurrency_limit,
        )


@pytest.mark.parametrize("concurrency_limit", [None, 1, 2, 8])
@pytest.mark.parametrize("tokenized", [False, True])
def test_beam_batch_preserves_generation(beam_batch_llm, concurrency_limit, tokenized):
    prompts = (
        [
            {"prompt_token_ids": [1, 5, 6]},
            {"prompt_token_ids": [1, 8]},
            {"prompt_token_ids": [1, 11, 12, 13]},
        ]
        if tokenized
        else [
            {"prompt": "word1 word2"},
            {"prompt": "word4"},
            {"prompt": "word7 word8 word9"},
        ]
    )
    params = BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True)
    reference = [beam_batch_llm.beam_search([prompt], params)[0] for prompt in prompts]
    outputs = beam_batch_llm.beam_search(
        prompts, params, concurrency_limit=concurrency_limit
    )
    assert len(outputs) == len(reference)
    for actual, expected in zip(outputs, reference):
        assert len(actual.sequences) == len(expected.sequences) == 2
        for actual_beam, expected_beam in zip(actual.sequences, expected.sequences):
            assert actual_beam.tokens == expected_beam.tokens
            assert actual_beam.text == expected_beam.text
            assert len(actual_beam.logprobs) == len(expected_beam.logprobs) == 3
            assert actual_beam.cum_logprob == pytest.approx(
                expected_beam.cum_logprob, rel=1e-6, abs=1e-6
            )
