# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare offline and HTTP beam search when a real model has tied logits."""

import torch
from tests.utils import RemoteOpenAIServer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, TokenizersBackend
from vllm.sampling_params import BeamSearchParams


def test_beam_tied_scores_native_outputs(tmp_path, vllm_runner):
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
    tokenizer.save_pretrained(tmp_path)
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
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(tmp_path)

    prompt = [1, 5, 6]
    allowed = list(range(4, 20))
    reference = {}
    with vllm_runner(
        str(tmp_path),
        dtype="float32",
        max_model_len=32,
        max_num_seqs=1,
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=16 * 1024 * 1024,
    ) as runner:
        for width in (2, 3, 5):
            for steps in (1, 3):
                params = BeamSearchParams(
                    beam_width=width,
                    max_tokens=steps,
                    ignore_eos=True,
                    allowed_token_ids=allowed,
                )
                beams = runner.llm.beam_search([{"prompt_token_ids": prompt}], params)[
                    0
                ].sequences
                reference[width, steps] = [
                    (beam.tokens[len(prompt) :], beam.cum_logprob) for beam in beams
                ]

    args = [
        "--dtype",
        "float32",
        "--max-model-len",
        "32",
        "--max-num-seqs",
        "1",
        "--no-async-scheduling",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.1",
        "--kv-cache-memory-bytes",
        str(16 * 1024 * 1024),
        "--served-model-name",
        "beam-ties-model",
    ]
    with (
        RemoteOpenAIServer(str(tmp_path), args) as server,
        server.get_client() as client,
    ):
        for (width, steps), expected in reference.items():
            response = client.completions.create(
                model="beam-ties-model",
                prompt=prompt,
                n=width,
                temperature=0,
                max_tokens=steps,
                extra_body={
                    "use_beam_search": True,
                    "ignore_eos": True,
                    "allowed_token_ids": allowed,
                    "return_token_ids": True,
                },
            )
            assert [choice.token_ids for choice in response.choices] == [
                tokens for tokens, _ in expected
            ], (width, steps)
            assert all(choice.finish_reason == "length" for choice in response.choices)
            assert len(response.choices) == width
            assert all(len(choice.token_ids) == steps for choice in response.choices)
