# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import PoolingParams
from vllm.utils.mistral import is_mistral_tokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
QUERY = "The quick brown fox jumps over the lazy dog"
DOCUMENT = "A fox is a small omnivorous mammal."


def test_token_limits_with_mistral_tokenizer(vllm_runner):
    """MistralTokenizer cannot return offset_mapping, but scoring token limits
    must still truncate the query and leave a short document unchanged.

    Only tokenization matters, so the model has dummy weights and one layer.
    """
    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        convert="embed",
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 1},
        enforce_eager=True,
    ) as runner:
        llm = runner.llm
        assert is_mistral_tokenizer(llm.get_tokenizer())

        params = PoolingParams(
            extra_kwargs={"max_tokens_per_query": 3, "max_tokens_per_doc": 100}
        )
        (limited,) = llm.score(QUERY, DOCUMENT, pooling_params=params, use_tqdm=False)
        (expected,) = llm.score("The quick brown", DOCUMENT, use_tqdm=False)

    assert limited.prompt_token_ids == expected.prompt_token_ids
