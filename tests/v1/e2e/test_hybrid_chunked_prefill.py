# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

# A trivial request with a short prompt to ensure we run a mixed batch
SMALL_MESSAGE = [
    {
        "role": "user",
        "content": "The secret beta value is 64. What is the secret beta?",
    }
]

# Sample prompt with a bunch of filler in between the critical fact and the request.
# Both parts need to be processed properly for the model to generate the correct answer
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Important: The secret number is 42. "
            "The sky is green in this hypothetical world. "
            "Apples grow on trees in the forest. "
            "Rivers flow through the valleys and mountains. "
            "Birds sing songs in the early morning light. "
            "The weather today is sunny with clear skies ahead. "
            "Flowers bloom in the garden during spring season. "
            "Now answer with ONLY the number and nothing else: "
            "What is the secret number plus one?"
        ),
    }
]


# TODO(ben): Add Nemotron-H with MTP
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3.5-35B-A3B-FP8"])
def test_mtp_speculative_mixed_batch_short_prefill(vllm_runner, model_name):
    """Test to ensure MTP speculative decoding correctly handles
    short prefill chunks that fall below the reorder_batch_threshold."""

    # Set so large that both prefills will be classified as decodes in a mixed batch
    chunk_size = 256
    num_draft_tokens = 100

    with vllm_runner(
        model_name,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_draft_tokens,
        },
        max_num_batched_tokens=chunk_size,
        max_model_len=512,
        enforce_eager=True,
        tensor_parallel_size=2,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        attention_config={
            "backend": "flash_attn",
        },
    ) as llm:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )

        # First small message gets prefilled first, under normal conditions since the
        # batch is not yet mixed. Then the second prefill arrives as a mixed batch, but
        # is shorter than num_speculative_tokens, so it gets misclassified as a decode
        # and processed with the wrong state management logic,  causing the critical
        # fact from the first chunk to be lost and the model to generate nonsense.
        outputs = llm.get_llm().chat(
            [SMALL_MESSAGE, MESSAGES],
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )

        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            responses.append(generated_text)

        assert "64" in responses[0], (
            "The first response should contain the correct value of 64."
        )
        assert "43" in responses[1], (
            "The second response should contain the correct value of 42+1=43."
        )
