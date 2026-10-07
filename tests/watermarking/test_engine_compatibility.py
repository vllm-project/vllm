# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
MAX_NUM_BATCHED_TOKENS = 32

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)


@pytest.fixture(scope="module")
def watermarked_llm(vllm_runner) -> Generator[LLM, None, None]:
    with vllm_runner(
        MODEL,
        dtype="half",
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=128,
        max_num_seqs=4,
        disable_log_stats=False,
        watermark_config={"algorithm": "gumbel", "key": 42},
    ) as runner:
        yield runner.llm


def test_watermark_with_combined_sampling_controls(watermarked_llm: LLM):
    params = SamplingParams(
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        min_p=0.01,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        logit_bias={10: 100.0},
        allowed_token_ids=[10],
        max_tokens=8,
        ignore_eos=True,
    )

    output = watermarked_llm.generate("Sampling smoke test", params)[0]

    assert list(output.outputs[0].token_ids) == [10] * 8

    # The assertion above holds with or without watermarking, because
    # allowed_token_ids leaves a single candidate. Repeat with a wider allowed
    # set, where only watermarked sampling is reproducible without a seed.
    wide = SamplingParams(
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        min_p=0.01,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        allowed_token_ids=list(range(10, 210)),
        max_tokens=8,
        ignore_eos=True,
    )

    first = watermarked_llm.generate("Sampling smoke test", wide)[0]
    second = watermarked_llm.generate("Sampling smoke test", wide)[0]

    assert set(first.outputs[0].token_ids) <= set(range(10, 210))
    assert first.outputs[0].token_ids == second.outputs[0].token_ids


def test_watermark_with_penalties_and_nucleus_sampling(watermarked_llm: LLM):
    params = SamplingParams(
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        min_p=0.01,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        max_tokens=8,
        ignore_eos=True,
    )

    first = watermarked_llm.generate("Penalty smoke test", params)[0]
    second = watermarked_llm.generate("Penalty smoke test", params)[0]

    assert len(first.outputs[0].token_ids) == 8
    assert first.outputs[0].token_ids == second.outputs[0].token_ids


def test_plain_gumbel_seeded_parallel_sampling_is_deterministic(
    watermarked_llm: LLM,
):
    output = watermarked_llm.generate(
        "Seeded parallel sampling smoke test",
        SamplingParams(
            n=2,
            seed=42,
            temperature=0.8,
            max_tokens=8,
            ignore_eos=True,
        ),
    )[0]

    assert len(output.outputs) == 2
    assert all(len(candidate.token_ids) == 8 for candidate in output.outputs)
    assert output.outputs[0].token_ids == output.outputs[1].token_ids

    # Plain Gumbel keys on the token context, so the request seed does not
    # change a watermarked draw. Pin that down rather than implying the seed
    # is what makes the candidates match.
    unseeded = watermarked_llm.generate(
        "Seeded parallel sampling smoke test",
        SamplingParams(
            n=2,
            temperature=0.8,
            max_tokens=8,
            ignore_eos=True,
        ),
    )[0]

    assert unseeded.outputs[0].token_ids == output.outputs[0].token_ids


@pytest.mark.parametrize(
    "structured_outputs",
    [
        StructuredOutputsParams(choice=["A", "B"]),
        StructuredOutputsParams(grammar='root ::= "A" | "B"'),
    ],
    ids=["choice", "grammar"],
)
def test_watermark_with_structured_output(
    watermarked_llm: LLM, structured_outputs: StructuredOutputsParams
):
    params = SamplingParams(
        temperature=0.8,
        max_tokens=8,
        structured_outputs=structured_outputs,
    )

    output = watermarked_llm.generate("Choose A or B:", params)[0]

    assert output.outputs[0].text.strip() in ("A", "B")


def test_watermark_with_wide_grammar_is_deterministic(watermarked_llm: LLM):
    # A wider grammar makes deterministic watermarking observable.
    params = SamplingParams(
        temperature=0.8,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(
            grammar="root ::= [A-Z] [A-Z] [A-Z] [A-Z] [A-Z] [A-Z]"
        ),
    )

    first = watermarked_llm.generate("Emit six letters:", params)[0]
    second = watermarked_llm.generate("Emit six letters:", params)[0]

    assert first.outputs[0].text.strip().isupper()
    assert first.outputs[0].token_ids == second.outputs[0].token_ids


def test_watermark_with_prefix_cache_and_chunked_prefill(watermarked_llm: LLM):
    prompt = {"prompt_token_ids": [101] * 48}
    params = SamplingParams(temperature=0.8, max_tokens=8, ignore_eos=True)

    assert len(prompt["prompt_token_ids"]) > MAX_NUM_BATCHED_TOKENS

    cold = watermarked_llm.generate(prompt, params)[0]
    cached = watermarked_llm.generate(prompt, params)[0]

    assert cached.num_cached_tokens > 0
    assert cold.outputs[0].token_ids == cached.outputs[0].token_ids


def test_mixed_watermarked_batch(watermarked_llm: LLM):
    prompts = [
        {"prompt_token_ids": [101] * 32 + [102]},
        {"prompt_token_ids": [201] * 32 + [202]},
    ]
    marked = SamplingParams(temperature=0.8, max_tokens=8, ignore_eos=True)
    ordinary = SamplingParams(
        temperature=0.8,
        seed=123,
        max_tokens=8,
        ignore_eos=True,
        watermarking=False,
    )

    batched = watermarked_llm.generate(prompts, [marked, ordinary])
    marked_alone = watermarked_llm.generate(prompts[0], marked)[0]
    ordinary_alone = watermarked_llm.generate(prompts[1], ordinary)[0]

    assert batched[0].outputs[0].token_ids == marked_alone.outputs[0].token_ids
    assert batched[1].outputs[0].token_ids == ordinary_alone.outputs[0].token_ids
