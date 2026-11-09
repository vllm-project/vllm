# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import shutil
from collections.abc import Sequence

import pytest
import torch
from safetensors.torch import load_file, save_file

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from ..utils import multi_gpu_test

MODEL_PATH = "allenai/OLMoE-1B-7B-0125-Instruct"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me. Do not return any additional explanation. Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.


###Input:
{context}

###Response:"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM candidate",
    "SELECT count(*) FROM candidate",
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]

EXPECTED_BASE_MODEL_OUTPUT = [
    "SELECT COUNT(Candidate_ID) FROM candidate",
    "SELECT COUNT(Candidate_ID) FROM candidate",
    (
        "SELECT Candidate_ID, COUNT(*) as Total_Candidates\nFROM candidate\nINNER JOIN people ON candidate.People_ID = people.People_ID",  # noqa: E501
        "SELECT Candidate_ID FROM candidate WHERE People_ID IN (SELECT People_ID FROM people)",  # noqa: E501
    ),
    # There are multiple acceptable responses
    (
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE People_ID IN (SELECT People_ID FROM people) ORDER BY COUNT(*) DESC LIMIT 1",  # noqa: E501
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE COUNT(People_ID) = (SELECT COUNT(People_ID) FROM people) ORDER BY Candidate_ID DESC LIMIT 1",  # noqa: E501
    ),
]


def _output_matches(generated: str, accepted: str | Sequence[str]) -> bool:
    if isinstance(accepted, str):
        accepted = (accepted,)
    return any(generated.startswith(s) for s in accepted)


PROMPTS = [
    PROMPT_TEMPLATE.format(context="How many candidates are there?"),
    PROMPT_TEMPLATE.format(context="Count the number of candidates."),
    PROMPT_TEMPLATE.format(
        context="Which poll resource provided the most number of candidate information?"  # noqa: E501
    ),
    PROMPT_TEMPLATE.format(
        context="Return the poll resource associated with the most candidates."
    ),
]


def generate_and_test(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: list[int | None] | int | None,
) -> None:
    prompts = PROMPTS
    lora_request = None
    if isinstance(lora_id, int):
        lora_request = LoRARequest(str(lora_id), lora_id, lora_path)
    elif isinstance(lora_id, list):
        lora_request = [
            LoRARequest(str(i), i, lora_path) if i is not None else None
            for i in lora_id
        ]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    for i in range(len(EXPECTED_LORA_OUTPUT)):
        req_lora_id = lora_id[i] if isinstance(lora_id, list) else lora_id
        generated_text = generated_texts[i]
        expected_output = (
            EXPECTED_LORA_OUTPUT[i]
            if req_lora_id is not None
            else EXPECTED_BASE_MODEL_OUTPUT[i]
        )
        assert _output_matches(generated_text, expected_output), (
            f"Output {i}: {generated_text!r} does not match any of {expected_output!r}"
        )


def test_olmoe_lora_mixed(olmoe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        hf_overrides={"head_dtype": "float32"},
        enable_chunked_prefill=True,
    )

    generate_and_test(llm, olmoe_lora_files, lora_id=[1, None, 3, None])


def test_olmoe_lora_mixed_random(
    olmoe_lora_files, tmp_path, maybe_enable_lora_dual_stream
):
    # Create a dummy LoRA with random weights based on the real one
    random_lora_path = tmp_path / "random_lora"
    shutil.copytree(olmoe_lora_files, random_lora_path)

    weights_path = random_lora_path / "adapter_model.safetensors"
    weights = load_file(str(weights_path))
    random_weights = {k: torch.randn_like(v) for k, v in weights.items()}
    save_file(random_weights, str(weights_path))

    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )

    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
    ]

    lora_requests = [
        LoRARequest("real", 1, olmoe_lora_files),
        LoRARequest("random", 2, str(random_lora_path)),
    ]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)
    assert outputs[0].outputs[0].text.strip().startswith(EXPECTED_LORA_OUTPUT[0])


@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_tp2(olmoe_lora_files, fully_sharded_loras):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
    )

    generate_and_test(llm, olmoe_lora_files, lora_id=1)
    generate_and_test(llm, olmoe_lora_files, lora_id=2)


async def _generate_on_rank(
    engine: AsyncLLM,
    tag: str,
    dp_rank: int,
    lora_request: LoRARequest | None,
    max_tokens: int = 64,
) -> list[str]:
    async def one(i: int, prompt: str) -> str:
        final = None
        async for out in engine.generate(
            prompt,
            vllm.SamplingParams(temperature=0, max_tokens=max_tokens),
            request_id=f"{tag}-{i}",
            lora_request=lora_request,
            data_parallel_rank=dp_rank,
        ):
            final = out
        assert final is not None
        return final.outputs[0].text.strip()

    return await asyncio.gather(*(one(i, p) for i, p in enumerate(PROMPTS)))


def _check(outputs: list[str], expected: list) -> None:
    # The last prompt's greedy output varies with how EP splits the batch.
    for i, (out, exp) in enumerate(zip(outputs[:3], expected)):
        assert _output_matches(out, exp), f"Output {i}: {out!r} vs {exp!r}"


async def _run_dp2_ep(lora_path: str) -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            MODEL_PATH,
            max_model_len=1024,
            enable_lora=True,
            max_loras=4,
            enforce_eager=True,
            trust_remote_code=True,
            hf_overrides={"head_dtype": "float32"},
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
    )
    try:
        lora = LoRARequest("1", 1, lora_path)
        # Load the adapter into the same slot on both ranks.
        await asyncio.gather(
            *(_generate_on_rank(engine, f"warmup{r}", r, lora, 1) for r in (0, 1))
        )

        base_outputs, lora_outputs = await asyncio.gather(
            _generate_on_rank(engine, "base", 0, None),
            _generate_on_rank(engine, "lora", 1, lora),
        )
        _check(base_outputs, EXPECTED_BASE_MODEL_OUTPUT)
        _check(lora_outputs, EXPECTED_LORA_OUTPUT)

        _check(
            await _generate_on_rank(engine, "lora-idle0", 1, lora),
            EXPECTED_LORA_OUTPUT,
        )
    finally:
        engine.shutdown()


@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_dp2_ep(olmoe_lora_files):
    """With DP + EP, MoE LoRA runs on tokens dispatched from the other rank.
    Rank 1 serves LoRA requests while rank 0 serves base requests or sits idle
    running DP dummy batches; neither may leak into or drop the other's LoRA.
    """
    asyncio.run(_run_dp2_ep(olmoe_lora_files))


async def _run_dp2_ep_slot_order(lora_path: str, opposite_path: str) -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            MODEL_PATH,
            max_model_len=1024,
            enable_lora=True,
            max_loras=4,
            enforce_eager=True,
            hf_overrides={"head_dtype": "float32"},
            data_parallel_size=2,
            enable_expert_parallel=True,
        )
    )
    try:
        sql = LoRARequest("sql", 1, lora_path)
        opposite = LoRARequest("opposite", 2, opposite_path)
        # Load in opposite orders: rank 0 slots are [sql, opposite], rank 1
        # slots are [opposite, sql].
        await _generate_on_rank(engine, "load0", 0, sql, 1)
        await _generate_on_rank(engine, "load1", 1, opposite, 1)
        await asyncio.gather(
            _generate_on_rank(engine, "load2", 0, opposite, 1),
            _generate_on_rank(engine, "load3", 1, sql, 1),
        )
        for rank in (0, 1):
            _check(
                await _generate_on_rank(engine, f"sql{rank}", rank, sql),
                EXPECTED_LORA_OUTPUT,
            )
    finally:
        engine.shutdown()


@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_dp2_ep_slot_order(olmoe_lora_files, tmp_path):
    """Tokens dispatched to a rank that holds the adapter in a different slot
    must still use that adapter's expert weights."""
    opposite = tmp_path / "opposite"
    shutil.copytree(olmoe_lora_files, opposite)
    weights = load_file(str(opposite / "adapter_model.safetensors"))
    for name in weights:
        if "experts" in name and "lora_B" in name:
            weights[name] = -weights[name]
    save_file(weights, str(opposite / "adapter_model.safetensors"))
    asyncio.run(_run_dp2_ep_slot_order(olmoe_lora_files, str(opposite)))
