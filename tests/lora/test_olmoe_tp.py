# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import requests
import torch
from safetensors.torch import load_file, save_file

import vllm
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from ..utils import RemoteOpenAIServer, multi_gpu_test

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

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
    "SELECT Candidate_ID, COUNT(*) as Total_Candidates\nFROM candidate\nINNER JOIN people ON candidate.People_ID = people.People_ID",  # noqa: E501
    # There are multiple acceptable responses
    (
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE People_ID IN (SELECT People_ID FROM people) ORDER BY COUNT(*) DESC LIMIT 1",  # noqa: E501
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE COUNT(People_ID) = (SELECT COUNT(People_ID) FROM people) ORDER BY Candidate_ID DESC LIMIT 1",  # noqa: E501
    ),
]

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


def _output_matches(generated: str, accepted: str | Sequence[str]) -> bool:
    if isinstance(accepted, str):
        accepted = (accepted,)
    return any(generated.startswith(s) for s in accepted)


def generate_and_test(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: list[int | None] | int | None,
    compare_lower: bool = False,
) -> None:
    lora_request = None
    if isinstance(lora_id, int):
        lora_request = LoRARequest(str(lora_id), lora_id, lora_path)
    elif isinstance(lora_id, list):
        lora_request = [
            LoRARequest(str(i), i, lora_path) if i is not None else None
            for i in lora_id
        ]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(PROMPTS, sampling_params, lora_request=lora_request)
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

        if compare_lower:
            generated_text = generated_text.lower()
            if isinstance(expected_output, str):
                expected_output = (expected_output.lower(),)
            else:
                expected_output = tuple(s.lower() for s in expected_output)
        assert _output_matches(generated_text, expected_output), (
            f"Output {i}: {generated_text!r} does not match any of {expected_output!r}"
        )


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
def test_olmoe_lora(olmoe_lora_files, maybe_enable_lora_dual_stream):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )

    generate_and_test(llm, olmoe_lora_files, lora_id=1)
    generate_and_test(llm, olmoe_lora_files, lora_id=2)


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


@pytest.mark.parametrize("use_v2", [False, True])
@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_idle_peer(olmoe_lora_files, use_v2):
    """Static expert adapters must survive first-idle and active-idle transitions."""
    args = [
        "--data-parallel-size",
        "2",
        "--enable-lora",
        "--max-loras",
        "1",
        "--max-lora-rank",
        "8",
        "--lora-modules",
        f"sql={olmoe_lora_files}",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "16",
        "--no-enable-prefix-caching",
        "--enforce-eager",
    ]
    with (
        RemoteOpenAIServer(
            MODEL_PATH,
            args,
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))},
        ) as server,
        server.get_client() as client,
    ):
        for rank in (0, 1, 0):
            completion = client.completions.create(
                model="sql",
                prompt=PROMPTS,
                max_tokens=64,
                temperature=0,
                extra_headers={"X-data-parallel-rank": str(rank)},
            )
            assert len(completion.choices) == len(EXPECTED_LORA_OUTPUT)
            for choice in completion.choices:
                assert _output_matches(
                    choice.text.strip(), EXPECTED_LORA_OUTPUT[choice.index]
                ), f"DP rank {rank}, prompt {choice.index}: {choice.text!r}"


class LoRASlotOrderWorkerExtension:
    def preload_lora_slots(self, paths: list[str], reverse_peer: bool):
        from vllm.distributed import get_dp_group

        worker = cast("Worker", self)
        group = get_dp_group()
        for adapter_id in list(worker.list_loras()):
            worker.remove_lora(adapter_id)
        order = [1, 2]
        if reverse_peer and group.rank_in_group == 1:
            order.reverse()
        for adapter_id in order:
            assert worker.add_lora(
                LoRARequest(str(adapter_id), adapter_id, paths[adapter_id - 1])
            )
        orders = [None] * group.world_size
        torch.distributed.all_gather_object(orders, order, group=group.cpu_group)
        return orders


@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_slot_order(olmoe_lora_files, tmp_path):
    """Different remote slots must preserve each adapter's teacher-forced scores."""
    opposite = tmp_path / "opposite"
    shutil.copytree(olmoe_lora_files, opposite)
    weights = load_file(str(Path(olmoe_lora_files) / "adapter_model.safetensors"))
    expert_b = [name for name in weights if "experts" in name and "lora_B" in name]
    assert any(weights[name].count_nonzero().item() for name in expert_b)
    for name in expert_b:
        weights[name] = -weights[name]
    save_file(weights, str(opposite / "adapter_model.safetensors"))
    paths = [str(olmoe_lora_files), str(opposite)]
    args = [
        "--api-server-count",
        "1",
        "--data-parallel-size",
        "2",
        "--enable-lora",
        "--max-loras",
        "2",
        "--max-lora-rank",
        "8",
        "--lora-modules",
        f"sql={paths[0]}",
        f"opposite={paths[1]}",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "16",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--worker-extension-cls",
        "tests.lora.test_olmoe_tp.LoRASlotOrderWorkerExtension",
    ]
    with (
        RemoteOpenAIServer(
            MODEL_PATH,
            args,
            env_dict={
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_BATCH_INVARIANT": "1",
            },
        ) as server,
        server.get_client() as client,
    ):
        scores: dict[tuple[int, str], list[float]] = {}
        for reverse in (False, True):
            requests.post(
                server.url_for("pause"), params={"mode": "wait"}, timeout=30
            ).raise_for_status()
            response = requests.post(
                server.url_for("collective_rpc"),
                json={"method": "preload_lora_slots", "args": [paths, reverse]},
                timeout=90,
            )
            response.raise_for_status()
            results = response.json()["results"]
            assert results
            for orders in results:
                assert orders == [[1, 2], [2, 1] if reverse else [1, 2]]
            requests.post(server.url_for("resume"), timeout=30).raise_for_status()
            for rank in (0, 1):
                for name in ("sql", "opposite"):
                    completion = client.completions.create(
                        model=name,
                        prompt=PROMPTS,
                        temperature=0,
                        max_tokens=1,
                        extra_headers={"X-data-parallel-rank": str(rank)},
                        extra_body={"prompt_logprobs": 0},
                    )
                    current = [
                        next(iter(entry.values()))["logprob"]
                        for choice in sorted(completion.choices, key=lambda c: c.index)
                        for entry in choice.model_dump()["prompt_logprobs"]
                        if entry is not None
                    ]
                    assert current
                    if reverse:
                        assert current == scores[rank, name], (
                            f"Slot-order changed scores on DP rank {rank}, "
                            f"adapter {name}"
                        )
                    else:
                        scores[rank, name] = current
            assert scores[0, "sql"] != scores[0, "opposite"]


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
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


@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@multi_gpu_test(num_gpus=4)
def test_olmoe_lora_tp4(olmoe_lora_files, fully_sharded_loras):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=4,
        fully_sharded_loras=fully_sharded_loras,
    )
    generate_and_test(
        llm, olmoe_lora_files, lora_id=1, compare_lower=fully_sharded_loras
    )
    generate_and_test(
        llm, olmoe_lora_files, lora_id=2, compare_lower=fully_sharded_loras
    )
