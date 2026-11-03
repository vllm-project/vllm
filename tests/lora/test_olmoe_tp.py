# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TYPE_CHECKING, cast

import pytest
import requests
import torch
from safetensors.torch import load_file, save_file

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from ..utils import RemoteOpenAIServer, multi_gpu_test

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
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
    def start_lora_trace(self):
        worker = cast("Worker", self)
        runner = cast("GPUModelRunner", worker.model_runner)
        manager = runner.cudagraph_manager
        assert manager is not None and manager.graphs
        self.lora_graph_replays = 0

        def count_replay(graph):
            self.lora_graph_replays += 1

        self.lora_graph_handles = [
            graph.register_replay_end_hook(count_replay)
            for graph in manager.graphs.values()
        ]

    def finish_lora_trace(self):
        from vllm.distributed import get_dp_group

        for handle in self.lora_graph_handles:
            handle.remove()
        group = get_dp_group()
        reports = [None] * group.world_size
        torch.distributed.all_gather_object(
            reports, self.lora_graph_replays, group=group.cpu_group
        )
        return reports

    def preload_lora_slots(self, paths: list[str], reverse_peer: bool):
        reports = self.configure_lora_residency(
            paths, [[1, 2], [2, 1] if reverse_peer else [1, 2]]
        )
        return [report["slots"] for report in reports]

    def configure_lora_residency(self, paths: list[str], orders: list[list[int]]):
        from vllm.distributed import get_dp_group

        worker = cast("Worker", self)
        group = get_dp_group()
        for adapter_id in list(worker.list_loras()):
            worker.remove_lora(adapter_id)
        order = orders[group.rank_in_group]
        for adapter_id in order:
            assert worker.add_lora(
                LoRARequest(str(adapter_id), adapter_id, paths[adapter_id - 1])
            )
        slots = list(worker.model_runner.get_model().lora_manager.lora_index_to_id)
        assert {item for item in slots if item is not None} == set(order)
        report = {
            "rank": group.rank_in_group,
            "slots": [x for x in slots if x is not None],
        }
        reports = [None] * group.world_size
        torch.distributed.all_gather_object(reports, report, group=group.cpu_group)
        return reports


def run_group_admission_server(
    model: str, paths: list[str], tp: int, graph: bool = False
):
    """Compare repaired cold/divergent receivers and concurrent adapter cohorts."""
    args = [
        "--host",
        "127.0.0.1",
        "--api-server-count",
        "1",
        "--data-parallel-size",
        "2",
        "--tensor-parallel-size",
        str(tp),
        "--all2all-backend",
        "allgather_reducescatter",
        "--enable-lora",
        "--enable-group-lora-admission",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "4",
        "--max-lora-rank",
        "8",
        "--lora-modules",
        *[f"adapter-{i}={p}" for i, p in enumerate(paths, 1)],
        "--max-model-len",
        "256",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "8",
        "--no-enable-prefix-caching",
        "--worker-extension-cls",
        "tests.lora.test_olmoe_tp.LoRASlotOrderWorkerExtension",
    ]
    if not graph:
        args.append("--enforce-eager")
    with (
        RemoteOpenAIServer(
            model,
            args,
            max_wait_seconds=300,
            env_dict={
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_GPU_SYNC_CHECK": "error",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        ) as server,
        server.get_client(timeout=90) as client,
    ):

        def score(adapter, rank, tokens=None, start=None):
            if start is not None:
                start.wait(timeout=30)
            if tokens is None:
                tokens = {1: 64, 2: 8, 3: 16, 4: 32}[adapter]
            response = client.completions.create(
                model=f"adapter-{adapter}",
                prompt=[[1, 42, 67, 105, 325, 556, 880, 1042]],
                temperature=0,
                max_tokens=tokens,
                extra_headers={"X-data-parallel-rank": str(rank)},
                extra_body={"prompt_logprobs": 0, "ignore_eos": True},
            )
            logprobs = [
                next(iter(item.values()))["logprob"]
                for choice in response.choices
                for item in choice.model_dump()["prompt_logprobs"]
                if item is not None
            ]
            assert len(logprobs) == 7
            return {
                "scores": logprobs,
                "text": response.choices[0].text,
                "finish_reason": response.choices[0].finish_reason,
            }

        reference = {adapter: score(adapter, 0) for adapter in range(1, 5)}
        assert reference[1]["scores"] != reference[2]["scores"]
        for orders in ([[1, 2], [2, 1]], [[1], []], [[1, 2], [3, 4]]):
            requests.post(
                server.url_for("pause"), params={"mode": "wait"}, timeout=60
            ).raise_for_status()
            configured = requests.post(
                server.url_for("collective_rpc"),
                json={
                    "method": "configure_lora_residency",
                    "args": [paths, orders],
                    "timeout": 60,
                },
                timeout=90,
            )
            configured.raise_for_status()
            requests.post(server.url_for("resume"), timeout=30).raise_for_status()
            assert score(1, 0) == reference[1]
            assert score(2, 1) == reference[2]
        with ThreadPoolExecutor(max_workers=4) as pool:
            start = Barrier(4)
            futures = {
                adapter: pool.submit(score, adapter, (adapter - 1) // 2, None, start)
                for adapter in range(1, 5)
            }
            assert {
                adapter: future.result() for adapter, future in futures.items()
            } == reference
        replays = []
        if graph:
            requests.post(
                server.url_for("pause"), params={"mode": "wait"}, timeout=60
            ).raise_for_status()
            requests.post(
                server.url_for("collective_rpc"),
                json={"method": "start_lora_trace"},
                timeout=90,
            ).raise_for_status()
            requests.post(server.url_for("resume"), timeout=30).raise_for_status()
            with ThreadPoolExecutor(max_workers=2) as pool:
                start = Barrier(2)
                probes = {
                    adapter: pool.submit(score, adapter, adapter - 1, None, start)
                    for adapter in (1, 2)
                }
                assert {
                    adapter: future.result() for adapter, future in probes.items()
                } == {adapter: reference[adapter] for adapter in (1, 2)}
            requests.post(
                server.url_for("pause"), params={"mode": "wait"}, timeout=60
            ).raise_for_status()
            replay = requests.post(
                server.url_for("collective_rpc"),
                json={
                    "method": "finish_lora_trace",
                    "timeout": 60,
                },
                timeout=90,
            )
            replay.raise_for_status()
            replays = [count for lane in replay.json()["results"] for count in lane]
            assert len(replays) == 2 * tp and all(count > 0 for count in replays), (
                replays
            )
        return {
            "dp": 2,
            "tp": tp,
            "graph_requested": graph,
            "completed_graph_replays_by_worker": replays,
            "reference": reference,
            "common_permuted_cold_divergent_and_concurrent": "passed_exact_scores",
        }


def _group_adapters(source: str, destination: Path) -> list[str]:
    weights = load_file(str(Path(source) / "adapter_model.safetensors"))
    expert_b = [name for name in weights if "experts" in name and "lora_B" in name]
    assert any(weights[name].count_nonzero().item() for name in expert_b)
    paths = []
    for index, scale in enumerate((1, -1, 0.5, -0.5), 1):
        target = destination / str(index)
        shutil.copytree(source, target)
        scaled = {
            name: value * scale if name in expert_b else value
            for name, value in weights.items()
        }
        save_file(scaled, str(target / "adapter_model.safetensors"))
        paths.append(str(target))
    return paths


@multi_gpu_test(num_gpus=2)
def test_olmoe_group_lora_admission(olmoe_lora_files, tmp_path):
    run_group_admission_server(
        MODEL_PATH, _group_adapters(olmoe_lora_files, tmp_path), 1
    )


@multi_gpu_test(num_gpus=4)
def test_olmoe_group_lora_admission_dp2_tp2(olmoe_lora_files, tmp_path):
    run_group_admission_server(
        MODEL_PATH, _group_adapters(olmoe_lora_files, tmp_path), 2
    )


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
