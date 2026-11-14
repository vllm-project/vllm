# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import shutil
from collections.abc import Sequence
from pathlib import Path

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
            enable_prefix_caching=False,
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


def _olmoe_slot_args(*, dp_size=1, eager=True):
    return AsyncEngineArgs(
        MODEL_PATH,
        max_model_len=1024,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=1024**3,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        enforce_eager=eager,
        hf_overrides={"head_dtype": "float32"},
        enable_prefix_caching=False,
        data_parallel_size=dp_size,
        enable_expert_parallel=dp_size > 1,
        all2all_backend="allgather_reducescatter",
        attention_config={"backend": "TRITON_ATTN"},
        kernel_config={"moe_backend": "triton", "enable_flashinfer_autotune": False},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
    )


async def _olmoe_slot_outputs(engine, tag, scenario, paths, gold=None):
    from vllm.sampling_params import RequestOutputKind

    # Queue the whole batch before scheduling to control request arrival order.
    await engine.pause_generation(mode="keep", clear_cache=False)
    collectors = []
    for group_index, (rank, adapter) in enumerate(scenario):
        request = (
            LoRARequest(str(adapter), adapter, paths[adapter - 1]) if adapter else None
        )
        group = []
        for index, prompt in enumerate(PROMPTS):
            if gold is not None:
                reference = gold[adapter][index]
                prompt = {
                    "prompt_token_ids": (
                        reference.prompt_token_ids
                        + list(reference.outputs[0].token_ids)
                    )
                }
            group.append(
                await engine.add_request(
                    prompt=prompt,
                    params=vllm.SamplingParams(
                        temperature=0,
                        max_tokens=64 if gold is None else 1,
                        logprobs=5 if gold is None else None,
                        prompt_logprobs=None if gold is None else 0,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                    ),
                    request_id=f"{tag}-{group_index}-{rank}-{index}",
                    lora_request=request,
                    data_parallel_rank=rank,
                )
            )
        collectors.append(group)
    # The utility RPC is processed after the adds on each DP engine.
    assert await engine.is_paused()
    await engine.resume_generation()
    return [await asyncio.gather(*(c.get() for c in group)) for group in collectors]


def _olmoe_continuation_scores(outputs, gold):
    scores: list[float] = []
    for output, reference in zip(outputs, gold):
        start = len(reference.prompt_token_ids)
        token_ids = reference.outputs[0].token_ids
        assert output.prompt_logprobs is not None
        assert len(output.prompt_logprobs) == start + len(token_ids)
        scores.extend(
            output.prompt_logprobs[start + i][token].logprob
            for i, token in enumerate(token_ids)
        )
    result = torch.tensor(scores, dtype=torch.float32)
    assert torch.isfinite(result).all()
    return result


@pytest.fixture(scope="module")
def olmoe_two_lora_paths(olmoe_lora_files, tmp_path_factory):
    path = tmp_path_factory.mktemp("olmoe_lora_pair") / "random_lora"
    shutil.copytree(olmoe_lora_files, path)
    weights_file = path / "adapter_model.safetensors"
    weights = load_file(str(weights_file))
    generator = torch.Generator().manual_seed(123)
    # Match each trained tensor's RMS so random weights do not cause overflow.
    weights = {
        name: (
            torch.randn(weight.shape, generator=generator)
            * weight.float().square().mean().sqrt()
        ).to(weight.dtype)
        for name, weight in weights.items()
    }
    save_file(weights, str(weights_file))
    return [olmoe_lora_files, str(path)]


async def _run_olmoe_preloaded_slots(paths, eager):
    from tests.models.utils import check_logprobs_close

    def convert(group):
        return [
            (
                list(out.outputs[0].token_ids),
                out.outputs[0].text,
                out.outputs[0].logprobs,
            )
            for out in group
        ]

    engine = AsyncLLM.from_engine_args(_olmoe_slot_args())
    gold = {}
    try:
        for adapter in (0, 1, 2):
            (gold[adapter],) = await _olmoe_slot_outputs(
                engine, f"single-{adapter}", ((0, adapter),), paths
            )
            if adapter != 2:
                _check(
                    [out.outputs[0].text.strip() for out in gold[adapter]],
                    EXPECTED_BASE_MODEL_OUTPUT
                    if adapter == 0
                    else EXPECTED_LORA_OUTPUT,
                )
    finally:
        engine.shutdown()
    assert any(
        a.outputs[0].token_ids != b.outputs[0].token_ids
        for a, b in zip(gold[1], gold[2])
    ), "The two adapters must have distinguishable outputs"

    scenarios = [
        ((0, 1), (1, 2)),
        ((0, 2), (1, 1)),
        ((0, 0), (1, 1)),
        ((1, 2),),
        ((0, 1),),
    ]
    reference = {}
    # Compare the same topology and kernels: EP reductions need not match DP1.
    # Compare public preloading with on-demand loading in the same coordinator.
    for baseline in (True, False):
        engine = AsyncLLM.from_engine_args(_olmoe_slot_args(dp_size=2, eager=eager))
        try:
            if baseline:
                for adapter, path in enumerate(paths, 1):
                    assert await engine.add_lora(
                        LoRARequest(str(adapter), adapter, path)
                    )
            for step, scenario in enumerate(scenarios):
                groups = await _olmoe_slot_outputs(
                    engine, f"generate-{step}", scenario, paths
                )
                scored = await _olmoe_slot_outputs(
                    engine, f"score-{step}", scenario, paths, gold=gold
                )
                for (rank, adapter), outputs, scores in zip(scenario, groups, scored):
                    if adapter == 1:
                        _check(
                            [out.outputs[0].text.strip() for out in outputs],
                            EXPECTED_LORA_OUTPUT,
                        )
                    check_logprobs_close(
                        outputs_0_lst=convert(gold[adapter]),
                        outputs_1_lst=convert(outputs),
                        name_0="standalone",
                        name_1="DP-EP",
                    )
                    actual = _olmoe_continuation_scores(scores, gold[adapter])
                    tokens = [list(out.outputs[0].token_ids) for out in outputs]
                    if baseline:
                        reference[step, rank] = (tokens, actual)
                    else:
                        expected_tokens, expected = reference[step, rank]
                        assert tokens == expected_tokens
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        print(
                            f"OLMoE eager={eager} step={step} rank={rank} "
                            f"adapter={adapter} scored_tokens={len(actual)} "
                            "max_logprob_error=0"
                        )
        finally:
            engine.shutdown()


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=2)
def test_olmoe_preloaded_slots_dp2_ep(olmoe_two_lora_paths, eager, monkeypatch):
    """On-demand loading matches preloading across remote and idle EP ranks."""
    # Default BF16 LoRA split-K atomics vary even for repeated identical batches.
    # Deterministic kernels let this placement-only change require exact results.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    asyncio.run(_run_olmoe_preloaded_slots(olmoe_two_lora_paths, eager))


@pytest.fixture(scope="module")
def olmoe_dynamic_lora_paths(olmoe_two_lora_paths, tmp_path_factory):
    path = tmp_path_factory.mktemp("olmoe_dynamic_slots") / "third_lora"
    shutil.copytree(olmoe_two_lora_paths[0], path)
    weights_file = path / "adapter_model.safetensors"
    weights = load_file(str(weights_file))
    generator = torch.Generator().manual_seed(456)
    save_file(
        {
            name: (
                torch.randn(weight.shape, generator=generator)
                * weight.float().square().mean().sqrt()
            ).to(weight.dtype)
            for name, weight in weights.items()
        },
        str(weights_file),
    )
    bad_path = path.parent / "bad_attention_lora"
    shutil.copytree(olmoe_two_lora_paths[0], bad_path)
    # CPU loading accepts this tensor, but activation cannot copy it into QKV.
    name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    weights[name] = torch.cat([weights[name], weights[name][:, :1]], dim=1)
    save_file(weights, str(bad_path / "adapter_model.safetensors"))
    return [*olmoe_two_lora_paths, str(path)]


async def _check_dp_slot_layout(engine):
    layouts = await engine.engine_core.call_utility_all_async(
        "collective_rpc", "get_lora_slot_ids"
    )
    assert len(layouts) == 2
    assert layouts[0] == layouts[1], layouts
    return layouts[0][0]


async def _run_olmoe_dynamic_slots(paths, eager):
    scenarios = [
        ((0, 0), (0, 1), (0, 2), (1, 2), (1, 0), (1, 1)),
        ((0, 3), (0, 0), (0, 2), (1, 1), (1, 0), (1, 3)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (0, 3), (1, 0)),
        ((0, 1), (0, 0), (1, 1), (1, 2)),
    ]
    gold = {}
    reference_scores = {}

    def args(baseline):
        config = _olmoe_slot_args(dp_size=2, eager=eager)
        config.max_num_seqs = 16
        config.max_num_batched_tokens = 4096
        config.async_scheduling = False
        config.max_loras = 3 if baseline else 2
        return config

    engine = AsyncLLM.from_engine_args(args(True))
    try:
        for adapter, path in enumerate(paths, 1):
            assert await engine.add_lora(LoRARequest(str(adapter), adapter, path))
        scenario = ((0, 0), (0, 1), (0, 2), (1, 3))
        groups = await _olmoe_slot_outputs(engine, "gold", scenario, paths)
        gold = {adapter: group for (_, adapter), group in zip(scenario, groups)}
        _check([o.outputs[0].text.strip() for o in gold[1]], EXPECTED_LORA_OUTPUT)
        for a, b in ((0, 1), (1, 2), (2, 3)):
            assert any(
                x.outputs[0].token_ids != y.outputs[0].token_ids
                for x, y in zip(gold[a], gold[b])
            ), "Adapters must have distinguishable outputs"
        scored = await _olmoe_slot_outputs(
            engine, "gold-score", scenario, paths, gold=gold
        )
        reference_scores = {
            adapter: _olmoe_continuation_scores(group, gold[adapter])
            for (_, adapter), group in zip(scenario, scored)
        }
    finally:
        engine.shutdown()

    async def check(engine, tag, scenario, current_paths, aliases=None):
        aliases = aliases or {}
        expected_gold = {i: gold[aliases.get(i, i)] for i in gold}
        groups = await _olmoe_slot_outputs(engine, tag, scenario, current_paths)
        scored = await _olmoe_slot_outputs(
            engine, f"{tag}-score", scenario, current_paths, gold=expected_gold
        )
        for (rank, adapter), group, scores in zip(scenario, groups, scored):
            expected_adapter = aliases.get(adapter, adapter)
            assert [list(o.outputs[0].token_ids) for o in group] == [
                list(o.outputs[0].token_ids) for o in gold[expected_adapter]
            ]
            actual = _olmoe_continuation_scores(scores, expected_gold[adapter])
            torch.testing.assert_close(
                actual, reference_scores[expected_adapter], atol=0, rtol=0
            )
            print(
                f"OLMoE dynamic eager={eager} phase={tag} rank={rank} "
                f"adapter={adapter} scored_tokens={len(actual)} max_logprob_error=0"
            )
        await _check_dp_slot_layout(engine)

    engine = AsyncLLM.from_engine_args(args(False))
    try:
        for step, scenario in enumerate(scenarios):
            await check(engine, f"mixed-{step}", scenario, paths)

        # A failed prepare must not partially replace any rank's resident weights.
        before = await _check_dp_slot_layout(engine)
        with pytest.raises(Exception, match="DP LoRA preparation failed"):
            await engine.add_lora(LoRARequest("missing", 4, "/missing/adapter"))
        assert await _check_dp_slot_layout(engine) == before
        bad_path = Path(paths[2]).parent / "bad_attention_lora"
        with pytest.raises(Exception, match="DP LoRA commit failed"):
            await engine.add_lora(LoRARequest("bad-shape", 4, str(bad_path)))
        assert await _check_dp_slot_layout(engine) == before
        await check(engine, "after-failed-load", scenarios[-1], paths)

        # Replace ID 1 with adapter C, then evict/reload other adapters around it.
        assert await engine.add_lora(
            LoRARequest("c-as-a", 1, paths[2], load_inplace=True)
        )
        current_paths = [paths[2], paths[1], paths[2]]
        assert await engine.pin_lora(1)
        await check(engine, "after-replace", scenarios[1], current_paths, {1: 3})
        assert 1 in await _check_dp_slot_layout(engine)
        assert await engine.remove_lora(1)
        assert 1 not in await _check_dp_slot_layout(engine)
        assert await engine.add_lora(LoRARequest("a-restored", 1, paths[0]))
        await check(engine, "after-restore", scenarios[0], paths)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=2)
def test_olmoe_dynamic_slots_mixed_base(olmoe_dynamic_lora_paths, eager, monkeypatch):
    """Mixed base/A/B/C stay correct when global slots overflow, reload or change."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    asyncio.run(
        asyncio.wait_for(_run_olmoe_dynamic_slots(olmoe_dynamic_lora_paths, eager), 600)
    )


async def olmoe_queue_observer(request, call_next):
    """Test-only HTTP observation; keep the real inference path unchanged."""
    from fastapi.responses import JSONResponse

    if request.url.path not in ("/test/queued", "/test/slots"):
        return await call_next(request)
    engine = request.app.state.engine_client
    if request.url.path == "/test/slots":
        # Gather independent RPC responses in the frontend. A new worker-side
        # collective could interleave with a peer's final dummy forward pass.
        return JSONResponse(
            await engine.engine_core.call_utility_all_async(
                "collective_rpc", "get_lora_slot_ids"
            )
        )
    # This RPC is ordered after request adds on each engine's input socket.
    paused = await engine.is_paused()
    core = engine.engine_core
    return JSONResponse(
        {
            "paused": paused,
            "count": engine.get_num_unfinished_requests(),
            "per_rank": [core.engine_inflight[e] for e in core.core_engines],
        }
    )


def _olmoe_online_args(paths, dp, eager, baseline, prefix_caching=False):
    import json

    args = [
        "--host",
        "127.0.0.1",
        "--api-server-count",
        "1",
        "--served-model-name",
        "base",
        "--enable-lora",
        "--max-loras",
        "3" if baseline else "2",
        "--max-lora-rank",
        "8",
        "--data-parallel-size",
        str(dp),
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--enable-prefix-caching"
        if prefix_caching and not baseline
        else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--no-async-scheduling",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "4096",
        "--kv-cache-memory-bytes",
        str(1024**3),
        "--hf-overrides",
        json.dumps({"head_dtype": "float32"}),
        "--attention-config",
        json.dumps({"backend": "TRITON_ATTN"}),
        "--kernel-config",
        json.dumps({"moe_backend": "triton", "enable_flashinfer_autotune": False}),
        "--compilation-config",
        json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]}),
        "--middleware",
        f"{__name__}.olmoe_queue_observer",
    ]
    if eager:
        args.append("--enforce-eager")
    if baseline:
        args.extend(
            [
                "--lora-modules",
                *[f"{name}={path}" for name, path in zip(("a", "b", "c"), paths)],
            ]
        )
    return args


async def _olmoe_online_checks(
    server, paths, dp, eager, gold=None, prefix_caching=False
):
    import httpx

    async with httpx.AsyncClient(base_url=server.url_for(""), timeout=120) as client:

        async def post(path, body=None, **kwargs):
            response = await client.post(path, json=body, **kwargs)
            assert response.status_code == 200, response.text
            return response

        async def layout():
            response = await client.get("/test/slots")
            response.raise_for_status()
            layouts = [workers[0] for workers in response.json()]
            assert len(layouts) == dp, layouts
            assert all(row == layouts[0] for row in layouts), layouts
            return layouts[0]

        async def completion(model, prompt, rank=None):
            headers = {} if rank is None else {"X-data-parallel-rank": str(rank)}
            response = await post(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": PROMPTS[prompt],
                    "temperature": 0,
                    "max_tokens": 64,
                    "logprobs": 0,
                    "return_tokens_as_token_ids": True,
                },
                headers=headers,
            )
            body = response.json()
            choice = body["choices"][0]
            choice["cached_tokens"] = (
                body["usage"].get("prompt_tokens_details") or {}
            ).get("cached_tokens", 0)
            return choice

        if gold is None:
            # Static registration works with runtime HTTP updates disabled.
            for endpoint, body in (
                (
                    "/v1/load_lora_adapter",
                    {"lora_name": "extra", "lora_path": paths[0]},
                ),
                ("/v1/unload_lora_adapter", {"lora_name": "a"}),
            ):
                response = await client.post(endpoint, json=body)
                assert response.status_code == 404, response.text
            gold = {}
            for model in ("base", "a", "b", "c"):
                for prompt in range(len(PROMPTS)):
                    gold[model, prompt] = await completion(model, prompt, 0)
            _check(
                [gold["a", i]["text"].strip() for i in range(4)], EXPECTED_LORA_OUTPUT
            )
            for left, right in (("base", "a"), ("a", "b"), ("b", "c")):
                assert any(
                    gold[left, i]["text"] != gold[right, i]["text"] for i in range(4)
                )
            await layout()
            return gold

        async def load(name, path, inplace=False):
            await post(
                "/v1/load_lora_adapter",
                {
                    "lora_name": name,
                    "lora_path": path,
                    "load_inplace": inplace,
                },
            )

        async def wait_queued(count, expected_ranks=None):
            async with asyncio.timeout(30):
                while True:
                    response = await client.get("/test/queued")
                    response.raise_for_status()
                    state = response.json()
                    assert state["paused"]
                    if state["count"] == count and sum(state["per_rank"]) == count:
                        if expected_ranks is not None:
                            assert state["per_rank"] == expected_ranks, state
                        return
                    await asyncio.sleep(0.05)

        outputs_checked = 0
        scores_checked = 0

        async def batch(tag, requests, aliases=None, *, require_cache=False):
            nonlocal outputs_checked, scores_checked
            aliases = aliases or {}
            await post("/pause?mode=keep&clear_cache=false")
            tasks = [asyncio.create_task(completion(*request)) for request in requests]
            try:
                expected_ranks = None
                if all(rank is not None for _, _, rank in requests):
                    expected_ranks = [
                        sum(rank == r for _, _, rank in requests) for r in range(dp)
                    ]
                await wait_queued(len(tasks), expected_ranks)
                await post("/resume")
                outputs = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await post("/resume")
            for (model, prompt, _), output in zip(requests, outputs):
                expected = gold[aliases.get(model, model), prompt]
                if require_cache:
                    assert output["cached_tokens"] > 0, (tag, model, prompt, output)
                assert output["text"] == expected["text"], (tag, model, prompt)
                assert output["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
                actual_scores = output["logprobs"]["token_logprobs"]
                assert actual_scores == expected["logprobs"]["token_logprobs"], (
                    tag,
                    model,
                    prompt,
                )
                outputs_checked += 1
                scores_checked += len(actual_scores)
            await layout()
            print(
                f"ONLINE_PASS dp={dp} eager={eager} "
                f"phase={tag} requests={len(requests)}",
                flush=True,
            )

        # Check unload really reaches workers, not only the API name registry.
        await load("a", paths[0])
        old_slots = await layout()
        adapter_id = next(slot for slot in old_slots if slot is not None)
        await post("/v1/unload_lora_adapter", {"lora_name": "a"})
        assert adapter_id not in await layout(), "HTTP unload left adapter in GPU slots"
        response = await client.post(
            "/v1/completions", json={"model": "a", "prompt": "test", "max_tokens": 1}
        )
        assert response.status_code == 404, response.text
        # Reverse load order relative to the preloaded reference.
        await load("b", paths[1])
        await load("a", paths[0])
        cached_batch = [
            ("base", 0, dp - 1),
            ("a", 1, dp - 1),
            ("b", 2, dp - 1),
            ("base", 3, dp - 1),
        ]
        if prefix_caching:
            await batch("cache-warm", cached_batch)
            await batch("cache-hit", cached_batch, require_cache=True)
        for count in (max(1, dp // 2), dp, dp * 2, dp * 3):
            for kind, models in (
                ("base", ("base",)),
                ("single", ("a",)),
                ("multi", ("a", "b")),
                ("mixed", ("a", "b", "base", "base")),
            ):
                if count < len(models):
                    continue
                requests = [
                    (models[i % len(models)], i % 4, i % dp) for i in range(count)
                ]
                await batch(f"{kind}-{count}-spread", requests)
        for route in ("last", "partial", "auto"):
            requests = [
                (
                    ("a", "b", "base", "base")[i % 4],
                    i % 4,
                    dp - 1
                    if route == "last"
                    else i % max(1, dp // 2)
                    if route == "partial"
                    else None,
                )
                for i in range(dp * 2)
            ]
            await batch(f"mixed-{route}", requests)
        # Every model also runs with only one active rank and one request.
        for model in ("base", "a", "b"):
            await batch(f"isolated-{model}", [(model, 0, dp - 1)])
        before = await layout()
        response = await client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": "missing", "lora_path": "/missing/adapter"},
        )
        assert response.status_code >= 400
        assert await layout() == before
        await load("c", paths[2])
        mixed = [
            (model, i % 4, i % dp)
            for i, model in enumerate(("a", "b", "c", "base", "base", "a", "b", "c"))
        ]
        await batch("oversubscribed-slots", mixed)
        if prefix_caching:
            await batch("cache-hit-after-eviction", cached_batch, require_cache=True)
        await load("a", paths[1], inplace=True)
        await batch("replaced-a-with-b", mixed, {"a": "b"})
        await post("/v1/unload_lora_adapter", {"lora_name": "a"})
        await batch(
            "after-unload", [("base", 2, 0), ("base", 3, dp - 1), ("b", 1, dp - 1)]
        )
        await load("a", paths[0])
        await batch("reloaded-a", mixed)
        if prefix_caching:
            await batch("cache-hit-after-reload", cached_batch, require_cache=True)
        # Queued requests on a remote rank must prevent unload and replacement.
        await post("/pause?mode=keep&clear_cache=false")
        protected = asyncio.create_task(completion("a", 0, dp - 1))
        try:
            await wait_queued(1)
            for endpoint, body in (
                ("/v1/unload_lora_adapter", {"lora_name": "a"}),
                (
                    "/v1/load_lora_adapter",
                    {"lora_name": "a", "lora_path": paths[1], "load_inplace": True},
                ),
            ):
                response = await client.post(endpoint, json=body)
                assert response.status_code >= 400, response.text
                assert "unfinished requests" in response.text
            models = (await client.get("/v1/models")).json()["data"]
            assert "a" in {model["id"] for model in models}
            await post("/resume")
            output = await protected
            assert output["logprobs"] == gold["a", 0]["logprobs"]
        finally:
            if not protected.done():
                protected.cancel()
            await asyncio.gather(protected, return_exceptions=True)
            await post("/resume")
        await layout()
        print(
            f"ONLINE_TOTAL dp={dp} eager={eager} outputs={outputs_checked} "
            f"logprobs={scores_checked} max_error=0",
            flush=True,
        )
        return gold


def _test_olmoe_online(paths, dp, eager, prefix_caching=False):
    from tests.utils import RemoteOpenAIServer

    env = {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_LOG_STATS_INTERVAL": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    gold = None
    for baseline in (True, False):
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0" if baseline else "1"
        with RemoteOpenAIServer(
            MODEL_PATH,
            _olmoe_online_args(paths, dp, eager, baseline, prefix_caching),
            env_dict=env,
        ) as server:
            gold = asyncio.run(
                _olmoe_online_checks(server, paths, dp, eager, gold, prefix_caching)
            )


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=2)
def test_olmoe_online_dp2(olmoe_dynamic_lora_paths, eager):
    _test_olmoe_online(olmoe_dynamic_lora_paths, 2, eager)


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=4)
def test_olmoe_online_dp4(olmoe_dynamic_lora_paths, eager):
    _test_olmoe_online(olmoe_dynamic_lora_paths, 4, eager)


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=8)
def test_olmoe_online_dp8(olmoe_dynamic_lora_paths, eager):
    _test_olmoe_online(olmoe_dynamic_lora_paths, 8, eager)


@pytest.mark.parametrize("eager", [True, False], ids=["eager", "cudagraph"])
@multi_gpu_test(num_gpus=2)
def test_olmoe_online_prefix_cache_dp2(olmoe_dynamic_lora_paths, eager):
    """Real cache hits preserve base/multi-LoRA outputs across slot changes."""
    _test_olmoe_online(olmoe_dynamic_lora_paths, 2, eager, prefix_caching=True)
