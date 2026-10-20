# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=2,
        max_num_batched_tokens=6,
        max_num_seqs=3,
        block_size=16,
    )


def test_concurrent_partial_prefill(llm):
    outputs = llm.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded(llm):
    # 17 tokens will make sure first 16 tokens are cached in a block
    input_tokens = {"prompt_token_ids": [101] * 17}
    _ = llm.generate([input_tokens])
    outputs = llm.generate([input_tokens])
    assert outputs[0].num_cached_tokens == 16


@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize(
    "model,tp,pp,ep",
    [
        ("Qwen/Qwen3-0.6B", 1, 1, False),
        ("Qwen/Qwen3-0.6B", 1, 2, False),
        ("Qwen/Qwen3-30B-A3B", 2, 2, True),
    ],
)
def test_layered_prefill_queued_decode_and_prefill_match_greedy_reference(
    vllm_runner, monkeypatch, model, tp, pp, ep, enforce_eager
):
    """Queued PP preserves greedy tokens and logprobs while prefills overlap."""
    if torch.accelerator.device_count() < tp * pp:
        pytest.skip(f"Requires {tp * pp} GPUs")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    steps = 2 * pp
    params = SamplingParams(
        temperature=0,
        max_tokens=2 * steps + 4,
        ignore_eos=True,
        detokenize=False,
        logprobs=2,
    )
    with vllm_runner(
        model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        enable_expert_parallel=ep,
        distributed_executor_backend="mp",
        num_layer_groups=2,
        async_scheduling=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
        },
    ) as runner:
        prompt = {"prompt_token_ids": [101] * 8}
        reference = runner.llm.generate([prompt], params.clone(), use_tqdm=False)[
            0
        ].outputs[0]
        engine = runner.llm.llm_engine
        core = engine.engine_core.engine_core
        scheduler = core.scheduler
        assert core.batch_queue_size == pp
        plans = []
        original_schedule = scheduler.schedule

        def schedule(*args, **kwargs):
            plan = original_schedule(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(scheduler, "schedule", schedule)
        engine.add_request("decode", prompt, params)
        for _ in range(steps + pp + 2):
            outputs = engine.step()
            if outputs:
                break
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == 1
        for req_id, length in (("prefill-a", 17), ("prefill-b", 46)):
            engine.add_request(req_id, {"prompt_token_ids": [102] * length}, params)
        plans.clear()
        previous_tokens = 1
        for tick in range(8 * steps + 16):
            outputs = {out.request_id: out for out in engine.step()}
            if tick == 0:
                engine.add_request("late", {"prompt_token_ids": [103] * 8}, params)
            if "decode" in outputs:
                decode = outputs["decode"].outputs[0]
                count = len(decode.token_ids)
                assert count == previous_tokens + 1
                previous_tokens = count
            if not engine.has_unfinished_requests():
                break
        assert not engine.has_unfinished_requests()
        assert previous_tokens == params.max_tokens
        check_logprobs_close(
            outputs_0_lst=[(reference.token_ids, "", reference.logprobs)],
            outputs_1_lst=[(decode.token_ids, "", decode.logprobs)],
            name_0="alone",
            name_1="layered",
        )
        layered = [p for p in plans if p.layered_prefill_outputs is not None]
        for plan in layered:
            assert all(
                stage.total_num_scheduled_tokens
                + plan.layered_decode_output.total_num_scheduled_tokens
                <= scheduler.max_num_scheduled_tokens
                for stage in plan.layered_prefill_outputs
            )

        def external(stage):
            # The engine appends a random suffix to each external request ID.
            return {
                req_id.rsplit("-", 1)[0]: num_tokens
                for req_id, num_tokens in stage.num_scheduled_tokens.items()
            }

        assert any(
            external(p.layered_prefill_outputs[0]) == {"prefill-a": 17, "prefill-b": 46}
            for p in layered
        )
        if pp == 2:
            assert any(
                "late" in external(p.layered_prefill_outputs[0])
                and external(p.layered_prefill_outputs[1]).keys()
                >= {"prefill-a", "prefill-b"}
                for p in layered
            )
