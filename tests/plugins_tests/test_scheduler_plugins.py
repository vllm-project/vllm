# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from examples.features.adaptive_budget.scheduler import (
    AdaptiveBudgetScheduler,
    BudgetConfig,
    BudgetController,
)
from tests.models.utils import check_logprobs_close
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


class DummyV1Scheduler(Scheduler):
    def schedule(self, throttle_prefills: bool = False):
        raise Exception("Exception raised by DummyV1Scheduler")


def test_scheduler_plugins_v1(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        with pytest.raises(Exception) as exception_info:
            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV1Scheduler,
            )

            engine = LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(exception_info.value) == "Exception raised by DummyV1Scheduler"


@pytest.mark.parametrize(
    "config",
    [
        {"low_budget": 1.5},
        {"low_budget": True},
        {"high_budget": 512},
        {"max_stall_ms": float("nan")},
        {"max_stall_ms": float("inf")},
        {"enter_prefills": 0},
        {"exit_prefills": 2},
        {"guard_interval_steps": 0},
        {"guard_enabled": "false"},
        {"misspelled_budget": 1024},
    ],
)
def test_adaptive_budget_rejects_invalid_config(config):
    with pytest.raises(ValueError):
        BudgetConfig(**config)


def test_adaptive_budget_dwell_and_unmixed_release():
    controller = BudgetController(
        BudgetConfig(
            enter_prefills=3, exit_prefills=1, min_dwell_steps=3, guard_enabled=False
        )
    )
    budgets = [controller.decide(p, 4, 0) for p in [3, 1, 1, 1, 3, 3, 3]]
    assert budgets == [1024] * 3 + [8192] * 3 + [1024]
    assert controller.decide(3, 0, 0) == 8192


def test_adaptive_budget_guard_is_rate_limited():
    controller = BudgetController(BudgetConfig(guard_interval_steps=3))
    assert [controller.decide(3, 4, 500) for _ in range(7)] == [
        4096,
        1024,
        1024,
        4096,
        1024,
        1024,
        4096,
    ]


def test_adaptive_budget_preserves_capacity_and_request_lifecycle():
    scheduler = create_scheduler(
        model="Qwen/Qwen3-1.7B",
        scheduler_cls=AdaptiveBudgetScheduler,
        additional_config={"adaptive_budget": {"enter_prefills": 1}},
    )
    try:
        decode = create_requests(1, num_tokens=32, req_ids=["decode"])[0]
        scheduler.add_request(decode)
        first = scheduler.schedule()
        scheduler.update_from_output(
            first,
            ModelRunnerOutput(
                req_ids=["decode"],
                req_id_to_index={"decode": 0},
                sampled_token_ids=[[42]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        prefill = create_requests(1, num_tokens=8000, req_ids=["prefill"])[0]
        scheduler.add_request(prefill)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"decode": 1, "prefill": 1023}
        assert scheduler.max_num_scheduled_tokens == 8192
        assert scheduler.scheduler_config.max_num_batched_tokens == 8192
        scheduler.finish_requests(["decode", "prefill"], RequestStatus.FINISHED_ABORTED)
        assert scheduler.get_num_unfinished_requests() == 0
        assert scheduler.schedule().total_num_scheduled_tokens == 0
    finally:
        scheduler.shutdown()


def test_adaptive_budget_rejects_capacity_overflow():
    with pytest.raises(ValueError, match="capacity"):
        create_scheduler(
            model="Qwen/Qwen3-1.7B",
            scheduler_cls=AdaptiveBudgetScheduler,
            max_num_batched_tokens=4096,
        )


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_adaptive_budget_generation(vllm_runner, enable_prefix_caching):
    """Compare native generation across different mixed prefill/decode shapes."""
    params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, logprobs=5)
    outputs = []
    for scheduler_cls in (Scheduler, AdaptiveBudgetScheduler):
        with vllm_runner(
            maybe_model_redirect("Qwen/Qwen3-1.7B"),
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=512,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            async_scheduling=False,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            scheduler_cls=scheduler_cls,
            additional_config={
                "adaptive_budget": {
                    "high_budget": 512,
                    "low_budget": 128,
                    "guard_budget": 256,
                    "enter_prefills": 1,
                }
            },
        ) as runner:
            tokenizer = runner.llm.get_tokenizer()
            short = tokenizer.encode("What is the capital of France? Explain briefly.")
            body = tokenizer.encode(
                "A scheduler shares compute between input and output tokens. "
            )
            prompts = [
                {"prompt_token_ids": short},
                {"prompt_token_ids": (body * 200)[:1536]},
                {"prompt_token_ids": (body * 220)[:2048]},
            ]
            batches = [runner.llm.generate(prompts, params)]
            if enable_prefix_caching:
                batches.append(runner.llm.generate(prompts, params))
                assert sum(out.num_cached_tokens for out in batches[1]) > 0
            completions = [out.outputs[0] for batch in batches for out in batch]
            assert all(
                len(out.token_ids) == 64 and out.finish_reason == "length"
                for out in completions
            )
            outputs.append(
                [(list(out.token_ids), out.text, out.logprobs) for out in completions]
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="native",
        name_1="adaptive",
    )
