# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor.abstract import Executor

from ...utils import create_new_process_for_each_test


def test_parent_request_to_output_stream() -> None:
    parent_request = ParentRequest(make_request(SamplingParams(n=2)))
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)

    # output_1 finished
    output_1.finish_reason = "ended"
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    # Finished output_1 had already returned, DO NOT returned again
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)

    # output_0 finished
    output_0.finish_reason = "ended"
    assert ([output_0], True) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)
    # Finished output_0 had already returned, DO NOT returned again
    assert parent_request.get_outputs("child_id_0", output_0) == ([], True)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)


def test_parent_request_to_output_final_only() -> None:
    parent_request = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY))
    )
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished, return nothing
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_1 finished, but outputs won't be returned until all child requests finished
    output_1.finish_reason = "ended"
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_0 finished, as all child requests finished, the output would be returned
    output_0.finish_reason = "ended"
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_1", output_1
    )


@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
@create_new_process_for_each_test()
def test_llm_engine_add_request_step_parallel_sampling(
    output_kind: RequestOutputKind,
) -> None:
    """Regression test for
    https://github.com/vllm-project/vllm/issues/21948: exercise parallel
    sampling (n>1) directly through `LLMEngine.add_request()`/`step()` for
    every `output_kind`. Previously only `LLM.generate()` (which always
    enforces `FINAL_ONLY`, see `test_llm_engine.py::test_parallel_sampling`)
    had coverage at this layer, so the `CUMULATIVE`/`DELTA` combinations here
    -- and specifically the "`LLMEngine` + `CUMULATIVE`" combination the
    issue calls out as suspect -- were unverified.
    """
    n = 2
    engine_args = EngineArgs(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=64,
    )
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine = LLMEngine(
        vllm_config=vllm_config, executor_class=executor_class, log_stats=False
    )

    sampling_params = SamplingParams(
        n=n, output_kind=output_kind, max_tokens=8, temperature=0.9, seed=0
    )
    engine.add_request("req-0", "Hello, my name is", sampling_params)

    texts: dict[int, str] = {i: "" for i in range(n)}
    finished: set[int] = set()
    steps = 0
    while len(finished) < n and steps < 200:
        for out in engine.step():
            for comp in out.outputs:
                if output_kind == RequestOutputKind.DELTA:
                    # DELTA: each update carries only the newly generated
                    # fragment -- reconstruct the full text by concatenating.
                    texts[comp.index] += comp.text
                elif output_kind == RequestOutputKind.CUMULATIVE:
                    # CUMULATIVE: each update must extend (never diverge
                    # from) the text already seen for this index -- this is
                    # exactly the property #21948 suspected was broken for
                    # LLMEngine + CUMULATIVE specifically.
                    assert comp.text.startswith(texts[comp.index]), (
                        f"CUMULATIVE text for index {comp.index} did not "
                        f"extend monotonically: {texts[comp.index]!r} -> "
                        f"{comp.text!r}"
                    )
                    texts[comp.index] = comp.text
                else:
                    # FINAL_ONLY: one update per index, carrying its full
                    # final text.
                    texts[comp.index] = comp.text
                if comp.finish_reason is not None:
                    finished.add(comp.index)
        steps += 1

    assert steps < 200, "engine never finished all child requests"
    assert finished == set(range(n))
    assert set(texts) == set(range(n))
    assert all(texts[i] for i in range(n)), texts


def make_request(sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent_id",
        external_req_id="ext_parent_id",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
