# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm import SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest

pytestmark = pytest.mark.skip_global_cleanup


class _FakeAsyncEngineCore:
    def __init__(self) -> None:
        self.added_requests: list[EngineCoreRequest] = []
        self.resources = SimpleNamespace(engine_dead=False)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self.added_requests.append(request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        return None

    def shutdown(self, timeout: float | None = None) -> None:
        return None


class _FakeSyncEngineCore:
    def __init__(self) -> None:
        self.added_requests: list[EngineCoreRequest] = []
        self.output_batches: list[EngineCoreOutputs] = []
        self.aborted_requests: list[list[str]] = []
        self.executed_dummy_batch = False

    def add_request(self, request: EngineCoreRequest) -> None:
        self.added_requests.append(request)

    def get_output(self) -> EngineCoreOutputs:
        assert self.output_batches
        return self.output_batches.pop(0)

    def abort_requests(self, request_ids: list[str]) -> None:
        self.aborted_requests.append(request_ids)

    def execute_dummy_batch(self) -> None:
        self.executed_dummy_batch = True

    def dp_engines_running(self) -> bool:
        return False


def _make_mock_async_llm() -> AsyncLLM:
    llm = MagicMock(spec=AsyncLLM)
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    llm.model_config = SimpleNamespace(max_model_len=2048)
    llm.log_requests = False
    llm.errored = False
    llm.output_handler = None
    llm.input_processor = SimpleNamespace(assign_request_id=lambda _: None)
    llm.output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=False,
        stream_interval=1,
        tracing_enabled=False,
    )
    llm.engine_core = _FakeAsyncEngineCore()
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()
    llm._add_request = AsyncLLM._add_request.__get__(llm, AsyncLLM)
    llm.add_request = AsyncLLM.add_request.__get__(llm, AsyncLLM)
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)
    return llm


def _make_mock_llm_engine() -> LLMEngine:
    llm_engine = LLMEngine.__new__(LLMEngine)
    llm_engine.model_config = SimpleNamespace(max_model_len=2048)
    llm_engine.log_stats = False
    llm_engine.logger_manager = None
    llm_engine.should_execute_dummy_batch = False
    llm_engine.input_processor = SimpleNamespace(assign_request_id=lambda _: None)
    llm_engine.output_processor = OutputProcessor(
        tokenizer=None,
        log_stats=False,
        stream_interval=1,
        tracing_enabled=False,
    )
    llm_engine.engine_core = _FakeSyncEngineCore()
    return llm_engine


def _make_child_outputs(
    child_req_0: str,
    child_req_1: str,
) -> list[EngineCoreOutput]:
    return [
        EngineCoreOutput(request_id=child_req_0, new_token_ids=[11]),
        EngineCoreOutput(request_id=child_req_1, new_token_ids=[21]),
        EngineCoreOutput(
            request_id=child_req_0,
            new_token_ids=[12],
            finish_reason=FinishReason.LENGTH,
        ),
        EngineCoreOutput(
            request_id=child_req_1,
            new_token_ids=[22],
            finish_reason=FinishReason.LENGTH,
        ),
    ]


def _assert_child_request_fanout(
    added_requests: list[EngineCoreRequest], parent_request_id: str
) -> None:
    assert len(added_requests) == 2
    children = {req.request_id: req for req in added_requests}
    assert set(children) == {f"0_{parent_request_id}", f"1_{parent_request_id}"}

    child_0 = children[f"0_{parent_request_id}"]
    child_1 = children[f"1_{parent_request_id}"]
    assert child_0.sampling_params is not None
    assert child_1.sampling_params is not None
    assert child_0.sampling_params.n == 1
    assert child_1.sampling_params.n == 1
    assert child_0.sampling_params.seed == 123
    assert child_1.sampling_params.seed == 124


def _assert_parallel_sampling_outputs(
    outputs: list[RequestOutput],
    output_kind: RequestOutputKind,
    request_id: str,
) -> None:
    assert all(output.request_id == request_id for output in outputs)

    if output_kind == RequestOutputKind.FINAL_ONLY:
        assert len(outputs) == 1
        final_output = outputs[0]
        assert final_output.finished
        assert [out.index for out in final_output.outputs] == [0, 1]
        assert [list(out.token_ids) for out in final_output.outputs] == [
            [11, 12],
            [21, 22],
        ]
        return

    assert len(outputs) == 4
    assert [output.finished for output in outputs] == [False, False, False, True]

    token_history: dict[int, list[list[int]]] = {0: [], 1: []}
    for output in outputs:
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        token_history[completion.index].append(list(completion.token_ids))

    if output_kind == RequestOutputKind.DELTA:
        assert token_history == {0: [[11], [12]], 1: [[21], [22]]}
    else:
        assert output_kind == RequestOutputKind.CUMULATIVE
        assert token_history == {0: [[11], [11, 12]], 1: [[21], [21, 22]]}
        for history in token_history.values():
            assert len(history[0]) < len(history[1])


async def _emit_async_outputs(llm: AsyncLLM, request_id: str) -> None:
    while len(llm.output_processor.request_states) < 2:
        await asyncio.sleep(0)

    req_state = next(iter(llm.output_processor.request_states.values()))
    assert req_state.queue is not None
    queue = req_state.queue
    child_outputs = _make_child_outputs(f"0_{request_id}", f"1_{request_id}")
    for idx, child_output in enumerate(child_outputs):
        llm.output_processor.process_outputs(
            [child_output], engine_core_timestamp=float(idx + 1)
        )
        while queue.output is not None:
            await asyncio.sleep(0)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
async def test_async_llm_generate_parallel_sampling_output_kind_matrix(
    output_kind: RequestOutputKind,
) -> None:
    request_id = "async-parent-id"
    sampling_params = SamplingParams(
        n=2,
        seed=123,
        max_tokens=2,
        detokenize=False,
        output_kind=output_kind,
    )
    request = make_request(
        sampling_params, request_id=request_id, external_req_id=request_id
    )

    llm = _make_mock_async_llm()
    output_task = asyncio.create_task(_emit_async_outputs(llm, request_id))
    outputs = [
        output
        async for output in llm.generate(
            prompt=request,
            sampling_params=sampling_params,
            request_id=request_id,
        )
    ]
    await output_task

    _assert_child_request_fanout(llm.engine_core.added_requests, request_id)
    _assert_parallel_sampling_outputs(outputs, output_kind, request_id)


@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
def test_llm_engine_parallel_sampling_output_kind_matrix(
    output_kind: RequestOutputKind,
) -> None:
    request_id = "llm-engine-parent-id"
    sampling_params = SamplingParams(
        n=2,
        seed=123,
        max_tokens=2,
        detokenize=False,
        output_kind=output_kind,
    )
    request = make_request(
        sampling_params, request_id=request_id, external_req_id=request_id
    )

    llm_engine = _make_mock_llm_engine()
    returned_request_id = llm_engine.add_request(request_id, request, sampling_params)
    assert returned_request_id == request_id

    _assert_child_request_fanout(llm_engine.engine_core.added_requests, request_id)

    llm_engine.engine_core.output_batches = [
        EngineCoreOutputs(
            outputs=_make_child_outputs(f"0_{request_id}", f"1_{request_id}")[:2],
            timestamp=1.0,
        ),
        EngineCoreOutputs(
            outputs=_make_child_outputs(f"0_{request_id}", f"1_{request_id}")[2:],
            timestamp=2.0,
        ),
    ]

    request_outputs = [*llm_engine.step(), *llm_engine.step()]
    assert all(isinstance(output, RequestOutput) for output in request_outputs)
    _assert_parallel_sampling_outputs(request_outputs, output_kind, request_id)


def make_request(
    sampling_params: SamplingParams,
    request_id: str = "parent_id",
    external_req_id: str = "ext_parent_id",
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=external_req_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
