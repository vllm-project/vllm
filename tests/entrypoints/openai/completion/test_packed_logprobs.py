# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed wire path preserves engine scores without candidate objects/text."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pybase64
import pytest
import torch

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.packed_logprobs import (
    create_packed_completion_logprobs,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


@pytest.fixture
def should_do_global_cleanup_after_test():
    # These tests never initialize an accelerator or distributed state.
    return False


def test_packed_head_tail_ties_and_infinity_without_materialization():
    scores = FlatLogprobs()
    for ids, values, rank in [
        ([9, 2, 3], [-4.0, -0.5, -1.5], 3),
        ([2, 2, 3], [-0.5, -0.5, -1.5], 1),
        ([9, 2, 3], [-1.0, -1.0, -1.0], 3),
        ([2, 2, 3], [0.0, 0.0, -np.inf], 1),
    ]:
        append_logprobs_for_next_position(scores, ids, values, [None] * 3, rank, 2)
    with patch.object(FlatLogprobs, "__getitem__", side_effect=AssertionError):
        result = create_packed_completion_logprobs([9, 2, 9, 2], scores, 2)
    packed = result.top_k
    ids = np.frombuffer(pybase64.b64decode(packed.token_ids), "<i4").reshape(4, 2)
    values = np.frombuffer(pybase64.b64decode(packed.logprobs), "<f4").reshape(4, 2)
    np.testing.assert_array_equal(ids, [[2, 3], [2, 3], [2, 3], [2, 3]])
    np.testing.assert_array_equal(
        values, [[-0.5, -1.5], [-0.5, -1.5], [-1, -1], [0, -np.inf]]
    )
    assert result.token_logprobs == [-4.0, -0.5, -1.0, 0.0]
    assert result.top_logprobs == []
    assert "Infinity" not in result.model_dump_json()
    assert "top_k" not in CompletionLogProbs().model_dump()

    scores.logprobs[0] = -np.inf
    assert (
        create_packed_completion_logprobs([9, 2, 9, 2], scores, 2).token_logprobs[0]
        == -9999.0
    )


def test_empty_packed_completion():
    result = create_packed_completion_logprobs([], FlatLogprobs(), 2)
    assert (result.top_k.num_positions, result.top_k.k) == (0, 2)
    assert result.top_k.token_ids == ""
    assert result.text_offset == []


@pytest.mark.parametrize(
    "override",
    [
        {"echo": True},
        {"use_beam_search": True},
        {"logprobs": 0},
        {"logprobs": -1},
        {"logprobs": None},
        {"logprob_token_ids": [1]},
        {"return_tokens_as_token_ids": False},
    ],
)
def test_reject_unsupported_packed_requests(override):
    request = dict(
        model="test",
        prompt=[1],
        logprobs=2,
        return_top_k_logprobs=True,
        return_tokens_as_token_ids=True,
    )
    with pytest.raises(ValueError):
        CompletionRequest(**(request | override))


@pytest.mark.parametrize("stream", [False, True])
def test_packed_request_uses_flat_engine_scores_without_candidate_decoding(stream):
    request = CompletionRequest(
        model="test",
        prompt=[1],
        logprobs=2,
        return_top_k_logprobs=True,
        return_tokens_as_token_ids=True,
        stream=stream,
        prompt_logprobs=None if stream else 1,
    )
    params = request.to_sampling_params(16)
    tokenizer = Mock()
    processor = LogprobsProcessor.from_new_request(
        tokenizer, Mock(sampling_params=params)
    )
    processor._update_sample_logprobs(
        LogprobsLists(
            np.array([[9, 2, 3]], dtype=np.int32),
            np.array([[-4.0, -0.5, -1.5]], dtype=np.float32),
            np.array([3], dtype=np.int32),
        )
    )
    assert isinstance(processor.logprobs, FlatLogprobs)
    assert processor.logprobs.token_ids == [9, 2, 3]
    assert processor.logprobs.decoded_tokens == [None, None, None]
    assert tokenizer.mock_calls == []
    if not stream:
        with patch(
            "vllm.v1.engine.logprobs.convert_ids_list_to_tokens",
            return_value=["prompt", "prompt"],
        ) as decode:
            processor._update_prompt_logprobs(
                LogprobsTensors(
                    torch.tensor([[1, 1]]),
                    torch.tensor([[-0.5, -0.5]]),
                    torch.tensor([1]),
                )
            )
        decode.assert_called_once()
        assert processor.prompt_logprobs[1][1].decoded_token == "prompt"


@pytest.mark.asyncio
async def test_packed_stream_slices_preserve_tail_scores_offsets_and_usage():
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_force_include_usage = False
    serving.enable_prompt_tokens_details = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None
    scores = FlatLogprobs()
    for _ in range(4):
        append_logprobs_for_next_position(
            scores, [9, 2, 3], [-4.0, -0.5, -1.5], [None] * 3, 3, 2
        )

    async def outputs():
        for start, end in ((0, 1), (1, 4), (4, 4)):
            yield (
                0,
                RequestOutput(
                    request_id="test",
                    prompt="",
                    prompt_token_ids=[1],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="x" * (end - start),
                            token_ids=[9] * (end - start),
                            cumulative_logprob=None,
                            logprobs=scores[start:end],
                            finish_reason="length" if start == end else None,
                        )
                    ],
                    finished=start == end,
                ),
            )

    request = CompletionRequest(
        model="test",
        prompt=[1],
        max_tokens=4,
        stream=True,
        stream_options={"include_usage": True},
        logprobs=2,
        return_top_k_logprobs=True,
        return_token_ids=True,
        return_tokens_as_token_ids=True,
    )
    # Slicing stays flat; converting any candidate row to Logprob objects fails.
    original = FlatLogprobs.__getitem__

    def slice_only(self, index):
        assert isinstance(index, slice)
        return original(self, index)

    with patch.object(FlatLogprobs, "__getitem__", slice_only):
        events = [
            event
            async for event in serving.completion_stream_generator(
                request,
                [],
                outputs(),
                "test",
                0,
                "test",
                1,
                None,
                RequestResponseMetadata(request_id="test"),
            )
        ]
    assert events[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
    assert [c["token_ids"] for c in choices] == [[9], [9, 9, 9], []]
    assert choices[-1]["finish_reason"] == "length"
    assert choices[1]["logprobs"]["text_offset"][0] == len("token_id:9")
    assert chunks[-1]["usage"]["completion_tokens"] == 4
    for choice in choices:
        lp = choice["logprobs"]
        count = len(choice["token_ids"])
        assert lp["token_logprobs"] == [-4.0] * count
        assert not lp.get("top_logprobs")
        packed = lp["top_k"]
        assert packed["num_positions"] == count
        assert packed["k"] == 2
        ids = np.frombuffer(pybase64.b64decode(packed["token_ids"]), "<i4").reshape(
            count, 2
        )
        np.testing.assert_array_equal(ids, np.tile([2, 3], (count, 1)))

    response = serving.request_output_to_completion_response(
        [
            RequestOutput(
                request_id="test",
                prompt="",
                prompt_token_ids=[1],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="xxxx",
                        token_ids=[9] * 4,
                        cumulative_logprob=None,
                        logprobs=scores,
                        finish_reason="length",
                    )
                ],
                finished=True,
            )
        ],
        request.model_copy(update={"stream": False}),
        "test",
        0,
        "test",
        None,
        RequestResponseMetadata(request_id="test"),
    )
    full = response.choices[0].logprobs
    assert full is not None
    assert full.token_logprobs == [-4.0] * 4
    assert full.text_offset == [0, 10, 20, 30]
    assert [
        offset for choice in choices for offset in choice["logprobs"]["text_offset"]
    ] == full.text_offset
    assert response.usage.completion_tokens == 4
