# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded sessions retain one reply's frontend state and preserve turn outputs."""

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.outputs import STREAM_FINISHED
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    RequestState,
    StreamingInputOutputCollector,
)
from vllm.v1.outputs import LogprobsLists
from vllm.v1.streaming.retention import StreamingRetentionParams

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tokenizer():
    vocab = {
        char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    return TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")


def request(
    tokenizer, turn=0, *, bounded=True, resumable=True, kind=RequestOutputKind.DELTA
):
    return EngineCoreRequest(
        request_id="internal",
        external_req_id="external",
        prompt_token_ids=tokenizer.encode(f"prompt{turn}", add_special_tokens=False),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=8,
            output_kind=kind,
            logprobs=0,
            extra_args=(
                {"streaming_retention": StreamingRetentionParams()} if bounded else None
            ),
        ),
        pooling_params=None,
        arrival_time=float(turn),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


def output(token_id, *, finished=False):
    return EngineCoreOutput(
        request_id="internal",
        new_token_ids=[token_id],
        new_logprobs=LogprobsLists(
            logprob_token_ids=np.array([[token_id]], dtype=np.int32),
            logprobs=np.array([[-0.25]], dtype=np.float32),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        ),
        finish_reason=FinishReason.LENGTH if finished else None,
    )


@pytest.mark.parametrize("queued", [False, True])
def test_many_turns_bound_frontend_history_and_keep_utf8_deltas_and_logprobs(
    tokenizer, queued
):
    processor = OutputProcessor(tokenizer, log_stats=False, stream_interval=2)
    processor.add_request(request(tokenizer), "prompt0")
    byte_ids = tokenizer.encode("é", add_special_tokens=False)
    assert len(byte_ids) == 2
    snapshots = []

    for turn in range(20):
        current_prompt = tokenizer.encode(f"prompt{turn}", add_special_tokens=False)
        if queued and turn < 19:
            processor.add_request(request(tokenizer, turn + 1), f"prompt{turn + 1}")
        [first] = processor.process_outputs([output(byte_ids[0])]).request_outputs
        [last] = processor.process_outputs(
            [output(byte_ids[1], finished=True)]
        ).request_outputs
        assert first.outputs[0].text == ""
        assert last.outputs[0].text == "é"
        assert last.outputs[0].token_ids == byte_ids[1:]
        assert last.outputs[0].finish_reason == "length"
        assert not last.finished
        assert len(last.outputs[0].logprobs) == 1
        assert last.outputs[0].cumulative_logprob == -0.5 * (turn + 1)
        snapshots.append((last.prompt_token_ids, current_prompt))
        if not queued and turn < 19:
            processor.add_request(request(tokenizer, turn + 1), f"prompt{turn + 1}")
        state = processor.request_states["internal"]
        assert len(state.prompt_token_ids) <= len("prompt19")
        assert state.detokenizer.num_output_tokens() <= len(byte_ids)
        assert len(state.logprobs_processor.logprobs) <= len(byte_ids)
        assert state.prompt_len == len(state.prompt_token_ids)

    # Earlier RequestOutputs must not alias a list extended by later turns.
    assert all(tokens == expected for tokens, expected in snapshots)


def test_unbounded_streaming_keeps_its_existing_output_state(tokenizer):
    processor = OutputProcessor(tokenizer, log_stats=False)
    initial = request(tokenizer, bounded=False)
    continuation = request(tokenizer, 1, bounded=False)
    expected = list(initial.prompt_token_ids) + continuation.prompt_token_ids
    processor.add_request(initial, "prompt0")
    state = processor.request_states["internal"]
    decoder = state.detokenizer
    processor.process_outputs([output(tokenizer.encode("A")[0], finished=True)])
    processor.add_request(continuation, "prompt1")
    assert type(state) is RequestState
    assert state.detokenizer is decoder
    assert state.prompt_token_ids == expected


def test_bounded_session_keeps_the_terminal_sentinel_contract(tokenizer):
    processor = OutputProcessor(tokenizer, log_stats=False)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "internal")
    processor.add_request(request(tokenizer), "prompt0", queue=queue)
    processor.process_outputs([output(tokenizer.encode("A")[0], finished=True)])
    boundary = queue.get_nowait()
    assert boundary.outputs[0].finish_reason == "length"
    assert not boundary.finished
    processor.add_request(request(tokenizer, resumable=False), None, queue=queue)
    assert queue.get_nowait() is STREAM_FINISHED
    assert not processor.has_request("internal")


def test_bounded_output_rejects_cumulative_history(tokenizer):
    processor = OutputProcessor(tokenizer, log_stats=False)
    with pytest.raises(ValueError, match="DELTA"):
        processor.add_request(
            request(tokenizer, kind=RequestOutputKind.CUMULATIVE), "prompt0"
        )
    assert not processor.has_request("internal")
