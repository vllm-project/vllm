# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.hidden_states import (
    HiddenStatePlan,
    PendingHiddenOutput,
)
from vllm.entrypoints.generate.aux_outputs import hidden_states_response_fields
from vllm.exceptions import GenerationError
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "step, expected",
    [
        ((0, 16), (31, 31)),
        ((31, 32), (31, 32)),
        ((32, 36), (32, 34)),
        ((34, 38), (34, 34)),
    ],
)
def test_hidden_range_intersects_generated_logits(step, expected):
    assert HiddenStatePlan(32, 0, 34).select(*step) == expected


def test_rejected_suffix_is_relative_to_full_step_not_selected_window():
    rows = torch.arange(8).reshape(4, 2)
    pending = PendingHiddenOutput(
        ["r"],
        {"r": (100, 104, rows[:2])},
        rows[:0],
        rows={"r": rows[:2]},
        num_sampled=np.array([2]),
        num_rejected=np.array([2]),
    )
    start, accepted = pending.process_output()["r"]
    assert start == 100
    torch.testing.assert_close(accepted, rows[:2], rtol=0, atol=0)


def test_hidden_tensor_roundtrips_through_engine_ipc():
    rows = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    output = EngineCoreOutput(
        request_id="r",
        new_token_ids=[8, 9],
        hidden_states=rows,
        prompt_hidden_states=(3, rows),
    )
    decoded = MsgpackDecoder(EngineCoreOutput).decode(MsgpackEncoder().encode(output))
    torch.testing.assert_close(decoded.hidden_states, rows, rtol=0, atol=0)
    assert decoded.prompt_hidden_states[0] == 3
    torch.testing.assert_close(decoded.prompt_hidden_states[1], rows, rtol=0, atol=0)


def test_prompt_rows_cross_chunks_keep_first_versions_and_reject_missing_rows():
    processor = OutputProcessor(None, log_stats=True)
    params = SamplingParams(
        detokenize=False,
        max_tokens=1,
        aux_outputs=("prompt_hidden_states",),
        hidden_states_start=1,
        hidden_states_end=3,
    )
    assert params.skip_reading_prefix_cache
    request = EngineCoreRequest(
        request_id="p",
        external_req_id="p",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=params,
        pooling_params=None,
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    rows = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    processor.add_request(request, prompt=None)
    for version in ("first", "recomputed"):
        stats = IterationStats()
        partial = processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="p", new_token_ids=[], prompt_hidden_states=(1, rows[:1])
                )
            ],
            weight_version=version,
            engine_core_timestamp=1.0,
            iteration_stats=stats,
        )
        assert not partial.request_outputs
        assert not stats.time_to_first_tokens_iter
    stats = IterationStats()
    result = (
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="p",
                    new_token_ids=[5],
                    prompt_hidden_states=(2, rows[1:]),
                    finish_reason=FinishReason.LENGTH,
                )
            ],
            weight_version="second",
            engine_core_timestamp=2.0,
            iteration_stats=stats,
        )
        .request_outputs[0]
        .outputs[0]
    )
    assert len(stats.time_to_first_tokens_iter) == 1
    assert result.hidden_states is None
    assert result.prompt_hidden_states_start == 1
    torch.testing.assert_close(result.prompt_hidden_states, rows, atol=0, rtol=0)
    assert [
        (s.version, s.start, s.end) for s in result.prompt_hidden_states_weight_versions
    ] == [
        ("first", 0, 1),
        ("second", 1, 2),
    ]
    processor.add_request(request, prompt=None)
    with pytest.raises(GenerationError, match="prompt hidden-state rows"):
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="p",
                    new_token_ids=[5],
                    prompt_hidden_states=(1, rows[:1]),
                    finish_reason=FinishReason.LENGTH,
                )
            ]
        )
    assert processor.get_num_unfinished_requests() == 0


def test_frontend_delivers_window_only_at_completion_and_rejects_missing_rows():
    processor = OutputProcessor(None, log_stats=False)
    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        pooling_params=None,
        sampling_params=SamplingParams(
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
            aux_outputs=("hidden_states",),
            hidden_states_start=3,
            hidden_states_end=5,
        ),
    )
    processor.add_request(request, prompt=None)
    first = processor.process_outputs(
        [EngineCoreOutput(request_id="r", new_token_ids=[4])]
    ).request_outputs[0]
    assert first.outputs[0].hidden_states is None
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    last = (
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="r",
                    new_token_ids=[5, 6],
                    hidden_states=rows,
                    finish_reason=FinishReason.LENGTH,
                ),
            ]
        )
        .request_outputs[0]
        .outputs[0]
    )
    assert last.hidden_states_start == 3
    torch.testing.assert_close(last.hidden_states, rows, rtol=0, atol=0)
    processor.add_request(request, prompt=None)
    with pytest.raises(GenerationError, match="not delivered"):
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="r",
                    new_token_ids=[4, 5, 6],
                    hidden_states=rows[:1],
                    finish_reason=FinishReason.LENGTH,
                ),
            ]
        )
    assert processor.get_num_unfinished_requests() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_http_hidden_payload_preserves_bits(dtype):
    import io

    import pybase64 as base64

    rows = torch.tensor([[1.0, -2.0, 0.25]], dtype=dtype)
    output = CompletionOutput(
        0, "", [], None, None, hidden_states=rows, hidden_states_start=8
    )
    fields = hidden_states_response_fields(output)
    array = np.load(
        io.BytesIO(base64.b64decode(fields["hidden_states"])), allow_pickle=False
    )
    decoded = torch.from_numpy(array)
    if dtype == torch.bfloat16:
        decoded = decoded.view(dtype)
    assert fields["hidden_states_dtype"] == str(dtype).removeprefix("torch.")
    torch.testing.assert_close(decoded, rows, atol=0, rtol=0)
