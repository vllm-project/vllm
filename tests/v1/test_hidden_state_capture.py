# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.hidden_state_capture import (
    HiddenStateCaptureBuffer,
    HiddenStateCapturePlan,
    HiddenStateCaptureState,
    accepted_hidden_range,
    capture_scheduled_hidden_states,
    drop_incompatible_aux_plans,
    hidden_state_capture_capability,
)


def test_hidden_capture_window_uses_teacher_forcing_positions():
    response = HiddenStateCapturePlan.from_window("req", 17, 1000, 2000, "collection")
    absolute = HiddenStateCapturePlan.from_window(
        "req", 17, 1016, 2016, "collection", coordinate="absolute"
    )
    assert (response.window_start_abs, response.window_end_abs) == (1016, 2016)
    assert response == absolute
    with pytest.raises(ValueError):
        HiddenStateCapturePlan.from_window("req", 17, 2000, 1000, "collection")


def test_hidden_capture_selects_only_each_requests_window_rows():
    plans = {
        "a": HiddenStateCapturePlan.from_window("a", 4, 0, 2, "a"),
        "b": HiddenStateCapturePlan.from_window("b", 2, 1, 3, "b"),
    }
    hidden = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    chunks = capture_scheduled_hidden_states(
        plans,
        ["a", "b", "c"],
        {"a": 4, "b": 3, "c": 2},
        {"a": 0, "b": 2, "c": 8},
        hidden,
    )
    np.testing.assert_array_equal(chunks["a"].positions, [3])
    torch.testing.assert_close(chunks["a"].hidden_states, hidden[3:4])
    np.testing.assert_array_equal(chunks["b"].positions, [2, 3])
    torch.testing.assert_close(chunks["b"].hidden_states, hidden[4:6])
    assert "c" not in chunks

    after_window = capture_scheduled_hidden_states(
        plans, ["a"], {"a": 2}, {"a": 5}, hidden
    )
    assert after_window == {}


def test_hidden_capture_discards_rejected_speculative_rows_and_short_samples():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 10, 13, "collection", min_rows=2
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 3}, {"req": 10}, hidden
    )["req"]

    buffer = HiddenStateCaptureBuffer(plan)
    assert buffer.state == HiddenStateCaptureState.ARMED
    buffer.add(chunk.accepted(10, 12))
    assert buffer.state == HiddenStateCaptureState.CAPTURING
    result = buffer.finish()
    assert buffer.state == HiddenStateCaptureState.FINISHED
    assert result is not None
    np.testing.assert_array_equal(result.hidden_positions, [10, 11])
    torch.testing.assert_close(result.hidden_states, hidden[:2])
    assert (result.hidden_position_start, result.hidden_position_end) == (10, 12)
    assert (result.hidden_window_start, result.hidden_window_end) == (10, 13)

    early_eos = HiddenStateCaptureBuffer(plan)
    early_eos.add(chunk.accepted(10, 11))
    assert early_eos.finish() is None


def test_hidden_capture_acceptance_uses_last_prefill_or_first_speculative_rows():
    assert accepted_hidden_range(100, 100, 1, speculative=False) == (99, 100)
    assert accepted_hidden_range(105, 5, 2, speculative=True) == (100, 102)


def test_hidden_capture_preserves_aux_final_layout_and_deduplicates_retries():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 0, 2, "collection", aux_layer_ids=(3,)
    )
    final = torch.tensor([[1.0], [2.0]])
    aux = torch.tensor([[10.0], [20.0]])
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 2}, {"req": 0}, final, [aux]
    )["req"]
    torch.testing.assert_close(
        chunk.hidden_states, torch.tensor([[10.0, 1.0], [20.0, 2.0]])
    )
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk)
    buffer.add(chunk.accepted(1, 2))
    result = buffer.finish()
    assert result is not None
    assert result.hidden_layout == "aux_final"
    np.testing.assert_array_equal(result.hidden_positions, [0, 1])
    torch.testing.assert_close(result.hidden_states, chunk.hidden_states)


def test_hidden_capture_dflash_layout_excludes_final_state():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 0, 1, "collection", aux_layer_ids=(3,), hidden_layout="dflash_aux"
    )
    chunk = capture_scheduled_hidden_states(
        {"req": plan},
        ["req"],
        {"req": 1},
        {"req": 0},
        torch.tensor([[1.0]]),
        [torch.tensor([[10.0]])],
    )["req"]
    torch.testing.assert_close(chunk.hidden_states, torch.tensor([[10.0]]))


def test_hidden_capture_keeps_bfloat16_on_cpu():
    plan = HiddenStateCapturePlan.from_window("req", 1, 0, 1, "collection")
    hidden = torch.tensor([[1.0]], dtype=torch.bfloat16)
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 1}, {"req": 0}, hidden
    )["req"]
    assert chunk.hidden_states.dtype == torch.bfloat16
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk)
    result = buffer.finish()
    assert result is not None
    assert result.hidden_states.dtype == torch.bfloat16


def test_hidden_capture_reports_sparse_positions_without_filling_gaps():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 10, 13, "collection", min_rows=2
    )
    hidden = torch.tensor([[1.0], [2.0], [3.0]])
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 3}, {"req": 10}, hidden
    )["req"]
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk.accepted(10, 11))
    buffer.add(chunk.accepted(12, 13))
    result = buffer.finish()
    assert result is not None
    np.testing.assert_array_equal(result.hidden_positions, [10, 12])
    torch.testing.assert_close(result.hidden_states, hidden[[0, 2]])


def test_hidden_capture_aux_capability_is_request_local():
    plan = HiddenStateCapturePlan.from_window(
        "a", 1, 0, 1, "collection", aux_layer_ids=(3,)
    )
    plans = {"a": plan, "b": plan}
    errors = drop_incompatible_aux_plans(plans, ["a"], (), None)
    assert errors == {"a": "aux_layers_unavailable"}
    assert plans == {"b": plan}


def test_hidden_capture_fails_closed_for_unmapped_runners():
    config = SimpleNamespace(
        device_config=SimpleNamespace(device_type="cuda"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        speculative_config=None,
    )
    assert hidden_state_capture_capability(config) is None
    config.scheduler_config.async_scheduling = True
    assert "async scheduling" in hidden_state_capture_capability(config)
    config.scheduler_config.async_scheduling = False
    config.device_config.device_type = "npu"
    assert "CUDA" in hidden_state_capture_capability(config)
