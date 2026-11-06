# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the hidden-state window helpers."""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.hidden_state_window import (
    HiddenStateWindow,
    block_window_ranges,
    extraction_layout,
    validate_window_request,
)

pytestmark = pytest.mark.cpu_test


def _extra_config(enabled=True, **extra):
    def get(key, default=None):
        if key == "enable_window_capture":
            return enabled
        return extra.get(key, default)

    return get


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"start": -1, "end": 3},
        {"start": 3, "end": 3},
        {"start": 0, "end": True},
        {"start": 0.0, "end": 3},
        {"start": 0, "end": 3, "min_rows": 0},
        {"start": 0, "end": 3, "min_rows": 4},
        {"start": 0, "end": 3, "coordinate": "absolute"},
        {"start": 0, "end": 2**63},
        [0, 3],
    ],
)
def test_invalid_windows_fail_before_capture(value):
    with pytest.raises(ValueError):
        HiddenStateWindow.from_params({"hidden_state_window": value})


@pytest.mark.parametrize(
    "prompt,length,start,end,expected",
    [
        (10, 8, 0, 4, (9, 13)),
        (10, 3, 1, 8, (10, 12)),
        (1, 1, 0, 8, (0, 1)),
        (10, 0, 0, 8, (9, 9)),
        (20, 3, 8, 16, (27, 27)),
        (17, 32, 0, 8192, (16, 48)),
        (17, 8192, 0, 8192, (16, 8208)),
    ],
)
def test_only_accepted_response_predictors_are_exported(
    prompt, length, start, end, expected
):
    window = HiddenStateWindow.from_params(
        {"hidden_state_window": {"start": start, "end": end}}
    )
    assert window.accepted_bounds(prompt, length) == expected


@pytest.mark.parametrize("width", [32, 128, 512, 2048, 8192])
def test_window_writes_only_selected_offsets_with_allocator_padding(width):
    begin = 129
    size = 16
    blocks = list(range(1, (begin + width + size - 1) // size + 1))
    ranges = block_window_ranges(blocks, size, (begin, begin + width))
    assert sum(hi - lo for _, lo, hi in ranges) == width
    assert all(0 <= lo <= hi <= size for _, lo, hi in ranges)


def test_mixed_batch_physical_blocks_keep_independent_windows():
    a = block_window_ranges([7, 2], 4, (3, 6))
    b = block_window_ranges([9, 4], 4, (0, 1))
    assert a + b == [(7, 3, 4), (2, 0, 2), (9, 0, 1), (4, 0, 0)]
    assert block_window_ranges([5], 4, (3, 6), first_block=1) == [(5, 0, 2)]
    assert block_window_ranges([5], 4, None) == [(5, 0, 4)]
    assert block_window_ranges([0], 4, None) == []


@pytest.mark.parametrize(
    "layout,ids",
    [
        ("auxiliary", [0, 2]),
        ("final", [-1]),
        ("auxiliary_plus_final", [0, 2, -1]),
    ],
)
def test_layout_distinguishes_pre_norm_residual_and_final_model_output(layout, ids):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    eagle_aux_hidden_state_layer_ids=[0, 2], hidden_states_layout=layout
                )
            )
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 2),
    )
    assert extraction_layout(config) == (layout, ids)


def test_absent_window_preserves_legacy_capture():
    assert HiddenStateWindow.from_params({"include_output_tokens": True}) is None
    with pytest.raises(ValueError, match="nonempty"):
        HiddenStateWindow(0, 1).absolute_bounds(0)


@pytest.mark.parametrize(
    "problem",
    [
        "disabled",
        "connector",
        "stop",
        "n",
        "resumable",
        "embeds",
        "empty",
        "invalid",
        "too_long",
    ],
)
def test_admission_rejects_unsupported_request_before_scheduler(problem):
    transfer = SimpleNamespace(
        kv_connector="ExampleHiddenStatesConnector",
        get_from_extra_config=_extra_config(problem != "disabled"),
    )
    config = SimpleNamespace(
        kv_transfer_config=None if problem == "connector" else transfer,
        model_config=SimpleNamespace(max_model_len=4096),
    )
    params = SimpleNamespace(
        n=2 if problem == "n" else 1,
        stop=["stop"] if problem == "stop" else [],
        extra_args={
            "kv_transfer_params": {
                "hidden_state_window": {
                    "start": -1 if problem == "invalid" else 0,
                    "end": 4097 if problem == "too_long" else 3,
                }
            }
        },
    )
    with pytest.raises(ValueError):
        validate_window_request(
            config,
            params,
            [] if problem == "empty" else [1],
            object() if problem == "embeds" else None,
            problem == "resumable",
        )


def test_admission_keeps_legacy_and_valid_window_requests():
    validate = validate_window_request
    params = SimpleNamespace(extra_args=None)
    validate(SimpleNamespace(kv_transfer_config=None), params, None, None, True)
    params = SimpleNamespace(
        n=1,
        stop=[],
        extra_args={
            "kv_transfer_params": {
                "hidden_state_window": {"start": 128, "end": 160, "min_rows": 16}
            }
        },
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="ExampleHiddenStatesConnector",
            get_from_extra_config=_extra_config(),
        ),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    validate(config, params, [1, 2], None, False)
