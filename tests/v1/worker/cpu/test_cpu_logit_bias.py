# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.worker.cpu.logits_processors import CPULogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")

if current_platform.is_cpu():
    import vllm.v1.worker.cpu.shm  # noqa: F401

VOCAB_SIZE = 128
STOP_TOKEN = 3
OTHER_TOKEN = 4
PROMPT_LEN = 2
MIN_TOKENS = 10
POS = 5


def _make_state(params: list[SamplingParams]) -> CPULogitBiasState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    req_states.prompt_len.np[:] = PROMPT_LEN
    state = CPULogitBiasState(
        None, LogitsProcRequestState.from_request_state(req_states)
    )
    for req_idx, sampling_params in enumerate(params):
        state.add_request(req_idx, sampling_params)
    state.apply_staged_writes()
    return state


def _apply(
    logits: torch.Tensor,
    params: list[SamplingParams],
    idx_mapping: list[int] | None = None,
    positions: list[int] | None = None,
) -> torch.Tensor:
    state = _make_state(params)
    if idx_mapping is None:
        idx_mapping = list(range(len(params)))
    if positions is None:
        positions = [POS] * len(idx_mapping)

    idx_mapping_tensor = torch.tensor(idx_mapping, dtype=torch.int64)
    num_tokens = len(idx_mapping)
    state.apply(
        logits,
        LogitsContext(
            expanded_idx_mapping=idx_mapping_tensor,
            idx_mapping=idx_mapping_tensor,
            idx_mapping_np=np.array(idx_mapping, dtype=np.intp),
            expanded_local_pos=torch.zeros(num_tokens, dtype=torch.int32),
            input_ids=torch.zeros(num_tokens, dtype=torch.int32),
            pos=torch.tensor(positions, dtype=torch.int64),
            seq_lens_upper_bound_np=np.ones(num_tokens, dtype=np.int64),
        ),
    )
    return logits


def _min_tokens_params(structured: bool) -> SamplingParams:
    return SamplingParams(
        min_tokens=MIN_TOKENS,
        stop_token_ids=[STOP_TOKEN],
        structured_outputs=(
            StructuredOutputsParams(json={"type": "boolean"}) if structured else None
        ),
    )


def _only_stop_token_left(num_rows: int) -> torch.Tensor:
    logits = torch.full((num_rows, VOCAB_SIZE), -float("inf"))
    logits[:, STOP_TOKEN] = 1.0
    return logits


def test_cpu_allowed_tokens_and_additive_bias_follow_request_mapping() -> None:
    logits = torch.arange(24, dtype=torch.float32).view(3, 8)
    params = [
        SamplingParams(logit_bias={1: 2.0, 4: -1.0}),
        SamplingParams(),
        SamplingParams(
            allowed_token_ids=[0, 3, 5],
            logit_bias={3: 0.5, 4: 100.0},
        ),
    ]

    actual = _apply(logits, params, idx_mapping=[2, 0, 2])
    expected = torch.tensor(
        [
            [0.0, -torch.inf, -torch.inf, 3.5, -torch.inf, 5.0, -torch.inf, -torch.inf],
            [8.0, 11.0, 10.0, 11.0, 11.0, 13.0, 14.0, 15.0],
            [
                16.0,
                -torch.inf,
                -torch.inf,
                19.5,
                -torch.inf,
                21.0,
                -torch.inf,
                -torch.inf,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_cpu_min_tokens_restores_terminal_structured_output() -> None:
    actual = _apply(_only_stop_token_left(1), [_min_tokens_params(True)])
    assert actual[0, STOP_TOKEN] == 1.0


def test_cpu_min_tokens_keeps_empty_unstructured_row() -> None:
    actual = _apply(_only_stop_token_left(1), [_min_tokens_params(False)])
    assert torch.isneginf(actual).all()


def test_cpu_min_tokens_keeps_masking_when_another_token_remains() -> None:
    logits = _only_stop_token_left(1)
    logits[0, OTHER_TOKEN] = 2.0

    actual = _apply(logits, [_min_tokens_params(True)])

    assert torch.isneginf(actual[0, STOP_TOKEN])
    assert actual[0, OTHER_TOKEN] == 2.0


def test_cpu_min_tokens_restores_only_structured_rows_in_mixed_batch() -> None:
    actual = _apply(
        _only_stop_token_left(2),
        [_min_tokens_params(True), _min_tokens_params(False)],
    )

    assert actual[0, STOP_TOKEN] == 1.0
    assert torch.isneginf(actual[1]).all()


def test_cpu_min_tokens_uses_strict_position_boundary() -> None:
    logits = torch.zeros((2, VOCAB_SIZE))
    min_len = PROMPT_LEN + MIN_TOKENS

    actual = _apply(
        logits,
        [_min_tokens_params(False)],
        idx_mapping=[0, 0],
        positions=[min_len - 2, min_len - 1],
    )

    assert torch.isneginf(actual[0, STOP_TOKEN])
    assert actual[1, STOP_TOKEN] == 0.0
