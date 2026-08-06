# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _set_max_n(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", str(value))
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def test_direct_beam_width_rejects_values_over_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)

    with pytest.raises(VLLMValidationError, match="beam_width must be at most 4"):
        BeamSearchParams(beam_width=5, max_tokens=1)


def test_chat_beam_conversion_rejects_n_before_stream_state_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=5,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    with pytest.raises(VLLMValidationError, match="beam_width must be at most 4"):
        request.to_beam_search_params(max_tokens=1, default_sampling_params={})


def test_chat_beam_conversion_accepts_n_at_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=4,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    params = request.to_beam_search_params(max_tokens=1, default_sampling_params={})

    assert params.beam_width == 4
