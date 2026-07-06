# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.sampler import Sampler


def _array_state(values: list[float] | list[int]) -> SimpleNamespace:
    return SimpleNamespace(np=np.asarray(values))


def _noop(*args, **kwargs) -> None:
    pass


def _make_sampler(num_reqs: int = 3, vocab_size: int = 128) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        temperature=_array_state([1.0] * num_reqs),
        top_k=_array_state([vocab_size] * num_reqs),
        top_p=_array_state([1.0] * num_reqs),
        min_p=_array_state([0.0] * num_reqs),
        apply_temperature=_noop,
        apply_min_p=_noop,
        apply_top_k_top_p=lambda logits, *args: logits,
    )
    sampler.logit_bias_state = SimpleNamespace(
        use_logit_bias=np.zeros(num_reqs, dtype=bool),
        apply_logit_bias=_noop,
    )
    sampler.penalties_state = SimpleNamespace(
        use_penalty=np.zeros(num_reqs, dtype=bool),
        apply_penalties=_noop,
    )
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=_array_state([0] * num_reqs),
        apply_bad_words=_noop,
    )
    return sampler


@pytest.mark.parametrize(
    ("state_name", "field_name", "value"),
    [
        ("logit_bias_state", "use_logit_bias", True),
        ("penalties_state", "use_penalty", True),
        ("bad_words_state", "num_bad_words", 1),
        ("sampling_states", "temperature", 0.5),
        ("sampling_states", "min_p", 0.1),
        ("sampling_states", "top_k", 10),
        ("sampling_states", "top_p", 0.9),
    ],
)
def test_requires_logits_processing(
    state_name: str,
    field_name: str,
    value: bool | float | int,
) -> None:
    sampler = _make_sampler()
    state = getattr(sampler, state_name)
    values = getattr(state, field_name)
    if hasattr(values, "np"):
        values = values.np
    values[0] = value

    idx_mapping_np = np.asarray([0, 1], dtype=np.int32)
    assert sampler._requires_logits_processing(idx_mapping_np)


def test_inactive_request_does_not_require_logits_processing() -> None:
    sampler = _make_sampler()
    sampler.penalties_state.use_penalty[2] = True

    idx_mapping_np = np.asarray([0, 1], dtype=np.int32)
    assert not sampler._requires_logits_processing(idx_mapping_np)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_noop_sampling_params_reuses_logits(temperature: float) -> None:
    sampler = _make_sampler()
    sampler.sampling_states.temperature.np[:] = temperature
    logits = torch.randn(2, 16, dtype=torch.bfloat16)
    unused = torch.empty(0, dtype=torch.int32)

    processed = sampler.apply_sampling_params(
        logits,
        unused,
        np.asarray([0, 1], dtype=np.int32),
        unused,
        unused,
        unused,
        skip_top_k_top_p=True,
    )

    assert processed is logits
    assert processed.dtype == torch.bfloat16


def test_active_sampling_params_materializes_fp32_logits() -> None:
    sampler = _make_sampler()
    sampler.sampling_states.temperature.np[0] = 0.5
    logits = torch.randn(2, 16, dtype=torch.bfloat16)
    unused = torch.empty(0, dtype=torch.int32)

    processed = sampler.apply_sampling_params(
        logits,
        unused,
        np.asarray([0, 1], dtype=np.int32),
        unused,
        unused,
        unused,
        skip_top_k_top_p=True,
    )

    assert processed is not logits
    assert processed.dtype == torch.float32
    torch.testing.assert_close(processed, logits.float())
