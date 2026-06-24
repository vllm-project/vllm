# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    _get_diffusion_gemma_sampler_memory_reserve_bytes,
)

DEFAULT_SHAPE = dict(
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    canvas_length=256,
    vocab_size=262144,
)


@pytest.mark.parametrize(
    ("reserve_mib", "scale", "shape", "expected"),
    [
        ("", 1.0, DEFAULT_SHAPE, 0),
        ("512", 1.0, DEFAULT_SHAPE, 512 * (1 << 20)),
        ("512", -1.0, DEFAULT_SHAPE, 512 * (1 << 20)),
        (
            " auto ",
            1.0,
            dict(
                max_num_seqs=1,
                max_num_batched_tokens=256,
                canvas_length=256,
                vocab_size=1024,
            ),
            256 * 1024 * 4 * 2,
        ),
        (
            " AUTO ",
            1.0,
            dict(
                max_num_seqs=1,
                max_num_batched_tokens=256,
                canvas_length=256,
                vocab_size=1024,
            ),
            256 * 1024 * 4 * 2,
        ),
    ],
)
def test_diffusion_sampler_memory_reserve_parses_supported_modes(
    reserve_mib, scale, shape, expected
):
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            reserve_mib, scale, **shape
        )
        == expected
    )


def test_diffusion_sampler_memory_reserve_auto_estimate():
    reserve = _get_diffusion_gemma_sampler_memory_reserve_bytes(
        "auto",
        1.0,
        canvas_length=256,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        vocab_size=262144,
    )

    assert reserve == 16 * 256 * 262144 * 4 * 2


def test_diffusion_sampler_memory_reserve_auto_respects_token_cap():
    reserve = _get_diffusion_gemma_sampler_memory_reserve_bytes(
        "auto",
        1.25,
        canvas_length=256,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        vocab_size=1024,
    )

    # max_num_batched_tokens limits the materialized decode shape to 16
    # diffusion requests, not max_num_seqs=64.
    assert reserve == int(16 * 256 * 1024 * 4 * 2 * 1.25)


@pytest.mark.parametrize(
    ("reserve_mib", "scale", "shape_override", "match"),
    [
        ("-1", 1.0, {}, None),
        ("not-a-number", 1.0, {}, "auto"),
        ("auto", -0.1, {}, None),
        ("auto", 1.0, {"canvas_length": 0}, "canvas_length"),
        ("auto", 1.0, {"max_num_seqs": 0}, "max_num_seqs"),
        ("auto", 1.0, {"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
        ("auto", 1.0, {"vocab_size": 0}, "vocab_size"),
    ],
)
def test_diffusion_sampler_memory_reserve_rejects_invalid_inputs(
    reserve_mib, scale, shape_override, match
):
    shape = {**DEFAULT_SHAPE, **shape_override}
    with pytest.raises(ValueError, match=match):
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            reserve_mib, scale, **shape
        )


def test_diffusion_model_state_reports_global_vocab_reserve(monkeypatch):
    state = DiffusionGemmaModelState.__new__(DiffusionGemmaModelState)
    state.max_num_reqs = 64
    state.max_num_tokens = 4096
    state.diffusion_states = SimpleNamespace(canvas_length=256)
    state.model_config = SimpleNamespace(get_vocab_size=lambda: 262144)

    monkeypatch.setattr(
        envs,
        "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB",
        "auto",
        raising=False,
    )
    monkeypatch.setattr(
        envs,
        "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE",
        1.1,
        raising=False,
    )

    assert state.get_extra_non_kv_cache_memory_bytes() == int(
        16 * 256 * 262144 * 4 * 2 * 1.1
    )
