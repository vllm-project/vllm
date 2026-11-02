# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.prfs.base import uint32_to_uniform


def test_philox_compatibility_vector():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([7, 8])
    expected_mantissas = [[1661833, 10571167], [9742331, 13375724]]
    expected = (
        (torch.tensor(expected_mantissas, dtype=torch.float64) + 1) / (2**24 + 1)
    ).to(torch.float32)

    assert torch.equal(PhiloxPRF(42).uniform(contexts, token_ids), expected)


def test_philox_raw_words_match_uniform_and_change_with_inputs():
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    token_ids = torch.arange(32)
    prf = PhiloxPRF(42)
    words = prf.uint32(contexts, token_ids)

    assert words.shape == (2, 32)
    assert words.dtype == torch.int64
    assert torch.equal(words, prf.uint32(contexts, token_ids))
    assert torch.equal(uint32_to_uniform(words), prf.uniform(contexts, token_ids))
    assert not torch.equal(words, PhiloxPRF(43).uint32(contexts, token_ids))
    assert not torch.equal(words, prf.uint32(contexts + 1, token_ids))
    assert not torch.equal(words, prf.uint32(contexts, token_ids + 1))


def test_philox_raw_word_compatibility_vector():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([7, 8])
    expected = torch.tensor(
        [[425429266, 2706218805], [2494036831, 3424185538]], dtype=torch.int64
    )

    assert torch.equal(PhiloxPRF(42).uint32(contexts, token_ids), expected)


def test_prf_pairs_contexts_with_target_tokens():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([[7], [8]])

    assert PhiloxPRF(42).uniform(contexts, token_ids).shape == (2, 1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
def test_philox_accelerator_matches_cpu(key: int, context_width: int):
    contexts = torch.arange(2 * context_width, dtype=torch.int64).reshape(
        2, context_width
    )
    token_ids = torch.arange(1024, dtype=torch.int64)
    prf = PhiloxPRF(key)

    expected = prf.uniform(contexts, token_ids)
    actual = prf.uniform(contexts.cuda(), token_ids.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    ("key", "context_width", "stream"),
    [
        (42, 1, 0),
        (15726070495360670683, 4, 1),
        (42, 16, 2),
        (15726070495360670683, 4, 2**32),
    ],
)
def test_philox_uint32_accelerator_matches_cpu(
    key: int,
    context_width: int,
    stream: int,
):
    contexts = torch.arange(2 * context_width, dtype=torch.int64).reshape(
        2, context_width
    )
    token_ids = torch.arange(1024, dtype=torch.int64)
    prf = PhiloxPRF(key)

    expected = prf.uint32(contexts, token_ids, stream=stream)
    actual = prf.uint32(
        contexts.cuda(),
        token_ids.cuda(),
        stream=stream,
    ).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_philox_streams_are_deterministic_and_distinct():
    contexts = torch.tensor([[-1, -1, 7], [11, 12, 13]])
    token_ids = torch.arange(32)
    prf = PhiloxPRF(42)

    stream_0 = prf.uint32(contexts, token_ids, stream=0)
    stream_1 = prf.uint32(contexts, token_ids, stream=1)
    stream_2 = prf.uint32(contexts, token_ids, stream=2)

    assert torch.equal(
        stream_1,
        prf.uint32(contexts, token_ids, stream=1),
    )
    assert not torch.equal(stream_0, stream_1)
    assert not torch.equal(stream_1, stream_2)
