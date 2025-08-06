# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.selector import choose_attention_backend

NUM_BACKENDS = [1, 2, 3, 4]
HEAD_SIZES = [256]
ATTENTION_DTYPES = [torch.bfloat16]
KVCACHE_DTYPES = ["auto"]
BLOCK_SIZES = [1]


class MockUnsupportedAttentionBackend(AttentionBackend):

    def __init__(self):
        pass

    @classmethod
    def get_name(cls) -> str:
        return "MOCK_UNSUPPORTED_BACKEND"

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return []

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []


class MockSupportedAttentionBackend(AttentionBackend):

    def __init__(self):
        pass

    @classmethod
    def get_name(cls) -> str:
        return "MOCK_SUPPORTED_BACKEND"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return HEAD_SIZES

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return ATTENTION_DTYPES


def get_full_qualname(cls):
    """
    Returns the fully qualified class path, e.g. 'package.module.ClassName'
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def generate_unsupported_backend_mapping(num_backends: int) -> dict[str, str]:
    backend_qualname = get_full_qualname(MockUnsupportedAttentionBackend)
    return {
        f"{MockUnsupportedAttentionBackend.get_name()}_{index}":
        backend_qualname
        for index in range(num_backends)
    }


def generate_supported_backend_mapping(num_backends: int) -> dict[str, str]:
    backend_qualname = get_full_qualname(MockSupportedAttentionBackend)
    return {
        f"{MockSupportedAttentionBackend.get_name()}_{index}": backend_qualname
        for index in range(num_backends)
    }


@pytest.mark.parametrize("num_backends", NUM_BACKENDS)
@pytest.mark.parametrize("arbitrary_head_size", HEAD_SIZES)
@pytest.mark.parametrize("arbitrary_dtype", ATTENTION_DTYPES)
@pytest.mark.parametrize("arbitrary_kvcache_dtype", KVCACHE_DTYPES)
@pytest.mark.parametrize("arbitrary_block_size", BLOCK_SIZES)
def test_choose_attention_backend_raises_on_no_supported_backend(
        num_backends: int, arbitrary_head_size: int,
        arbitrary_dtype: torch.dtype, arbitrary_kvcache_dtype: str,
        arbitrary_block_size: int) -> None:

    unsupported_backends = generate_unsupported_backend_mapping(num_backends)

    with pytest.raises(ValueError):
        choose_attention_backend(unsupported_backends, arbitrary_head_size,
                                 arbitrary_dtype, arbitrary_kvcache_dtype,
                                 arbitrary_block_size)


@pytest.mark.parametrize("num_backends", NUM_BACKENDS)
@pytest.mark.parametrize("arbitrary_head_size", HEAD_SIZES)
@pytest.mark.parametrize("arbitrary_dtype", ATTENTION_DTYPES)
@pytest.mark.parametrize("arbitrary_kvcache_dtype", KVCACHE_DTYPES)
@pytest.mark.parametrize("arbitrary_block_size", BLOCK_SIZES)
def test_choose_attention_backend_returns_qualname_for_supported_backend_only(
        num_backends: int, arbitrary_head_size: int,
        arbitrary_dtype: torch.dtype, arbitrary_kvcache_dtype: str,
        arbitrary_block_size: int) -> None:
    supported_backend_qual_name = get_full_qualname(
        MockSupportedAttentionBackend)
    supported_backends = generate_supported_backend_mapping(num_backends)
    _, chosen_backend_qualname = choose_attention_backend(
        supported_backends, arbitrary_head_size, arbitrary_dtype,
        arbitrary_kvcache_dtype, arbitrary_block_size)
    assert chosen_backend_qualname == supported_backend_qual_name


@pytest.mark.parametrize("num_backends", NUM_BACKENDS)
@pytest.mark.parametrize("arbitrary_head_size", HEAD_SIZES)
@pytest.mark.parametrize("arbitrary_dtype", ATTENTION_DTYPES)
@pytest.mark.parametrize("arbitrary_kvcache_dtype", KVCACHE_DTYPES)
@pytest.mark.parametrize("arbitrary_block_size", BLOCK_SIZES)
def test_choose_attention_backend_returns_supported_backend_qualname(
        num_backends: int, arbitrary_head_size: int,
        arbitrary_dtype: torch.dtype, arbitrary_kvcache_dtype: str,
        arbitrary_block_size: int) -> None:
    supported_backend_qual_name = get_full_qualname(
        MockSupportedAttentionBackend)
    unsupported_backends = generate_unsupported_backend_mapping(num_backends)
    supported_backends = generate_supported_backend_mapping(num_backends)
    all_backends = supported_backends | unsupported_backends
    _, chosen_backend_qualname = choose_attention_backend(
        all_backends, arbitrary_head_size, arbitrary_dtype,
        arbitrary_kvcache_dtype, arbitrary_block_size)

    assert chosen_backend_qualname == supported_backend_qual_name


@pytest.mark.parametrize("num_backends", NUM_BACKENDS)
@pytest.mark.parametrize("arbitrary_head_size", HEAD_SIZES)
@pytest.mark.parametrize("arbitrary_dtype", ATTENTION_DTYPES)
@pytest.mark.parametrize("arbitrary_kvcache_dtype", KVCACHE_DTYPES)
@pytest.mark.parametrize("arbitrary_block_size", BLOCK_SIZES)
def test_choose_attention_backend_forces_backend_via_env(
        num_backends: int, arbitrary_head_size: int,
        arbitrary_dtype: torch.dtype, arbitrary_kvcache_dtype: str,
        arbitrary_block_size: int, monkeypatch) -> None:
    supported_backends = generate_supported_backend_mapping(num_backends)

    # Force an arbitrary supported backend
    arbitrary_backend_name, _ = random.choice(list(supported_backends.items()))
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", arbitrary_backend_name)

    chosen_backend_name, _ = choose_attention_backend(supported_backends,
                                                      arbitrary_head_size,
                                                      arbitrary_dtype,
                                                      arbitrary_kvcache_dtype,
                                                      arbitrary_block_size)
    assert chosen_backend_name == arbitrary_backend_name
