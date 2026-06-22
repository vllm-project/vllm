# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend registry."""

import pytest
import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)


class CustomMLAPrefillBackend(MLAPrefillBackend):
    """Mock custom MLA prefill backend for testing."""

    supported_dtypes = [torch.bfloat16, torch.float16]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
        raise NotImplementedError

    def run_prefill_context_chunk(self, chunk_idx, q, k, v):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clear any overrides after each test."""
    yield
    for member in MLAPrefillBackendEnum:
        member.clear_override()


def test_custom_is_not_alias_of_any_backend():
    all_backends = list(MLAPrefillBackendEnum)

    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MLAPrefillBackendEnum.CUSTOM:
            aliases.append(backend.name)

    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MLAPrefillBackendEnum.CUSTOM.value)}\n"
        f"All MLA prefill backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    assert MLAPrefillBackendEnum.CUSTOM.name == "CUSTOM"


def test_custom_unregistered_raises():
    with pytest.raises(ValueError, match="must be registered before use"):
        MLAPrefillBackendEnum.CUSTOM.get_path()


def test_register_custom_backend_with_class_path():
    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.CUSTOM,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()

    class_path = MLAPrefillBackendEnum.CUSTOM.get_path()
    assert class_path == (
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
    )

    backend_cls = MLAPrefillBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_register_custom_backend_as_decorator():
    @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
    class DecoratedPrefillBackend(MLAPrefillBackend):
        supported_dtypes = [torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            return "DECORATED"

        def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
            raise NotImplementedError

        def run_prefill_context_chunk(self, chunk_idx, q, k, v):
            raise NotImplementedError

    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()
    assert "DecoratedPrefillBackend" in MLAPrefillBackendEnum.CUSTOM.get_path()


def test_override_existing_backend():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )

    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() != original_path

    backend_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
    assert backend_cls.get_name() == "CUSTOM"


def test_clear_override():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()

    register_mla_prefill_backend(
        backend=MLAPrefillBackendEnum.FLASH_ATTN,
        class_path=(
            "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
        ),
    )
    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()

    MLAPrefillBackendEnum.FLASH_ATTN.clear_override()
    assert not MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() == original_path


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
        MLAPrefillBackendEnum["NONEXISTENT"]
