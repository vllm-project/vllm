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
    requires_r1_mla_dimensions = False

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
    for member in list(MLAPrefillBackendEnum):
        member.clear_override()


def test_custom_unregistered_raises():
    with pytest.raises(ValueError, match="must be registered before use"):
        MLAPrefillBackendEnum.CUSTOM.get_path()


def test_register_custom_backend_with_class_path():
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.CUSTOM,
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend",
    )
    assert MLAPrefillBackendEnum.CUSTOM.is_overridden()
    assert MLAPrefillBackendEnum.CUSTOM.get_class().get_name() == "CUSTOM"


def test_register_custom_backend_as_decorator():
    @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
    class DecoratedPrefillBackend(MLAPrefillBackend):
        supported_dtypes = [torch.bfloat16]
        requires_r1_mla_dimensions = False

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
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend",
    )
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_name() != "FLASH_ATTN"
    assert MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()


def test_clear_override():
    original_path = MLAPrefillBackendEnum.FLASH_ATTN.get_path()
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend",
    )
    MLAPrefillBackendEnum.FLASH_ATTN.clear_override()
    assert not MLAPrefillBackendEnum.FLASH_ATTN.is_overridden()
    assert MLAPrefillBackendEnum.FLASH_ATTN.get_path() == original_path


def test_register_dynamic_member():
    p = "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend"
    member = MLAPrefillBackendEnum.register("DYNAMIC_TEST", p)
    assert member.name == "DYNAMIC_TEST"
    assert member is MLAPrefillBackendEnum["DYNAMIC_TEST"]

    MLAPrefillBackendEnum._member_map_.pop("DYNAMIC_TEST", None)
    MLAPrefillBackendEnum._member_names_.remove("DYNAMIC_TEST")
    delattr(MLAPrefillBackendEnum, "DYNAMIC_TEST")


def test_register_dynamic_member_duplicate_raises():
    path = "some.module.Class"
    MLAPrefillBackendEnum.register("DUP_TEST", path)
    with pytest.raises(ValueError, match="already exists"):
        MLAPrefillBackendEnum.register("DUP_TEST", "other.module.OtherClass")

    MLAPrefillBackendEnum._member_map_.pop("DUP_TEST", None)
    MLAPrefillBackendEnum._member_names_.remove("DUP_TEST")
    delattr(MLAPrefillBackendEnum, "DUP_TEST")


def test_register_mla_prefill_backend_with_string_name_direct():
    register_mla_prefill_backend(
        "STRING_DIRECT",
        "tests.v1.attention.test_mla_prefill_registry.CustomMLAPrefillBackend",
    )
    member = MLAPrefillBackendEnum.STRING_DIRECT
    assert member.is_overridden()
    assert member.get_class().get_name() == "CUSTOM"

    member.clear_override()
    MLAPrefillBackendEnum._member_map_.pop("STRING_DIRECT", None)
    MLAPrefillBackendEnum._member_names_.remove("STRING_DIRECT")
    delattr(MLAPrefillBackendEnum, "STRING_DIRECT")


def test_register_mla_prefill_backend_with_string_name_decorator():
    @register_mla_prefill_backend("STRING_DECORATOR")
    class MLAPrefillDecorated(MLAPrefillBackend):
        supported_dtypes = [torch.bfloat16]
        requires_r1_mla_dimensions = False

        @staticmethod
        def get_name() -> str:
            return "DECORATED"

        def run_prefill_new_tokens(self, q, k, v, return_softmax_lse):
            raise NotImplementedError

        def run_prefill_context_chunk(self, chunk_idx, q, k, v):
            raise NotImplementedError

    member = MLAPrefillBackendEnum.STRING_DECORATOR
    assert member.is_overridden()
    assert "MLAPrefillDecorated" in member.get_path()
    assert member.get_class().get_name() == "DECORATED"

    member.clear_override()
    MLAPrefillBackendEnum._member_map_.pop("STRING_DECORATOR", None)
    MLAPrefillBackendEnum._member_names_.remove("STRING_DECORATOR")
    delattr(MLAPrefillBackendEnum, "STRING_DECORATOR")
