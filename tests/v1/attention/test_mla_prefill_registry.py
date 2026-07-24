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


def test_prefill_backend_clone_has_isolated_metadata():
    backend = CustomMLAPrefillBackend(
        num_heads=4,
        scale=0.5,
        kv_lora_rank=8,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=32,
        vllm_config=object(),
    )

    clone = backend.clone()

    assert isinstance(clone, CustomMLAPrefillBackend)
    assert clone is not backend
    assert clone.num_heads == backend.num_heads
    assert clone.scale == backend.scale
    backend._prefill_metadata = object()
    clone._prefill_metadata = object()
    assert clone._prefill_metadata is not backend._prefill_metadata


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


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError, match="Unknown MLA prefill backend"):
        MLAPrefillBackendEnum["NONEXISTENT"]


def test_rocm_aiter_fa_registered():
    """ROCM_AITER_FA is a known backend pointing at the AITER FA class."""
    assert "ROCM_AITER_FA" in MLAPrefillBackendEnum.__members__

    path = MLAPrefillBackendEnum.ROCM_AITER_FA.get_path()
    assert path == (
        "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn."
        "AiterFlashAttnPrefillBackend"
    )

    backend_cls = MLAPrefillBackendEnum.ROCM_AITER_FA.get_class()
    assert backend_cls.get_name() == "ROCM_AITER_FA"
    # The AITER FA path is the fp16/bf16 generic-varlen prefill path.
    assert backend_cls.supports_dtype(torch.bfloat16)
    assert backend_cls.supports_dtype(torch.float16)


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
