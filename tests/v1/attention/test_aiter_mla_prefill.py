# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AiterAsmPrefillBackend (gfx950 FP8 MLA prefill)."""

from unittest.mock import patch

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import (
    MLAPrefillSelectorConfig,
    _auto_select_mla_prefill_backend,
)


def _aiter_asm_class():
    try:
        return MLAPrefillBackendEnum.AITER_ASM.get_class()
    except ImportError:
        return None


@pytest.fixture(autouse=True)
def _clear_selector_cache():
    _auto_select_mla_prefill_backend.cache_clear()
    yield
    _auto_select_mla_prefill_backend.cache_clear()


GFX950 = DeviceCapability(major=9, minor=5)


class TestValidation:
    """validate_configuration should refuse non-FP8 KV cache."""

    def test_fp8_cache_accepted(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        with patch.object(cls, "is_available", return_value=True):
            reasons = cls.validate_configuration(
                GFX950,
                MLAPrefillSelectorConfig(
                    dtype=torch.bfloat16,
                    is_r1_compatible=True,
                    cache_dtype="fp8",
                ),
            )
            assert reasons == []

    def test_non_fp8_cache_rejected(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        with patch.object(cls, "is_available", return_value=True):
            reasons = cls.validate_configuration(
                GFX950,
                MLAPrefillSelectorConfig(
                    dtype=torch.bfloat16,
                    is_r1_compatible=True,
                    cache_dtype="auto",
                ),
            )
            assert any("FP8" in r or "fp8" in r for r in reasons)

    def test_non_r1_dims_rejected(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        with patch.object(cls, "is_available", return_value=True):
            reasons = cls.validate_configuration(
                GFX950,
                MLAPrefillSelectorConfig(
                    dtype=torch.bfloat16,
                    is_r1_compatible=False,
                    cache_dtype="fp8",
                ),
            )
            assert any("R1" in r for r in reasons)

    def test_non_gfx950_rejected(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        with patch.object(cls, "is_available", return_value=True):
            reasons = cls.validate_configuration(
                DeviceCapability(major=9, minor=0),  # Hopper
                MLAPrefillSelectorConfig(
                    dtype=torch.bfloat16,
                    is_r1_compatible=True,
                    cache_dtype="fp8",
                ),
            )
            assert any("compute capability" in r for r in reasons)


class TestSelectorPriority:
    """On gfx950, AITER_ASM should win over FLASH_ATTN when FP8 KV is on."""

    def test_aiter_asm_wins_on_gfx950_fp8(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        cfg = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=True,
            cache_dtype="fp8",
        )
        with patch.object(cls, "is_available", return_value=True):
            selected = _auto_select_mla_prefill_backend(GFX950, cfg)
            assert selected.get_name() == "AITER_ASM"

    def test_falls_through_to_flash_attn_when_not_fp8(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        try:
            fa_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not importable")
        cfg = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=True,
            cache_dtype="auto",
        )
        with (
            patch.object(cls, "is_available", return_value=True),
            patch.object(fa_cls, "validate_configuration", return_value=[]),
        ):
            selected = _auto_select_mla_prefill_backend(GFX950, cfg)
            assert selected.get_name() == "FLASH_ATTN"

    def test_falls_through_when_aiter_unavailable(self):
        cls = _aiter_asm_class()
        if cls is None:
            pytest.skip("AITER_ASM backend not importable")
        try:
            fa_cls = MLAPrefillBackendEnum.FLASH_ATTN.get_class()
        except ImportError:
            pytest.skip("FLASH_ATTN backend not importable")
        cfg = MLAPrefillSelectorConfig(
            dtype=torch.bfloat16,
            is_r1_compatible=True,
            cache_dtype="fp8",
        )
        with (
            patch.object(cls, "is_available", return_value=False),
            patch.object(fa_cls, "validate_configuration", return_value=[]),
        ):
            selected = _auto_select_mla_prefill_backend(GFX950, cfg)
            assert selected.get_name() == "FLASH_ATTN"
