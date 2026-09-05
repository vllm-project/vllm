# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def fa_utils_on_hopper(monkeypatch):
    fake_flash_attn = ModuleType("vllm.vllm_flash_attn")
    fake_flash_attn_any = cast(Any, fake_flash_attn)
    fake_flash_attn_any.compile_flash_attn_varlen_func_from_specs = object()
    fake_flash_attn_any.flash_attn_varlen_func = object()
    fake_flash_attn_any.get_scheduler_metadata = object()
    fake_flash_attn_interface = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fake_flash_attn_interface_any = cast(Any, fake_flash_attn_interface)
    fake_flash_attn_interface_any.flash_attn_varlen_func = object()
    fake_flash_attn_interface_any.fa_version_unsupported_reason = lambda _: None
    fake_flash_attn_interface_any.is_fa_version_supported = lambda version: version in (
        2,
        3,
    )
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_flash_attn)
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_flash_attn_interface,
    )

    from vllm.v1.attention.backends import fa_utils

    monkeypatch.setattr(
        fa_utils,
        "current_platform",
        SimpleNamespace(
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            get_device_capability=lambda: SimpleNamespace(major=9),
        ),
    )
    return fa_utils


@pytest.mark.parametrize("configured_version", [None, 3])
def test_fa3_falls_back_on_hopper_with_batch_invariance(
    monkeypatch, fa_utils_on_hopper, configured_version
):
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=configured_version),
        model_config=None,
    )
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none", lambda: vllm_config
    )

    assert fa_utils_on_hopper.get_flash_attn_version() == 2


def test_fa3_remains_default_without_batch_invariance(monkeypatch, fa_utils_on_hopper):
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)

    assert fa_utils_on_hopper.get_flash_attn_version() == 3


def test_flashattention_mla_rejects_batch_invariance(fa_utils_on_hopper):
    from vllm.v1.attention.backends.mla.flashattn_mla import (
        FlashAttnMLABackend,
    )

    assert not FlashAttnMLABackend.supports_batch_invariance()
