# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from typing import NamedTuple

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: str | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = "decoder"
    use_non_causal: bool = False


class _FakeMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FAKE_MLA"

    @staticmethod
    def get_impl_cls():
        return object

    @staticmethod
    def get_builder_cls():
        return object

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True


class FakeRocmAiterMLABackend(_FakeMLABackend):
    @staticmethod
    def get_name() -> str:
        return "FAKE_ROCM_AITER_MLA"

    @classmethod
    def supports_num_heads(cls, num_heads: int | None) -> str | None:
        if num_heads is None:
            return None

        valid_heads = num_heads in (4, 8) or (
            num_heads % 16 == 0 and 16 <= num_heads <= 128
        )
        if valid_heads:
            return None

        return "num_heads not supported"


class FakeRocmAiterTritonMLABackend(FakeRocmAiterMLABackend):
    @staticmethod
    def get_name() -> str:
        return "FAKE_ROCM_AITER_TRITON_MLA"


class FakeTritonMLABackend(_FakeMLABackend):
    @staticmethod
    def get_name() -> str:
        return "FAKE_TRITON_MLA"


def _install_fake_amdsmi(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_amdsmi = types.ModuleType("amdsmi")

    class AmdSmiException(Exception):
        pass

    fake_amdsmi.AmdSmiException = AmdSmiException
    fake_amdsmi.amdsmi_init = lambda: None
    fake_amdsmi.amdsmi_shut_down = lambda: None
    fake_amdsmi.amdsmi_get_processor_handles = lambda: ["gpu0"]
    fake_amdsmi.amdsmi_get_gpu_asic_info = lambda handle: {
        "target_graphics_version": "gfx942"
    }
    fake_amdsmi.amdsmi_get_gpu_device_uuid = lambda handle: "uuid"
    fake_amdsmi.amdsmi_topo_get_link_type = lambda src, dst: {"hops": 1, "type": 2}
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)


def _install_fake_aiter_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_aiter_ops = types.ModuleType("vllm._aiter_ops")

    class FakeRocmAiterOps:
        @staticmethod
        def is_mla_enabled() -> bool:
            return True

        @staticmethod
        def is_mha_enabled() -> bool:
            return False

        @staticmethod
        def is_unified_attention_enabled() -> bool:
            return False

    fake_aiter_ops.rocm_aiter_ops = FakeRocmAiterOps()
    fake_aiter_ops.is_aiter_found_and_supported = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)


def _override_backends() -> None:
    register_backend(
        AttentionBackendEnum.ROCM_AITER_MLA,
        f"{__name__}.FakeRocmAiterMLABackend",
    )
    register_backend(
        AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
        f"{__name__}.FakeRocmAiterTritonMLABackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_MLA,
        f"{__name__}.FakeTritonMLABackend",
    )


def _clear_backend_overrides() -> None:
    AttentionBackendEnum.ROCM_AITER_MLA.clear_override()
    AttentionBackendEnum.ROCM_AITER_TRITON_MLA.clear_override()
    AttentionBackendEnum.TRITON_MLA.clear_override()


def _load_rocm_platform(monkeypatch: pytest.MonkeyPatch):
    _install_fake_amdsmi(monkeypatch)
    _install_fake_aiter_ops(monkeypatch)
    _override_backends()

    for module_name in ("vllm.platforms.rocm",):
        sys.modules.pop(module_name, None)

    return importlib.import_module("vllm.platforms.rocm")


@pytest.fixture(autouse=True)
def _cleanup_backend_overrides():
    yield
    _clear_backend_overrides()


def _make_selector_config() -> AttentionSelectorConfig:
    return AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
    )


def test_auto_selection_falls_back_for_unsupported_dense_aiter_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rocm = _load_rocm_platform(monkeypatch)

    backend_path = rocm.RocmPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=_make_selector_config(),
        num_heads=24,
    )

    assert backend_path == AttentionBackendEnum.TRITON_MLA.get_path()


def test_auto_selection_keeps_supported_dense_aiter_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rocm = _load_rocm_platform(monkeypatch)

    backend_path = rocm.RocmPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=_make_selector_config(),
        num_heads=32,
    )

    assert backend_path == AttentionBackendEnum.ROCM_AITER_MLA.get_path()


@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.ROCM_AITER_MLA,
        AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
    ],
)
def test_explicit_dense_aiter_selection_rejects_unsupported_heads(
    monkeypatch: pytest.MonkeyPatch,
    backend: AttentionBackendEnum,
) -> None:
    rocm = _load_rocm_platform(monkeypatch)

    with pytest.raises(ValueError, match="num_heads not supported"):
        rocm.RocmPlatform.get_attn_backend_cls(
            selected_backend=backend,
            attn_selector_config=_make_selector_config(),
            num_heads=24,
        )
