# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import AttentionConfig, replace
from vllm.model_executor.models.config import Gemma4Config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _config(
    *,
    version: int | None = 4,
    kv_cache_dtype: str = "auto",
    backend: AttentionBackendEnum | None = None,
    sparse_mla: bool = False,
):
    hf_text_config = SimpleNamespace()
    if sparse_mla:
        hf_text_config.index_topk = 2048
    return SimpleNamespace(
        attention_config=AttentionConfig(
            flash_attn_version=version,
            backend=backend,
        ),
        cache_config=SimpleNamespace(cache_dtype=kv_cache_dtype),
        model_config=SimpleNamespace(
            use_mla=sparse_mla,
            is_diffusion=False,
            architecture="DeepseekV3ForCausalLM",
            hf_text_config=hf_text_config,
            get_head_size=lambda: 128,
        ),
    )


@pytest.fixture
def hopper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fa_utils.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(fa_utils, "_fa4_cute_import_error", lambda: None)


def test_default_hopper_flash_attn_version_is_unchanged(hopper):
    config = _config(version=None)

    assert fa_utils.resolve_flash_attn_version(config) is None
    assert config.attention_config.flash_attn_version is None


def test_fa4_request_without_kv_cache_config_does_not_crash(hopper):
    config = _config()
    config.cache_config = None

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_non_fa3_fp8_cache_format_does_not_fallback(hopper):
    config = _config(kv_cache_dtype="fp8_e5m2")

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert not config.attention_config._flash_attn_version_fallback


def test_internal_resolution_state_does_not_change_config_hash():
    config = AttentionConfig(flash_attn_version=3)
    original_hash = config.compute_hash()

    config._flash_attn_version_fallback = True
    config._flash_attn_version_required = True

    assert config.compute_hash() == original_hash


def test_internal_resolution_state_survives_config_replace():
    config = AttentionConfig(flash_attn_version=3)
    config._flash_attn_version_fallback = True
    config._flash_attn_version_required = True

    replacement = replace(config, use_non_causal=True)

    assert replacement is not config
    assert replacement._flash_attn_version_fallback
    assert replacement._flash_attn_version_required


@pytest.mark.parametrize("required_by_model", [False, True])
def test_fa4_only_route_rejects_incompatible_fallback(
    hopper, required_by_model: bool
):
    config = _config(kv_cache_dtype="fp8")
    config.attention_config._flash_attn_version_required = required_by_model
    if not required_by_model:
        config.model_config.get_head_size = lambda: 512

    with pytest.raises(ValueError, match="model requires FA4"):
        fa_utils.resolve_flash_attn_version(config)
    assert not config.attention_config._flash_attn_version_fallback


def test_model_required_fa4_rejects_fallback(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(version=None, kv_cache_dtype="fp8")
    config.model_config.hf_text_config = SimpleNamespace(
        head_dim=128,
        global_head_dim=256,
    )
    monkeypatch.setattr(fa_utils, "is_fa_version_supported", lambda version: True)

    Gemma4Config.verify_and_update_config(config)

    assert config.attention_config.flash_attn_version == 4
    assert config.attention_config._flash_attn_version_required
    with pytest.raises(ValueError, match="model requires FA4"):
        fa_utils.resolve_flash_attn_version(config)


@pytest.mark.parametrize("version", (3, 4))
def test_explicit_fa_version_is_frozen_and_logged(
    hopper, monkeypatch: pytest.MonkeyPatch, version
):
    config = _config(version=version)
    info_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "info_once", info_once)

    assert fa_utils.resolve_flash_attn_version(config) == version
    assert config.attention_config.flash_attn_version == version
    message = f"requested=FA{version}, effective=FA{version}"
    assert message in info_once.call_args.args[0]
    assert info_once.call_args.kwargs == {"scope": "global"}


@pytest.mark.parametrize(
    ("config_kwargs", "batch_invariant", "import_error", "reason"),
    [
        ({"kv_cache_dtype": "fp8"}, False, None, "FP8 KV cache"),
        ({}, True, None, "batch-invariant serving"),
        (
            {"backend": AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE},
            False,
            None,
            "generic sparse-MLA FA3 route",
        ),
        ({}, False, "ModuleNotFoundError: no cutlass", "failed to import"),
    ],
)
def test_fa4_gap_falls_back_the_whole_server(
    hopper,
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs,
    batch_invariant,
    import_error,
    reason,
):
    config = _config(**config_kwargs)
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(fa_utils, "_fa4_cute_import_error", lambda: import_error)
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config.flash_attn_version == 3
    assert config.attention_config._flash_attn_version_fallback
    assert reason in warning_once.call_args.args[1]
    assert "whole server is using FA3" in warning_once.call_args.args[0]
    assert warning_once.call_args.kwargs == {"scope": "global"}


def test_auto_selected_generic_sparse_mla_falls_back(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(sparse_mla=True)
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config.flash_attn_version == 3
    assert "generic sparse-MLA FA3 route" in warning_once.call_args.args[1]


def test_explicit_non_fa_sparse_mla_route_does_not_fallback(hopper):
    config = _config(
        sparse_mla=True,
        backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    )

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_deepseek_v4_sparse_route_is_outside_fa3_policy(hopper):
    config = _config(sparse_mla=True)
    config.model_config.architecture = "DeepseekV4ForCausalLM"

    assert fa_utils.resolve_flash_attn_version(config) == 4
    assert config.attention_config.flash_attn_version == 4


def test_each_distinct_fallback_reason_uses_warning_once(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(kv_cache_dtype="fp8")
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        fa_utils,
        "_fa4_cute_import_error",
        lambda: "ModuleNotFoundError: no cutlass",
    )
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)

    fa_utils.resolve_flash_attn_version(config)

    reasons = [call.args[1] for call in warning_once.call_args_list]
    assert len(reasons) == len(set(reasons)) == 3
    assert all(
        call.kwargs == {"scope": "global"}
        for call in warning_once.call_args_list
    )


def test_fallback_reasons_are_collected_from_all_ranks(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config()
    warning_once = MagicMock()
    monkeypatch.setattr(fa_utils.logger, "warning_once", warning_once)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def all_gather_object(gathered, local):
        gathered[:] = [local, ["remote FA4 dependency failure"]]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    assert fa_utils.resolve_flash_attn_version(config) == 3
    assert config.attention_config._flash_attn_version_fallback
    assert "remote FA4 dependency failure" in warning_once.call_args.args[1]


def test_cute_probe_imports_kernel_entrypoint(monkeypatch: pytest.MonkeyPatch):
    import_module = MagicMock(side_effect=ModuleNotFoundError("no cutlass"))
    monkeypatch.setattr(fa_utils, "import_module", import_module)

    assert fa_utils._fa4_cute_import_error() == "ModuleNotFoundError: no cutlass"
    import_module.assert_called_once_with("vllm.vllm_flash_attn.cute.interface")


@pytest.mark.parametrize(
    ("configured_version", "head_size", "effective_version"),
    [
        (None, 128, 3),
        (3, 512, 4),
        (4, 512, 4),
    ],
)
def test_standard_selector_consumes_frozen_version(
    hopper,
    monkeypatch: pytest.MonkeyPatch,
    configured_version: int | None,
    head_size: int,
    effective_version: int,
):
    config = _config(version=configured_version)
    fake_interface = types.ModuleType(
        "vllm.vllm_flash_attn.flash_attn_interface"
    )
    fake_interface.is_fa_version_supported = lambda version: version in (3, 4)
    fake_interface.fa_version_unsupported_reason = lambda version: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_interface,
    )
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: config,
    )

    assert fa_utils.get_flash_attn_version(head_size=head_size) == effective_version


def test_fallback_rejects_layer_that_requires_fa4(
    hopper, monkeypatch: pytest.MonkeyPatch
):
    config = _config(version=3)
    config.attention_config._flash_attn_version_fallback = True
    fake_interface = types.ModuleType(
        "vllm.vllm_flash_attn.flash_attn_interface"
    )
    fake_interface.is_fa_version_supported = lambda version: version in (3, 4)
    fake_interface.fa_version_unsupported_reason = lambda version: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_interface,
    )
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: config,
    )

    with pytest.raises(ValueError, match="resolved FA4 to FA3"):
        fa_utils.get_flash_attn_version(head_size=512)


def test_mla_decode_consumes_frozen_version(hopper, monkeypatch: pytest.MonkeyPatch):
    from vllm.v1.attention.backends.mla import flashattn_mla

    config = _config(version=3)
    monkeypatch.setattr(flashattn_mla, "is_fa_version_supported", lambda version: True)

    assert flashattn_mla._get_mla_fa_version(config) == 3

    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    config.attention_config.flash_attn_version = 4
    assert flashattn_mla._get_mla_fa_version(config) == 4

    monkeypatch.setattr(
        flashattn_mla.current_platform, "is_device_capability", lambda _: False
    )
    assert flashattn_mla._get_mla_fa_version(config) == 3

    monkeypatch.setattr(
        flashattn_mla.current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    assert flashattn_mla._get_mla_fa_version(config) == 4
