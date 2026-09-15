# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
from vllm.model_executor.layers.fused_moe.oracle.fp8_registry import (
    _REGISTERED_FP8_MOE_BACKENDS,
    get_registered_fp8_moe_backend,
    iter_auto_fp8_moe_backends,
    register_fp8_moe_backend,
    registered_fp8_moe_backend_names,
    resolve_fp8_moe_experts,
)


class _SupportedExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def is_supported_config(cls, *args, **kwargs):
        return True, None


class _UnsupportedExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def is_supported_config(cls, *args, **kwargs):
        return False, "test incompatibility"


NOT_AN_EXPERT = object()


@pytest.fixture(autouse=True)
def reset_fp8_moe_backend_registry():
    saved_registry = dict(_REGISTERED_FP8_MOE_BACKENDS)
    _REGISTERED_FP8_MOE_BACKENDS.clear()
    resolve_fp8_moe_experts.cache_clear()
    yield
    _REGISTERED_FP8_MOE_BACKENDS.clear()
    _REGISTERED_FP8_MOE_BACKENDS.update(saved_registry)
    resolve_fp8_moe_experts.cache_clear()


def _class_path(cls: type) -> str:
    return f"{__name__}.{cls.__name__}"


def _moe_config(backend: str):
    parallel_config = SimpleNamespace(use_batched_activation_format=False)
    return SimpleNamespace(
        moe_backend=backend,
        moe_parallel_config=parallel_config,
    )


def test_register_and_resolve_backend():
    register_fp8_moe_backend("My-Backend", _class_path(_SupportedExperts))

    backend = get_registered_fp8_moe_backend("my_backend")
    assert backend is not None
    assert backend.name == "my_backend"
    assert resolve_fp8_moe_experts(backend) == (_SupportedExperts,)
    assert registered_fp8_moe_backend_names() == ("my_backend",)


def test_register_multiple_expert_classes():
    register_fp8_moe_backend(
        "multiple",
        [_class_path(_UnsupportedExperts), _class_path(_SupportedExperts)],
    )

    backend = get_registered_fp8_moe_backend("multiple")
    assert backend is not None
    assert resolve_fp8_moe_experts(backend) == (
        _UnsupportedExperts,
        _SupportedExperts,
    )


def test_registration_is_idempotent():
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))

    assert registered_fp8_moe_backend_names() == ("custom",)


def test_conflicting_registration_fails():
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))

    with pytest.raises(ValueError, match="different configuration"):
        register_fp8_moe_backend("custom", _class_path(_UnsupportedExperts))


@pytest.mark.parametrize("name", ["auto", "triton", "flashinfer-cutlass"])
def test_builtin_name_collision_fails(name: str):
    with pytest.raises(ValueError, match="reserved"):
        register_fp8_moe_backend(name, _class_path(_SupportedExperts))


@pytest.mark.parametrize("paths", ["", [], [""]])
def test_empty_class_paths_fail(paths):
    with pytest.raises(
        ValueError,
        match="At least one expert class path is required",
    ):
        register_fp8_moe_backend("custom", paths)


def test_invalid_expert_class_fails_during_resolution():
    register_fp8_moe_backend("custom", f"{__name__}.NOT_AN_EXPERT")
    backend = get_registered_fp8_moe_backend("custom")
    assert backend is not None

    with pytest.raises(TypeError, match="must be a subclass"):
        resolve_fp8_moe_experts(backend)


def test_auto_selection_is_opt_in():
    register_fp8_moe_backend(
        "manual",
        _class_path(_SupportedExperts),
        auto_select=False,
    )
    register_fp8_moe_backend(
        "automatic",
        _class_path(_SupportedExperts),
        auto_select=True,
    )

    assert [backend.name for backend in iter_auto_fp8_moe_backends()] == ["automatic"]


def test_kernel_config_accepts_registered_backend_name():
    config = KernelConfig(moe_backend="My-Backend")
    assert config.moe_backend == "my_backend"


def test_explicit_registered_backend_selection():
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))

    backend, experts_cls = fp8_oracle.select_fp8_moe_backend(
        _moe_config("custom"),
        weight_key=None,
        activation_key=None,
    )

    assert backend == get_registered_fp8_moe_backend("custom")
    assert experts_cls is _SupportedExperts


def test_explicit_unsupported_registered_backend_fails():
    register_fp8_moe_backend("custom", _class_path(_UnsupportedExperts))

    with pytest.raises(ValueError, match="test incompatibility"):
        fp8_oracle.select_fp8_moe_backend(
            _moe_config("custom"),
            weight_key=None,
            activation_key=None,
        )


def test_registered_backend_is_auto_fallback(monkeypatch):
    register_fp8_moe_backend(
        "custom",
        _class_path(_SupportedExperts),
        auto_select=True,
    )
    monkeypatch.setattr(fp8_oracle, "_get_priority_backends", lambda *args: [])

    backend, experts_cls = fp8_oracle.select_fp8_moe_backend(
        _moe_config("auto"),
        weight_key=None,
        activation_key=None,
    )

    assert backend == get_registered_fp8_moe_backend("custom")
    assert experts_cls is _SupportedExperts


def test_builtin_auto_selection_keeps_priority(monkeypatch):
    register_fp8_moe_backend(
        "custom",
        _class_path(_SupportedExperts),
        auto_select=True,
    )
    monkeypatch.setattr(
        fp8_oracle,
        "_get_priority_backends",
        lambda *args: [fp8_oracle.Fp8MoeBackend.TRITON],
    )
    monkeypatch.setattr(
        fp8_oracle,
        "backend_to_kernel_cls",
        lambda backend: [_SupportedExperts],
    )

    backend, experts_cls = fp8_oracle.select_fp8_moe_backend(
        _moe_config("auto"),
        weight_key=None,
        activation_key=None,
    )

    assert backend == fp8_oracle.Fp8MoeBackend.TRITON
    assert experts_cls is _SupportedExperts


def test_registered_backend_uses_canonical_weights():
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))
    backend = get_registered_fp8_moe_backend("custom")
    assert backend is not None

    tensors = tuple(torch.empty(1) for _ in range(4))
    converted = fp8_oracle.convert_to_fp8_moe_kernel_format(
        backend,
        SimpleNamespace(),
        *tensors,
        None,
        None,
    )

    assert all(got is expected for got, expected in zip(converted, tensors))


def test_unknown_backend_error_lists_registered_names():
    register_fp8_moe_backend("custom", _class_path(_SupportedExperts))

    with pytest.raises(ValueError, match=r"Registered backends: \['custom'\]"):
        fp8_oracle.map_fp8_backend("missing")
