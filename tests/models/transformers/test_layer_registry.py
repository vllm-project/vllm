# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for resolving layers between the in-tree and hw-agnostic paths.

`hw_agnostic.resolve` returns a layer from
`vllm.model_executor.hw_agnostic.layers.<module>` when `VLLM_USE_HW_AGNOSTIC`
is set and the layer exists, and otherwise from
`vllm.model_executor.layers.<module>`. The Transformers backend and out-of-tree
plugins both use it, so a plugin's override reaches the class the backend
builds.

CPU-only: no engine is built. The end-to-end counterpart that serves a model
through `vllm_runner` lives in `test_layer_registry_e2e.py` (GPU job).
"""

import importlib
import sys
import types

import pytest
import torch

from vllm.model_executor import hw_agnostic
from vllm.model_executor.models.transformers import layers

HW_MODULE = "vllm.model_executor.hw_agnostic.layers.layernorm"


@pytest.fixture
def fake_hw_layernorm(monkeypatch):
    """Inject a hw-agnostic `layernorm` module exposing a sentinel `RMSNorm`.

    A `SimpleNamespace` stands in for the module: `importlib.import_module`
    returns it from `sys.modules` and `getattr` resolves `RMSNorm`, while its
    attributes are set at construction (no `ModuleType` attribute-set that mypy
    rejects, no constant `setattr` that ruff rejects)."""
    module = types.SimpleNamespace(RMSNorm=type("HwRMSNorm", (), {}))
    monkeypatch.setitem(sys.modules, HW_MODULE, module)
    return module


@pytest.fixture
def resolve_logs(monkeypatch):
    """Messages `resolve` logs, captured directly since the `*_once` methods
    deduplicate."""
    from vllm.model_executor.hw_agnostic import _resolve

    logged: list[str] = []
    for method in ("info_once", "warning_once"):
        monkeypatch.setattr(
            _resolve.logger, method, lambda msg, *args, **_: logged.append(msg % args)
        )
    return logged


def test_falls_back_to_vllm_when_disabled(monkeypatch, fake_hw_layernorm):
    """Disabled: the in-tree class is used even if a hw-agnostic one exists."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    assert hw_agnostic.resolve("layernorm", "RMSNorm") is VllmRMSNorm


def test_uses_hw_agnostic_when_enabled(monkeypatch, fake_hw_layernorm, resolve_logs):
    """Enabled and available: the hw-agnostic class is used and logged."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    resolved = hw_agnostic.resolve("layernorm", "RMSNorm")
    assert resolved is fake_hw_layernorm.RMSNorm
    assert resolve_logs == ["Using hardware agnostic layer layernorm.RMSNorm"]


def test_falls_back_when_symbol_missing(monkeypatch, resolve_logs):
    """Enabled but the symbol is not ported: fall back to in-tree and warn."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    # A hw-agnostic module without the requested attribute triggers fallback.
    empty = types.ModuleType(HW_MODULE)
    monkeypatch.setitem(sys.modules, HW_MODULE, empty)
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    resolved = hw_agnostic.resolve("layernorm", "RMSNorm")
    assert resolved is VllmRMSNorm
    assert resolve_logs == [
        "hw-agnostic layer layernorm.RMSNorm is not available; using the in-tree layer"
    ]


def test_falls_back_when_module_missing(monkeypatch):
    """Enabled but the module is not ported: fall back to in-tree."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    # `None` in `sys.modules` makes the import fail as for a missing module.
    monkeypatch.setitem(sys.modules, HW_MODULE, None)
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    assert hw_agnostic.resolve("layernorm", "RMSNorm") is VllmRMSNorm


def test_broken_hw_agnostic_module_raises(monkeypatch, tmp_path):
    """A port that fails to import raises instead of falling back silently."""
    (tmp_path / "broken_port.py").write_text("import not_a_real_dependency\n")
    hw_pkg = importlib.import_module("vllm.model_executor.hw_agnostic.layers")
    monkeypatch.setattr(hw_pkg, "__path__", [*hw_pkg.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency"):
        hw_agnostic.resolve("broken_port", "Layer")


def test_act_and_mul_falls_back_for_unknown_activation(
    monkeypatch, default_vllm_config
):
    """An activation with no hw-agnostic equivalent falls back to vLLM's.

    `default_vllm_config` supplies the config context the CustomOp needs.
    """
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    from vllm.model_executor.layers.activation import GeluAndMul

    assert isinstance(layers.get_act_and_mul_fn("gelu"), GeluAndMul)


@pytest.fixture
def oot_registries():
    """The in-tree and hw-agnostic `op_registry_oot`s, restored afterwards."""
    from vllm.model_executor.custom_op import op_registry_oot as in_tree
    from vllm.model_executor.hw_agnostic.custom_op import op_registry_oot as hw

    saved = [(registry, dict(registry)) for registry in (in_tree, hw)]
    yield in_tree, hw
    for registry, before in saved:
        registry.clear()
        registry.update(before)


@pytest.fixture
def reload_backend():
    """Re-import the backend modules that resolve layers at import, under the
    test's `VLLM_USE_HW_AGNOSTIC`. Their original namespaces are restored
    afterwards, so classes other tests imported stay current."""
    from vllm.model_executor.models.transformers.fusers import rms_norm

    saved = [(module, dict(vars(module))) for module in (layers, rms_norm)]
    yield lambda: [importlib.reload(module) for module, _ in saved]
    for module, namespace in saved:
        vars(module).clear()
        vars(module).update(namespace)


def _fused_rms_norm(backend, rms_norm, vllm_config):
    fuser = rms_norm.RMSNormFuser(
        zero_centered=False, source_cls="LlamaRMSNorm", eps=1e-6
    )
    return fuser.fuse(torch.nn.RMSNorm(8), "norm", vllm_config)


@pytest.mark.parametrize(
    "module,name,build",
    [
        pytest.param(
            "layernorm",
            "RMSNorm",
            _fused_rms_norm,
            id="RMSNorm",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="The fuser builds `TPAwareRMSNorm`, a subclass, and "
                "overrides are keyed on the exact class name.",
            ),
        ),
        pytest.param(
            "activation",
            "SiluAndMul",
            lambda backend, *_: backend.get_act_and_mul_fn("silu"),
            id="SiluAndMul",
        ),
        pytest.param(
            "activation",
            "GeluAndMul",
            lambda backend, *_: backend.get_act_and_mul_fn("gelu"),
            id="GeluAndMul",
        ),
    ],
)
@pytest.mark.parametrize("use_hw_agnostic", ["0", "1"])
def test_plugin_override_reaches_the_backend(
    monkeypatch,
    oot_registries,
    reload_backend,
    default_vllm_config,
    use_hw_agnostic,
    module,
    name,
    build,
):
    """A plugin that subclasses `resolve`'s result overrides what the backend
    builds, on both paths and for a layer with no hw-agnostic port (GeluAndMul).
    """
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", use_hw_agnostic)
    backend, rms_norm = reload_backend()
    for registry in oot_registries:
        registry.pop(name, None)  # e.g. already owned by a loaded plugin
    # The in-tree `get_act_and_mul_fn` caches the op it builds first; start empty,
    # as a process does where plugins load before any model is built.
    from vllm.model_executor.layers import activation

    monkeypatch.setattr(activation._ACTIVATION_AND_MUL_REGISTRY, "_dict", {})

    base = hw_agnostic.resolve(module, name)
    override = base.register_oot(type(f"Plugin{name}", (base,), {}))

    assert isinstance(build(backend, rms_norm, default_vllm_config), override)


def test_in_tree_override_warns_on_hw_agnostic_path(monkeypatch):
    """An override registered only for the in-tree class is not used on the
    hw-agnostic path, and the warning says how to opt in."""
    from vllm.model_executor.custom_op import op_registry_oot as in_tree_registry
    from vllm.model_executor.hw_agnostic import custom_op as hw_custom_op
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    plugin_cls = type("PluginRMSNorm", (VllmRMSNorm,), {})
    monkeypatch.setitem(in_tree_registry, "RMSNorm", plugin_cls)
    monkeypatch.delitem(hw_custom_op.op_registry_oot, "RMSNorm", raising=False)
    warned: list[str] = []
    monkeypatch.setattr(
        hw_custom_op.logger,
        "warning_once",
        lambda msg, *args, **_: warned.append(msg % args),
    )

    assert type(HwRMSNorm.__new__(HwRMSNorm)) is HwRMSNorm
    assert len(warned) == 1
    assert "PluginRMSNorm is registered for the in-tree RMSNorm" in warned[0]


def test_override_not_derived_from_hw_agnostic_class_raises(monkeypatch):
    """Python would skip such an override's `__init__`, so it is rejected."""
    from vllm.model_executor.hw_agnostic import custom_op as hw_custom_op
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    plugin_cls = type("PluginRMSNorm", (VllmRMSNorm,), {})
    monkeypatch.setitem(hw_custom_op.op_registry_oot, "RMSNorm", plugin_cls)

    with pytest.raises(TypeError, match="PluginRMSNorm.*does not derive from"):
        HwRMSNorm.__new__(HwRMSNorm)
