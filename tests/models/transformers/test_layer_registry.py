# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's hw-agnostic layer resolution.

`layers._resolve` imports a layer symbol from
`vllm.model_executor.hw_agnostic.layers.<module>` when `VLLM_USE_HW_AGNOSTIC`
is set and the symbol exists, and otherwise falls back to
`vllm.model_executor.layers.<module>`. These tests pin that contract and the
logging that reports which source was used.
"""

import ast
import importlib
import inspect
import logging
import pkgutil
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm.model_executor.models.transformers import layers

HW_MODULE = "vllm.model_executor.hw_agnostic.layers.layernorm"
_HW_PKG = "vllm.model_executor.hw_agnostic.layers"
_VLLM_PKG = "vllm.model_executor.layers"


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


def test_falls_back_to_vllm_when_disabled(monkeypatch, fake_hw_layernorm):
    """Disabled: the vLLM class is used even if a hw-agnostic one exists."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    assert layers._resolve("layernorm", "RMSNorm") is VllmRMSNorm


def test_uses_hw_agnostic_when_enabled(monkeypatch, fake_hw_layernorm, caplog):
    """Enabled and available: the hw-agnostic class is used and logged."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    with caplog.at_level(logging.INFO):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is fake_hw_layernorm.RMSNorm
    assert "Using hw-agnostic layer: RMSNorm" in caplog.text


def test_falls_back_when_symbol_missing(monkeypatch, caplog):
    """Enabled but the symbol is not ported: fall back to vLLM and warn."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    # A hw-agnostic module without the requested attribute triggers fallback.
    empty = types.ModuleType(HW_MODULE)
    monkeypatch.setitem(sys.modules, HW_MODULE, empty)
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    with caplog.at_level(logging.WARNING):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is VllmRMSNorm
    assert "falling back to default" in caplog.text


def test_act_and_mul_falls_back_for_unknown_activation(
    monkeypatch, default_vllm_config
):
    """An activation with no hw-agnostic equivalent falls back to vLLM's.

    `default_vllm_config` supplies the config context the CustomOp needs.
    """
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    from vllm.model_executor.layers.activation import GeluAndMul

    assert isinstance(layers.get_act_and_mul_fn("gelu"), GeluAndMul)


@pytest.fixture(scope="module")
def tiny_llama_path(tmp_path_factory):
    """A randomly-initialized microscopic Llama saved to disk (with an ungated
    tokenizer) so vLLM can load it like any local checkpoint."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)

    path = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


# Registered names of the layers the backend can
# currently route to hw-agnostic implementations.
_COVERED_LAYERS = ("rms_norm", "silu_and_mul")


def _layer_providers(model) -> dict[str, str]:
    """Map each covered layer type present in the model to the provider its
    implementation came from (``hw_agnostic`` or ``vllm``).
    """

    def provider_of(module) -> str | None:
        for cls in type(module).__mro__:
            if "hw_agnostic.layers" in cls.__module__:
                return "hw_agnostic"
            if ".model_executor.layers." in cls.__module__:
                return "vllm"
        return None

    providers: dict[str, str] = {}
    for module in model.modules():
        name = getattr(module, "name", None)
        if (
            name in _COVERED_LAYERS
            and name not in providers
            and (prov := provider_of(module)) is not None
        ):
            providers[name] = prov
    return providers


def _serve(vllm_runner, model_path, prompts):
    """Serve the model through the backend; return (layer_providers, logprobs)."""
    with vllm_runner(
        model_path,
        model_impl="transformers",
        max_model_len=64,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
    ) as runner:
        assert runner.llm.llm_engine.model_config.using_transformers_backend()
        providers = runner.apply_model(_layer_providers)[0]
        outputs = runner.generate_greedy_logprobs(
            prompts, max_tokens=32, num_logprobs=5
        )
        return providers, outputs


def test_hw_agnostic_matches_vllm_end_to_end(monkeypatch, vllm_runner, tiny_llama_path):
    """Serving the tiny model with hw-agnostic layers matches the vLLM baseline."""
    # spawn: worker re-imports layers with the env set (see docstring).
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model pickles the introspection function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from ..utils import check_logprobs_close

    prompts = ["The capital of France is", "vLLM is"]

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    vllm_providers, vllm_outputs = _serve(vllm_runner, tiny_llama_path, prompts)
    # Both replaceable layers present in a Llama block must be vLLM's here.
    assert vllm_providers == {"rms_norm": "vllm", "silu_and_mul": "vllm"}

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    hw_providers, hw_outputs = _serve(vllm_runner, tiny_llama_path, prompts)
    assert hw_providers == {"rms_norm": "hw_agnostic", "silu_and_mul": "hw_agnostic"}

    check_logprobs_close(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=hw_outputs,
        name_0="vllm",
        name_1="hw_agnostic",
    )


_HW_AGNOSTIC_LAYERS = (
    ("layernorm", "RMSNorm"),
    ("activation", "SiluAndMul"),
)


def _hw_layer(module: str, name: str) -> Any:
    """The hw-agnostic class `name` from `hw_agnostic.layers.<module>`."""
    return getattr(
        importlib.import_module(f"vllm.model_executor.hw_agnostic.layers.{module}"),
        name,
    )


def _vllm_layer(module: str, name: str) -> Any:
    """The in-tree class `name` from `model_executor.layers.<module>`."""
    return getattr(
        importlib.import_module(f"vllm.model_executor.layers.{module}"), name
    )


def _mirrored(module: str) -> dict[str, str]:
    """`_mirrored_modules()` narrowed to one module, for validating in isolation."""
    return {
        f"vllm.model_executor.layers.{module}": (
            f"vllm.model_executor.hw_agnostic.layers.{module}"
        )
    }


@pytest.fixture
def oot_registries():
    """The in-tree and hw-agnostic `op_registry_oot`s, restored afterwards.

    Tests register through `register_oot` rather than writing a dict, because
    which dict the write lands in is under test.
    """
    from vllm.model_executor.custom_op import op_registry_oot as in_tree
    from vllm.model_executor.hw_agnostic.custom_op import op_registry_oot as hw_agnostic

    saved = [(reg, dict(reg)) for reg in (in_tree, hw_agnostic)]
    yield in_tree, hw_agnostic
    for registry, before in saved:
        registry.clear()
        registry.update(before)


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_hw_agnostic_layer_is_standalone(module, name):
    """Each hw-agnostic layer is an independent implementation, not a subclass.

    So the two classes are interchangeable only by name, which is what the rest
    of these tests are about.
    """
    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)

    assert hw_cls is not vllm_cls
    assert not issubclass(hw_cls, vllm_cls)
    assert not issubclass(vllm_cls, hw_cls)
    # No in-tree import anywhere in the hw-agnostic layer's own ancestry.
    assert not any(
        c.__module__.startswith("vllm.model_executor.layers.") for c in hw_cls.__mro__
    )


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_hw_agnostic_layer_keeps_the_registered_names(module, name):
    """`__name__` and `name` match the in-tree layer, each in its own registry.

    `__name__` keys the out-of-tree swap in `__new__`; `name` identifies the op
    to `CompilationConfig.custom_ops`. The registries are separate, so both
    classes can register the same name.
    """
    from vllm.model_executor.custom_op import op_registry as in_tree_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry as hw_agnostic_registry,
    )

    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)

    assert hw_cls.__name__ == vllm_cls.__name__
    # `name` may be inherited -- `MergedColumnParallelLinear` and
    # `QKVParallelLinear` both report `column_parallel_linear` -- which is fine,
    # because the swap keys on `__name__`.
    assert hw_cls.name == vllm_cls.name
    assert issubclass(hw_cls, hw_agnostic_registry[hw_cls.name])
    assert issubclass(vllm_cls, in_tree_registry[vllm_cls.name])
    # Same name, two owners, because they are two registries.
    assert hw_agnostic_registry[hw_cls.name] is not in_tree_registry[vllm_cls.name]
    assert hw_agnostic_registry is not in_tree_registry


def test_layer_names_scope_is_a_noop_when_disabled(monkeypatch):
    """Disabled: the in-tree names are untouched, so a plugin overrides in-tree."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    with hw_agnostic_layer_names():
        for module, name in _HW_AGNOSTIC_LAYERS:
            assert _vllm_layer(module, name) is _vllm_layer(module, name)
            assert _vllm_layer(module, name) is not _hw_layer(module, name)


def test_layer_names_scope_rebinds_and_restores(monkeypatch):
    """Enabled: mirrored names resolve to the hw-agnostic class inside the block,
    and to the in-tree class again outside it.

    Restoring matters: in-tree modules import these names, and a permanent swap
    would make `isinstance` checks depend on import order.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    before = {(m, n): _vllm_layer(m, n) for m, n in _HW_AGNOSTIC_LAYERS}

    with hw_agnostic_layer_names():
        for module, name in _HW_AGNOSTIC_LAYERS:
            assert _vllm_layer(module, name) is _hw_layer(module, name)

    for key, cls in before.items():
        assert _vllm_layer(*key) is cls


def test_layer_names_scope_restores_on_exception(monkeypatch):
    """A plugin that raises mid-import must not leave the namespace rebound."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    original = _vllm_layer("layernorm", "RMSNorm")

    with (
        pytest.raises(RuntimeError, match="plugin blew up"),
        hw_agnostic_layer_names(),
    ):
        raise RuntimeError("plugin blew up")

    assert _vllm_layer("layernorm", "RMSNorm") is original


def test_layer_names_scope_leaves_unported_layers_alone(monkeypatch):
    """Names with no hw-agnostic implementation keep their in-tree class.

    The same fallback `_resolve` takes, so a plugin's `GeluAndMul` override
    lands on the class that gets built.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    unported = (("activation", "GeluAndMul"), ("layernorm", "GemmaRMSNorm"))
    before = {(m, n): _vllm_layer(m, n) for m, n in unported}

    with hw_agnostic_layer_names():
        for key, cls in before.items():
            assert _vllm_layer(*key) is cls


def test_layer_names_scope_rebinds_classes_only(monkeypatch):
    """A mirrored *function* name keeps its in-tree implementation.

    The two `get_act_and_mul_fn`s are not interchangeable: the in-tree one takes
    `compile_native` and raises `ValueError` for an unsupported activation, the
    hw-agnostic one takes neither and raises `KeyError`.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    vllm_fn = _vllm_layer("activation", "get_act_and_mul_fn")
    assert vllm_fn is not _hw_layer("activation", "get_act_and_mul_fn")

    with hw_agnostic_layer_names():
        # The class next to it in the same module is rebound, so this is the
        # class/function distinction and not the scope failing to open.
        assert _vllm_layer("activation", "SiluAndMul") is _hw_layer(
            "activation", "SiluAndMul"
        )
        assert _vllm_layer("activation", "get_act_and_mul_fn") is vllm_fn
        # The in-tree signature, still callable as in-tree callers expect.
        inspect.signature(vllm_fn).bind("silu", compile_native=False)


@pytest.mark.parametrize("case", ["function", "op_name"])
def test_layer_names_scope_skips_non_counterparts(monkeypatch, case):
    """An in-tree name that is not the ported class keeps its object.

    Covers an in-tree function (`layernorm.poly_norm`) and a mismatched op name.
    Skipping is per name and only warns; `test_mirrored_names_are_counterparts`
    fails on it.
    """
    from vllm.model_executor.hw_agnostic.layers import _layer_names
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    hw_module = importlib.import_module(f"{_HW_PKG}.layernorm")
    if case == "function":
        name = "poly_norm"
        fake = type(name, (), {"__module__": hw_module.__name__})
        monkeypatch.setattr(hw_module, name, fake, raising=False)
    else:
        name = "RMSNorm"
        monkeypatch.setattr(_hw_layer("layernorm", name), "name", "not_rms_norm")
    warned: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        _layer_names.logger, "warning_once", lambda *args, **_: warned.append(args)
    )
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    original = _vllm_layer("layernorm", name)

    with hw_agnostic_layer_names():
        assert _vllm_layer("layernorm", name) is original
        assert _vllm_layer("activation", "SiluAndMul") is _hw_layer(
            "activation", "SiluAndMul"
        )

    assert [args[2] for args in warned] == [name]


@pytest.fixture
def extra_hw_layers(tmp_path, monkeypatch):
    """Extend the hw-agnostic package with files written under `tmp_path`.

    Yields a writer `(relpath, source, in_tree=False)`; `in_tree=True` extends
    the in-tree package instead. The new modules are dropped from `sys.modules`
    afterwards, so nothing leaks into later tests.
    """
    roots = {}
    for in_tree, pkg_name in ((False, _HW_PKG), (True, _VLLM_PKG)):
        pkg = importlib.import_module(pkg_name)
        roots[in_tree] = tmp_path / ("vllm" if in_tree else "hw_agnostic")
        monkeypatch.setattr(pkg, "__path__", [*pkg.__path__, str(roots[in_tree])])

    def write(relpath: str, source: str, in_tree: bool = False) -> None:
        path = roots[in_tree] / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        importlib.invalidate_caches()

    yield write
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", None) or "").startswith(str(tmp_path)):
            del sys.modules[name]


def _hw_module_names() -> list[str]:
    """Every public hw-agnostic layer module, mirrored or not."""
    hw_pkg = importlib.import_module(_HW_PKG)
    return [
        info.name
        for info in pkgutil.walk_packages(hw_pkg.__path__, prefix=f"{_HW_PKG}.")
        if not any(p.startswith("_") for p in info.name.split("."))
    ]


def test_discovery_finds_the_ported_modules():
    """The pairs follow the package layout, the convention `_resolve` uses."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
    )

    mirrored = _mirrored_modules()
    for module, _ in _HW_AGNOSTIC_LAYERS:
        assert mirrored[f"{_VLLM_PKG}.{module}"] == f"{_HW_PKG}.{module}"
    for vllm_name, hw_name in mirrored.items():
        assert vllm_name.removeprefix(_VLLM_PKG) == hw_name.removeprefix(_HW_PKG)
        assert "._" not in hw_name  # `_layer_names` itself is private


def test_new_hw_agnostic_module_reaches_plugins(
    monkeypatch, extra_hw_layers, oot_registries
):
    """A new hw-agnostic module is discovered and reached by a plugin's override."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
        hw_agnostic_layer_names,
    )

    extra_hw_layers(
        "vocab_parallel_embedding.py",
        "from vllm.model_executor.hw_agnostic.custom_op import PluggableLayer\n"
        "\n\n"
        "class VocabParallelEmbedding(PluggableLayer):\n"
        '    name = "vocab_parallel_embedding"\n',
    )
    extra_hw_layers("hw_only_helpers.py", "class Helper:\n    pass\n")

    mirrored = _mirrored_modules()
    assert (
        mirrored[f"{_VLLM_PKG}.vocab_parallel_embedding"]
        == f"{_HW_PKG}.vocab_parallel_embedding"
    )
    assert f"{_HW_PKG}.hw_only_helpers" not in mirrored.values()

    in_tree_registry, hw_agnostic_registry = oot_registries
    in_tree_registry.pop("VocabParallelEmbedding", None)
    hw_agnostic_registry.pop("VocabParallelEmbedding", None)
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")

    with hw_agnostic_layer_names():
        imported = _vllm_layer("vocab_parallel_embedding", "VocabParallelEmbedding")
        override = imported.register_oot(name="VocabParallelEmbedding")(
            type("PluginVocabParallelEmbedding", (imported,), {})
        )

    hw_cls = _hw_layer("vocab_parallel_embedding", "VocabParallelEmbedding")
    assert imported is hw_cls
    assert hw_agnostic_registry["VocabParallelEmbedding"] is override
    assert "VocabParallelEmbedding" not in in_tree_registry
    assert type(hw_cls.__new__(hw_cls)) is override


def test_subpackage_without_init_is_invisible(extra_hw_layers):
    """A directory without an `__init__.py` is invisible to discovery."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
    )

    extra_hw_layers("quantization/fp8.py", "")
    assert f"{_HW_PKG}.quantization.fp8" not in _mirrored_modules().values()

    extra_hw_layers("quantization/__init__.py", "")
    assert f"{_HW_PKG}.quantization.fp8" in _mirrored_modules().values()


def test_private_modules_are_never_mirrored(extra_hw_layers):
    """A `_`-prefixed hw-agnostic module stays private even with an in-tree twin."""
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
    )

    extra_hw_layers("_helpers.py", "class Helper:\n    pass\n")
    extra_hw_layers("_helpers.py", "class Helper:\n    pass\n", in_tree=True)
    assert f"{_HW_PKG}._helpers" not in _mirrored_modules().values()


def test_broken_subpackage_fails_discovery(extra_hw_layers):
    """A subpackage that fails to import raises instead of vanishing.

    `walk_packages` would otherwise swallow the `ImportError` and skip every
    module under it.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
    )

    extra_hw_layers("quantization/__init__.py", "raise ImportError('broken port')")
    extra_hw_layers("quantization/fp8.py", "")
    with pytest.raises(ImportError, match="broken port"):
        _mirrored_modules()


def test_every_hw_agnostic_directory_is_a_package():
    """Each directory holding a hw-agnostic module has an `__init__.py`."""
    root = Path(importlib.import_module(_HW_PKG).__file__).parent
    dirs = {
        d
        for py in root.rglob("*.py")
        for d in (py.parent, *py.parent.parents)
        if d.is_relative_to(root)
    }
    missing = sorted(
        str(d.relative_to(root)) for d in dirs if not (d / "__init__.py").is_file()
    )
    assert not missing


def test_mirrored_names_are_counterparts():
    """Every hw-agnostic class whose name exists in-tree is that class's port.

    At runtime a mismatch only warns
    (`test_layer_names_scope_skips_non_counterparts`); here it fails.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _hw_agnostic_classes,
        _is_counterpart,
        _mirrored_modules,
    )

    mismatched = []
    for vllm_name, hw_name in _mirrored_modules().items():
        vllm_module = importlib.import_module(vllm_name)
        hw_module = importlib.import_module(hw_name)
        for name, hw_cls in _hw_agnostic_classes(hw_module).items():
            if hasattr(vllm_module, name) and not _is_counterpart(
                getattr(vllm_module, name), hw_cls, name
            ):
                mismatched.append(f"{hw_name}.{name}")
    assert not mismatched


def test_hw_agnostic_classes_reuse_in_tree_names_only_at_the_mirrored_path():
    """A hw-agnostic class named like an in-tree class lives at its mirrored path.

    Registries key on the bare name, so e.g. a `RotaryEmbedding` in hw-agnostic
    `layernorm` could not be told apart from the in-tree one in
    `rotary_embedding`. Reads the in-tree source to avoid importing it all.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import _hw_agnostic_classes

    vllm_root = Path(importlib.import_module(_VLLM_PKG).__file__).parent
    defined_in: dict[str, set[str]] = defaultdict(set)
    for py in vllm_root.rglob("*.py"):
        rel = py.relative_to(vllm_root).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                defined_in[node.name].add(".".join((_VLLM_PKG, *parts)))

    misplaced = []
    for hw_name in _hw_module_names():
        mirror = _VLLM_PKG + hw_name.removeprefix(_HW_PKG)
        for name in _hw_agnostic_classes(importlib.import_module(hw_name)):
            homes = defined_in.get(name)
            if homes and mirror not in homes:
                misplaced.append(f"{hw_name}.{name} (in-tree: {sorted(homes)})")
    assert not misplaced


@pytest.mark.parametrize("hw_agnostic", ["0", "1"])
def test_general_plugins_open_the_scope_only_when_enabled(monkeypatch, hw_agnostic):
    """`load_general_plugins` opens the scope only with `VLLM_USE_HW_AGNOSTIC`.

    With it off, plugins see the in-tree classes.
    """
    import vllm.plugins as plugins

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", hw_agnostic)
    monkeypatch.setattr(plugins, "plugins_loaded", False)

    seen: dict[tuple[str, str], Any] = {}
    monkeypatch.setattr(
        plugins,
        "_run_general_plugins",
        lambda: seen.update(
            {(m, n): _vllm_layer(m, n) for m, n in _HW_AGNOSTIC_LAYERS}
        ),
    )
    plugins.load_general_plugins()

    assert seen, "the plugin-loading body never ran"
    for module, name in _HW_AGNOSTIC_LAYERS:
        expected = (
            _hw_layer(module, name) if hw_agnostic == "1" else _vllm_layer(module, name)
        )
        assert seen[(module, name)] is expected


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
@pytest.mark.parametrize("hw_agnostic", ["0", "1"])
def test_registration_follows_the_imported_class(
    monkeypatch, oot_registries, module, name, hw_agnostic
):
    """The class a plugin imports decides which registry its override lands in.

    `register_oot` writes to the `op_registry_oot` of the module defining the
    class, so base class and registry always match. Rebinding the name is
    therefore enough: an unchanged `from vllm.model_executor.layers.<mod> import
    X` registers where the active path reads.
    """
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    in_tree_registry, hw_agnostic_registry = oot_registries
    # A plugin may already own this name in this process (on Spyre, "RMSNorm"
    # resolves to `SpyreRMSNorm`); `register_oot` asserts on a duplicate.
    in_tree_registry.pop(name, None)
    hw_agnostic_registry.pop(name, None)

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", hw_agnostic)

    with hw_agnostic_layer_names():
        imported = _vllm_layer(module, name)  # what the plugin's import yields
        # Built with `type()` and decorated by hand because the base class is a
        # variable, which mypy rejects in a `class` header; same idiom as
        # `test_plugin_import_pattern_reaches_the_entry_point` below.
        override = imported.register_oot(name=name)(type("Override", (imported,), {}))

    if hw_agnostic == "1":
        assert hw_agnostic_registry[name] is override
        assert name not in in_tree_registry
    else:
        assert in_tree_registry[name] is override
        assert name not in hw_agnostic_registry


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
@pytest.mark.parametrize("hw_agnostic", ["0", "1"])
def test_plugin_import_pattern_reaches_the_entry_point(
    monkeypatch, module, name, hw_agnostic
):
    """The end-to-end contract, both ways round.

    A plugin subclasses the name it imported from
    `vllm.model_executor.layers.<mod>` under `hw_agnostic_layer_names()`; the
    result replaces the class the modeling code instantiates on that path, with
    `__init__` reachable. So a plugin needs no `VLLM_USE_HW_AGNOSTIC`-dependent
    imports.
    """
    from vllm.model_executor.custom_op import op_registry_oot as in_tree_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        hw_agnostic_layer_names,
    )

    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", hw_agnostic)

    with hw_agnostic_layer_names():
        imported = _vllm_layer(module, name)
        plugin_cls = type(f"Plugin{name}", (imported,), {})

    # Where `register_oot` on `imported` would have put it; see
    # `test_registration_follows_the_imported_class`.
    registry = hw_agnostic_registry if hw_agnostic == "1" else in_tree_registry
    monkeypatch.setitem(registry, name, plugin_cls)

    # What the modeling code instantiates on this path.
    entry_cls = _hw_layer(module, name) if hw_agnostic == "1" else imported
    assert plugin_cls.__bases__ == (entry_cls,)

    obj = entry_cls.__new__(entry_cls)
    assert type(obj) is plugin_cls
    # `type.__call__` runs `__init__` only because this holds.
    assert isinstance(obj, entry_cls)


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_override_registered_against_wrong_base_raises(monkeypatch, module, name):
    """In the hw-agnostic registry but derived from the in-tree class.

    `__new__` matches on the bare name and returns the override, which is not an
    instance of the called class, so `type.__call__` skips `__init__`. The first
    symptom would be an unrelated `AttributeError` on `_backward_hooks`; the
    validator names both classes instead.
    """
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        validate_registered_overrides,
    )

    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)
    # Captured the in-tree class, but the hw-agnostic one gets instantiated.
    plugin_cls = type(f"Plugin{name}", (vllm_cls,), {})
    monkeypatch.setitem(hw_agnostic_registry, name, plugin_cls)

    # Correct on the in-tree path, and silently broken on the hw-agnostic one.
    assert isinstance(vllm_cls.__new__(vllm_cls), vllm_cls)
    assert type(hw_cls.__new__(hw_cls)) is plugin_cls
    assert not isinstance(hw_cls.__new__(hw_cls), hw_cls)

    with pytest.raises(TypeError, match=f"Plugin{name}.*does not derive"):
        validate_registered_overrides(_mirrored(module))


@pytest.mark.parametrize("module,name", _HW_AGNOSTIC_LAYERS)
def test_override_stranded_in_the_in_tree_registry_raises(monkeypatch, module, name):
    """Registered against the in-tree class: dropped from the hw-agnostic path.

    The hw-agnostic `__new__` reads only its own registry, so the plain
    hw-agnostic layer runs as if no plugin were loaded. This is why the
    validator checks both registries.
    """
    from vllm.model_executor.custom_op import op_registry_oot as in_tree_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        validate_registered_overrides,
    )

    hw_cls = _hw_layer(module, name)
    vllm_cls = _vllm_layer(module, name)
    plugin_cls = type(f"Plugin{name}", (vllm_cls,), {})
    monkeypatch.setitem(in_tree_registry, name, plugin_cls)
    monkeypatch.delitem(hw_agnostic_registry, name, raising=False)

    # The hw-agnostic path never sees the override.
    assert type(hw_cls.__new__(hw_cls)) is hw_cls

    with pytest.raises(TypeError, match=f"Plugin{name}.*invisible to the hw-agnostic"):
        validate_registered_overrides(_mirrored(module))


def test_correctly_based_override_passes_validation(monkeypatch):
    """The validator rejects only the two mismatches, and only for ported layers.

    A layer with no hw-agnostic implementation is instantiated from its in-tree
    class on both paths, so its override legitimately lives in the in-tree
    registry.
    """
    from vllm.model_executor.custom_op import op_registry_oot as in_tree_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )
    from vllm.model_executor.hw_agnostic.layers._layer_names import (
        _mirrored_modules,
        validate_registered_overrides,
    )

    # What a plugin loaded under `hw_agnostic_layer_names()` registers.
    monkeypatch.setitem(
        hw_agnostic_registry,
        "RMSNorm",
        type("PluginRMSNorm", (_hw_layer("layernorm", "RMSNorm"),), {}),
    )
    # An unported layer, registered against the class both paths instantiate.
    monkeypatch.setitem(
        in_tree_registry,
        "GeluAndMul",
        type("PluginGeluAndMul", (_vllm_layer("activation", "GeluAndMul"),), {}),
    )

    validate_registered_overrides(_mirrored_modules())


def test_hw_agnostic_ops_skip_vendor_forwards(monkeypatch):
    """Hw-agnostic dispatch has two branches: portable, or out-of-tree.

    The in-tree `CustomOp` has a `forward_<vendor>` per platform; the hw-agnostic
    one has none, so a hw-agnostic layer cannot inherit a vendor kernel.
    """
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import CustomOp as VllmCustomOp
    from vllm.model_executor.hw_agnostic.custom_op import (
        CustomOp as HwCustomOp,
    )
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.platforms import current_platform

    vendors = ("cuda", "hip", "cpu", "tpu", "xpu")
    assert all(hasattr(VllmCustomOp, f"forward_{v}") for v in vendors)
    assert not any(hasattr(HwCustomOp, f"forward_{v}") for v in vendors)
    # Inherited, so a plugin's override keeps the property too.
    assert issubclass(HwRMSNorm, HwCustomOp)
    assert not issubclass(HwRMSNorm, VllmCustomOp)
    assert not any(
        hasattr(type("PluginRMSNorm", (HwRMSNorm,), {}), f"forward_{v}")
        for v in vendors
    )

    # Register the hw-agnostic class as its own override, so this builds the class
    # under test whether or not an out-of-tree plugin is loaded in this process.
    monkeypatch.setitem(hw_agnostic_registry, "RMSNorm", HwRMSNorm)
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        # `dispatch_forward` consults only `is_out_of_tree`, so pinning it pins
        # the choice on any host.
        monkeypatch.setattr(current_platform, "is_out_of_tree", lambda: False)
        assert HwRMSNorm(8)._forward_method.__name__ == "forward_native"
        monkeypatch.setattr(current_platform, "is_out_of_tree", lambda: True)
        op = HwRMSNorm(8)
        assert op._forward_method.__name__ == "forward_oot"
        # `forward_oot` is where a plugin hooks in; unoverridden it is the
        # portable implementation, not a vendor one.
        x = torch.randn(2, 8)
        torch.testing.assert_close(op.forward_oot(x), op.forward_native(x))


def test_hw_agnostic_rms_norm_matches_vllm_native():
    """The hw-agnostic `forward_native` computes what vLLM's does; it differs
    only in bypassing the `vllm.ir` op registry, which can resolve to a vendor
    kernel."""
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.hw_agnostic.layers.layernorm import RMSNorm as HwRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        vllm_norm, hw_norm = VllmRMSNorm(64), HwRMSNorm(64)
    hw_norm.load_state_dict(vllm_norm.state_dict())

    x = torch.randn(4, 64)
    torch.testing.assert_close(vllm_norm.forward_native(x), hw_norm.forward_native(x))

    residual = torch.randn(4, 64)
    expected = vllm_norm.forward_native(x.clone(), residual.clone())
    actual = hw_norm.forward_native(x.clone(), residual.clone())
    for want, got in zip(expected, actual):
        torch.testing.assert_close(want, got)
