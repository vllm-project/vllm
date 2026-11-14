# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
import warnings

import pytest
import torch.cuda

from vllm.model_executor.models import (
    is_pooling_model,
    is_text_generation_model,
    supports_multimodal,
)
from vllm.model_executor.models.adapters import (
    as_embedding_model,
    as_seq_cls_model,
)
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    _VLLM_MODELS,
    ModelRegistry,
    _LazyRegisteredModel,
    _ModelRegistry,
    _resolve_module_name,
)
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    # Skip if transformers version is incompatible
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    if model_arch == "Terratorch":
        import importlib.util

        if importlib.util.find_spec("terratorch") is None:
            pytest.skip(
                "terratorch is not installed; "
                "temporarily skipped while PyPI has `lightning` quarantined "
                "(see #41376)"
            )

    # DSpark draft model is supported on CUDA and ROCm; stubbed to None on XPU.
    if model_arch == "DSparkDraftModel" and not (
        current_platform.is_cuda() or current_platform.is_rocm()
    ):
        pytest.skip("DSparkDraftModel is only supported on CUDA and ROCm")

    if model_arch in ("Dots3NoteForCausalLM", "Dots3NoteMTPModel") and not (
        current_platform.is_cuda()
    ):
        pytest.skip("Dots3 NOTE is only supported on CUDA")

    if model_arch in ("HYV4ForCausalLM", "HYV4MTPModel") and not (
        current_platform.is_cuda() or current_platform.is_rocm()
    ):
        pytest.skip("HY V4 is only supported on CUDA and ROCm")

    if (
        model_arch == "DeepseekV4ForConditionalGeneration"
        and not current_platform.is_cuda_alike()
    ):
        pytest.skip("Deepseek V4 vision is only supported on CUDA and ROCm")

    # Ensure all model classes can be imported successfully
    model_cls = ModelRegistry._try_load_model_cls(model_arch)
    assert model_cls is not None

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        return  # Ignore these models which do not have a unified format

    if model_arch in _TEXT_GENERATION_MODELS or model_arch in _MULTIMODAL_MODELS:
        assert is_text_generation_model(model_cls)

    # All vLLM models should be convertible to a pooling model
    assert is_pooling_model(as_seq_cls_model(model_cls))
    assert is_pooling_model(as_embedding_model(model_cls))

    if model_arch in _MULTIMODAL_MODELS:
        assert supports_multimodal(model_cls)


@pytest.mark.parametrize(
    "model_arch,is_mm,score_type",
    [
        ("LlamaForCausalLM", False, "bi-encoder"),
        ("LlavaForConditionalGeneration", True, "bi-encoder"),
        ("Step5ForCausalLM", False, "bi-encoder"),
        ("Step5ForConditionalGeneration", True, "bi-encoder"),
        ("DeepseekV41ForCausalLM", True, "bi-encoder"),
        ("BertForSequenceClassification", False, "cross-encoder"),
        ("RobertaForSequenceClassification", False, "cross-encoder"),
        ("XLMRobertaForSequenceClassification", False, "cross-encoder"),
        ("GteNewModel", False, "bi-encoder"),
        ("GteNewForSequenceClassification", False, "cross-encoder"),
        ("HF_ColBERT", False, "late-interaction"),
    ],
)
def test_registry_model_property(model_arch, is_mm, score_type):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_multimodal is is_mm
    assert model_info.score_type == score_type


@pytest.mark.parametrize(
    "model_arch,is_pp",
    [
        ("DeepseekV2ForCausalLM", True),
        ("Qwen2VLForConditionalGeneration", True),
    ],
)
def test_registry_is_pp(model_arch, is_pp):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp


@pytest.mark.parametrize(
    "model_arch,supported",
    [
        ("Qwen4ExpForCausalLM", True),
        ("Qwen4ExpForConditionalGeneration", True),
        ("Qwen4ExpMTP", True),
        ("LlamaForCausalLM", False),
    ],
)
def test_registry_supports_hc_sp(model_arch, supported):
    """HC SP is opt-in for each model implementation, including MTP."""
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.supports_hc_sp is supported


@pytest.mark.parametrize(
    "model_arch,may_use",
    [
        # Its checkpoint has no MTP embedding, so under PP it needs the target's.
        ("DeepSeekV4MTPModel", True),
        # Not an EAGLE-capable model, so it never declares the attribute.
        ("OPTForCausalLM", False),
    ],
)
def test_registry_may_use_target_embed(model_arch, may_use):
    """A drafter opts in to sharing the target's input embedding; under PP the
    last rank then keeps a copy of it for the drafter."""
    if model_arch == "DeepSeekV4MTPModel" and not current_platform.is_cuda():
        pytest.skip("Only the NVIDIA DeepSeek-V4 MTP opts in")
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.may_use_target_embed is may_use


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch",
    ["LlavaForConditionalGeneration", "Qwen2VLForConditionalGeneration"],
)
def test_registry_inspect_does_not_init_cuda(model_arch):
    """Inspecting a model must not import it: these architectures initialize
    CUDA at import time, so the inspection path has to stay out-of-process."""
    if not current_platform.is_cuda_alike():
        pytest.skip("Requires a CUDA-like platform")

    assert ModelRegistry._try_inspect_model_cls(model_arch) is not None
    assert not torch.cuda.is_initialized()

    ModelRegistry._try_load_model_cls(model_arch)
    if not torch.cuda.is_initialized():
        warnings.warn(
            "This model no longer initializes CUDA on import. "
            "Please test using a different one.",
            stacklevel=2,
        )


@pytest.mark.parametrize(
    "model_arch,supported",
    [
        # ReplaySSM is opt-in per model.
        ("NemotronHForCausalLM", True),
        ("NemotronH_Nano_VL_V2", True),
        ("NemotronH_Omni_Reasoning_V3", True),
        (
            "KimiLinearForCausalLM",
            not (current_platform.is_rocm() or current_platform.is_xpu()),
        ),
        ("KimiK3ForConditionalGeneration", not current_platform.is_rocm()),
        ("Mamba2ForCausalLM", False),
        ("Zamba2ForCausalLM", False),
    ],
)
def test_registry_supports_replayssm(model_arch, supported):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.supports_replayssm is supported


def test_lazy_modelinfo_package_hash_includes_submodules(tmp_path):
    package_dir = tmp_path / "model_package"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.write_text("from .model import Model\n", encoding="utf-8")
    model_file = package_dir / "model.py"
    model_file.write_text("class Model: pass\n", encoding="utf-8")

    first_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    model_file.write_text("class Model:\n    supports_pp = True\n", encoding="utf-8")
    second_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    assert first_hash != second_hash


def test_lazy_modelinfo_package_attempts_cache_load(monkeypatch):
    cached_model_info = object()
    loaded_hashes = []

    def fake_load_cache(self, module_hash):
        loaded_hashes.append(module_hash)
        return cached_model_info

    monkeypatch.setattr(
        _LazyRegisteredModel,
        "_load_modelinfo_from_cache",
        fake_load_cache,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry._run_in_subprocess",
        lambda _: pytest.fail("Package-backed model should use the cache path"),
    )

    registered_model = _LazyRegisteredModel(
        module_name="vllm.model_executor.models.transformers",
        class_name="TransformersForCausalLM",
    )

    result = registered_model.inspect_model_cls()

    assert result is cached_model_info
    assert len(loaded_hashes) == 1
    assert loaded_hashes[0]


def test_fill_model_info_cache(monkeypatch, tmp_path):
    """After one fill, inspecting a model reads the cache instead of starting a
    subprocess."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    arch = "LlamaForCausalLM"
    registry = _ModelRegistry({arch: ModelRegistry.models[arch]})
    registry.fill_model_info_cache()

    monkeypatch.setattr(
        "vllm.model_executor.models.registry._run_in_subprocess",
        lambda _: pytest.fail("Model info should come from the filled cache"),
    )
    assert registry.models[arch].inspect_model_cls().is_text_generation_model


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )


# Persistent state: the registry reports `has_persistent_state` from a
# model class's `supports_persistent_state` declaration.

_STATE_MODEL_SOURCE = """\
from torch import nn


class OutOfTreeStateModel(nn.Module):
    supports_persistent_state = True
"""


def test_registry_in_tree_models_have_no_persistent_state(monkeypatch, tmp_path):
    """Every in-tree architecture reports `has_persistent_state` False. An
    architecture that cannot be inspected here is accounted for only when its
    class cannot be imported in this environment either."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    # The canonical in-tree table, not the process-wide registry plugins extend.
    registry = _ModelRegistry(
        {
            arch: _LazyRegisteredModel(
                module_name=_resolve_module_name(module), class_name=class_name
            )
            for arch, (module, class_name) in _VLLM_MODELS.items()
        }
    )
    registry.fill_model_info_cache()
    results = {
        arch: registry._try_inspect_model_cls(arch) for arch in sorted(_VLLM_MODELS)
    }
    uninspectable = sorted(arch for arch, info in results.items() if info is None)
    assert [
        arch for arch in uninspectable if registry._try_load_model_cls(arch) is not None
    ] == []
    declared = {
        arch: info.has_persistent_state
        for arch, info in results.items()
        if info is not None and info.has_persistent_state is not False
    }
    assert declared == {}


def _out_of_tree_state_model(monkeypatch, tmp_path):
    package = tmp_path / "oot_state_models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "model.py").write_text(_STATE_MODEL_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    registry = _ModelRegistry({})
    registry.register_model(
        "OutOfTreeStateModel", "oot_state_models.model:OutOfTreeStateModel"
    )
    return registry, package / "model.py"


def test_registry_lazy_out_of_tree_model_reports_persistent_state(
    monkeypatch, tmp_path
):
    """A lazily registered out-of-tree class that declares persistent state is
    reported with `has_persistent_state` True."""
    registry, _ = _out_of_tree_state_model(monkeypatch, tmp_path)
    model_info = registry._try_inspect_model_cls("OutOfTreeStateModel")
    assert model_info is not None
    assert model_info.has_persistent_state is True


def test_registry_stale_model_info_cache_cannot_hide_persistent_state(
    monkeypatch, tmp_path
):
    """A model-info cache entry written before the field existed is not read
    as False for an unchanged out-of-tree module: it is inspected again."""
    from dataclasses import asdict

    from vllm.model_executor.models.registry import _ModelInfo

    registry, module_path = _out_of_tree_state_model(monkeypatch, tmp_path)
    lazy = registry.models["OutOfTreeStateModel"]
    module = importlib.import_module("oot_state_models.model")

    stale = asdict(_ModelInfo.from_model_cls(module.OutOfTreeStateModel))
    stale.pop("has_persistent_state", None)
    cache_dir = lazy._get_cache_dir()
    cache_dir.mkdir(parents=True)
    (cache_dir / lazy._get_cache_filename()).write_text(
        json.dumps(
            {
                "hash": _LazyRegisteredModel._get_modelinfo_module_hash(module_path),
                "modelinfo": stale,
            }
        )
    )
    model_info = registry._try_inspect_model_cls("OutOfTreeStateModel")
    assert model_info is not None
    assert model_info.has_persistent_state is True
