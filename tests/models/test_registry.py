# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch.cuda

from vllm.model_executor.models import (
    is_pooling_model,
    is_text_generation_model,
    supports_multimodal,
)
from vllm.model_executor.models import registry as model_registry_module
from vllm.model_executor.models.adapters import (
    as_embedding_model,
    as_seq_cls_model,
)
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    ModelRegistry,
    _LazyRegisteredModel,
    _ModelInfo,
    _ModelRegistry,
    _RegisteredModel,
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

    if model_arch in ("PrithviGeoSpatialMAE", "Terratorch"):
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


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_mm,init_cuda,score_type",
    [
        ("LlamaForCausalLM", False, False, "bi-encoder"),
        ("LlavaForConditionalGeneration", True, True, "bi-encoder"),
        ("BertForSequenceClassification", False, False, "cross-encoder"),
        ("RobertaForSequenceClassification", False, False, "cross-encoder"),
        ("XLMRobertaForSequenceClassification", False, False, "cross-encoder"),
        ("GteNewModel", False, False, "bi-encoder"),
        ("GteNewForSequenceClassification", False, False, "cross-encoder"),
        ("HF_ColBERT", False, False, "late-interaction"),
    ],
)
def test_registry_model_property(model_arch, is_mm, init_cuda, score_type):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_multimodal is is_mm
    assert model_info.score_type == score_type

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_pp,init_cuda",
    [
        # TODO(woosuk): Re-enable this once the MLP Speculator is supported
        # in V1.
        # ("MLPSpeculatorPreTrainedModel", False, False),
        ("DeepseekV2ForCausalLM", True, False),
        ("Qwen2VLForConditionalGeneration", True, True),
    ],
)
def test_registry_is_pp(model_arch, is_pp, init_cuda):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,supported",
    [
        # ReplaySSM is opt-in per model; only Nemotron-H sets the flag today.
        ("NemotronHForCausalLM", True),
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


def test_prepare_model_info_inspects_builtin_lazy_registration():
    registry = _ModelRegistry(
        {"Qwen3ForCausalLM": ModelRegistry.models["Qwen3ForCausalLM"]}
    )
    with patch.object(_LazyRegisteredModel, "prepare_model_info") as prepare_model_info:
        registry.prepare_model_info("Qwen3ForCausalLM")

    prepare_model_info.assert_called_once_with()


def test_lazy_prepare_model_info_requires_readable_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    registered_model = ModelRegistry.models["Qwen3ForCausalLM"]

    with (
        patch.object(_LazyRegisteredModel, "inspect_model_cls"),
        pytest.raises(RuntimeError, match="was not persisted"),
    ):
        registered_model.prepare_model_info()


def test_inspect_model_info_keeps_best_effort_cache_write(tmp_path, monkeypatch):
    cache_root = tmp_path / "not-a-directory"
    cache_root.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(cache_root))
    registered_model = ModelRegistry.models["Qwen3ForCausalLM"]
    model_info = _ModelInfo.from_model_cls(torch.nn.Module)

    with patch(
        "vllm.model_executor.models.registry._run_in_subprocess",
        return_value=model_info,
    ):
        assert registered_model.inspect_model_cls() is model_info


def test_lazy_prepare_model_info_accepts_readable_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    registered_model = ModelRegistry.models["Qwen3ForCausalLM"]
    model_info = _ModelInfo.from_model_cls(torch.nn.Module)
    model_path = Path(model_registry_module.__file__).parent / "qwen3.py"
    module_hash = registered_model._get_modelinfo_module_hash(model_path)
    registered_model._save_modelinfo_to_cache(model_info, module_hash)

    with patch.object(
        _LazyRegisteredModel, "inspect_model_cls", return_value=model_info
    ):
        registered_model.prepare_model_info()


@pytest.mark.parametrize(
    ("registration", "model_arch", "error"),
    [
        (None, "UnknownForCausalLM", "is not registered"),
        (
            _LazyRegisteredModel("plugin.models", "PluginModel"),
            "PluginForCausalLM",
            "is not a built-in model",
        ),
        (
            _RegisteredModel(interfaces=None, model_cls=torch.nn.Module),
            "Qwen3ForCausalLM",
            "is not registered lazily",
        ),
        (
            _LazyRegisteredModel("plugin.models", "PluginModel"),
            "Qwen3ForCausalLM",
            "does not use its built-in registration",
        ),
    ],
)
def test_prepare_model_info_rejects_unsupported_registration(
    registration, model_arch, error
):
    models = {} if registration is None else {model_arch: registration}
    registry = _ModelRegistry(models)

    with pytest.raises(ValueError, match=error):
        registry.prepare_model_info(model_arch)


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )
