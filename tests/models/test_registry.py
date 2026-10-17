# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from pathlib import Path

import pytest
import regex as re
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
    _EMBEDDING_MODELS,
    _LATE_INTERACTION_MODELS,
    _MULTIMODAL_MODELS,
    _OOT_SUPPORTED_MODELS,
    _PREVIOUSLY_SUPPORTED_MODELS,
    _REWARD_MODELS,
    _SEQUENCE_CLASSIFICATION_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    _TOKEN_CLASSIFICATION_MODELS,
    _TRANSFORMERS_BACKEND_MODELS,
    _TRANSFORMERS_SUPPORTED_MODELS,
    ModelRegistry,
    _LazyRegisteredModel,
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
        ("KimiLinearForCausalLM", not current_platform.is_rocm()),
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


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )


_DOCS_MODELS_DIR = Path(__file__).parents[2] / "docs" / "models"

# Pooling-mode and truncation-policy values show up as first-column tokens in
# the pooling docs tables; they are not architecture names.
_POOLING_ENUM_TOKENS = frozenset(
    {"ALL", "CLS", "END", "LAST", "MEAN", "NONE", "START", "STEP"}
)

_REGISTRY_ARCHS = (
    set(_TEXT_GENERATION_MODELS)
    | set(_EMBEDDING_MODELS)
    | set(_LATE_INTERACTION_MODELS)
    | set(_REWARD_MODELS)
    | set(_TOKEN_CLASSIFICATION_MODELS)
    | set(_SEQUENCE_CLASSIFICATION_MODELS)
    | set(_MULTIMODAL_MODELS)
)

# Architectures served by other paths (out-of-tree plugins, the Transformers
# fallback backend, speculative decoding) are listed in the docs without
# being part of the model registry.
_NON_REGISTRY_DOCS_ARCHS = (
    set(_SPECULATIVE_DECODING_MODELS)
    | set(_TRANSFORMERS_SUPPORTED_MODELS)
    | set(_TRANSFORMERS_BACKEND_MODELS)
    | set(_OOT_SUPPORTED_MODELS)
)


def _iter_docs_arch_tokens():
    """Yield (token, page, is_converted_row) for every backticked token in the
    first column of a model docs table."""
    pages = [
        _DOCS_MODELS_DIR / "supported_models.md",
        _DOCS_MODELS_DIR / "structured_decisions_models.md",
        *sorted((_DOCS_MODELS_DIR / "pooling_models").glob("*.md")),
    ]
    for page in pages:
        for line in page.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or line.startswith("| -"):
                continue
            cells = line.split("|")
            if len(cells) < 3:
                continue
            arch_cell = cells[1]
            for token in re.findall(r"`([A-Za-z][A-Za-z0-9_.]+)`", arch_cell):
                if any(c.isupper() for c in token):
                    yield token, page.name, "<sup>C</sup>" in arch_cell


def _require_docs_dir():
    if not _DOCS_MODELS_DIR.exists():
        pytest.skip("model docs are not available in this installation")


# These tests only parse the registry dicts and the docs tables; no global
# vLLM state is created, so the post-test cleanup is unnecessary.
@pytest.mark.skip_global_cleanup
def test_docs_architectures_are_registered():
    """Docs table architectures must be registry architectures, converted
    models (rows marked ``<sup>C</sup>``), or documented non-registry archs."""
    _require_docs_dir()
    unknown: dict[str, list[str]] = {}
    for token, page, is_converted_row in _iter_docs_arch_tokens():
        if token in _REGISTRY_ARCHS or token in _NON_REGISTRY_DOCS_ARCHS:
            continue
        if is_converted_row or token in _POOLING_ENUM_TOKENS:
            continue
        unknown.setdefault(page, []).append(token)

    assert not unknown, (
        "The model docs list architectures the registry does not know; fix "
        f"the rows or register the architectures: {unknown}. Rows marked "
        "with <sup>C</sup> are models converted from a generative "
        "architecture and are exempt."
    )


@pytest.mark.skip_global_cleanup
def test_registry_architectures_are_documented():
    """Every architecture in the documented registry dicts must appear in the
    model docs tables."""
    _require_docs_dir()
    documented = {token for token, _, _ in _iter_docs_arch_tokens()}
    missing = sorted(_REGISTRY_ARCHS - documented)

    assert not missing, (
        "Registry architectures are missing from the model docs; add them "
        "to docs/models/supported_models.md, the relevant "
        "docs/models/pooling_models/ page, or "
        f"docs/models/structured_decisions_models.md: {missing}"
    )


@pytest.mark.skip_global_cleanup
def test_removed_architectures_are_absent_from_docs():
    """Architectures dropped from vLLM must not remain in the docs tables."""
    _require_docs_dir()
    resurrected: dict[str, list[str]] = {}
    for token, page, _ in _iter_docs_arch_tokens():
        if token in _PREVIOUSLY_SUPPORTED_MODELS and token not in _REGISTRY_ARCHS:
            resurrected.setdefault(token, []).append(page)

    assert not resurrected, (
        "Architectures removed from vLLM are still listed as supported in "
        "the docs; drop those rows (the docs keep no previously-supported "
        f"section): {resurrected}"
    )
