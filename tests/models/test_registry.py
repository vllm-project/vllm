# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
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
        # TODO(woosuk): Re-enable this once the MLP Speculator is supported
        # in V1.
        # ("MLPSpeculatorPreTrainedModel", False),
        ("DeepseekV2ForCausalLM", True),
        ("Qwen2VLForConditionalGeneration", True),
    ],
)
def test_registry_is_pp(model_arch, is_pp):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp


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


# Runs in a fresh interpreter (see the test below) so that nothing imported by
# this test module or its conftest can mask what the registry itself imports.
_MODELINFO_CACHE_IMPORT_BOUNDARY_SCRIPT = """
import json
import sys
from dataclasses import asdict

# A model-info cache miss is handled in the parent (API-server) process. Its
# JSON save/read must stay stdlib-only: neither the model loader nor the
# attention layer may be imported as a side effect.
FORBIDDEN = (
    "vllm.model_executor.model_loader",
    "vllm.model_executor.layers.attention.attention",
)


def assert_clean(stage):
    imported = [name for name in FORBIDDEN if name in sys.modules]
    assert not imported, f"{stage}: unexpectedly imported {imported}"


assert_clean("before importing vLLM")

from vllm.model_executor.models import registry

assert_clean("after importing the registry")

model = registry._LazyRegisteredModel(
    module_name="vllm.model_executor.models.llama",
    class_name="LlamaForCausalLM",
)
cache_file = model._get_cache_dir() / model._get_cache_filename()
assert not cache_file.exists(), "VLLM_CACHE_ROOT is not empty"

# Cache miss: the model class is inspected in a subprocess and the result is
# written to VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json.
saved = model.inspect_model_cls()
assert cache_file.exists(), "model-info cache file was not written"
assert_clean("after the model-info cache save")


def _no_subprocess(fn):
    raise AssertionError("second inspect_model_cls() should hit the cache")


# Cache hit: must be served from the JSON file, not by inspecting again.
registry._run_in_subprocess = _no_subprocess
loaded = model.inspect_model_cls()
assert_clean("after the model-info cache read")


def normalize(mi):
    # JSON turns tuples into lists; compare the JSON-normalized dicts.
    return json.loads(json.dumps(asdict(mi)))


assert normalize(loaded) == normalize(saved)
"""


def test_lazy_modelinfo_cache_roundtrip_stays_light(tmp_path):
    """The model-info JSON save/read must not import the model loader or the
    attention layer into the parent process. Importing attention before the
    engine has finished configuring itself (e.g. before breakable CUDA graphs
    are enabled) leaks into forked workers, so the boundary is checked in a
    genuinely clean subprocess interpreter."""
    env = {
        **os.environ,
        "VLLM_CACHE_ROOT": str(tmp_path),
        "VLLM_LOGGING_LEVEL": "ERROR",
    }
    subprocess.run(
        [sys.executable, "-c", _MODELINFO_CACHE_IMPORT_BOUNDARY_SCRIPT],
        check=True,
        env=env,
    )


def test_weight_utils_atomic_writer_reexport():
    """`weight_utils.atomic_writer` keeps working for existing users. Checked
    in its own process: importing `weight_utils` pulls in the model loader,
    which must not happen in the import-boundary test above."""
    code = (
        "from vllm.model_executor.model_loader import weight_utils\n"
        "from vllm.utils.file_utils import atomic_writer\n"
        "assert weight_utils.atomic_writer is atomic_writer\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )
