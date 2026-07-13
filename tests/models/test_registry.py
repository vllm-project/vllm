# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
import warnings
from types import SimpleNamespace

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
from vllm.transformers_utils.config import get_config

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


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )


def test_rwkv7_hf_registry_uses_blinkdl_raw_pth(tmp_path):
    model_info = HF_EXAMPLE_MODELS.get_hf_info("RWKV7ForCausalLM")

    assert model_info.default == (
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    )
    assert model_info.tokenizer_mode == "rwkv"
    assert model_info.is_available_online is False

    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()
    hf_config = get_config(checkpoint, trust_remote_code=False)
    model_cls, arch = ModelRegistry.resolve_model_cls(
        hf_config.architectures,
        SimpleNamespace(
            model_impl="auto",
            convert_type="none",
            runner_type="generate",
        ),
    )

    assert arch == "RWKV7ForCausalLM"
    assert model_cls.__name__ == "RWKV7ForCausalLM"
    assert is_text_generation_model(model_cls)


def test_rwkv7_registry_load_does_not_import_ops_with_unspecified_platform():
    code = """
import sys
import types

for name in (
    "vllm._C",
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
):
    sys.modules[name] = types.ModuleType(name)

import vllm.platforms as platforms
from vllm.platforms.interface import UnspecifiedPlatform

platforms.current_platform = UnspecifiedPlatform()
from vllm.model_executor.models.registry import ModelRegistry

assert "vllm.rwkv7_ops" not in sys.modules
model_cls = ModelRegistry._try_load_model_cls("RWKV7ForCausalLM")
assert model_cls.__name__ == "RWKV7ForCausalLM"
assert "vllm.rwkv7_ops" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", code], check=True)
