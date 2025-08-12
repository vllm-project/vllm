import warnings

import pytest
import torch.cuda

from vllm.model_executor.models import (is_pooling_model,
                                        is_text_generation_model,
                                        supports_multimodal)
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.registry import (_MULTIMODAL_MODELS,
                                                 _SPECULATIVE_DECODING_MODELS,
                                                 _TEXT_GENERATION_MODELS,
                                                 ModelRegistry)
from vllm.platforms import current_platform

from ..utils import fork_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    # Ensure all model classes can be imported successfully
    model_cls, _ = ModelRegistry.resolve_model_cls(model_arch)

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        return  # Ignore these models which do not have a unified format

    if (model_arch in _TEXT_GENERATION_MODELS
            or model_arch in _MULTIMODAL_MODELS):
        assert is_text_generation_model(model_cls)

    # All vLLM models should be convertible to an embedding model
    embed_model = as_embedding_model(model_cls)
    assert is_pooling_model(embed_model)

    if model_arch in _MULTIMODAL_MODELS:
        assert supports_multimodal(model_cls)


@fork_new_process_for_each_test
@pytest.mark.parametrize("model_arch,is_mm,init_cuda,is_ce", [
    ("LlamaForCausalLM", False, False, False),
    ("MllamaForConditionalGeneration", True, False, False),
    ("LlavaForConditionalGeneration", True, True, False),
    ("BertForSequenceClassification", False, False, True),
    ("RobertaForSequenceClassification", False, False, True),
    ("XLMRobertaForSequenceClassification", False, False, True),
])
def test_registry_model_property(model_arch, is_mm, init_cuda, is_ce):
    assert ModelRegistry.is_multimodal_model(model_arch) is is_mm

    assert ModelRegistry.is_cross_encoder_model(model_arch) is is_ce

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry.resolve_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2)


@fork_new_process_for_each_test
@pytest.mark.parametrize("model_arch,is_pp,init_cuda", [
    ("MLPSpeculatorPreTrainedModel", False, False),
    ("DeepseekV2ForCausalLM", True, False),
    ("Qwen2VLForConditionalGeneration", True, True),
])
def test_registry_is_pp(model_arch, is_pp, init_cuda):
    assert ModelRegistry.is_pp_supported_model(model_arch) is is_pp

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry.resolve_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2)


def test_hf_registry_coverage():
    untested_archs = (ModelRegistry.get_supported_archs() -
                      HF_EXAMPLE_MODELS.get_supported_archs())

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}")
