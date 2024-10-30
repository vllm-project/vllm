import warnings

import pytest
import torch.cuda

from vllm.model_executor.models import (is_embedding_model,
                                        is_text_generation_model,
                                        supports_multimodal)
from vllm.model_executor.models.registry import (_EMBEDDING_MODELS,
                                                 _MULTIMODAL_MODELS,
                                                 _SPECULATIVE_DECODING_MODELS,
                                                 _TEXT_GENERATION_MODELS,
                                                 ModelRegistry)
from vllm.platforms import current_platform

from ..utils import fork_new_process_for_each_test


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    # Ensure all model classes can be imported successfully
    model_cls, _ = ModelRegistry.resolve_model_cls(model_arch)

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        pass  # Ignore these models which do not have a unified format
    else:
        assert is_text_generation_model(model_cls) is (
            model_arch in _TEXT_GENERATION_MODELS
            or model_arch in _MULTIMODAL_MODELS)

        assert is_embedding_model(model_cls) is (model_arch
                                                 in _EMBEDDING_MODELS)

        assert supports_multimodal(model_cls) is (model_arch
                                                  in _MULTIMODAL_MODELS)


@fork_new_process_for_each_test
@pytest.mark.parametrize("model_arch,is_mm,init_cuda", [
    ("LlamaForCausalLM", False, False),
    ("MllamaForConditionalGeneration", True, False),
    ("LlavaForConditionalGeneration", True, True),
])
def test_registry_is_multimodal(model_arch, is_mm, init_cuda):
    assert ModelRegistry.is_multimodal_model(model_arch) is is_mm

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
