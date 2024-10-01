import pytest

from vllm.model_executor.models import _MODELS, ModelRegistry


@pytest.mark.parametrize("model_cls", _MODELS)
def test_registry_imports(model_cls):
    # Ensure all model classes can be imported successfully
    ModelRegistry.resolve_model_cls([model_cls])


@pytest.mark.parametrize("model_cls,is_mm", [
    ("LlamaForCausalLM", False),
    ("MllamaForConditionalGeneration", True),
])
def test_registry_is_multimodal(model_cls, is_mm):
    assert ModelRegistry.is_multimodal_model(model_cls) is is_mm


@pytest.mark.parametrize("model_cls,is_pp", [
    ("MLPSpeculatorPreTrainedModel", False),
    ("DeepseekV2ForCausalLM", True),
])
def test_registry_is_pp(model_cls, is_pp):
    assert ModelRegistry.is_pp_supported_model(model_cls) is is_pp
