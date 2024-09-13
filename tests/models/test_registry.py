import pytest
import transformers

from vllm.model_executor.models import _MODELS, ModelRegistry


@pytest.mark.parametrize("model_cls", _MODELS)
def test_registry_imports(model_cls):
    if (model_cls == "Qwen2VLForConditionalGeneration"
            and transformers.__version__ < "4.45"):
        pytest.skip("Waiting for next transformers release")

    # Ensure all model classes can be imported successfully
    ModelRegistry.resolve_model_cls([model_cls])
