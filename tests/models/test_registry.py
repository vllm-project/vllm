import pytest

from vllm.model_executor.models import _MODELS, ModelRegistry


@pytest.mark.parametrize("model_cls", _MODELS)
def test_registry_imports(model_cls):
    # Ensure all model classes can be imported successfully
    ModelRegistry.resolve_model_cls([model_cls])
