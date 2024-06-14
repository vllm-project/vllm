import pytest

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.model_executor.models import _MODELS, ModelRegistry

if should_skip_test_group(group_name="TEST_MODELS"):
    pytest.skip("TEST_MODELS=DISABLE, skipping model test group",
                allow_module_level=True)


@pytest.mark.parametrize("model_cls", _MODELS)
def test_registry_imports(model_cls):
    # Ensure all model classes can be imported successfully
    ModelRegistry.load_model_cls(model_cls)
