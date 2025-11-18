"""Lab 08: Multi-Model Serving - Tests"""

import pytest
from solution import ModelRegistry


def test_model_registry():
    """Test model registry."""
    registry = ModelRegistry()
    assert len(registry.models) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
