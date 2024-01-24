import pytest
from vllm.utils import create_kv_caches


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches
