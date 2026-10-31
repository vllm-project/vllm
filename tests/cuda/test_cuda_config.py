# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA not available"
)


def test_fp8_inc_kv_cache_dtype_is_rejected_on_cuda():
    with pytest.raises(ValueError) as exc_info:
        VllmConfig(cache_config=CacheConfig(cache_dtype="fp8_inc"))

    message = str(exc_info.value)
    assert "fp8_inc" in message
    assert "CUDA" in message
    assert "HPU" in message
    assert "auto" in message


@pytest.mark.parametrize(
    "cache_dtype",
    ["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"],
)
def test_cuda_kv_cache_dtype_is_not_rejected_by_platform_validation(cache_dtype):
    config = VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))

    assert config.cache_config.cache_dtype == cache_dtype
