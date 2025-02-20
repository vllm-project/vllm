# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.utils import (create_kv_caches_with_random,
                        create_kv_caches_with_random_flash)


def pytest_configure(config: pytest.Config):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(1 / 8)


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches_with_random


@pytest.fixture()
def kv_cache_factory_flashinfer():
    return create_kv_caches_with_random_flash
