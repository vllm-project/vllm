# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch


@pytest.fixture
def clean_gpu_memory_between_tests():
    """Free GPU memory before and after each test that uses a real GPU.

    Call gc.collect() + empty_cache() before the test so that allocations
    from previous tests (in the same session) don't prevent this test from
    reserving the memory it needs.  Repeat after the test so that the next
    test starts with a clean slate.
    """
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    yield
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
