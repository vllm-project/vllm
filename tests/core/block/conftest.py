# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    """Disable the global cleanup fixture for tests in this directory. This
    provides a ~10x speedup for unit tests that don't load a model to GPU.

    This requires that tests in this directory clean up after themselves if they
    use the GPU.
    """
    return False
