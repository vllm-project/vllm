# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope='module')
def adapter_cache(request, tmpdir_factory):
    # Create dir that mimics the structure of the adapter cache
    adapter_cache = tmpdir_factory.mktemp(
        request.module.__name__) / "adapter_cache"
    return adapter_cache
