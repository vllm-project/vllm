# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.v1.attention.backends.rocm_aiter_fa.known_defects import (
    mark_known_defects,
)


def pytest_collection_modifyitems(config, items):
    mark_known_defects(items)
