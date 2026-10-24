# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False
