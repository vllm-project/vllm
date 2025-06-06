# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass
