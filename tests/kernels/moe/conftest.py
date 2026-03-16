# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--subtests", action="store", type=str, default=None, help="subtest ids"
    )


@pytest.fixture
def subtests(request):
    return request.config.getoption("--subtests")
