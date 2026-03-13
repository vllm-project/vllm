# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--subtest", action="store", type=str, default=None, help="subtest id"
    )


@pytest.fixture
def subtest(request):
    return request.config.getoption("--subtest")
