# SPDX-License-Identifier: Apache-2.0

import pytest

from ..utils import compare_two_settings


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V1', '0')


def test_cpu_offload():
    compare_two_settings("meta-llama/Llama-3.2-1B-Instruct", [],
                         ["--cpu-offload-gb", "1"])
