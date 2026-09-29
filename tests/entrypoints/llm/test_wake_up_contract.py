# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm import LLM

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("fully_awake", [True, False])
@pytest.mark.parametrize("tags", [None, ["weights"], ["kv_cache", "scheduling"]])
def test_llm_wake_up_returns_engine_result(fully_awake, tags):
    engine = SimpleNamespace(wake_up=Mock(return_value=fully_awake))
    llm = SimpleNamespace(llm_engine=engine)

    assert LLM.wake_up(llm, tags) is fully_awake
    engine.wake_up.assert_called_once_with(tags)
