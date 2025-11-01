# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import os

import pytest


def _reload_llm_engine_with_env(val: str | None):
    # envs.VLLM_USE_V1 is computed at import time; reload after setting env
    if val is None:
        os.environ.pop("VLLM_USE_V1", None)
    else:
        os.environ["VLLM_USE_V1"] = val

    import vllm.envs as envs
    import vllm.v1.engine.llm_engine as le

    importlib.reload(envs)
    importlib.reload(le)
    return le


def test_v1_env_zero_raises_clear_message():
    le = _reload_llm_engine_with_env("0")
    with pytest.raises(ValueError) as excinfo:
        le._ensure_v1_env_or_raise()  # called at the start of LLMEngine.__init__
    msg = str(excinfo.value)
    assert "V0 engine was removed" in msg
    assert "unset VLLM_USE_V1 or set VLLM_USE_V1=1" in msg


def test_v1_env_unset_and_one_ok():
    le = _reload_llm_engine_with_env(None)
    le._ensure_v1_env_or_raise()  # should not raise
    le = _reload_llm_engine_with_env("1")
    le._ensure_v1_env_or_raise()  # should not raise
