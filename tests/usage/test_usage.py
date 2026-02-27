# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.usage.usage_lib import is_usage_stats_enabled


def test_is_usage_stats_enabled(monkeypatch):
    with monkeypatch.context() as m:
        # Reset the global state to ensure test isolation.
        # The original value will be restored after the test.
        monkeypatch.setattr("vllm.usage.usage_lib._USAGE_STATS_ENABLED", False)

        m.setattr("vllm.envs.VLLM_DO_NOT_TRACK", True)
        m.setattr("vllm.envs.VLLM_NO_USAGE_STATS", True)
        m.setattr("os.path.exists", lambda x: True)
        assert is_usage_stats_enabled() is False

        m.setattr("vllm.envs.VLLM_DO_NOT_TRACK", False)
        m.setattr("vllm.envs.VLLM_NO_USAGE_STATS", False)
        m.setattr("os.path.exists", lambda x: False)
        assert is_usage_stats_enabled() is True
