# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import tests.utils as test_utils
from tests.utils import RemoteOpenAIServer


def test_openai_server_shutdown_wait_covers_engine_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    server = object.__new__(RemoteOpenAIServer)
    server._request_shutdown_timeout = 0.0
    server.proc = Mock(pid=1234)
    engine_timeout_args = []

    def get_engine_timeout(request_timeout, process_timeout):
        engine_timeout_args.append((request_timeout, process_timeout))
        return 60.0

    monkeypatch.setattr(
        test_utils, "get_engine_process_shutdown_timeout", get_engine_timeout
    )
    monkeypatch.setattr(test_utils.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(server, "_kill_process_group_survivors", Mock())

    server._terminate_process_tree()

    assert engine_timeout_args == [(0.0, 0.0)]
    server.proc.wait.assert_called_once_with(timeout=75.0)
