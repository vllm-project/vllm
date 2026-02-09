# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.request import RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"
