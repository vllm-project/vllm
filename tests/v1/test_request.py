# SPDX-License-Identifier: Apache-2.0
from vllm.v1.request import RequestStatus

def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert "%s" % RequestStatus.WAITING == "WAITING"
    assert "%s" % RequestStatus.WAITING_FOR_FSM == "WAITING_FOR_FSM"
    assert "%s" % RequestStatus.WAITING_FOR_REMOTE_KVS == "WAITING_FOR_REMOTE_KVS"
    assert "%s" % RequestStatus.RUNNING == "RUNNING"
    assert "%s" % RequestStatus.PREEMPTED == "PREEMPTED"
    assert "%s" % RequestStatus.FINISHED_STOPPED == "FINISHED_STOPPED"
    assert "%s" % RequestStatus.FINISHED_LENGTH_CAPPED == "FINISHED_LENGTH_CAPPED"
    assert "%s" % RequestStatus.FINISHED_ABORTED == "FINISHED_ABORTED"
    assert "%s" % RequestStatus.FINISHED_IGNORED == "FINISHED_IGNORED"
