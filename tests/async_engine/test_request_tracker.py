import pytest

from vllm.engine.async_llm_engine import RequestTracker
from vllm.outputs import RequestOutput


def test_request_tracker():
    tracker = RequestTracker()
    tracker.add_request("1")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(new) == 1
    assert new[0]["request_id"] == "1"
    assert not finished

    tracker.add_request("2")
    tracker.add_request("3")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(new) == 2
    assert new[0]["request_id"] == "2"
    assert new[1]["request_id"] == "3"
    assert not finished

    # request_ids must be unique
    with pytest.raises(KeyError):
        tracker.add_request("1")

    tracker.abort_request("1")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "1" in finished
    assert not new

    tracker.add_request("4")
    tracker.abort_request("4")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "4" in finished
    assert not new

    tracker.add_request("5")
    tracker.process_request_output(
        RequestOutput("2", "output", [], [], finished=True))
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "2" in finished
    assert len(new) == 1
    assert new[0]["request_id"] == "5"
