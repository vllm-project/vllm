import pytest

from vllm.engine.async_llm_engine import RequestTracker
from vllm.outputs import RequestOutput


class DummyEvent:

    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False


def test_request_tracker():
    tracker = RequestTracker()
    tracker.new_requests_event = DummyEvent()
    stream_1 = tracker.add_request("1")
    assert tracker.new_requests_event.flag
    new, finished = tracker.get_new_and_finished_requests()
    assert not tracker.new_requests_event.flag
    assert len(new) == 1
    assert new[0]["request_id"] == "1"
    assert not finished
    assert not stream_1.finished

    stream_2 = tracker.add_request("2")
    stream_3 = tracker.add_request("3")
    assert tracker.new_requests_event.flag
    new, finished = tracker.get_new_and_finished_requests()
    assert not tracker.new_requests_event.flag
    assert len(new) == 2
    assert new[0]["request_id"] == "2"
    assert new[1]["request_id"] == "3"
    assert not finished
    assert not stream_2.finished
    assert not stream_3.finished

    # request_ids must be unique
    with pytest.raises(KeyError):
        tracker.add_request("1")
    assert not tracker.new_requests_event.flag

    tracker.abort_request("1")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "1" in finished
    assert not new
    assert stream_1.finished

    stream_4 = tracker.add_request("4")
    tracker.abort_request("4")
    assert tracker.new_requests_event.flag
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "4" in finished
    assert not new
    assert stream_4.finished

    stream_5 = tracker.add_request("5")
    assert tracker.new_requests_event.flag
    tracker.process_request_output(
        RequestOutput("2", "output", [], [], [], bool(finished)))
    new, finished = tracker.get_new_and_finished_requests()
    assert not tracker.new_requests_event.flag
    assert len(finished) == 1
    assert "2" in finished
    assert len(new) == 1
    assert new[0]["request_id"] == "5"
    assert stream_2.finished
    assert not stream_5.finished
