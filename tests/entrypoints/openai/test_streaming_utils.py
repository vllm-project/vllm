from vllm.entrypoints.openai.streaming_utils import (EmptyDeltaTracker,
                                                     _is_empty_content_only_delta)


class _DummyDelta:

    def __init__(self, **payload) -> None:
        self._payload = payload

    def model_dump(self, *, exclude_none: bool, exclude_unset: bool):
        del exclude_none, exclude_unset
        return {k: v for k, v in self._payload.items() if v is not None}


class _DummyChoice:

    def __init__(self, delta, *, index=0, finish_reason=None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason
        self.index = index


def test_is_empty_content_only_delta_true_for_content_only():
    choice = _DummyChoice(delta=_DummyDelta(content=""))

    assert _is_empty_content_only_delta(choice) is True


def test_is_empty_content_only_delta_false_when_other_fields_present():
    choice_with_role = _DummyChoice(
        delta=_DummyDelta(role="assistant", content=""))
    choice_without_delta = _DummyChoice(delta=None)

    assert _is_empty_content_only_delta(choice_with_role) is False
    assert _is_empty_content_only_delta(choice_without_delta) is False


def test_empty_delta_tracker_tracks_per_choice_index():
    tracker = EmptyDeltaTracker()

    first_empty = _DummyChoice(delta=_DummyDelta(content=""), index=0)
    second_empty_same_choice = _DummyChoice(delta=_DummyDelta(content=""),
                                            index=0)
    first_empty_other_choice = _DummyChoice(delta=_DummyDelta(content=""),
                                            index=1)

    assert tracker.should_suppress(first_empty) is False
    assert tracker.should_suppress(second_empty_same_choice) is True
    assert tracker.should_suppress(first_empty_other_choice) is False


def test_empty_delta_tracker_allows_final_empty_chunk():
    tracker = EmptyDeltaTracker()

    assert tracker.should_suppress(
        _DummyChoice(delta=_DummyDelta(content=""), index=2)) is False

    final_chunk = _DummyChoice(delta=_DummyDelta(content=""),
                               index=2,
                               finish_reason="stop")
    assert tracker.should_suppress(final_chunk) is False


def test_empty_delta_tracker_handles_missing_index():
    tracker = EmptyDeltaTracker()

    first = _DummyChoice(delta=_DummyDelta(content=""), index=None)
    second = _DummyChoice(delta=_DummyDelta(content=""), index=None)

    assert tracker.should_suppress(first) is False
    assert tracker.should_suppress(second) is True
