from vllm.entrypoints.openai.streaming_utils import _is_empty_content_only_delta


class _DummyDelta:

    def __init__(self, **payload) -> None:
        self._payload = payload

    def model_dump(self, *, exclude_none: bool, exclude_unset: bool):
        del exclude_none, exclude_unset
        return {k: v for k, v in self._payload.items() if v is not None}


class _DummyChoice:

    def __init__(self, delta, finish_reason=None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


def test_is_empty_content_only_delta_true_for_content_only():
    choice = _DummyChoice(delta=_DummyDelta(content=""))

    assert _is_empty_content_only_delta(choice) is True


def test_is_empty_content_only_delta_false_when_other_fields_present():
    choice_with_role = _DummyChoice(
        delta=_DummyDelta(role="assistant", content=""))
    choice_without_delta = _DummyChoice(delta=None)

    assert _is_empty_content_only_delta(choice_with_role) is False
    assert _is_empty_content_only_delta(choice_without_delta) is False
