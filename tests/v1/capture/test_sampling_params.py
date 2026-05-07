# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural validation for the new ``SamplingParams.capture`` field.

Phase F declares ``capture: dict[str, Any] | None`` on
:class:`vllm.sampling_params.SamplingParams` and wires a structural-only
validator into ``__post_init__``. Full per-consumer validation — layers
in range, byte budgets, prefix-cache positions — happens at the OpenAI
entrypoint's ``_admit_capture`` against the serving-layer consumer
cache, not here.

These tests exercise only the structural contract: the default, accepted
dict shapes, and the rejected shapes.
"""

from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams


class TestCaptureStructuralValidation:
    """The ``capture`` field must accept ``None`` or a string-keyed dict."""

    def test_default_is_none(self) -> None:
        params = SamplingParams()
        assert params.capture is None

    def test_none_is_accepted_explicitly(self) -> None:
        params = SamplingParams(capture=None)
        assert params.capture is None

    def test_empty_dict_is_accepted(self) -> None:
        params = SamplingParams(capture={})
        assert params.capture == {}

    def test_dict_with_string_keys_is_accepted(self) -> None:
        spec = {
            "filesystem": {"tag": "t", "hooks": {"post_mlp": [0]}},
            "logging": {"level": "INFO"},
        }
        params = SamplingParams(capture=spec)
        assert params.capture == spec
        # The dict must be stored by reference — the entrypoint mutates
        # values in place to the validated CaptureSpec. If this test
        # starts copying the dict the entrypoint contract breaks.
        assert params.capture is spec

    def test_payload_values_are_opaque(self) -> None:
        # Values are ``Any`` — the framework forwards them to the
        # consumer's validator which applies its own schema.
        class _Arbitrary:
            pass

            def __eq__(self, other: object) -> bool:
                return isinstance(other, _Arbitrary)

        arbitrary = _Arbitrary()
        params = SamplingParams(capture={"filesystem": arbitrary})
        assert params.capture == {"filesystem": arbitrary}

    @pytest.mark.parametrize(
        "bad",
        [
            [],  # list
            "not a dict",
            42,
            ("filesystem", {"x": 1}),
        ],
    )
    def test_non_dict_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="capture must be a dict"):
            SamplingParams(capture=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_key",
        [
            1,
            (1, 2),
            None,
            3.14,
        ],
    )
    def test_non_string_keys_rejected(self, bad_key: object) -> None:
        with pytest.raises(ValueError, match="capture keys must be strings"):
            SamplingParams(capture={bad_key: {}})  # type: ignore[dict-item]

    def test_mixed_key_types_rejected(self) -> None:
        # One non-string key anywhere in the dict is enough to fail.
        with pytest.raises(ValueError, match="capture keys must be strings"):
            SamplingParams(capture={"filesystem": {}, 1: {}})
