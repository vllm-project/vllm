# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the steering request protocol."""

from vllm.entrypoints.serve.steering.protocol import SetSteeringRequest


class TestSetSteeringRequest:
    """Validate SetSteeringRequest Pydantic model."""

    def test_basic_vectors(self):
        req = SetSteeringRequest(vectors={0: [1.0, 2.0], 5: [3.0, 4.0]})
        assert req.vectors == {0: [1.0, 2.0], 5: [3.0, 4.0]}
        assert req.scales is None
        assert req.replace is False

    def test_with_scales(self):
        req = SetSteeringRequest(
            vectors={0: [1.0, 2.0]},
            scales={0: 2.5},
        )
        assert req.scales == {0: 2.5}

    def test_replace_flag(self):
        req = SetSteeringRequest(
            vectors={0: [1.0]},
            replace=True,
        )
        assert req.replace is True

    def test_replace_defaults_false(self):
        req = SetSteeringRequest(vectors={0: [1.0]})
        assert req.replace is False

    def test_empty_vectors_allowed(self):
        """Empty dict is a valid request (no-op)."""
        req = SetSteeringRequest(vectors={})
        assert req.vectors == {}

    def test_vectors_defaults_to_empty(self):
        """vectors defaults to empty dict (hook_vectors can supply vectors)."""
        req = SetSteeringRequest()
        assert req.vectors == {}

    def test_string_keys_coerced_to_int(self):
        """JSON dict keys are strings; Pydantic should coerce to int."""
        req = SetSteeringRequest.model_validate(
            {"vectors": {"0": [1.0, 2.0], "5": [3.0, 4.0]}}
        )
        assert 0 in req.vectors
        assert 5 in req.vectors

    def test_scales_none_by_default(self):
        req = SetSteeringRequest(vectors={0: [1.0]})
        assert req.scales is None

    def test_full_request(self):
        req = SetSteeringRequest(
            vectors={0: [1.0, 0.5], 3: [0.0, 1.0]},
            scales={0: 2.0, 3: 0.5},
            replace=True,
        )
        assert req.vectors[0] == [1.0, 0.5]
        assert req.scales[3] == 0.5
        assert req.replace is True
