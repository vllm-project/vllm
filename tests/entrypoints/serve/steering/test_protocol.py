# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the steering request protocol."""

from vllm.entrypoints.serve.steering.protocol import SetSteeringRequest


class TestSetSteeringRequest:
    """Validate SetSteeringRequest Pydantic model."""

    def test_basic_vectors(self):
        req = SetSteeringRequest(
            vectors={"post_mlp_pre_ln": {0: [1.0, 2.0], 5: [3.0, 4.0]}}
        )
        assert req.vectors is not None
        assert req.vectors["post_mlp_pre_ln"][0] == [1.0, 2.0]
        assert req.prefill_vectors is None
        assert req.decode_vectors is None
        assert req.replace is False

    def test_with_co_located_scale(self):
        req = SetSteeringRequest(
            vectors={"post_mlp_pre_ln": {0: {"vector": [1.0, 2.0], "scale": 2.5}}},
        )
        assert req.vectors is not None
        entry = req.vectors["post_mlp_pre_ln"][0]
        assert isinstance(entry, dict)
        assert entry["vector"] == [1.0, 2.0]
        assert entry["scale"] == 2.5

    def test_replace_flag(self):
        req = SetSteeringRequest(
            vectors={"pre_attn": {0: [1.0]}},
            replace=True,
        )
        assert req.replace is True

    def test_replace_defaults_false(self):
        req = SetSteeringRequest(vectors={"post_mlp_pre_ln": {0: [1.0]}})
        assert req.replace is False

    def test_empty_vectors_allowed(self):
        """Empty dict is a valid request (no-op)."""
        req = SetSteeringRequest(vectors={})
        assert req.vectors == {}

    def test_all_fields_none_by_default(self):
        """All vector fields default to None."""
        req = SetSteeringRequest()
        assert req.vectors is None
        assert req.prefill_vectors is None
        assert req.decode_vectors is None

    def test_string_keys_coerced_to_int(self):
        """JSON dict keys are strings; Pydantic should coerce to int."""
        req = SetSteeringRequest.model_validate(
            {"vectors": {"post_mlp_pre_ln": {"0": [1.0, 2.0]}}}
        )
        assert req.vectors is not None
        assert 0 in req.vectors["post_mlp_pre_ln"]

    def test_full_request(self):
        req = SetSteeringRequest(
            vectors={
                "pre_attn": {0: [1.0, 0.5]},
                "post_mlp_pre_ln": {3: [0.0, 1.0]},
            },
            prefill_vectors={
                "pre_attn": {0: {"vector": [0.1, 0.2], "scale": 2.0}},
            },
            decode_vectors={
                "post_mlp_pre_ln": {3: [0.5, 0.5]},
            },
            replace=True,
        )
        assert req.vectors is not None
        assert req.vectors["pre_attn"][0] == [1.0, 0.5]
        assert req.prefill_vectors is not None
        assert req.decode_vectors is not None
        assert req.replace is True

    def test_multiple_hook_points(self):
        req = SetSteeringRequest(
            vectors={
                "pre_attn": {0: [1.0]},
                "post_attn": {0: [2.0]},
                "post_mlp_pre_ln": {0: [3.0]},
                "post_mlp_post_ln": {0: [4.0]},
            }
        )
        assert req.vectors is not None
        assert len(req.vectors) == 4

    def test_phase_specific_only(self):
        """Can set prefill/decode vectors without base vectors."""
        req = SetSteeringRequest(
            prefill_vectors={"pre_attn": {0: [1.0]}},
            decode_vectors={"post_attn": {0: [2.0]}},
        )
        assert req.vectors is None
        assert req.prefill_vectors is not None
        assert req.decode_vectors is not None
