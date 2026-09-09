# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.utils.routed_expert_session import normalize_routed_expert_cache_salt


@pytest.mark.parametrize("salt", [None, "original", '{"session_id":"user-data"}'])
def test_session_salt_preserves_original_and_is_idempotent(salt):
    normalized = normalize_routed_expert_cache_salt(salt, "session-1")
    assert json.loads(normalized.split(":", 3)[3]) == ["session-1", salt]
    assert normalize_routed_expert_cache_salt(normalized, "session-1") == normalized
    assert normalize_routed_expert_cache_salt(salt, "session-2") != normalized


@pytest.mark.parametrize("session_id", [None, "", "  "])
def test_missing_session_is_rejected(session_id):
    with pytest.raises(ValueError, match="requires x-session-id"):
        normalize_routed_expert_cache_salt("original", session_id)


def test_render_to_generate_preserves_identity_and_rejects_conflicts():
    rendered = normalize_routed_expert_cache_salt("original", "session-1")
    assert (
        normalize_routed_expert_cache_salt(rendered, None, allow_encoded_session=True)
        == rendered
    )
    with pytest.raises(ValueError, match="conflicts"):
        normalize_routed_expert_cache_salt(
            rendered, "session-2", allow_encoded_session=True
        )
    with pytest.raises(ValueError, match="requires x-session-id"):
        normalize_routed_expert_cache_salt(rendered, None)


@pytest.mark.parametrize("encoded", ["broken", "null", '["",null]', '["a",1]'])
def test_malformed_encoded_session_is_rejected(encoded):
    with pytest.raises(ValueError, match="Malformed"):
        normalize_routed_expert_cache_salt(
            "vllm:routed-expert-session:v1:" + encoded,
            None,
            allow_encoded_session=True,
        )
