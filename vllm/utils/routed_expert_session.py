# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

_SESSION_SALT_PREFIX = "vllm:routed-expert-session:v1:"


def normalize_routed_expert_cache_salt(
    cache_salt: str | None,
    session_id: str | None,
    *,
    allow_encoded_session: bool = False,
) -> str:
    """Preserve the original salt and namespace it once by a stable session."""
    encoded = None
    if cache_salt is not None and cache_salt.startswith(_SESSION_SALT_PREFIX):
        try:
            encoded = json.loads(cache_salt[len(_SESSION_SALT_PREFIX) :])
        except ValueError as exc:
            raise ValueError("Malformed routed-expert session cache_salt") from exc
        if not (
            isinstance(encoded, list)
            and len(encoded) == 2
            and isinstance(encoded[0], str)
            and encoded[0].strip()
            and (encoded[1] is None or isinstance(encoded[1], str))
        ):
            raise ValueError("Malformed routed-expert session cache_salt")

    if not session_id or not session_id.strip():
        if allow_encoded_session and encoded is not None:
            assert cache_salt is not None
            return cache_salt
        raise ValueError("Prefix routed-expert omission requires x-session-id")
    if encoded is not None:
        if encoded[0] != session_id:
            raise ValueError("x-session-id conflicts with the session in cache_salt")
        assert cache_salt is not None
        return cache_salt
    return _SESSION_SALT_PREFIX + json.dumps(
        [session_id, cache_salt], ensure_ascii=True, separators=(",", ":")
    )
