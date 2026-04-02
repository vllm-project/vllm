"""Helpers for working with transport-scoped request IDs."""

from functools import lru_cache

_PUBLIC_ID_PREFIXES = (
    "cmpl-",
    "chatcmpl-",
    "generate-tokens-",
)
_TRANSPORT_PREFIX = "___prefill_addr_"
_DECODE_PREFIX = "___decode_addr_"


@lru_cache(maxsize=4096)
def normalize_request_id(request_id: str) -> str:
    """Return a transport-stable request ID for KV connector lookups.

    The disaggregated prefill proxy injects a transport-aware request ID:

        ___prefill_addr_<host:port>___decode_addr_<host:port>_<hash>

    API layers may wrap that value with a public prefix such as ``cmpl-`` or
    ``chatcmpl-``, and the engine may append an internal suffix (for example
    ``-<idx>-<rand>`` or ``-<rand>``). This helper strips those wrappers while
    leaving unrelated or malformed request IDs unchanged.
    """
    if not request_id:
        return request_id

    candidate = request_id
    for prefix in _PUBLIC_ID_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    transport_start = candidate.find(_TRANSPORT_PREFIX)
    if transport_start == -1:
        return request_id

    candidate = candidate[transport_start:]
    decode_start = candidate.find(_DECODE_PREFIX)
    if decode_start == -1:
        return request_id

    hash_sep = candidate.find("_", decode_start + len(_DECODE_PREFIX))
    if hash_sep == -1 or hash_sep == len(candidate) - 1:
        return request_id

    transport_end = hash_sep + 1
    while transport_end < len(candidate) and candidate[transport_end].isalnum():
        transport_end += 1

    if transport_end == hash_sep + 1:
        return request_id

    return candidate[:transport_end]
