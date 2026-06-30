# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def merge_default_stop_token_ids(
    request_stop_token_ids: list[int] | None,
    default_sampling_params: dict | None,
) -> list[int] | None:
    """Merge server-default and request stop token ids deterministically."""
    default_stop_token_ids = (default_sampling_params or {}).get("stop_token_ids") or ()
    if not default_stop_token_ids:
        return request_stop_token_ids

    # Default stop token ids are model/server requirements, so request stop ids
    # should extend them rather than replace them.
    stop_token_ids = list(default_stop_token_ids)
    seen_token_ids = set(stop_token_ids)
    for token_id in request_stop_token_ids or ():
        if token_id not in seen_token_ids:
            stop_token_ids.append(token_id)
            seen_token_ids.add(token_id)

    return stop_token_ids
