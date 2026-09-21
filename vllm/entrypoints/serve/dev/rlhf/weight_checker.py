# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless comparison and request handling for the Weight Checker.

The endpoint keeps no state: with more than one API worker process, a request
lands on an arbitrary process, so the caller holds the baseline.
"""

from http import HTTPStatus

from fastapi import HTTPException

from vllm.engine.protocol import EngineClient
from vllm.utils.weight_checksum import combine_weight_checksums

_ACTIONS = ("checksum", "reset", "compare")


def compare_weight_checksums(
    baseline: dict[str, str],
    current: dict[str, str],
) -> tuple[bool, list[str]]:
    """Return whether every tensor matches, and the keys that differ.

    Keys present in only one of the two maps count as mismatches. The caller
    owns the baseline: with several API processes, any of them may serve any
    request, so no baseline can live server-side.
    """
    mismatches = sorted(
        key
        for key in baseline.keys() | current.keys()
        if baseline.get(key) != current.get(key)
    )
    return not mismatches, mismatches


def _require_action(body: dict) -> str:
    action = body.get("action")
    if action not in _ACTIONS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"action must be one of checksum|reset|compare, got {action!r}",
        )
    return action


def _require_baseline(body: dict) -> dict[str, str]:
    baseline = body.get("baseline")
    if not isinstance(baseline, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="action='compare' requires a 'baseline' object",
        )
    return baseline


async def _require_awake_engine(client: EngineClient) -> None:
    # A pause or sleep drops the weight storage, so hashing it or rewriting it
    # is meaningless. Checked before the per-action arguments so that every
    # action reports the engine state rather than a missing argument.
    if await client.is_paused():
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail="weight_checker requires an awake, unpaused engine",
        )


async def handle_weight_checker(body: dict, client: EngineClient) -> dict:
    """Run one Weight Checker request and return its response body.

    Request body::

        {"action": "checksum"} -> return SHA-256 digests of all weights
        {"action": "reset"}    -> overwrite GPU weights with random values
        {"action": "compare", "baseline": {name: hex_str}}
                               -> diff current weights against the baseline

    Responses:

    * **checksum**: ``{"checksums": {name: hex_str}}``
    * **reset**:    ``{"status": "reset"}``
    * **compare**:  ``{"match": bool, "mismatches": [str]}``

    Sleep level 2 drops the weight storage, so a paused or sleeping engine
    returns HTTP 409. The RL workflow is in docs/features/weight_checker.md.

    Raises:
        HTTPException: For an unknown action, a missing baseline, a paused or
            sleeping engine, or duplicate keys from the workers.
    """
    action = _require_action(body)
    await _require_awake_engine(client)

    if action == "reset":
        await client.reset_weights()
        return {"status": "reset"}

    baseline = _require_baseline(body) if action == "compare" else None
    per_engine = await client.compute_weight_checksums_all()
    try:
        checksums = combine_weight_checksums(per_engine)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(exc),
        ) from exc

    if baseline is None:
        return {"checksums": checksums}

    match, mismatches = compare_weight_checksums(baseline, checksums)
    return {"match": match, "mismatches": mismatches}
