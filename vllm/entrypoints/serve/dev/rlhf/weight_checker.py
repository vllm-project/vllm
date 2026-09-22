# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless comparison and request handling for the Weight Checker.

The endpoint keeps no state: with more than one API worker process, a request
lands on an arbitrary process, so the caller holds the baseline.
"""

from http import HTTPStatus

from fastapi import HTTPException

from vllm.engine.protocol import EngineClient
from vllm.utils.weight_checksum import (
    combine_weight_checksums,
    compare_weight_checksum_reports,
)

_ACTIONS = ("checksum", "reset", "compare")


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


def _optional_extra_reports(body: dict) -> list[dict[str, str]]:
    """Return the caller's other checksum reports, if it sent any.

    These are what turns ``compare`` from "does this engine match its baseline"
    into "do all of these agree", so a malformed value is rejected rather than
    ignored.
    """
    reports = body.get("checksums")
    if reports is None:
        return []
    if not isinstance(reports, list) or not all(
        isinstance(report, dict) for report in reports
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'checksums' must be a list of checksum response objects",
        )
    return reports


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
        {"action": "compare", "baseline": {name: hex_str},
                             "checksums": [{name: hex_str}, ...]}
                               -> also diff the caller's other checksum
                                  reports against the baseline

    Responses:

    * **checksum**: ``{"checksums": {name: hex_str}}``
    * **reset**:    ``{"status": "reset"}``
    * **compare**:  ``{"match": bool, "mismatches": [str], "ranks": [str]}``

    ``ranks`` lists the rank prefixes the comparison covered, so a caller that
    passed several reports can tell whether they reached the ranks it expected
    rather than only that the ones they did reach agree.

    A paused engine returns HTTP 409, and so does a duplicate key from the
    workers. A sleeping engine is the caller's responsibility to wake first.
    The RL workflow is in docs/features/weight_checker.md.

    Raises:
        HTTPException: For an unknown action, a missing baseline or a
            malformed checksum list, a paused engine, or duplicate keys from
            the workers.
    """
    action = _require_action(body)
    await _require_awake_engine(client)

    if action == "reset":
        await client.reset_weights()
        return {"status": "reset"}

    baseline = _require_baseline(body) if action == "compare" else None
    # Validated before the RPC: a malformed request should not cost a hashing
    # pass over every weight.
    extra_reports = _optional_extra_reports(body) if action == "compare" else []
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

    # More than one report compares replicas against each other, which needs no
    # baseline of its own: every report is held to every other one.
    match, mismatches, ranks = compare_weight_checksum_reports(
        [baseline, checksums, *extra_reports]
    )
    return {"match": match, "mismatches": mismatches, "ranks": ranks}
