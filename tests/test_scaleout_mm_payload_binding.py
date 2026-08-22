# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.scale_out.token_in_token_out.serving import (
    _bind_mm_hashes_to_kwargs_data,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_scaleout_mm_hashes_are_bound_to_serialized_payload() -> None:
    first = _bind_mm_hashes_to_kwargs_data(
        {"image": ["victim-hash"]},
        {"image": ["payload-a"]},
    )
    second = _bind_mm_hashes_to_kwargs_data(
        {"image": ["attacker-chosen-hash"]},
        {"image": ["payload-a"]},
    )
    different_payload = _bind_mm_hashes_to_kwargs_data(
        {"image": ["victim-hash"]},
        {"image": ["payload-b"]},
    )

    assert first == second
    assert first != different_payload


@pytest.mark.parametrize(
    ("kwargs_data", "match"),
    [
        (None, "must include kwargs_data"),
        ({"image": [None]}, "serialized kwargs for every item"),
        ({"video": ["payload"]}, "modalities must match"),
    ],
)
def test_scaleout_mm_cache_only_or_mismatched_payloads_are_rejected(
    kwargs_data: dict[str, list[str | None]] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _bind_mm_hashes_to_kwargs_data({"image": ["hash"]}, kwargs_data)
