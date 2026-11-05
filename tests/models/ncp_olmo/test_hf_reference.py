# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json

import pytest

from tests.models.ncp_olmo.hf_reference import HF_VERSIONS, PROTOCOL, validate_reference


@pytest.fixture
def reference_pair():
    request = {
        "nonce": "fresh-request",
        "source_sha256": "source",
        "test_sha256": "test-source",
        "commit": "commit",
        "runtime_hashes": {"config.json": "model"},
        "protocol": PROTOCOL,
        "shared_versions": {"torch": "2.14.0+cu130"},
        "prompts": ["prompt"],
        "max_tokens": 16,
        "num_logprobs": 5,
    }
    result = {
        **copy.deepcopy(request),
        "request_sha256": "request-digest",
        "hf_versions": HF_VERSIONS,
        "input_ids": [[1, 2]],
        "outputs": [
            ([3], "text", [[[3, -0.1], [4, -1.0], [5, -2.0], [6, -3.0], [7, -4.0]]])
        ],
    }
    return request, json.loads(json.dumps(result))


def test_reference_transport_preserves_integer_logprob_keys(reference_pair):
    request, result = reference_pair
    assert validate_reference(request, result, "request-digest") == [
        ([3], "text", [{3: -0.1, 4: -1.0, 5: -2.0, 6: -3.0, 7: -4.0}])
    ]


@pytest.mark.parametrize(
    "field",
    [
        "nonce",
        "request_sha256",
        "source_sha256",
        "runtime_hashes",
        "shared_versions",
        "hf_versions",
        "protocol",
        "test_sha256",
        "commit",
    ],
)
def test_reference_rejects_stale_or_wrong_environment(reference_pair, field):
    request, result = reference_pair
    result[field] = "wrong"
    with pytest.raises(AssertionError):
        validate_reference(request, result, "request-digest")


@pytest.mark.parametrize("invalid", ["duplicate", "nonfinite", "missing", "string-key"])
def test_reference_rejects_corrupted_scores(reference_pair, invalid):
    request, result = reference_pair
    pairs = result["outputs"][0][2][0]
    if invalid == "duplicate":
        pairs[1][0] = pairs[0][0]
    elif invalid == "nonfinite":
        pairs[0][1] = float("nan")
    elif invalid == "missing":
        pairs.pop()
    else:
        pairs[0][0] = "3"
    with pytest.raises(AssertionError):
        validate_reference(request, result, "request-digest")
