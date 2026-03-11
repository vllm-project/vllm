# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tools.resolve_kv_connector_requirements import (
    resolve_kv_connector_requirements,
)

KV_CONNECTOR_REQUIREMENTS = """\
# Existing comment
lmcache >= 0.3.9
nixl >= 0.7.1, < 0.10.0 # Required for disaggregated prefill
mooncake-transfer-engine >= 0.3.8
"""


@pytest.mark.parametrize(
    ("cuda_major", "expected_package"),
    [
        (12, "nixl-cu12"),
        (13, "nixl-cu13"),
        (14, "nixl-cu13"),
    ],
)
def test_resolve_kv_connector_requirements(cuda_major, expected_package):
    resolved = resolve_kv_connector_requirements(
        KV_CONNECTOR_REQUIREMENTS,
        cuda_major,
    )

    assert "# Existing comment\n" in resolved
    assert "lmcache >= 0.3.9\n" in resolved
    assert "mooncake-transfer-engine >= 0.3.8\n" in resolved
    assert (
        f"{expected_package} >= 0.7.1, < 0.10.0 # Required for disaggregated prefill\n"
    ) in resolved


def test_resolve_kv_connector_requirements_requires_nixl():
    requirements = "lmcache >= 0.3.9\n"

    with pytest.raises(ValueError, match="Could not find a nixl requirement"):
        resolve_kv_connector_requirements(requirements, cuda_major=12)
