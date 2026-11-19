# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do not merge: fails on a job's first attempt and on its automatic retry, then
passes on a manual retry, to check that the step's GitHub status goes red, then
green."""

import os


def test_status_probe_passes_only_on_a_manual_retry():
    retries = int(os.environ.get("BUILDKITE_RETRY_COUNT", "0"))
    assert retries >= 2, "deliberate failure until the manual retry"
