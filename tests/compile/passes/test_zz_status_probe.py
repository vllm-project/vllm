# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do not merge: fails on a job's first attempt and passes once the job is
retried, to check that the step's GitHub status goes red, then green."""

import os


def test_status_probe_passes_only_when_retried():
    retries = int(os.environ.get("BUILDKITE_RETRY_COUNT", "0"))
    assert retries >= 1, "deliberate failure on the first attempt"
