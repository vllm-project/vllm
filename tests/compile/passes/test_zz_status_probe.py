# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do not merge: fails on purpose so one shard of this step fails, to check
that the step reports a single failed GitHub status."""


def test_status_probe_fails_on_purpose():
    raise AssertionError("deliberate failure: step status validation")
