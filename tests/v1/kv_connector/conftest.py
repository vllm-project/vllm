# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import pytest


@pytest.fixture
def socket_dir():
    """Keep AF_UNIX paths short regardless of CI's TMPDIR."""
    with tempfile.TemporaryDirectory(prefix="umbp-", dir="/tmp") as directory:
        yield directory
