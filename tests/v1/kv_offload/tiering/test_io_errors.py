# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""errno classification of failed tier I/O jobs."""

import errno

import pytest

from vllm.v1.kv_offload.tiering.io_errors import (
    IOFailureKind,
    classify_errno,
    indicates_tier_pressure,
)


@pytest.mark.parametrize(
    "err, kind",
    [
        (None, IOFailureKind.UNKNOWN),
        (errno.ENOSPC, IOFailureKind.CAPACITY),
        (errno.EDQUOT, IOFailureKind.CAPACITY),
        (errno.EFBIG, IOFailureKind.CAPACITY),
        (errno.EINTR, IOFailureKind.TRANSIENT),
        (errno.EAGAIN, IOFailureKind.TRANSIENT),
        (errno.EMFILE, IOFailureKind.TRANSIENT),
        (errno.ENFILE, IOFailureKind.TRANSIENT),
        (errno.ENOMEM, IOFailureKind.TRANSIENT),
        (errno.EIO, IOFailureKind.PERSISTENT),
        (errno.EACCES, IOFailureKind.PERSISTENT),
        (errno.EROFS, IOFailureKind.PERSISTENT),
    ],
)
def test_classify_errno(err, kind):
    assert classify_errno(err) is kind


def test_only_transient_failures_skip_back_pressure():
    assert [k for k in IOFailureKind if not indicates_tier_pressure(k)] == [
        IOFailureKind.TRANSIENT
    ]
