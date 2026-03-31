# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.v1.attention.backends.triton_attn import (
    TRITON_ATTN_SEQ_THRESHOLD_3D_ENV,
    _maybe_override_seq_threshold_3d,
)


def test_seq_threshold_override_uses_env_value():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_SEQ_THRESHOLD_3D_ENV: "8"},
        clear=False,
    ):
        assert _maybe_override_seq_threshold_3d(16) == 8


def test_seq_threshold_override_keeps_default_when_unset():
    with patch.dict("os.environ", {}, clear=True):
        assert _maybe_override_seq_threshold_3d(16) == 16


def test_seq_threshold_override_requires_positive_integer():
    with patch.dict(
        "os.environ",
        {TRITON_ATTN_SEQ_THRESHOLD_3D_ENV: "0"},
        clear=False,
    ):
        try:
            _maybe_override_seq_threshold_3d(16)
        except ValueError as exc:
            assert TRITON_ATTN_SEQ_THRESHOLD_3D_ENV in str(exc)
        else:
            raise AssertionError(
                "Expected ValueError for invalid seq-threshold override"
            )
