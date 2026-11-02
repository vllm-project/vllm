# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ORCA endpoint load report headers."""

from unittest.mock import MagicMock, patch

from vllm.entrypoints.serve.utils.orca_metrics import create_orca_header


def test_unsupported_format_warning_names_the_requested_format():
    mock_logger = MagicMock()
    with patch("vllm.entrypoints.serve.utils.orca_metrics.logger", mock_logger):
        assert create_orca_header("xml", [("kv_cache_usage_perc", 0.4)]) is None

    mock_logger.warning.assert_called_once()
    msg, *args = mock_logger.warning.call_args.args
    assert msg % tuple(args) == (
        "Warning: `xml` format is not supported in the ORCA response header"
    )
