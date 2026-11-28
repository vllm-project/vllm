# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dotted `--group.field` argument handling in FlexibleArgumentParser."""

import json

import pytest

from vllm.utils.argparse_utils import FlexibleArgumentParser


def _parser():
    parser = FlexibleArgumentParser()
    parser.add_argument("--kv-transfer-config", type=str, default=None)
    return parser


def test_dotted_argument_without_value_reports_usage_error(capsys):
    """A trailing `--group.field` with no value used to run off the end of the
    argv list and raise IndexError instead of argparse's usage error."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["--kv-transfer-config.cpu_bytes_to_use"])

    err = capsys.readouterr().err
    assert "expected one argument" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--kv-transfer-config.cpu_bytes_to_use", "1000"],
        ["--kv-transfer-config.cpu_bytes_to_use=1000"],
    ],
)
def test_dotted_argument_with_value_still_expands(argv):
    """The guard must not disturb the two working spellings."""
    args = _parser().parse_args(argv)

    assert json.loads(args.kv_transfer_config) == {"cpu_bytes_to_use": 1000}
