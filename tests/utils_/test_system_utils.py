# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from vllm.envs import disable_envs_cache
from vllm.utils.system_utils import _add_prefix, decorate_logs, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


class TestDecorateLogsEnablePrefix:
    """Tests for the --enable-log-prefix feature."""

    def test_add_prefix_decorates_output(self):
        """Verify _add_prefix adds the expected prefix to writes."""
        buf = io.StringIO()
        _add_prefix(buf, "TestProc", 9999)
        buf.write("hello\n")
        assert buf.getvalue() == "(TestProc pid=9999) hello\n"

    def test_decorate_logs_applies_prefix_by_default(self):
        """decorate_logs should add prefix when not disabled."""
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with (
            mock.patch.object(sys, "stdout", fake_stdout),
            mock.patch.object(sys, "stderr", fake_stderr),
        ):
            decorate_logs("TestWorker")

            fake_stdout.write("stdout line\n")
            fake_stderr.write("stderr line\n")

        assert "(TestWorker pid=" in fake_stdout.getvalue()
        assert "(TestWorker pid=" in fake_stderr.getvalue()

    def test_decorate_logs_skipped_when_enable_prefix_false(self):
        """decorate_logs should be a no-op when enable_prefix=False."""
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with (
            mock.patch.object(sys, "stdout", fake_stdout),
            mock.patch.object(sys, "stderr", fake_stderr),
        ):
            decorate_logs("TestWorker", enable_prefix=False)

            fake_stdout.write("stdout line\n")
            fake_stderr.write("stderr line\n")

        # No prefix should have been added
        assert fake_stdout.getvalue() == "stdout line\n"
        assert fake_stderr.getvalue() == "stderr line\n"

    def test_decorate_logs_skipped_when_configure_logging_off(self):
        """decorate_logs should be a no-op when VLLM_CONFIGURE_LOGGING=0."""
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with (
            mock.patch.object(sys, "stdout", fake_stdout),
            mock.patch.object(sys, "stderr", fake_stderr),
            mock.patch.dict(
                os.environ,
                {"VLLM_CONFIGURE_LOGGING": "0"},
                clear=False,
            ),
        ):
            disable_envs_cache()

            decorate_logs("TestWorker")

            fake_stdout.write("stdout line\n")
            fake_stderr.write("stderr line\n")

        assert fake_stdout.getvalue() == "stdout line\n"
        assert fake_stderr.getvalue() == "stderr line\n"
