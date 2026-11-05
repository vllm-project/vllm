# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound complete NCP tests, including cleanup after a failed assertion."""

import faulthandler
import json
import os
import threading
from pathlib import Path

import pytest


class LiveReporter:
    def __init__(self, journal: Path, timeout: int):
        self.journal = journal
        self.timeout = timeout
        self.timer: threading.Timer | None = None
        self.skipped = 0

    def pytest_runtest_logstart(self, nodeid, location):
        self.timer = threading.Timer(self.timeout, self.hard_timeout, (nodeid,))
        self.timer.daemon = True
        self.timer.start()

    def pytest_runtest_logfinish(self, nodeid, location):
        if self.timer is not None:
            self.timer.cancel()

    def hard_timeout(self, nodeid):
        try:
            print(f"NCP_TEST_HARD_TIMEOUT {self.timeout}s {nodeid}", flush=True)
            faulthandler.dump_traceback(all_threads=True)
        finally:
            os._exit(124)

    def pytest_runtest_logreport(self, report):
        record = {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration": report.duration,
        }
        if report.failed:
            record["traceback"] = report.longreprtext
            print(
                f"\nNCP_IMMEDIATE_FAILURE {report.nodeid}\n{report.longreprtext}",
                flush=True,
            )
        if report.skipped:
            self.skipped += 1
            record["skip_reason"] = str(report.longrepr)
        with self.journal.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def pytest_sessionfinish(self, session, exitstatus):
        if self.timer is not None:
            self.timer.cancel()
        if self.skipped:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    journal = Path(os.environ["NCP_OLMO_CI_JOURNAL"])
    journal.parent.mkdir(parents=True, exist_ok=True)
    config.pluginmanager.register(LiveReporter(journal, 900), "ncp-live-reporter")
