# SPDX-License-Identifier: Apache-2.0
import logging
import re


class RedactFilter(logging.Filter):
    """Redacts sensitive information from log messages."""

    def __init__(self, patterns):
        super().__init__()
        self._patterns = [re.compile(pattern) for pattern in patterns]

    def filter(self, record):
        for pattern in self._patterns:
            record.msg = pattern.sub("[...]", record.getMessage())
        record.args = ()
        return True
