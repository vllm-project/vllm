# SPDX-License-Identifier: Apache-2.0
import logging
import re


class RedactFilter(logging.Filter):
    """Redacts sensitive information from log messages."""

    def __init__(self, patterns):
        super().__init__()
        self._patterns = [re.compile(pattern) for pattern in patterns]

    def filter(self, record):
        redacted_msg = record.getMessage()
        for pattern in self._patterns:
            redacted_msg = pattern.sub("[...]", redacted_msg)

        # Update msg and clear args only if redaction occurred
        if redacted_msg != record.getMessage():
            record.msg = redacted_msg
            record.args = ()

        return True
