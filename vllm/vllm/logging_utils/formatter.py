# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        logging.Formatter.__init__(self, fmt, datefmt, style)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
