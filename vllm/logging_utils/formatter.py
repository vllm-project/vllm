# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        logging.Formatter.__init__(self, fmt, datefmt, style)
        self.root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.."))

    def format(self, record):
        logger = logging.getLogger(record.name)
        if logger.getEffectiveLevel() == logging.DEBUG:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = os.path.relpath(abs_path, self.root_dir)
                except Exception:
                    relpath = record.filename
            else:
                relpath = record.filename
            record.fileinfo = relpath
        else:
            record.fileinfo = record.filename

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
