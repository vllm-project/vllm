# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os

from vllm import envs


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        filename = getattr(record, "filename", None)
        assert filename is not None
        name = getattr(record, "name", None)

        record.fileinfo = filename
        if envs.VLLM_LOGGING_IMPORT_PACKAGE_FILE and name is not None:
            # assume logger name is the package name, ie, __name__
            parts = name.split(".")
            if filename != "__init__.py":
                parts = parts[:-1]
            record.fileinfo = os.path.join(*parts, filename)

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
