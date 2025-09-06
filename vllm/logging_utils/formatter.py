# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from pathlib import Path

from vllm import envs


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

        self.use_relpath = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        if self.use_relpath:
            self.root_dir = Path(__file__).resolve().parent.parent.parent

    def format(self, record):

        def shrink_path(relpath: str) -> str:
            parts = relpath.split("/")
            # Remove the leading 'vllm' folder
            if parts and parts[0] == "vllm":
                parts = parts[1:]
            if parts and parts[0] == "v1":
                # If starts with 'v1', keep the first two and last two levels
                if len(parts) > 4:
                    return "/".join(parts[:2] + ["..."] + parts[-2:])
                else:
                    return "/".join(parts)
            else:
                # Otherwise, keep the first and last two levels
                if len(parts) > 3:
                    return "/".join([parts[0], "..."] + parts[-2:])
                else:
                    return "/".join(parts)

        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = str(
                        Path(abs_path).resolve().relative_to(self.root_dir))
                except Exception:
                    relpath = record.filename
            else:
                relpath = record.filename
            record.fileinfo = shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
