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
        def shrink_path(relpath: Path) -> str:
            """
            Shortens a file path for logging display:
            - Removes leading 'vllm' folder if present.
            - If path starts with 'v1',
            keeps the first two and last two levels,
            collapsing the middle as '...'.
            - Otherwise, keeps the first and last two levels,
            collapsing the middle as '...'.
            - If the path is short, returns it as-is.
            - Examples:
            vllm/model_executor/layers/quantization/utils/fp8_utils.py ->
            model_executor/.../quantization/utils/fp8_utils.py
            vllm/model_executor/layers/quantization/awq.py ->
            model_executor/layers/quantization/awq.py
            vllm/v1/attention/backends/mla/common.py ->
            v1/attention/backends/mla/common.py

            Args:
                relpath (Path): The relative path to be shortened.
            Returns:
                str: The shortened path string for display.
            """
            parts = list(relpath.parts)
            new_parts = []
            if parts and parts[0] == "vllm":
                parts = parts[1:]
            if parts and parts[0] == "v1":
                new_parts += parts[:2]
                parts = parts[2:]
            elif parts:
                new_parts += parts[:1]
                parts = parts[1:]
            if len(parts) > 2:
                new_parts += ["..."] + parts[-2:]
            else:
                new_parts += parts
            return "/".join(new_parts)

        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = Path(abs_path).resolve().relative_to(self.root_dir)
                except Exception:
                    relpath = Path(record.filename)
            else:
                relpath = Path(record.filename)
            record.fileinfo = shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ColoredFormatter(NewLineFormatter):
    """Adds ANSI color codes to log levels for terminal output.

    This formatter adds colors by injecting them into the format string for
    static elements (timestamp, filename, line number) and modifying the
    levelname attribute for dynamic color selection.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    GREY = "\033[90m"  # Grey for timestamp and file info
    RESET = "\033[0m"

    def __init__(self, fmt, datefmt=None, style="%"):
        # Inject grey color codes into format string for timestamp and file info
        if fmt:
            # Wrap %(asctime)s with grey
            fmt = fmt.replace("%(asctime)s", f"{self.GREY}%(asctime)s{self.RESET}")
            # Wrap [%(fileinfo)s:%(lineno)d] with grey
            fmt = fmt.replace(
                "[%(fileinfo)s:%(lineno)d]",
                f"{self.GREY}[%(fileinfo)s:%(lineno)d]{self.RESET}",
            )

        # Call parent __init__ with potentially modified format string
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # Store original levelname to restore later (in case record is reused)
        orig_levelname = record.levelname

        # Only modify levelname - it needs dynamic color based on severity
        if (color_code := self.COLORS.get(record.levelname)) is not None:
            record.levelname = f"{color_code}{record.levelname}{self.RESET}"

        # Call parent format which will handle everything else
        msg = super().format(record)

        # Restore original levelname
        record.levelname = orig_levelname

        return msg
