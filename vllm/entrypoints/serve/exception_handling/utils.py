# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re


def sanitize_message(message: str) -> str:
    """Strip memory addresses, tracebacks, and file paths from error messages."""
    message = re.sub(r" at 0x[0-9a-f]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    message = re.sub(r"(?:/[\w\-]+)+/[\w\-]+\.\w+", "<path>", message)
    return message.strip()
