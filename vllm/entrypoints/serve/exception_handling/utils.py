# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re


def sanitize_message(message: str) -> str:
    """Strip memory addresses, tracebacks, and file paths from error messages."""
    message = re.sub(r" at 0x[0-9a-fA-F]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    # Match each run of /segments whole and check for an extension afterwards:
    # requiring it inside the pattern backtracks superlinearly on long runs.
    message = re.sub(
        r"(?:/[\w\-]+)+(\.\w+)?",
        lambda m: "<path>" if m[1] and m[0].count("/") > 1 else m[0],
        message,
    )
    return message.strip()
