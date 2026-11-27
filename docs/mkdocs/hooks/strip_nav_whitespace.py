# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import regex as re

CONTENT_START = '<div class="md-content"'
INDENTATION = re.compile(r"\n\s+")


def on_post_page(output: str, **kwargs) -> str:
    """Strip indentation from the header and nav, leaving page content alone."""
    head, sep, content = output.partition(CONTENT_START)
    return INDENTATION.sub("\n", head) + sep + content
