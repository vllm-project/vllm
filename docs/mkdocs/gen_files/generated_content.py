# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline build-time generated content into existing docs pages.

Source pages mark where generated content goes with a snippet-style marker,
`--8<-- "gen:<key>"`, so the insertion point is explicit and readable. The
substitution happens here (at gen-files time, before mkdocs-gen-files shadows
the page), not via pymdownx.snippets, so the content can be generated at build
time without living in a real file on disk.

The `gen:` prefix keeps these markers distinct from real pymdownx.snippets
includes, and `fill_markers` fails loudly if a marker is missing or left behind
(pymdownx.snippets would otherwise silently drop an unsubstituted marker).
"""

from pathlib import Path

import mkdocs_gen_files
import regex as re

DOCS_DIR = Path(__file__).parent.parent.parent

_MARKER = '--8<-- "gen:{key}"'
_ANY_MARKER = re.compile(r'--8<-- "gen:[^"]*"')


def fill_markers(doc_path: str, blocks: dict[str, str]) -> None:
    """Replace `--8<-- "gen:<key>"` markers in a docs page with generated content.

    Args:
        doc_path: Docs-relative path of the source page to fill.
        blocks: Mapping of marker key to the markdown to insert in its place.

    Raises:
        FileNotFoundError: If the source page does not exist.
        ValueError: If an expected marker is missing, or any `gen:` marker is
            left unsubstituted after filling.
    """
    source = DOCS_DIR / doc_path
    if not source.exists():
        raise FileNotFoundError(f"Cannot fill markers in missing page: {doc_path}")

    text = source.read_text()
    for key, content in blocks.items():
        marker = _MARKER.format(key=key)
        if marker not in text:
            raise ValueError(f"{doc_path}: missing marker {marker}")
        text = text.replace(marker, content)

    if leftover := _ANY_MARKER.search(text):
        raise ValueError(f"{doc_path}: unsubstituted marker {leftover.group()}")

    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(text)
    # Keep the edit button pointing at the real source page
    mkdocs_gen_files.set_edit_path(doc_path, doc_path)
