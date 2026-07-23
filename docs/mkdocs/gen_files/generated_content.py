# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Append build-time generated content to existing docs pages.

Pages are regenerated with the content appended, shadowing the originals in
the build via mkdocs-gen-files.
"""

from pathlib import Path

import mkdocs_gen_files

DOCS_DIR = Path(__file__).parent.parent.parent


def append_to_page(doc_path: str, content: str) -> None:
    """Append generated markdown to the docs page at `doc_path`.

    Raises:
        FileNotFoundError: If the page does not exist, to catch broken
            references when pages are moved or renamed.
    """
    if not (DOCS_DIR / doc_path).exists():
        raise FileNotFoundError(f"Cannot append generated content to {doc_path}")
    with mkdocs_gen_files.open(doc_path, "a") as f:
        f.write(f"\n{content}")
    # Keep the edit button pointing at the real source page
    mkdocs_gen_files.set_edit_path(doc_path, doc_path)
