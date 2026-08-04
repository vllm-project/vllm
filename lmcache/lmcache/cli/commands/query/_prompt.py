# SPDX-License-Identifier: Apache-2.0
"""Prompt placeholder expansion helpers for ``lmcache query``."""

# Standard
from pathlib import Path
from typing import Optional
import re

PLACEHOLDER = re.compile(r"\{(\w+)\}")
BUILTIN_DOCUMENT_PATHS = {
    "lmcache": (
        Path(__file__).resolve().parent.parent.parent / "documents" / "lmcache.txt"
    ),
}


class PromptBuilder:
    """Build a complete prompt from CLI inputs."""

    def __init__(self, prompt: str, documents_args: Optional[list[str]] = None) -> None:
        self._complete_prompt = self.expand_prompt(prompt, documents_args or [])

    @property
    def complete_prompt(self) -> str:
        """Return the expanded prompt from constructor inputs."""
        return self._complete_prompt

    def expand_prompt(
        self,
        prompt: str,
        documents_args: Optional[list[str]] = None,
    ) -> str:
        """Expand placeholders and return the complete prompt text.

        Args:
            prompt: The prompt template string with {name} placeholders.
            documents_args: Optional list of document paths or NAME=PATH strings.

        Returns:
            The expanded prompt string.
        """
        docs_args = documents_args or []
        documents, appended_text = resolve_documents(prompt, docs_args)

        for key in _unique_placeholders(prompt):
            if key not in documents:
                unknown_documents(key)

        complete_prompt = PLACEHOLDER.sub(
            lambda match: documents[match.group(1)], prompt
        )
        if appended_text:
            complete_prompt = (
                f"{complete_prompt}\n{appended_text}"
                if complete_prompt
                else appended_text
            )
        return complete_prompt


def resolve_documents(
    prompt_template: str, documents_args: list[str]
) -> tuple[dict[str, str], str]:
    """Resolve ``--documents`` args into placeholder mapping and trailing text.

    Args:
        prompt_template: The prompt string containing placeholders.
        documents_args: List of document arguments from the CLI.

    Returns:
        A tuple containing a dictionary mapping placeholder names to document
        content and a string containing any appended document text.
    """
    placeholders = _unique_placeholders(prompt_template)
    documents: dict[str, str] = {}
    plain_docs: list[str] = []
    for item in documents_args:
        if "=" not in item:
            plain_docs.append(_read_document_file(item))
            continue
        name, path = [x.strip() for x in item.split("=", 1)]
        if not name:
            raise ValueError(f"Invalid --documents {item!r}; empty name")
        documents[name] = _read_document_file(path, name=name)

    unresolved = [key for key in placeholders if key not in documents]
    for idx, key in enumerate(unresolved[: len(plain_docs)]):
        documents[key] = plain_docs[idx]
    appended_docs = plain_docs[len(unresolved) :]

    for key in placeholders:
        if key in documents:
            continue
        builtin_path = BUILTIN_DOCUMENT_PATHS.get(key)
        if builtin_path is None:
            continue
        documents[key] = _read_document_file(str(builtin_path), name=key)

    return documents, "\n".join(appended_docs).strip()


def unknown_documents(key: str) -> None:
    """Raise an error for a missing documents placeholder.

    Args:
        key: The name of the missing placeholder.

    Raises:
        ValueError: Always raised with a descriptive message.
    """
    raise ValueError(
        f"Unknown documents {key!r}. Define it with --documents {key}=PATH "
    )


def _read_document_file(path: str, *, name: Optional[str] = None) -> str:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        if name is not None:
            raise ValueError(f"documents file not found for {name!r}: {file_path}")
        raise ValueError(f"documents file not found: {file_path}")
    return file_path.read_text(encoding="utf-8", errors="replace")


def _unique_placeholders(prompt_template: str) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for match in PLACEHOLDER.finditer(prompt_template):
        key = match.group(1)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys
