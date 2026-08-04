# SPDX-License-Identifier: Apache-2.0
"""Translate missing Chinese Sphinx gettext entries with a model endpoint."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from urllib import request
import json
import os
import sys

LOCALE_DIR = Path("docs/source/locale/zh_CN/LC_MESSAGES")
MAX_ENTRIES = int(os.environ.get("TRANSLATION_MAX_ENTRIES", "500"))

SYSTEM_PROMPT = (
    "You translate Sphinx gettext entries from English to Simplified Chinese. "
    "Each request contains one focal entry to translate, optionally preceded "
    "by a 'Source file:' line and surrounded by 'Previous entry' and "
    "'Next entry' blocks for context. Translate ONLY the focal msgid under "
    "'Translate this entry'. Return the Chinese translation as plain text "
    "with no labels, no surrounding quotes, no other entries, and no "
    "explanation. "
    "Use the neighboring entries only to infer the section topic and to "
    "keep terminology consistent; never include their text in your output. "
    "If the focal msgid is a code identifier (snake_case, "
    "SCREAMING_SNAKE_CASE, a config key, environment variable, file path, "
    "URL, or product name), return it unchanged. "
    "Do not add notes, explanations, summaries, or comments. "
    "Preserve all reStructuredText/Sphinx formatting exactly, including "
    "headings, indentation, directives, roles, labels, anchors, references, "
    "admonitions, lists, tables, and cross-references. "
    "Do not translate or change code blocks, inline code, shell commands, "
    "configuration snippets, Python/JSON/YAML/TOML, API signatures, "
    "environment variables, file paths, URLs, placeholders, package names, "
    "class names, function names, method names, parameter names, CLI flags, "
    "model names, feature names, product names, or values inside backticks. "
    "Preserve Sphinx roles exactly, such as :ref:`...`, :doc:`...`, "
    ":class:`...`, :func:`...`, :meth:`...`, :mod:`...`, :attr:`...`, "
    ":obj:`...`, and :term:`...`. "
    "Translate only normal explanatory prose. "
    "Use natural Simplified Chinese for developer documentation. "
    "Use these terms consistently: "
    "KV cache -> KV Cache; KV Cache -> KV Cache; "
    "inference -> 推理; serving -> 服务; "
    "prefill -> Prefill; decode -> 解码; decoding -> 解码; "
    "offload -> 卸载; offloading -> 卸载; "
    "prefetch -> 预取; prefetch module -> 预取模块; "
    "evict -> 逐出; eviction -> 逐出; "
    "recompute -> 重计算; recomputation -> 重计算; "
    "throughput -> 吞吐量; latency -> 延迟; "
    "GPU memory -> 显存; VRAM -> 显存; CPU memory -> CPU 内存; "
    "backend -> 后端; connector -> 连接器; storage -> 存储; "
    "cache hit -> 缓存命中; cache miss -> 缓存未命中; lookup -> 查找; "
    "chunk -> 块; layerwise -> 逐层; compression -> 压缩; "
    "quantization -> 量化; "
    "serialization -> 序列化; deserialization -> 反序列化; "
    "disaggregated prefill -> 分离式 Prefill; multi-tenant -> 多租户. "
    "Keep these terms unchanged: LMCache, vLLM, SGLang, TensorRT-LLM, NIXL, "
    "CUDA, ROCm, Redis, S3, GDS, POSIX, Hugging Face, CacheBlend, cache_salt, "
    "TTFT, LLM, GPU, CPU, API, token, prompt, system prompt, Attention. "
    "If unsure whether something is syntax, code, an identifier, or a "
    "product name, keep it unchanged."
)


class PoEntry:
    """One translatable message read from a ``.po`` translation file.

    A ``.po`` file is a list of messages. Each message pairs the original
    English text with its Chinese translation, plus some status markers.

    Attributes:
        start: Line number where this message begins in the file.
        end: Line number just after this message ends in the file.
        flags: Status markers on the message. ``"fuzzy"`` means the English
            text changed since the last translation, so the Chinese is stale
            and should be redone.
        msgid: The original English text.
        msgstr: The current Chinese translation (empty string if not yet
            translated).
    """

    def __init__(
        self,
        start: int,
        end: int,
        flags: set[str],
        msgid: str,
        msgstr: str,
    ) -> None:
        self.start = start
        self.end = end
        self.flags = flags
        self.msgid = msgid
        self.msgstr = msgstr


def decode_po_string(value: str) -> str:
    """Unwrap one quoted ``.po`` line into a plain Python string."""
    return json.loads(value)


def encode_po_string(value: str) -> list[str]:
    """Convert a Python string into one or more quoted ``.po`` lines."""
    if "\n" not in value:
        return [json.dumps(value, ensure_ascii=False)]

    lines = ['""']
    parts = value.splitlines(keepends=True)
    for part in parts:
        lines.append(json.dumps(part, ensure_ascii=False))
    return lines


def collect_field(lines: list[str], index: int) -> tuple[str, int]:
    """Read one ``msgid``/``msgstr`` value plus any continuation lines.

    Returns:
        The joined string value, and the next line number to read.
    """
    _, raw_value = lines[index].split(" ", 1)
    values = [decode_po_string(raw_value.strip())]
    index += 1

    while index < len(lines) and lines[index].startswith('"'):
        values.append(decode_po_string(lines[index].strip()))
        index += 1

    return "".join(values), index


def parse_entries(lines: list[str]) -> list[PoEntry]:
    """Parse a ``.po`` file into one :class:`PoEntry` per message."""
    entries: list[PoEntry] = []
    index = 0

    while index < len(lines):
        start = index
        flags: set[str] = set()

        while index < len(lines) and not lines[index].startswith("msgid "):
            if lines[index].startswith("#,"):
                flags.update(flag.strip() for flag in lines[index][2:].split(","))
            index += 1

        if index >= len(lines):
            break

        msgid, index = collect_field(lines, index)

        while index < len(lines) and not lines[index].startswith("msgstr "):
            index += 1

        if index >= len(lines):
            break

        msgstr, index = collect_field(lines, index)

        while index < len(lines) and lines[index].strip():
            index += 1

        entries.append(PoEntry(start, index, flags, msgid, msgstr))

    return entries


def replace_msgstr(lines: list[str], entry: PoEntry, translation: str) -> None:
    """Replace one entry's ``msgstr`` in ``lines``, in place."""
    msgstr_index = entry.start
    while msgstr_index < entry.end and not lines[msgstr_index].startswith("msgstr "):
        msgstr_index += 1

    msgstr_end = msgstr_index + 1
    while msgstr_end < len(lines) and lines[msgstr_end].startswith('"'):
        msgstr_end += 1

    encoded = encode_po_string(translation)
    lines[msgstr_index:msgstr_end] = ["msgstr " + encoded[0], *encoded[1:]]


def clean_fuzzy_flag(lines: list[str], entry: PoEntry) -> None:
    """Remove the ``"fuzzy"`` marker from a message, modifying ``lines`` in place."""
    for index in range(entry.start, entry.end):
        if not lines[index].startswith("#,"):
            continue
        flags = [flag.strip() for flag in lines[index][2:].split(",")]
        flags = [flag for flag in flags if flag != "fuzzy"]
        if flags:
            lines[index] = "#, " + ", ".join(flags)
        else:
            lines.pop(index)
        return


def should_translate(entry: PoEntry) -> bool:
    """Return True if the entry has English content and is empty or fuzzy."""
    return bool(entry.msgid.strip()) and (
        not entry.msgstr.strip() or "fuzzy" in entry.flags
    )


def build_user_message(
    target: PoEntry,
    previous: PoEntry | None,
    next_: PoEntry | None,
    source_file: str | None = None,
) -> str:
    """Build the chat-completions user message for one translation call.

    Wraps the focal entry in labeled blocks and prepends the neighboring
    entries (when present) so the model can disambiguate identifiers from
    prose and reuse established terminology.

    Args:
        target: The entry being translated.
        previous: The entry immediately before ``target``, or None at the
            start of the file or when the neighbor has an empty msgid (the
            gettext metadata header).
        next_: The entry immediately after ``target``, or None at the end.
        source_file: Optional ``.rst`` path the entries came from (e.g.,
            ``"api_reference/configurations.rst"``).

    Returns:
        A multi-line string suitable for the ``content`` field of a user
        message.
    """
    parts: list[str] = []

    if source_file:
        parts.append(f"Source file: {source_file}")
        parts.append("")

    if previous is not None and previous.msgid.strip():
        parts.append("Previous entry (context only, do not translate):")
        parts.append(f"  msgid: {json.dumps(previous.msgid, ensure_ascii=False)}")
        if previous.msgstr.strip():
            parts.append(f"  msgstr: {json.dumps(previous.msgstr, ensure_ascii=False)}")
        parts.append("")

    parts.append("Translate this entry:")
    parts.append(f"  msgid: {json.dumps(target.msgid, ensure_ascii=False)}")

    if next_ is not None and next_.msgid.strip():
        parts.append("")
        parts.append("Next entry (context only, do not translate):")
        parts.append(f"  msgid: {json.dumps(next_.msgid, ensure_ascii=False)}")
        if next_.msgstr.strip():
            parts.append(f"  msgstr: {json.dumps(next_.msgstr, ensure_ascii=False)}")

    return "\n".join(parts)


def clean_translation_response(text: str) -> str:
    """Strip incidental formatting the model occasionally adds to the output.

    Trims whitespace, removes a leading ``msgstr:`` / ``Translation:`` label
    if present, and unwraps a single pair of surrounding double quotes.

    Args:
        text: Raw ``message.content`` string returned by the endpoint.

    Returns:
        The Chinese translation only, ready to be written into ``msgstr``.
    """
    cleaned = text.strip()

    for prefix in ("msgstr:", "msgstr =", "translation:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
            break

    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        try:
            cleaned = json.loads(cleaned)
        except json.JSONDecodeError:
            cleaned = cleaned[1:-1]

    return cleaned


def endpoint_url() -> str:
    """Build the chat-completions URL from ``TRANSLATION_API_BASE_URL``."""
    base_url = os.environ["TRANSLATION_API_BASE_URL"].rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def translate_text(user_message: str) -> str:
    """Send a pre-built user message to the endpoint and return the cleaned translation.

    Raises:
        RuntimeError: If the response has no message content.
    """
    payload = {
        "model": os.environ["TRANSLATION_MODEL"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint_url(),
        data=body,
        headers={
            "Authorization": "Bearer " + os.environ["TRANSLATION_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(http_request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Translation endpoint returned no message content") from exc
    return clean_translation_response(raw)


def validate_environment() -> None:
    """Exit with an error if any required API env var is unset.

    Raises:
        SystemExit: If ``TRANSLATION_API_BASE_URL``, ``TRANSLATION_API_KEY``,
            or ``TRANSLATION_MODEL`` is missing.
    """
    missing = [
        name
        for name in (
            "TRANSLATION_API_BASE_URL",
            "TRANSLATION_API_KEY",
            "TRANSLATION_MODEL",
        )
        if not os.environ.get(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required translation secret(s): {joined}")


def update_file(path: Path, remaining_budget: int) -> int:
    """Translate up to ``remaining_budget`` empty/fuzzy messages in one ``.po`` file.

    For each entry needing translation, sends the focal msgid along with its
    immediate neighbors as contextual hints to the model.

    Returns:
        Number of messages actually translated.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = parse_entries(lines)
    translated = 0

    try:
        source_file = str(path.relative_to(LOCALE_DIR).with_suffix(".rst"))
    except ValueError:
        source_file = None

    for i in reversed(range(len(entries))):
        if translated >= remaining_budget:
            break
        entry = entries[i]
        if not should_translate(entry):
            continue

        previous = entries[i - 1] if i > 0 else None
        next_ = entries[i + 1] if i + 1 < len(entries) else None

        user_message = build_user_message(entry, previous, next_, source_file)
        translation = translate_text(user_message)

        replace_msgstr(lines, entry, translation)
        if "fuzzy" in entry.flags:
            clean_fuzzy_flag(lines, entry)
        translated += 1

    if translated:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return translated


def main() -> int:
    """Translate up to ``MAX_ENTRIES`` messages across all Chinese ``.po`` files."""
    po_files = sorted(LOCALE_DIR.glob("**/*.po"))
    if not po_files:
        print("No Chinese PO files found.")
        return 0

    needs_translation = False
    for path in po_files:
        entries = parse_entries(path.read_text(encoding="utf-8").splitlines())
        if any(should_translate(entry) for entry in entries):
            needs_translation = True
            break

    if not needs_translation:
        print("No missing or fuzzy Chinese translations found.")
        return 0

    validate_environment()

    translated = 0
    for path in po_files:
        if translated >= MAX_ENTRIES:
            break
        translated += update_file(path, MAX_ENTRIES - translated)

    print(f"Translated {translated} Chinese documentation entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
