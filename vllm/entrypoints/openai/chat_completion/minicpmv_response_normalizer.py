# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.generate.base.protocol import DeltaMessage

# Model types whose tokenizer decodes newlines as literal escape sequences.
# Add new MiniCPM-V generations here as they are validated.
MINICPMV_NORMALIZE_MODEL_TYPES = frozenset({"minicpmv4_6", "minicpmv4_7"})


class EscapedNewlineNormalizer:
    def __init__(self) -> None:
        self._closing_marker: str | None = None
        self._pending = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        text = self._pending + text
        self._pending = ""
        output: list[str] = []
        index = 0

        while index < len(text):
            if self._closing_marker is not None:
                marker = self._closing_marker
                remaining = text[index:]
                if remaining.startswith(marker):
                    output.append(marker)
                    index += len(marker)
                    self._closing_marker = None
                    continue
                if not final and marker.startswith(remaining):
                    self._pending = remaining
                    break
                output.append(text[index])
                index += 1
                continue

            remaining = text[index:]
            if text[index] == "`":
                if not final and remaining in ("`", "``"):
                    self._pending = remaining
                    break
                if remaining.startswith("```"):
                    output.append("```")
                    index += 3
                    self._closing_marker = "```"
                else:
                    output.append("`")
                    index += 1
                    self._closing_marker = "`"
                continue

            if text[index] == "$":
                if not final and remaining == "$":
                    self._pending = remaining
                    break
                if remaining.startswith("$$"):
                    output.append("$$")
                    index += 2
                    self._closing_marker = "$$"
                else:
                    output.append("$")
                    index += 1
                    self._closing_marker = "$"
                continue

            if text[index] != "\\":
                output.append(text[index])
                index += 1
                continue

            slash_end = index + 1
            while slash_end < len(text) and text[slash_end] == "\\":
                slash_end += 1
            if slash_end - index > 1:
                output.append(text[index:slash_end])
                index = slash_end
                continue

            if not final and remaining in ("\\", "\\r", "\\r\\"):
                self._pending = remaining
                break
            if remaining.startswith("\\r\\n"):
                output.append("\n")
                index += 4
            elif remaining.startswith(("\\n", "\\r")):
                output.append("\n")
                index += 2
            elif remaining.startswith("\\("):
                output.append("\\(")
                index += 2
                self._closing_marker = "\\)"
            elif remaining.startswith("\\["):
                output.append("\\[")
                index += 2
                self._closing_marker = "\\]"
            else:
                output.append("\\")
                index += 1

        if final and self._pending:
            output.append(self._pending)
            self._pending = ""
        return "".join(output)


def normalize_response_text(text: str | None) -> str | None:
    if text is None or "\\" not in text:
        return text
    return EscapedNewlineNormalizer().feed(text, final=True)


class MiniCPMVResponseNormalizer:
    def __init__(self) -> None:
        self._reasoning = EscapedNewlineNormalizer()
        self._content = EscapedNewlineNormalizer()

    def normalize_delta(
        self,
        delta: DeltaMessage | None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        reasoning_finished = finished or (
            delta is not None and (delta.content is not None or bool(delta.tool_calls))
        )
        reasoning = self._reasoning.feed(
            delta.reasoning if delta and delta.reasoning else "",
            final=reasoning_finished,
        )
        content = self._content.feed(
            delta.content if delta and delta.content else "",
            final=finished,
        )

        if delta is None:
            if not reasoning and not content:
                return None
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        if reasoning:
            delta.reasoning = reasoning
        elif delta.reasoning is not None:
            delta.reasoning = None
        if content:
            delta.content = content
        elif delta.content is not None:
            delta.content = None
        return delta
