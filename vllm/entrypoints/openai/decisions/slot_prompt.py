# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixed prompt of slot-readout decision models, byte for byte as trained."""

import json
from typing import Any

from .question_types import LABELS

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
#: Two-letter labels that are not one token in the Qwen3.5 vocabulary.
_NOT_ONE_TOKEN = frozenset(
    {
        "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV",
        "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN",
        "JQ",
    }
)  # fmt: skip
#: Slot i labels option i: A to Z, then the two-letter labels in order.
SLOT_LABELS = (
    LABELS
    + tuple(a + b for a in LABELS for b in LABELS if a + b not in _NOT_ONE_TOKEN)[
        : 256 - len(LABELS)
    ]
)
DEFAULT_INSTRUCTIONS = {
    "choice": "Which option applies to the state?",
    "noul": "Is the statement true, or is the answer to the question yes?",
    "score": "Which level describes the state?",
}
NOUL_DEFAULTS = {
    "true": "the statement is true / the answer is yes",
    "false": "the statement is false / the answer is no",
}
#: Placed where a suffix goes to find where the shared prefix ends.
PREFIX_MARK = "QEV_PREFIX_MARK"


def render_state(value: Any, strip: bool = False) -> str:
    """Strings as they are (stripped with ``strip``), others as indented JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip() if strip else value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_suffix(type_name: str, instructions: Any, criteria: Any) -> str:
    """The question, its options under slot labels and the answer line."""
    if type_name == "noul":
        criteria = criteria or {}
        names = ["yes", "no"]
        descriptions = [
            d if criteria.get(k) is None else criteria[k]
            for k, d in NOUL_DEFAULTS.items()
        ]
    elif type_name == "choice":
        names, descriptions = list(criteria), list(criteria.values())
    else:
        names, descriptions = [f"level {i}" for i in range(len(criteria))], criteria
    lines = []
    for label, name, desc in zip(SLOT_LABELS, names, descriptions):
        text = render_state(desc, strip=True)
        lines.append(f"{label}. {name}: {text}" if text else f"{label}. {name}")
    header = "Rate the state:" if type_name == "score" else "Question:"
    question = render_state(instructions, strip=True) or DEFAULT_INSTRUCTIONS[type_name]
    kind = "letter" if len(names) <= len(LABELS) else "label"
    options, labels = "\n".join(lines), ", ".join(SLOT_LABELS[: len(names)])
    return (
        f"{header} {question}\n\nOptions:\n{options}\n\n"
        f"Answer with one {kind}: {labels}."
    )
