# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated spatial answers, separate from next-token label probabilities."""

import json
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.openai.decisions.question_types import StructuredDecisionError

BOX_MAX_TOKENS = 96
BOX_MAX_QUESTIONS = 4
BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "bbox_2d": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        }
    },
    "required": ["bbox_2d"],
    "additionalProperties": False,
}


class BoxOutputError(ValueError):
    """The model did not return a complete, valid box."""


@dataclass(frozen=True)
class BoxQuestion:
    id: str
    target: str
    description: str | None = None
    exclude: str | None = None

    @classmethod
    def parse(cls, qid: str, instructions: Any, criteria: Any) -> "BoxQuestion":
        if not qid.strip():
            raise StructuredDecisionError("box question ids must be non-empty")
        if criteria is not None:
            raise StructuredDecisionError(f"question {qid!r}: box takes no criteria")
        if isinstance(instructions, str):
            fields = {"target": instructions}
        elif isinstance(instructions, dict) and instructions.keys() <= {
            "target",
            "description",
            "exclude",
        }:
            fields = instructions
        else:
            raise StructuredDecisionError(
                f"question {qid!r}: box instructions need a target string "
                "or a target/description/exclude object"
            )
        for key in ("target", "description", "exclude"):
            value = fields.get(key)
            if (key == "target" or value is not None) and (
                not isinstance(value, str) or not value.strip()
            ):
                raise StructuredDecisionError(
                    f"question {qid!r}: {key} must be a non-empty string"
                )
        return cls(qid, **fields)

    def prompt(self, state: str) -> str:
        task = f"Locate the {self.description or self.target} in this image."
        if self.exclude:
            task += f" Exclude the {self.exclude}."
        prompt = (
            task + " Use normalized integer coordinates from 0 to 1000. "
            "x=0 is the left edge and x=1000 is the right edge; "
            "y=0 is the top edge and y=1000 is the bottom edge. "
            "Return only a JSON object with bbox_2d in the order "
            "[x_min, y_min, x_max, y_max], enclosing the entire "
            + self.target
            + ". No explanation or reasoning."
        )
        if state:
            prompt += "\nContext: " + state
        return prompt


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BoxOutputError("duplicate coordinate key")
        value[key] = item
    return value


def parse_box(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError) as exc:
        raise BoxOutputError("invalid coordinate JSON") from exc
    if not isinstance(value, dict) or set(value) != {"bbox_2d"}:
        raise BoxOutputError("expected only bbox_2d")
    coords = value["bbox_2d"]
    if not isinstance(coords, list) or len(coords) != 4:
        raise BoxOutputError("expected four coordinates")
    if any(type(v) is not int or not 0 <= v <= 1000 for v in coords):
        raise BoxOutputError("coordinates must be integers in [0, 1000]")
    left, top, right, bottom = coords
    if left >= right or top >= bottom:
        raise BoxOutputError("box must have positive width and height")
    return {
        "type": "box",
        "box": {
            "left": left / 1000,
            "top": top / 1000,
            "right": right / 1000,
            "bottom": bottom / 1000,
        },
        "confidence": None,
    }
