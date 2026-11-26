# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixed packing of Drex v1.5 records, mirroring `code/kev/model.py`.

A row is the shared state followed by one question branch:

    <state> ... <q> instr <opt> option </opt> ... <decide>

The five delimiters are reused Qwen control tokens, exactly as the checkpoint
was trained; they are never generated, so no embedding rows are added.
"""

import re
from collections.abc import Sequence
from typing import Any, Protocol

#: Delimiters in `code/kev` SPECIAL order: state, q, opt, /opt, decide.
DELIMITERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)

#: Training limits from `code/kev`; serving passes its context window instead.
MAX_STATE = 384
MAX_BRANCH = 1024
MAX_PACKED = 2048

#: Caller text may contain `<|name|>`, which the tokenizer would turn into a
#: real control token. Rewrite it to the `¦` form so option boundaries cannot
#: be forged, exactly as `code/kev` does.
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Tokenizer(Protocol):
    def __call__(self, text: str, add_special_tokens: bool = False) -> Any: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def user_tokens(tokenizer: Tokenizer, text: str) -> list[int]:
    """Tokenize caller text so it can never emit a delimiter token."""
    rewritten = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tokenizer(rewritten, add_special_tokens=False).input_ids


def render(value: Any, indent: int = 0) -> str:
    """Flatten a JSON value into the text the reference model reads.

    Strings pass through unchanged; mappings become `key: value` lines and
    arrays become `- item` lines, indented by depth. `None` is empty.
    """
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, description: str | None) -> str:
    """Render one option as `name` or `name: description`."""
    if not description:
        return name
    return f"{name}: {description}"


def model_options(
    question_type: str, options: Sequence[tuple[str, str | None]]
) -> tuple[list[str], list[int]]:
    """Map a question's options to the texts the model reads and their order.

    `options` is in the question type's own order. The returned `order` maps
    each model position back to that order, so the read can be reordered to the
    caller's order before answering.

    Args:
        question_type: One of `choice`, `noul`, `score`.
        options: `(name, description)` pairs in the question type's order.

    Returns:
        The option texts in model order and the model-to-question permutation.

    Raises:
        ValueError: If the question type is not one Drex answers.

    """
    if question_type == "choice":
        return [option_text(name, desc) for name, desc in options], list(
            range(len(options))
        )
    if question_type == "noul":
        if len(options) != 2:
            raise ValueError("a noul question needs a true and a false description")
        # The type stores (yes, no); the model reads (no, yes) like the API.
        yes, no = options
        return [option_text(*no), option_text(*yes)], [1, 0]
    if question_type == "score":
        return [name for name, _ in options], list(range(len(options)))
    raise ValueError(f"Drex does not answer {question_type!r} questions")


def apply_order(probs: Sequence[float], order: Sequence[int]) -> list[float]:
    """Return ``probs`` in the caller's option order.

    ``model_options`` returns probabilities in the order the model reads them
    and the position of each one in the caller's order, so a noul question's
    ``(no, yes)`` read becomes ``(yes, no)`` here.
    """
    if len(probs) != len(order):
        raise ValueError(
            f"got {len(probs)} options but {len(order)} positions to place them"
        )
    return [probs[i] for i in order]


def pack(
    tokenizer: Tokenizer,
    state: str,
    instr: str,
    options: Sequence[str],
    *,
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    max_packed: int = MAX_PACKED,
) -> list[int]:
    """Pack one question as the causal row `[state][branch]`.

    The state is truncated to `max_state` tokens (its delimiter included), as
    the reference does for serving. A branch that does not fit raises, because
    silently dropping options would change the answer.
    """
    state_ids = [tokenizer.convert_tokens_to_ids(DELIMITERS[0])]
    state_ids += user_tokens(tokenizer, state)[: max_state - 1]

    q_id, opt_id, end_id, decide_id = (
        tokenizer.convert_tokens_to_ids(token) for token in DELIMITERS[1:]
    )
    branch = [q_id] + user_tokens(tokenizer, instr)
    for option in options:
        branch += [opt_id] + user_tokens(tokenizer, option) + [end_id]
    branch.append(decide_id)

    if len(branch) > max_branch - len(state_ids):
        raise ValueError(
            f"question branch has {len(branch)} tokens; the row limit leaves "
            f"{max_branch - len(state_ids)} after a {len(state_ids)}-token state"
        )
    if len(state_ids) + len(branch) > max_packed:
        raise ValueError(
            f"packed row has {len(state_ids) + len(branch)} tokens; "
            f"limit is {max_packed}"
        )
    return state_ids + branch


def find_positions(ids: Sequence[int], tokenizer: Tokenizer) -> tuple[list[int], int]:
    """Return each `</opt>` token's position and the `<decide>` position."""
    end_id = tokenizer.convert_tokens_to_ids(DELIMITERS[3])
    decide_id = tokenizer.convert_tokens_to_ids(DELIMITERS[4])
    opt_positions = [i for i, token_id in enumerate(ids) if token_id == end_id]
    return opt_positions, ids.index(decide_id)
