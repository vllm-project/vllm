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
