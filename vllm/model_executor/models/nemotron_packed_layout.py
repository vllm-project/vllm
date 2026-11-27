# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout of a packed read of NemotronForDecision: the questions of one
request in one bidirectional forward.

A packed prompt contains the shared template header and state tokens, then
what is left of each question's prompt. The shared tokens attend to
themselves. Each question attends to the shared tokens and to itself, never to
another question, at the positions it has in its own prompt.

A packed read is not the same computation as one read per question: there the
shared tokens also attend to the question."""

from collections.abc import Sequence

import numpy as np
import torch

from vllm.sampling_params import SamplingParams

SEGMENT_LENGTHS = "segment_lengths"
"""`SamplingParams.extra_args` key of a packed read: the number of shared
tokens, then of each question's tokens."""

MAX_KEY_ROWS = 1 << 20
"""Most key rows one attention call gathers, and one packed read in all (see
`key_rows`). With the Ministral 3 default of 8 key-value heads of 128
dimensions in bfloat16, that is 2 GiB of keys and 2 GiB of values."""

ATTENTION_LAYOUT = "decision_attention_layout"
"""HF config attribute that names how the state attends, in the checkpoint's
`config.json` or in `--hf-overrides`: `OWN_QUESTION`, also without the
attribute, or `NO_QUESTIONS`."""

OWN_QUESTION = "state_sees_own_question"
"""Each question is one read of its own prompt, in which the state and the
question attend to each other."""

NO_QUESTIONS = "state_sees_no_questions"
"""A packed read: the questions of a request share one state, which attends
only to itself. Each question attends to the state and to itself."""

ALL_QUESTIONS = "state_sees_all_questions"
"""Named by the checkpoint format and not served yet."""

STATE_END_TOKEN_ID = "state_end_token_id"
"""HF config attribute with the id of the token the prompt writes right after
the state value. The shared tokens of a packed read end with it."""


def attention_layout(hf_config: object) -> str:
    """The layout the server reads with: the checkpoint's, or `OWN_QUESTION`
    for a checkpoint that does not name one."""
    value = getattr(hf_config, ATTENTION_LAYOUT, OWN_QUESTION)
    if value == ALL_QUESTIONS:
        raise ValueError(
            f"NemotronForDecision does not serve {ATTENTION_LAYOUT} {ALL_QUESTIONS} "
            f"yet. Use {OWN_QUESTION} or {NO_QUESTIONS}."
        )
    if value not in (OWN_QUESTION, NO_QUESTIONS):
        raise ValueError(
            f"NemotronForDecision takes {ATTENTION_LAYOUT} {OWN_QUESTION} or "
            f"{NO_QUESTIONS}, not {value!r}."
        )
    return value


def packed_read(hf_config: object) -> bool:
    """Whether the questions of a request share one read: the layout
    `NO_QUESTIONS`."""
    return attention_layout(hf_config) == NO_QUESTIONS


def state_end_token_id(hf_config: object) -> int | None:
    """The id of the token that closes the state in the prompt. None for a
    checkpoint without one, whose prompt is the plain JSON object."""
    value = getattr(hf_config, STATE_END_TOKEN_ID, None)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(
            f"NemotronForDecision takes {STATE_END_TOKEN_ID} as a token id, not "
            f"{value!r} ({type(value).__name__})."
        )
    return value


def check_read_keys(hf_config: object) -> None:
    """Refuse a key that is a current key but for its case or a hyphen. It
    would be ignored, and the server would start with another read, or another
    prompt, than the one asked for. Also refuse a packed read of a prompt
    without a state-end token, which has no place to split."""
    for name in sorted(vars(hf_config)):
        for key in (ATTENTION_LAYOUT, STATE_END_TOKEN_ID):
            if name != key and name.lower().replace("-", "_") == key:
                raise ValueError(
                    f"NemotronForDecision reads {key}, so {name!r} would be ignored. "
                    f"Spell the key {key} in --hf-overrides and in the "
                    "checkpoint's config.json."
                )
    state_end = state_end_token_id(hf_config)
    if packed_read(hf_config) and state_end is None:
        raise ValueError(
            f"NemotronForDecision reads {ATTENTION_LAYOUT} {NO_QUESTIONS} only from "
            f"a checkpoint whose prompt closes the state with a token: its "
            f"config.json has no {STATE_END_TOKEN_ID}."
        )


def key_rows(lengths: Sequence[int]) -> int:
    """The key rows the attention gathers for a packed read: the shared tokens
    for themselves and once more for every question, and each question's own."""
    return lengths[0] * len(lengths) + sum(lengths[1:])


def segment_lengths(
    params: SamplingParams | None, prompt_token_ids: Sequence[int] | None
) -> list[int] | None:
    """The valid segment lengths a request sends, else None."""
    extra_args = params.extra_args if params is not None else None
    lengths = extra_args.get(SEGMENT_LENGTHS) if extra_args else None
    if (
        isinstance(lengths, list)
        and len(lengths) >= 2
        and all(type(n) is int and n > 0 for n in lengths)
        and prompt_token_ids is not None
        and sum(lengths) == len(prompt_token_ids)
        and key_rows(lengths) <= MAX_KEY_ROWS
    ):
        return lengths
    return None


def segment_positions(lengths: Sequence[int]) -> torch.Tensor:
    """The position of every token: each question restarts after the shared
    tokens."""
    shared = lengths[0]
    return torch.cat(
        [torch.arange(shared)] + [torch.arange(shared, shared + n) for n in lengths[1:]]
    )


def packed_sequences(
    query_start_loc: Sequence[int], layouts: Sequence[Sequence[int] | None]
) -> tuple[list[int], list[int], np.ndarray]:
    """The query length, key length and key rows of each attention sequence.

    A request without a layout is one sequence over its rows. A packed read is
    one sequence for its shared tokens and one per question, whose keys are the
    shared rows followed by the question's own. The sequences' queries are the
    step's rows in order."""
    q_lens: list[int] = []
    k_lens: list[int] = []
    kv_rows: list[np.ndarray] = []
    for start, end, lengths in zip(query_start_loc, query_start_loc[1:], layouts):
        if lengths is None:
            q_lens.append(end - start)
            k_lens.append(end - start)
            kv_rows.append(np.arange(start, end))
            continue
        shared = np.arange(start, start + lengths[0])
        q_lens.append(lengths[0])
        k_lens.append(lengths[0])
        kv_rows.append(shared)
        start += lengths[0]
        for n in lengths[1:]:
            q_lens.append(n)
            k_lens.append(lengths[0] + n)
            kv_rows += [shared, np.arange(start, start + n)]
            start += n
    return q_lens, k_lens, np.concatenate(kv_rows)


def sequence_groups(k_lens: Sequence[int], max_key_rows: int) -> list[tuple[int, int]]:
    """Consecutive sequences `[first, last)` for one attention call each, so
    that a call gathers at most `max_key_rows` key rows (or one sequence)."""
    groups: list[tuple[int, int]] = []
    first = rows = 0
    for i, k_len in enumerate(k_lens):
        if i > first and rows + k_len > max_key_rows:
            groups.append((first, i))
            first, rows = i, 0
        rows += k_len
    groups.append((first, len(k_lens)))
    return groups
