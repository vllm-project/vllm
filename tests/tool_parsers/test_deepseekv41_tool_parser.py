# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 output parsing through ParserManager with the real tokenizer.

The V4.1 tokenizer spells each DSML marker as ordinary text around the
``｜DSML｜`` token, so the model can also write any marker with ordinary
tokens. Only markers whose ``｜DSML｜`` arrived as the token are structure.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

pytestmark = pytest.mark.skip_global_cleanup

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
ATOM = "｜DSML｜"
THINK_START_ID = 128821
THINK_END_ID = 128822
CALL = (
    '<｜DSML｜ calls><｜DSML｜ invoke name="calc">'
    '<｜DSML｜ parameter name="value" string="false">42</｜DSML｜ parameter>'
    "</｜DSML｜ invoke></｜DSML｜ calls>"
)
TWIN = CALL.replace(">42<", ">99<")
INVOKE = CALL.removeprefix("<｜DSML｜ calls>").removesuffix("</｜DSML｜ calls>")
STRING_HEAD = (
    '<｜DSML｜ calls><｜DSML｜ invoke name="echo">'
    '<｜DSML｜ parameter name="text" string="true">'
)
STRING_TAIL = "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
CALC = ("calc", {"value": 42})
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": kind} for key, kind in props},
            },
        },
    }
    for name, props in (
        ("calc", [("value", "integer")]),
        ("echo", [("text", "string"), ("note", "string")]),
    )
]
SCHEDULES = ["nonstream", "one_chunk", "per_token", "text_lag3"]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL, tokenizer_mode="deepseek_v41")


def marker_ids(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.convert_tokens_to_ids(ATOM) in ids
    return ids


def text_ids(tokenizer, text):
    """Spell *text* with ordinary tokens only, as the model can."""
    added = set(tokenizer.get_added_vocab().values())
    ids = []
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        if token_id not in added:
            ids.append(token_id)
            continue
        for ch in tokenizer.decode([token_id]):
            ids += tokenizer.encode(ch, add_special_tokens=False)
    assert added.isdisjoint(ids) and tokenizer.decode(ids) == text
    return ids


def make_parser(tokenizer, thinking):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
        tokenizer=tokenizer,
    )
    assert cls is not None
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        include_reasoning=True,
        chat_template_kwargs={"thinking": thinking},
    )
    parser = cls(tokenizer, TOOLS, chat_template_kwargs={"thinking": thinking})
    return parser, request, [THINK_START_ID if thinking else THINK_END_ID]


def schedule_deltas(tokenizer, ids, prompt_ids, schedule):
    if schedule == "one_chunk":
        return [(tokenizer.decode(ids), ids)]
    deltas = []
    if schedule == "per_token":
        tokens, prefix_offset, read_offset = None, 0, 0
        for index in range(len(ids)):
            new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
                tokenizer,
                prompt_ids + ids[: index + 1],
                tokens,
                prefix_offset,
                read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            tokens = new_tokens if tokens is None else tokens + new_tokens
            deltas.append((text, [ids[index]]))
        return deltas
    # Text trails the token IDs by three characters, as while a stop string
    # is held back, so a delta's text can stop inside one of its tokens.
    assert schedule == "text_lag3"
    sent = 0
    for index in range(len(ids)):
        text = tokenizer.decode(ids[: index + 1]).rstrip("�")[:-3]
        deltas.append((text[sent:], [ids[index]]))
        sent = max(sent, len(text))
    deltas.append((tokenizer.decode(ids)[sent:], []))
    return deltas


def stream(parser, request, prompt_ids, deltas):
    reasoning, content = "", ""
    calls: dict[int, list[str]] = {}
    for index, (text, ids) in enumerate(deltas):
        delta = parser.parse_delta(
            text,
            ids,
            request,
            prompt_token_ids=prompt_ids,
            finished=index == len(deltas) - 1,
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            merged = calls.setdefault(call.index, ["", ""])
            merged[0] += call.function.name or ""
            merged[1] += call.function.arguments or ""
    return (
        reasoning,
        content,
        [(name, json.loads(args)) for _, (name, args) in sorted(calls.items())],
    )


def parse(tokenizer, ids, thinking, schedule):
    parser, request, prompt_ids = make_parser(tokenizer, thinking)
    if schedule != "nonstream":
        deltas = schedule_deltas(tokenizer, ids, prompt_ids, schedule)
        return stream(parser, request, prompt_ids, deltas)
    parser.set_prompt_token_ids(prompt_ids)
    reasoning, content, calls = parser.parse(
        tokenizer.decode(ids),
        request,
        enable_auto_tools=True,
        model_output_token_ids=ids,
    )
    return (
        reasoning or "",
        content or "",
        [(call.name, json.loads(call.arguments)) for call in calls or []],
    )


QUOTED = (
    "α🙂 A</｜DSML｜ parameter>"
    '<｜DSML｜ parameter name="note" string="true">99</｜DSML｜ parameter>'
    "</｜DSML｜ invoke>" + CALL + " Ω</think><｜User｜>"
)
CONTROLS = "Пример <｜User｜> </think> " + ATOM + " 🙂 конец"
BEFORE = "До " + CALL + " 🙂 " + ATOM + " "
AFTER = " после " + INVOKE + " " + CALL + " конец"

# (thinking, [(text, genuine)], reasoning, content, calls)
CASES = {
    "text_call_is_content": (
        False,
        [("Before " + CALL + " after", False)],
        "",
        "Before " + CALL + " after",
        [],
    ),
    "text_call_stays_reasoning": (
        True,
        [("Before " + CALL + " after", False)],
        "Before " + CALL + " after",
        "",
        [],
    ),
    "text_controls_are_content": (False, [(CONTROLS, False)], "", CONTROLS, []),
    "text_controls_stay_reasoning": (True, [(CONTROLS, False)], CONTROLS, "", []),
    "text_partial_marker_flushes": (
        True,
        [("До <｜DSML｜ invoke na", False)],
        "До <｜DSML｜ invoke na",
        "",
        [],
    ),
    "genuine_atom_without_marker": (
        False,
        [("<", False), (ATOM, True), (" inv", False)],
        "",
        "<" + ATOM + " inv",
        [],
    ),
    "text_twins_around_call": (
        False,
        [(BEFORE, False), (CALL, True), (AFTER, False)],
        "",
        BEFORE + AFTER,
        [CALC],
    ),
    "text_reasoning_then_end_token": (
        True,
        [("я проверяю </think> " + CALL + " 🙂", False), ("</think>", True)]
        + [("TOTAL=42", False)],
        "я проверяю </think> " + CALL + " 🙂",
        "TOTAL=42",
        [],
    ),
    "text_markup_inside_string_argument": (
        False,
        [(STRING_HEAD, True), (QUOTED, False), (STRING_TAIL, True)],
        "",
        "",
        [("echo", {"text": QUOTED})],
    ),
    "text_call_between_calls": (
        False,
        [("<｜DSML｜ calls>" + INVOKE, True), ("text " + INVOKE + " 🙂 ", False)]
        + [(INVOKE + "</｜DSML｜ calls>", True)],
        "",
        "text " + INVOKE + " 🙂 ",
        [CALC, CALC],
    ),
    # The reasoning-to-tool handoff must keep the IDs of the whole marker.
    "implicit_reasoning_end_then_twin": (
        True,
        [("Plan.", False), (CALL, True), (TWIN, False)],
        "Plan.",
        TWIN,
        [CALC],
    ),
    "explicit_reasoning_end_then_twin": (
        True,
        [("Plan.", False), ("</think>", True), (CALL, True), (TWIN, False)],
        "Plan.",
        TWIN,
        [CALC],
    ),
    "emoji_then_twin": (
        False,
        [("🙂", False), (CALL, True), (TWIN, False)],
        "",
        "🙂" + TWIN,
        [CALC],
    ),
    "delayed_bos_before_call": (
        False,
        [("<｜begin▁of▁sentence｜>" + CALL, True)],
        "",
        "",
        [CALC],
    ),
}


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("case", CASES)
def test_only_dsml_token_markers_are_structure(tokenizer, case, schedule):
    thinking, parts, reasoning, content, calls = CASES[case]
    ids = []
    for text, genuine in parts:
        if text == "</think>" and genuine:
            ids.append(THINK_END_ID)
        else:
            ids += (marker_ids if genuine else text_ids)(tokenizer, text)
    assert parse(tokenizer, ids, thinking, schedule) == (reasoning, content, calls)


def test_split_character_does_not_rebind_marker_to_twin(tokenizer):
    # The first delta ends inside the emoji, so its text arrives later.
    emoji = text_ids(tokenizer, "🙂")
    assert len(emoji) == 2
    ids = emoji + marker_ids(tokenizer, CALL) + text_ids(tokenizer, TWIN)
    parser, request, prompt_ids = make_parser(tokenizer, False)
    deltas = [("", emoji[:1]), (tokenizer.decode(ids), ids[1:])]
    assert stream(parser, request, prompt_ids, deltas) == ("", "🙂" + TWIN, [CALC])


def test_delta_text_inside_token_keeps_alignment(tokenizer):
    # A delta whose text stops inside its only token must not bypass the
    # token-ID scanner, or the reasoning end below is read as text.
    plan = tokenizer.encode("Plan", add_special_tokens=False)
    assert len(plan) == 1
    ids = plan + [THINK_END_ID] + marker_ids(tokenizer, CALL)
    parser, request, prompt_ids = make_parser(tokenizer, True)
    deltas = [("Pl", plan), (tokenizer.decode(ids)[2:], ids[1:])]
    assert stream(parser, request, prompt_ids, deltas) == ("Plan", "", [CALC])


def test_text_between_calls_keeps_order(tokenizer):
    # The second delta starts a call after text held over from the first;
    # the text deferred in the first delta must still come first.
    first = marker_ids(tokenizer, CALL)
    middle = text_ids(tokenizer, "note " + INVOKE)
    last = marker_ids(tokenizer, CALL)
    ids = first + middle + last
    cuts = [len(first) + len(text_ids(tokenizer, "note <")), len(ids) - 3, len(ids)]
    text, deltas, start = tokenizer.decode(ids), [], 0
    for cut in cuts:
        sent = len(tokenizer.decode(ids[:start]))
        deltas.append((text[sent : len(tokenizer.decode(ids[:cut]))], ids[start:cut]))
        start = cut
    parser, request, prompt_ids = make_parser(tokenizer, False)
    assert stream(parser, request, prompt_ids, deltas) == (
        "",
        "note " + INVOKE,
        [CALC, CALC],
    )


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_text_parameter_tag_does_not_extend_streamed_value(tokenizer, schedule):
    ids = (
        marker_ids(tokenizer, STRING_HEAD)
        + text_ids(tokenizer, "abc")
        + marker_ids(tokenizer, "</｜DSML｜ parameter>")
        + text_ids(tokenizer, '<｜DSML｜ parameter name="note" string="true">x y z')
        + marker_ids(tokenizer, "</｜DSML｜ invoke></｜DSML｜ calls>")
    )
    assert parse(tokenizer, ids, False, schedule) == (
        "",
        "",
        [("echo", {"text": "abc"})],
    )


def lagged_deltas(tokenizer, ids, group, lag, *, omit_bos=False):
    deltas, sent = [], 0
    for start in range(0, len(ids), group):
        end = min(start + group, len(ids))
        text = tokenizer.decode(ids[:end]).rstrip("�")
        if omit_bos:
            text = text.removeprefix(tokenizer.bos_token)
        stop = max(sent, len(text) - lag)
        deltas.append((text[sent:stop], ids[start:end]))
        sent = stop
    deltas.append((text[sent:], []))
    return deltas


@pytest.mark.parametrize("group", [1, 3, 1000])
@pytest.mark.parametrize("lag", [0, 3, 17])
@pytest.mark.parametrize("genuine", [False, True])
@pytest.mark.parametrize("name", ["note", "note>more"])
def test_unclosed_parameter_followed_by_header(tokenizer, group, lag, genuine, name):
    header = f'<｜DSML｜ parameter name="{name}" string="true">second'
    ids = (
        marker_ids(tokenizer, STRING_HEAD)
        + text_ids(tokenizer, "a > b")
        + (marker_ids if genuine else text_ids)(tokenizer, header)
        + marker_ids(tokenizer, STRING_TAIL)
    )
    expected = (
        {"text": "a > b", name: "second"} if genuine else {"text": "a > b" + header}
    )
    parser, request, prompt_ids = make_parser(tokenizer, False)
    assert stream(
        parser, request, prompt_ids, lagged_deltas(tokenizer, ids, group, lag)
    ) == ("", "", [("echo", expected)])


@pytest.mark.parametrize("group", [1, 3, 1000])
@pytest.mark.parametrize("lag", [0, 3, 17])
@pytest.mark.parametrize("omit_bos", [False, True])
def test_delayed_drop_keeps_following_call(tokenizer, group, lag, omit_bos):
    ids = [tokenizer.bos_token_id] + marker_ids(tokenizer, CALL)
    parser, request, prompt_ids = make_parser(tokenizer, False)
    assert stream(
        parser,
        request,
        prompt_ids,
        lagged_deltas(tokenizer, ids, group, lag, omit_bos=omit_bos),
    ) == ("", "", [CALC])


@pytest.mark.parametrize("count", [100, 200])
@pytest.mark.parametrize("lag", [3, 10000])
def test_delayed_drop_decode_work_is_linear(tokenizer, monkeypatch, count, lag):
    ids = [tokenizer.bos_token_id] + text_ids(tokenizer, " hello" * count)
    parser, request, prompt_ids = make_parser(tokenizer, False)
    deltas = lagged_deltas(tokenizer, ids, 1, lag)
    decode = tokenizer.decode
    decoded_ids = decode_calls = 0

    def counted_decode(token_ids, *args, **kwargs):
        nonlocal decoded_ids, decode_calls
        decode_calls += 1
        decoded_ids += len(token_ids)
        return decode(token_ids, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", counted_decode)
    actual = stream(parser, request, prompt_ids, deltas)
    assert decoded_ids <= 8 * len(ids), (decode_calls, decoded_ids, len(ids))
    assert actual == ("", " hello" * count, [])


@pytest.mark.parametrize("strict_calling", ["0", "1"])
def test_constrained_tool_choices_follow_adjusted_request(strict_calling):
    # Adapter capabilities are selected at import time from this env var.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import json
                import os
                from itertools import product
                from tests.tool_parsers.test_deepseekv41_tool_parser import (
                    CALL, CALC, SCHEDULES, THINK_END_ID, make_parser,
                    schedule_deltas, stream, text_ids,
                    tokenizer as tokenizer_fixture,
                )

                tokenizer = tokenizer_fixture.__wrapped__()
                strict = os.environ["VLLM_ENFORCE_STRICT_TOOL_CALLING"] == "1"
                failures = []
                named = {"type": "function", "function": {"name": "calc"}}
                for choice in ("required", named):
                    for thinking, schedule in product((False, True), SCHEDULES):
                        parser, request, prompt_ids = make_parser(tokenizer, thinking)
                        request = type(request).model_validate({
                            **request.model_dump(), "tool_choice": choice,
                        })
                        request = parser.adjust_request(request)
                        assert request.structured_outputs is not None
                        if strict:
                            assert request.structured_outputs.structural_tag is not None
                            text = CALL
                        else:
                            assert request.structured_outputs.json is not None
                            text = ('[{"name":"calc","parameters":{"value":42}}]'
                                    if choice == "required" else '{"value":42}')
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        plan = "Quote </think> as text." if thinking else ""
                        if thinking:
                            ids = text_ids(tokenizer, plan) + [THINK_END_ID] + ids
                            text = tokenizer.decode(ids)
                        if schedule == "nonstream":
                            parser.set_prompt_token_ids(prompt_ids)
                            reasoning, content, calls = parser.parse(
                                text, request, True, ids)
                            actual = (reasoning or "", content or "", [
                                (c.name, json.loads(c.arguments)) for c in calls or []
                            ])
                        else:
                            actual = stream(parser, request, prompt_ids,
                                schedule_deltas(tokenizer, ids, prompt_ids, schedule))
                        if actual != (plan, "", [CALC]):
                            failures.append((choice, thinking, schedule, actual))
                assert not failures, failures
                """),
        ],
        env={**os.environ, "VLLM_ENFORCE_STRICT_TOOL_CALLING": strict_calling},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("schedule", ["nonstream", "one_chunk", "per_token"])
def test_unified_parser_records_tool_invocations(tokenizer, schedule):
    from prometheus_client import REGISTRY

    from vllm.parser.metrics import init_parser_metrics

    model = MODEL + "-dsml-guards"
    init_parser_metrics(model_name=model)
    labels = {
        "model_name": model,
        "mode": "non_streaming" if schedule == "nonstream" else "streaming",
        "request_type": "chat_completions",
        "outcome": "tool_call",
    }

    def count(outcome):
        return REGISTRY.get_sample_value(
            "vllm:tool_call_parser_invocations_total", {**labels, "outcome": outcome}
        )

    before = count("tool_call")
    total = before + count("no_tool_call")
    ids = marker_ids(tokenizer, CALL)
    assert parse(tokenizer, ids, False, schedule) == ("", "", [CALC])
    assert count("tool_call") > before
    invocations = len(ids) if schedule == "per_token" else 1
    assert count("tool_call") + count("no_tool_call") == total + invocations


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "role_token", ["<｜Assistant｜>", "<｜User｜>", "<｜System｜>"]
)
def test_task_prompt_does_not_reuse_previous_turn_reasoning_end(
    tokenizer, schedule, role_token
):
    """Task prompts without a think marker cannot inherit an earlier closed block."""
    ids = text_ids(tokenizer, "new reasoning")
    parser, request, _ = make_parser(tokenizer, True)
    prompt_ids = [THINK_END_ID] + tokenizer.encode(
        role_token + "<｜query｜>", add_special_tokens=False
    )
    assert not parser.is_reasoning_end(prompt_ids)
    if schedule == "nonstream":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, _ = parser.parse(tokenizer.decode(ids), request, True, ids)
        actual = (reasoning or "", content or "")
    else:
        deltas = schedule_deltas(tokenizer, ids, prompt_ids, schedule)
        actual = stream(parser, request, prompt_ids, deltas)[:2]
    assert actual == ("new reasoning", "")
    assert parser.count_reasoning_tokens(ids) == len(ids)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_prompt_that_closes_reasoning_starts_in_content(tokenizer, schedule):
    ids = text_ids(tokenizer, "already content")
    parser, request, _ = make_parser(tokenizer, True)
    prompt_ids = [THINK_START_ID, THINK_END_ID]
    if schedule == "nonstream":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, _ = parser.parse(tokenizer.decode(ids), request, True, ids)
        actual = (reasoning or "", content or "")
    else:
        deltas = schedule_deltas(tokenizer, ids, prompt_ids, schedule)
        actual = stream(parser, request, prompt_ids, deltas)[:2]
    assert actual == ("", "already content")
    assert parser.count_reasoning_tokens(ids) == 0


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_tool_choice_none_keeps_call_as_content(tokenizer, schedule):
    ids = text_ids(tokenizer, "Sure. ") + marker_ids(tokenizer, CALL)
    parser, request, prompt_ids = make_parser(tokenizer, False)
    request.tool_choice = "none"
    if schedule == "nonstream":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, calls = parser.parse(
            tokenizer.decode(ids), request, True, ids
        )
        actual = (reasoning or "", content or "", calls or [])
    else:
        deltas = schedule_deltas(tokenizer, ids, prompt_ids, schedule)
        actual = stream(parser, request, prompt_ids, deltas)
    assert actual == ("", "Sure. " + CALL, [])
