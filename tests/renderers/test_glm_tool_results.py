# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small synthetic regressions; full model-template benchmarks live externally."""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.renderers.hf import safe_apply_chat_template
from vllm.transformers_utils.glm_tool_results import prepare_glm_tool_results
from vllm.utils.async_utils import make_async

# Deliberately synthetic: the stock sentinel makes accidental optimization of
# explicit templates/continuations observable. This is not a model golden file.
_TEMPLATE = """{%- macro render_tool_response(m) -%}{{ m.content }}{%- endmacro -%}
{%- for m in messages -%}
{%- if m.role == 'tool' -%}
{%- if loop.first or messages[loop.index0 - 1].role != 'tool' -%}
{%- set block_start = loop.index0 -%}
{%- set ns_blk = namespace(end=block_start) -%}
{%- for j in range(block_start, messages|length) -%}
{%- if messages[j].role == 'tool' -%}
{%- set ns_blk.end = j -%}
{%- else -%}{%- break -%}{%- endif -%}
{%- endfor -%}
    {%- set ns_a = namespace(tool_calls=none) -%}
{{- '<stock>' -}}
{% endif -%}
{%- elif m.role == 'system' -%}
{{- m.content -}}
{%- else -%}
{{- m.content or '' -}}
{%- endif -%}
{%- endfor -%}"""


def _messages(calls=("a", "b"), results=("b", "a")):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": key, "function": {"name": "lookup", "arguments": {}}}
                for key in calls
            ],
        },
        *[
            {"role": "tool", "tool_call_id": key, "content": str(key).upper()}
            for key in results
        ],
    ]


def _prepare(messages, template=_TEMPLATE, **kwargs):
    options = {"architectures": ["GlmMoeDsaForCausalLM"], **kwargs}
    return prepare_glm_tool_results(messages, template, **options)


@pytest.mark.parametrize(
    "calls,results,expected",
    [
        (("a", "b"), ("b", "a"), ("a", "b")),
        (("A", "a"), ("a", "A"), ("A", "a")),
        (("a", "b", "c"), ("c", "a"), ("a", "c")),
        (("a", "b"), ("a", "a"), ("a", "a")),
        (("a", "b"), ("unknown", "a"), ("unknown", "a")),
        (("a", "b"), (None, "a"), (None, "a")),
        (("a", "a", "b"), ("b", "a"), ("b", "a")),
        ((None, "a"), ("a",), ("a",)),
    ],
)
def test_ordering_and_all_or_nothing_fallback(calls, results, expected):
    messages = _messages(calls, results)
    before = copy.deepcopy(messages)
    ordered, template = _prepare(messages)
    assert tuple(m["tool_call_id"] for m in ordered[1:]) == expected
    assert "<stock>" not in template
    assert messages == before


def test_nested_outputs_expand_without_losing_following_messages():
    messages = _messages()
    messages[1:] = [
        {
            "role": "tool",
            "content": [
                {"tool_call_id": "b", "output": None},
                {"tool_call_id": "a", "output": "A", "type": "tool_reference"},
            ],
        },
        {"role": "user", "content": "tail"},
    ]
    before = copy.deepcopy(messages)
    ordered, _ = _prepare(messages)
    assert ordered[1:] == [
        {"role": "tool", "content": [{"output": "A"}]},
        {"role": "tool", "content": [{"output": None}]},
        {"role": "user", "content": "tail"},
    ]
    assert messages == before


@pytest.mark.parametrize(
    "options",
    [
        {"template_override": True},
        {"continue_final_message": True},
        {"architectures": ["LlamaForCausalLM"]},
        {"architectures": None},
    ],
)
def test_inapplicable_requests_keep_both_original_inputs(options):
    messages = _messages()
    ordered, template = _prepare(messages, **options)
    assert ordered is messages
    assert template is _TEMPLATE


@pytest.mark.parametrize(
    "template",
    [
        None,
        "unrelated",
        _TEMPLATE.replace("namespace(tool_calls=none)", "namespace(other=none)"),
        _TEMPLATE + _TEMPLATE,
    ],
)
def test_unrecognized_template_is_not_partially_applied(template):
    messages = _messages()
    ordered, selected = _prepare(messages, template)
    assert ordered is messages
    assert selected is template


@pytest.mark.parametrize(
    "architecture", ["GlmMoeDsaForCausalLM", "Glm5NextForConditionalGeneration"]
)
def test_template_text_edits_do_not_require_version_hashes(architecture):
    template = _TEMPLATE.replace("<stock>", "<changed>")
    ordered, selected = _prepare(_messages(), template, architectures=[architecture])
    assert [m["tool_call_id"] for m in ordered[1:]] == ["a", "b"]
    assert "<changed>" not in selected


@pytest.mark.parametrize("via_executor", [False, True])
@pytest.mark.parametrize("tokenize", [False, True])
@pytest.mark.parametrize("mode", ["default", "explicit", "continue"])
def test_shared_renderer_keeps_preparation_and_template_paired(
    via_executor, tokenize, mode
):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(vocab={char: i for i, char in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = TokenizersBackend(tokenizer_object=backend, chat_template=_TEMPLATE)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    )
    messages = _messages()
    continuing = mode == "continue"
    if continuing:
        messages.append({"role": "assistant", "content": "P"})
    before = copy.deepcopy(messages)
    expected = "AB" if mode == "default" else "<stock>" + ("P" if continuing else "")
    if tokenize:
        expected = tokenizer.encode(expected, add_special_tokens=False)
    kwargs = dict(
        tools=[],
        tokenize=tokenize,
        chat_template=_TEMPLATE if mode == "explicit" else None,
        continue_final_message=continuing,
        add_generation_prompt=not continuing,
    )
    if via_executor:
        with ThreadPoolExecutor(max_workers=2) as pool:
            async_call = make_async(safe_apply_chat_template, executor=pool)

            async def render():
                return await async_call(config, tokenizer, messages, **kwargs)

            actual = asyncio.run(render())
    else:
        actual = safe_apply_chat_template(config, tokenizer, messages, **kwargs)
    assert actual == expected
    assert messages == before
