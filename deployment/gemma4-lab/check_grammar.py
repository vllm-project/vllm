# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of the same native-tools OR response-schema shape used by vLLM."""

import json
import sys

import xgrammar as xg
from grammar import bound_reply_whitespace, gemma_tools, resolve_tool_token_ids
from xgrammar.structural_tag import JSONSchemaFormat, OrFormat


def obj(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


tools = [
    dict(type="function", function=dict(name=n, strict=True, parameters=obj(p)))
    for n, p in [
        ("hangup_call", {"closure_message": {"type": ["string", "null"]}}),
        (
            "switch_language",
            {
                "language_code": {"enum": ["hi", "en"], "type": "string"},
                "reason": {"type": "string"},
            },
        ),
    ]
]
functions, builtins, choice = xg.normalize_tool_choice(tools, "required")
native = gemma_tools(functions, builtins, choice, False, "")
text = {"type": "string"}
refs = {"type": "array", "items": text}
schema = obj(
    dict(
        response=text, user_language=text, reasoning=text, sources=refs, remove_ids=refs
    )
)
tag = xg.StructuralTag(
    format=OrFormat(elements=[native.format, JSONSchemaFormat(json_schema=schema)])
)
vocab = [bytes([i]) for i in range(256)] + [
    b"<|tool_call>",
    b"<tool_call|>",
    b"[EOS]",
]
compiler = xg.GrammarCompiler(xg.TokenizerInfo(vocab, stop_token_ids=[258]))
composed = tag.model_dump()
assert bound_reply_whitespace(composed)


def preflight(composed, vocab):
    resolved = json.loads(json.dumps(composed))
    resolve_tool_token_ids(resolved["format"], vocab)
    xg.Grammar.from_structural_tag(json.dumps(resolved))
    return xg.StructuralTag.model_validate(resolved)


compiled = compiler.compile_structural_tag(
    preflight(composed, {"<|tool_call>": 256, "<tool_call|>": 257})
)


def accepts(s):
    matcher = xg.GrammarMatcher(compiled)
    return matcher.accept_string(s) and matcher.is_completed()


assert accepts(
    json.dumps(
        dict(
            response="Hello!",
            user_language="en",
            reasoning="Greeting",
            sources=[],
            remove_ids=[],
        )
    )
)
for body in [
    "call:hangup_call{closure_message:null}",
    'call:switch_language{language_code:<|"|>en<|"|>,reason:<|"|>Asked<|"|>}',
]:
    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_token(256) and matcher.accept_string(body)
    assert matcher.accept_token(257) and matcher.is_completed()
    matcher = xg.GrammarMatcher(compiled)
    assert not matcher.accept_token(ord("<"))
assert not accepts('{"response":"Hello!"' + " \n" * 50)
assert accepts(
    json.dumps(
        dict(
            response="Hello:     नमस्ते!",
            user_language="hi",
            reasoning="A   good reason.",
            sources=[],
            remove_ids=[],
        )
    )
)
assert not accepts("<|tool_call>call:invented{}<tool_call|>")
assert not accepts(
    "speech before a tool "
    "<|tool_call>call:hangup_call{closure_message:null}<tool_call|>"
)
assert not accepts('{"response":"missing fields"}')
print("CPU grammar checks passed")

if len(sys.argv) == 2:
    from transformers import TokenizersBackend

    tokenizer = TokenizersBackend(tokenizer_file=sys.argv[1], eos_token="<turn|>")
    ids = tokenizer.get_vocab()
    assert (ids["<|tool_call>"], ids["<tool_call|>"]) == (48, 49)
    cc = xg.GrammarCompiler(xg.TokenizerInfo.from_huggingface(tokenizer))
    actual = cc.compile_structural_tag(preflight(composed, ids))
    m = xg.GrammarMatcher(actual)
    assert m.accept_token(48) and m.accept_string(
        "call:hangup_call{closure_message:null}"
    )
    assert m.accept_token(49) and m.is_completed()
    assert not xg.GrammarMatcher(actual).accept_token(ids["<"])
    print("Pinned checkpoint tokenizer checks passed")
