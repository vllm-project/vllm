# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit test the MuseGlimmer ATEM tool parser against realistic model output."""

import json

# Import parser class directly (avoid full vllm import cost where possible)
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

P = MuseGlimmerToolParser.__new__(
    MuseGlimmerToolParser
)  # skip __init__ (needs tokenizer)


def show(title, out):
    print(f"\n### {title}")
    print("  tools_called:", out.tools_called)
    for tc in out.tool_calls:
        print(f"    - {tc.function.name}({tc.function.arguments})")
    if out.content:
        print("  content:", repr(out.content[:80]))


# --- Case 1: single tool call (to=<tool>) ---
o1 = (
    "to=self<|message|>Let me check the weather.<|eom|>"
    "<|start|>assistant to=weather.get<|message|>"
    '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    '<atem:parameter name="units">celsius</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eot|>"
)
out1 = MuseGlimmerToolParser.extract_tool_calls(P, o1, None)
show("single call + reasoning", out1)
assert out1.tools_called and len(out1.tool_calls) == 1
assert out1.tool_calls[0].function.name == "weather.get"
a1 = json.loads(out1.tool_calls[0].function.arguments)
assert a1 == {"city": "Paris", "units": "celsius"}, a1

# --- Case 2: parallel calls (multiple invokes, <|eom|> separated) ---
o2 = (
    "<|start|>assistant to=math.add<|message|>"
    '<atem:function_calls>\n<atem:invoke name="math.add">\n'
    '<atem:parameter name="a">1</atem:parameter>\n'
    '<atem:parameter name="b">2</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eom|>"
    "<|start|>assistant to=math.mul<|message|>"
    '<atem:function_calls>\n<atem:invoke name="math.mul">\n'
    '<atem:parameter name="a">3</atem:parameter>\n'
    '<atem:parameter name="b">4</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eot|>"
)
out2 = MuseGlimmerToolParser.extract_tool_calls(P, o2, None)
show("parallel calls", out2)
assert out2.tools_called and len(out2.tool_calls) == 2, len(out2.tool_calls)
assert [t.function.name for t in out2.tool_calls] == ["math.add", "math.mul"]
# JSON-typed values decode to ints
assert json.loads(out2.tool_calls[0].function.arguments) == {"a": 1, "b": 2}

# --- Case 3: echoed invoke inside reasoning must NOT be parsed (channel scope) ---
o3 = (
    'to=self<|message|>I could call <atem:invoke name="evil.fn">'
    '<atem:parameter name="x">1</atem:parameter></atem:invoke> but I will not.<|eom|>'
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
)
out3 = MuseGlimmerToolParser.extract_tool_calls(P, o3, None)
show("echoed-invoke-in-reasoning (must be 0 calls)", out3)
assert not out3.tools_called, "channel scoping failed — echoed invoke parsed!"
assert out3.content == "The answer is 42.", repr(out3.content)

# --- Case 4: plain answer, no tools ---
o4 = "to=user<|message|>Just a plain answer.<|eot|>"
out4 = MuseGlimmerToolParser.extract_tool_calls(P, o4, None)
show("plain answer", out4)
assert not out4.tools_called

# --- Case 5: object/array JSON param value ---
o5 = (
    "<|start|>assistant to=api.call<|message|>"
    '<atem:function_calls>\n<atem:invoke name="api.call">\n'
    '<atem:parameter name="payload">{"nested": [1, 2, 3]}</atem:parameter>\n'
    '<atem:parameter name="flag">true</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eot|>"
)
out5 = MuseGlimmerToolParser.extract_tool_calls(P, o5, None)
show("json object/array/bool params", out5)
a5 = json.loads(out5.tool_calls[0].function.arguments)
assert a5 == {"payload": {"nested": [1, 2, 3]}, "flag": True}, a5

print("\nALL PARSER TESTS PASSED")
