# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Guards the reasoning-parser -> tool-parser handoff for MuseGlimmer.

Regression test for the bug where a reasoning+tool-call turn returned
content=None from the reasoning parser, starving the tool parser (no tool
calls surfaced). The reasoning parser must strip only the reasoning span and
forward the remaining tool-call channels as content.
"""

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

R = MuseGlimmerReasoningParser.__new__(MuseGlimmerReasoningParser)
T = MuseGlimmerToolParser.__new__(MuseGlimmerToolParser)

# --- Case 1: reasoning + tool call (the regression) ---
raw = (
    " to=self<|message|>Let me call the tool.<|eom|>"
    "<|start|>assistant to=weather.get<|message|>"
    '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)
reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(R, raw, None)
assert reasoning == "Let me call the tool.", repr(reasoning)
assert content is not None and "<atem:invoke" in content, repr(content)
out = MuseGlimmerToolParser.extract_tool_calls(T, content, None)
assert out.tools_called and len(out.tool_calls) == 1
assert out.tool_calls[0].function.name == "weather.get"
print("Case 1 (reasoning+tool handoff): PASS")

# --- Case 2: reasoning + user answer ---
raw2 = (
    " to=self<|message|>thinking<|eom|>"
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
)
r2, c2 = MuseGlimmerReasoningParser.extract_reasoning(R, raw2, None)
assert r2 == "thinking", repr(r2)
assert c2 == "The answer is 42.", repr(c2)
o2 = MuseGlimmerToolParser.extract_tool_calls(T, c2, None)
assert not o2.tools_called
print("Case 2 (reasoning+answer): PASS")

# --- Case 3: plain content, no framing ---
raw3 = "Just a direct answer."
r3, c3 = MuseGlimmerReasoningParser.extract_reasoning(R, raw3, None)
assert r3 is None and c3 == "Just a direct answer.", (r3, c3)
print("Case 3 (plain content): PASS")

# --- Case 4: parallel tool calls after reasoning ---
raw4 = (
    " to=self<|message|>need two calls<|eom|>"
    "<|start|>assistant to=math.add<|message|>"
    '<atem:function_calls>\n<atem:invoke name="math.add">\n'
    '<atem:parameter name="a">1</atem:parameter>\n</atem:invoke>\n'
    "</atem:function_calls><|eom|>"
    "<|start|>assistant to=math.mul<|message|>"
    '<atem:function_calls>\n<atem:invoke name="math.mul">\n'
    '<atem:parameter name="a">3</atem:parameter>\n</atem:invoke>\n'
    "</atem:function_calls><|eot|>"
)
r4, c4 = MuseGlimmerReasoningParser.extract_reasoning(R, raw4, None)
assert r4 == "need two calls", repr(r4)
o4 = MuseGlimmerToolParser.extract_tool_calls(T, c4, None)
assert [t.function.name for t in o4.tool_calls] == ["math.add", "math.mul"], (
    o4.tool_calls
)
print("Case 4 (reasoning+parallel calls): PASS")

print("\nALL REASONING-HANDOFF TESTS PASSED")
