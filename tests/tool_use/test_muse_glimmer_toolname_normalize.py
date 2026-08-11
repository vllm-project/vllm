# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MuseGlimmer tool-name normalization: collapse model-synthesized namespaces to the
client-registered tool name (so bare-named tools bind).

MuseGlimmer emits `get_weather.get_weather` for a bare-registered `get_weather`, and
`weather.get` verbatim for a namespaced one. The parser normalizes against the
tools actually registered on the request.
"""

from types import SimpleNamespace

from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

P = MuseGlimmerToolParser.__new__(MuseGlimmerToolParser)


def _req(*names):
    return SimpleNamespace(
        tools=[SimpleNamespace(function=SimpleNamespace(name=n)) for n in names]
    )


def _call(name):
    return (
        f"<|start|>assistant to={name}<|message|>"
        f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
        f'<atem:parameter name="city">Paris</atem:parameter>\n'
        f"</atem:invoke>\n</atem:function_calls>"
    )


def test_doubled_bare_name_collapses():
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("get_weather.get_weather"), _req("get_weather")
    )
    assert out.tools_called and out.tool_calls[0].function.name == "get_weather", (
        out.tool_calls[0].function.name
    )


def test_namespaced_name_preserved():
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("weather.get"), _req("weather.get")
    )
    assert out.tool_calls[0].function.name == "weather.get"


def test_trailing_segment_unambiguous():
    # emitted foo.get_weather; registered get_weather (bare) -> bind to get_weather
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("foo.get_weather"), _req("get_weather")
    )
    assert out.tool_calls[0].function.name == "get_weather"


def test_trailing_segment_ambiguous_left_alone():
    # two registered tools share leaf 'get' -> ambiguous -> do NOT rewrite
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("x.get"), _req("weather.get", "time.get")
    )
    assert out.tool_calls[0].function.name == "x.get"


def test_no_registered_tools_passthrough():
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("get_weather.get_weather"), None
    )
    assert out.tool_calls[0].function.name == "get_weather.get_weather"


def test_exact_match_kept():
    out = MuseGlimmerToolParser.extract_tool_calls(
        P, _call("get_weather"), _req("get_weather")
    )
    assert out.tool_calls[0].function.name == "get_weather"


if __name__ == "__main__":
    for fn in [
        test_doubled_bare_name_collapses,
        test_namespaced_name_preserved,
        test_trailing_segment_unambiguous,
        test_trailing_segment_ambiguous_left_alone,
        test_no_registered_tools_passthrough,
        test_exact_match_kept,
    ]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("\nALL TOOL-NAME NORMALIZATION TESTS PASSED")
