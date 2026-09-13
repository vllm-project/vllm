# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EBNF grammar generation for non-strict GLM-4.7 tool calls.

The grammar constrains a full assistant turn: an optional thinking block
(excluding tool-call markup), optional text, and any number of
``<tool_call>`` units whose arguments are shallowly typed from the tool
schemas. Arguments may be omitted, repeated, or emitted in any order.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = ["GlmSpecialTokenConfig", "generate_glm_grammar"]


class _GlmTrieNode:
    """Trie node with Aho-Corasick failure link."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.children: dict[str, _GlmTrieNode] = {}
        self.is_end = False
        self.fail: _GlmTrieNode | None = None


def _glm_build_trie_with_failure_links(
    patterns: list[str],
) -> tuple[_GlmTrieNode, list[_GlmTrieNode]]:
    """Build Trie and compute Aho-Corasick failure links."""
    root = _GlmTrieNode(0)
    all_nodes = [root]
    next_id = 1

    for pattern in patterns:
        node = root
        for char in pattern:
            if char not in node.children:
                new_node = _GlmTrieNode(next_id)
                next_id += 1
                all_nodes.append(new_node)
                node.children[char] = new_node
            node = node.children[char]
        node.is_end = True

    root.fail = root
    queue = deque[_GlmTrieNode]()
    for child in root.children.values():
        child.fail = root
        queue.append(child)
    while queue:
        node = queue.popleft()
        for char, child in node.children.items():
            queue.append(child)
            fail_node = node.fail
            while (
                fail_node is not None
                and fail_node is not root
                and char not in fail_node.children
            ):
                fail_node = fail_node.fail
            if (
                fail_node is not None
                and char in fail_node.children
                and fail_node.children[char] is not child
            ):
                child.fail = fail_node.children[char]
            else:
                child.fail = root
            # A suffix match also completes a forbidden pattern.
            if child.fail.is_end:
                child.is_end = True

    return root, all_nodes


def _glm_get_transition(
    node: _GlmTrieNode, char: str, root: _GlmTrieNode
) -> _GlmTrieNode:
    """Follow Aho-Corasick failure links to the next state."""
    current = node
    while True:
        if char in current.children:
            return current.children[char]
        if current is root:
            return root
        assert current.fail is not None
        current = current.fail


def _glm_escape_char_class(s: str) -> str:
    """Escape special characters for use in EBNF character class [...]."""
    result = []
    for c in s:
        if c in r"\]^-":
            result.append("\\" + c)
        elif c == "\n":
            result.append("\\n")
        elif c == "\t":
            result.append("\\t")
        elif c == "\r":
            result.append("\\r")
        elif ord(c) < 32 or ord(c) > 126:
            result.append(f"\\x{ord(c):02X}")
        else:
            result.append(c)
    return "".join(result)


def _glm_escape_string(c: str) -> str:
    """Escape a character for use in EBNF string literal "..."."""
    if c == '"':
        return '\\"'
    elif c == "\\":
        return "\\\\"
    elif c == "\n":
        return "\\n"
    elif c == "\t":
        return "\\t"
    elif c == "\r":
        return "\\r"
    elif ord(c) < 32 or ord(c) > 126:
        return f"\\x{ord(c):02X}"
    return c


def _glm_any_string_exclude(rule_name: str, negative_strings: list[str]) -> list[str]:
    return list(_glm_cached_string_exclude(rule_name, tuple(negative_strings)))


@lru_cache(maxsize=32)
def _glm_cached_string_exclude(
    rule_name: str, negative_strings: tuple[str, ...]
) -> tuple[str, ...]:
    """Build EBNF that excludes forbidden substrings using Aho-Corasick states."""

    if not negative_strings:
        return (f"{rule_name} ::= [^]*",)
    sorted_strings = sorted(set(s for s in negative_strings if s))
    if not sorted_strings:
        return (f"{rule_name} ::= [^]*",)
    hash_input = "\x00".join(sorted_strings)
    hash_prefix = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    root, all_nodes = _glm_build_trie_with_failure_links(sorted_strings)
    all_pattern_chars: set[str] = set()
    for pattern in sorted_strings:
        all_pattern_chars.update(pattern)

    def state_name(node: _GlmTrieNode) -> str:
        return f"s_{hash_prefix}_{node.id}"

    rules = []
    rules.append(f"{rule_name} ::= {state_name(root)}")

    for node in all_nodes:
        if node.is_end:
            continue

        excluded_chars: list[str] = []
        transitions_by_target: dict[int, list[str]] = {}

        for char in all_pattern_chars:
            target = _glm_get_transition(node, char, root)
            if target.is_end:
                excluded_chars.append(char)
            else:
                if target.id not in transitions_by_target:
                    transitions_by_target[target.id] = []
                transitions_by_target[target.id].append(char)

        alternatives = []

        all_explicit_chars = set(excluded_chars)
        for chars in transitions_by_target.values():
            all_explicit_chars.update(chars)

        if all_explicit_chars:
            escaped = _glm_escape_char_class("".join(sorted(all_explicit_chars)))
            alternatives.append(f"[^{escaped}] {state_name(root)}")
        else:
            alternatives.append(f"[^] {state_name(root)}")

        for target_id in sorted(transitions_by_target.keys()):
            chars = transitions_by_target[target_id]
            target_node = next(n for n in all_nodes if n.id == target_id)
            for char in sorted(chars):
                alternatives.append(
                    f'"{_glm_escape_string(char)}" {state_name(target_node)}'
                )

        alternatives.append('""')

        rules.append(f"{state_name(node)} ::= {' | '.join(alternatives)}")

    return tuple(rules)


_GLM_XML_GRAMMAR_RULES = [
    'basic_string ::= (([\\"] basic_string_1 [\\"]))',
    'basic_string_1 ::= "" | [^"\\\\\\x00-\\x1F] basic_string_1 '
    '| "\\\\" escape basic_string_1',
    'escape ::= ["\\\\//bfnrt] | "u" [A-Fa-f0-9]{4}',
    'basic_integer ::= "-"? ("0" | [1-9] [0-9]*) ".0"?',
    'basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
    'basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"',
    'basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any '
    '( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"',
    "ws ::= [ \\n\\t]*",
    "basic_any ::= basic_number | basic_string | basic_boolean | basic_null "
    "| basic_array | basic_object",
    'basic_boolean ::= "true" | "false"',
    'basic_null ::= "null"',
]

_GLM_TYPE_MAPPING = {
    "string": "text_without_special_tokens",
    "number": "basic_number",
    "integer": "basic_number",
    "boolean": "basic_boolean",
    "null": "basic_null",
    "array": "basic_array",
    "object": "basic_object",
}


def _glm_hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _glm_get_value_rule(prop: dict) -> str:
    if "enum" in prop:
        return _glm_handle_enum(prop)
    if "type" in prop:
        return _glm_handle_type(prop)
    return "text_without_special_tokens"


def _glm_escape_ebnf_string(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("\r", "\\r")
    return s


def _glm_handle_enum(prop: dict) -> str:
    enum_values = prop["enum"]
    prop_type = prop.get("type", "string")

    def format_enum_val(v: Any) -> str:
        if prop_type == "boolean":
            return '"true"' if v else '"false"'
        if prop_type == "string":
            return f'"{_glm_escape_ebnf_string(v)}"'
        return f'"{v}"'

    formatted_values = [format_enum_val(v) for v in enum_values]
    enum_rule = " | ".join(formatted_values)
    return f"({enum_rule})" if len(formatted_values) > 1 else enum_rule


def _glm_handle_type(prop: dict) -> str:
    prop_type = prop["type"]
    if isinstance(prop_type, list):
        type_rules = [
            _GLM_TYPE_MAPPING.get(t, "text_without_special_tokens") for t in prop_type
        ]
        return " | ".join(type_rules) if type_rules else "text_without_special_tokens"
    return _GLM_TYPE_MAPPING.get(prop_type, "text_without_special_tokens")


def _glm_build_tool_call_rules(
    non_terminal_name: str,
    functions: list[tuple[str, dict]],
    special_tokens: GlmSpecialTokenConfig,
) -> list[str]:
    """Build non-strict XML tool-call rules with shallow value constraints."""
    rules = [
        f"{non_terminal_name} ::= ( tool_call_unit )*",
        f'tool_call_unit ::= "{special_tokens.begin_of_tool_call}" '
        f'single_tool_call "{special_tokens.end_of_tool_call}"',
    ]

    # Include the index to distinguish duplicate function names.
    tool_alternatives = " | ".join(
        f"call_{_glm_hash_name(func_name + str(function_index))}"
        for function_index, (func_name, _) in enumerate(functions)
    )
    rules.append(f"single_tool_call ::= {tool_alternatives}")

    kv_template = (
        f'"{special_tokens.begin_of_key}{{key}}{special_tokens.end_of_key}" '
        f'"{special_tokens.begin_of_value}" ({{valrule}}) '
        f'"{special_tokens.end_of_value}"'
    )

    for function_index, (func_name, params) in enumerate(functions):
        namehash = _glm_hash_name(func_name + str(function_index))
        properties = (params or {}).get("properties", {})

        prop_kv_pairs = {
            prop_name: kv_template.format(
                key=prop_name, valrule=_glm_get_value_rule(prop_schema)
            )
            for prop_name, prop_schema in properties.items()
        }

        all_props = list(properties.keys())

        if all_props:
            all_choices = " | ".join(prop_kv_pairs[k] for k in all_props)
            arguments_rule = f"( ( {all_choices} ) ( ( {all_choices} ) )* )?"
        else:
            arguments_rule = '""'

        rules.append(
            f'call_{namehash} ::= "{_glm_escape_ebnf_string(func_name)}" '
            f"( arguments_{namehash} )?"
        )
        rules.append(f"arguments_{namehash} ::= {arguments_rule}")

    rules.extend(_GLM_XML_GRAMMAR_RULES)
    return rules


@dataclass
class GlmSpecialTokenConfig:
    begin_of_thinking: str = "<think>"
    end_of_thinking: str = "</think>"
    begin_of_tool_call: str = "<tool_call>"
    end_of_tool_call: str = "</tool_call>"
    begin_of_key: str = "<arg_key>"
    end_of_key: str = "</arg_key>"
    begin_of_value: str = "<arg_value>"
    end_of_value: str = "</arg_value>"
    assistant_token: str = "<|assistant|>"

    def all_special_tokens(self) -> list[str]:
        return list(vars(self).values())


def generate_glm_grammar(
    enable_thinking: bool,
    functions: list[tuple[str, dict]] | None,
    special_tokens: GlmSpecialTokenConfig | None = None,
    root_name: str = "root",
) -> str:
    """Generate an EBNF grammar for a full GLM-4.7 assistant turn.

    Args:
        enable_thinking: Whether the chat template leaves room for a
            thinking block before ``</think>``.
        functions: ``(name, parameters)`` pairs of the available function
            tools, or ``None`` when the request carries no tools.
        special_tokens: Tag configuration; defaults to GLM-4.7 tags.
        root_name: Name of the root rule.

    Returns:
        The EBNF grammar as a newline-joined rule list.
    """
    st = special_tokens or GlmSpecialTokenConfig()
    ebnf_lines = [
        f"{root_name} ::= assistant_turn",
        "assistant_turn ::= thinking_block text_block tool_call_blocks",
    ]

    thinking_exclusions = [
        st.begin_of_tool_call,
        st.end_of_tool_call,
        st.begin_of_key,
        st.end_of_key,
        st.begin_of_value,
        st.end_of_value,
        st.end_of_thinking,
    ]

    if enable_thinking:
        ebnf_lines.append(
            f'thinking_block ::= thinking_block_content "{st.end_of_thinking}"'
        )
        ebnf_lines.extend(
            _glm_any_string_exclude("thinking_block_content", thinking_exclusions)
        )
    else:
        # The chat template already emitted </think> for non-thinking mode.
        ebnf_lines.append('thinking_block ::= ""')

    ebnf_lines.extend(
        _glm_any_string_exclude("text_without_special_tokens", st.all_special_tokens())
    )

    ebnf_lines.append("text_block ::= ( text_without_special_tokens )?")

    if functions:
        ebnf_lines.extend(
            _glm_build_tool_call_rules(
                non_terminal_name="tool_call_blocks",
                functions=functions,
                special_tokens=st,
            )
        )
    else:
        ebnf_lines.append('tool_call_blocks ::= ""')

    non_terminals: dict[str, str] = {}
    deduped_lines = []
    for line in ebnf_lines:
        assert "\n" not in line, "Each EBNF rule should be in a single line."
        lhs = line.split("::=")[0].strip()
        if lhs in non_terminals:
            if non_terminals[lhs] == line:
                continue
            raise ValueError(f"Duplicate non-terminal found: {lhs}")
        non_terminals[lhs] = line
        deduped_lines.append(line)

    return "\n".join(deduped_lines)
