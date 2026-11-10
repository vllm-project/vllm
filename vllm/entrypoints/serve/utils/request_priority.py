# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache

import pybase64 as base64
import regex as re
from fastapi import Request
from lark import Lark, Token, Tree, UnexpectedInput

# RFC 8941 Dictionary syntax, including unknown members and item parameters.
# Use the existing Lark dependency rather than matching only a substring `u=`.
_DICTIONARY_GRAMMAR = r"""
    start: [member (_ows "," _ows member)*] _ows
    member: TOKEN ("=" value | parameters)
    ?value: item | inner_list
    item: bare_item parameters
    inner_list: "(" _SP* (item _SP+)* item? ")" parameters
    parameters: parameter*
    parameter: ";" _SP* TOKEN ["=" bare_item]
    ?bare_item: DECIMAL | INTEGER | STRING | TOKEN | BINARY | BOOLEAN
    _ows: (_SP | "\t")*
    _SP: " "
    DECIMAL: /-?[0-9]{1,12}\.[0-9]{1,3}/
    INTEGER: /-?[0-9]{1,15}/
    STRING: /"([ !#-\[\]-~]|\\["\\])*"/
    TOKEN: /[A-Za-z*][!#$%&'*+.^_`|~0-9A-Za-z:\/-]*/
    BINARY: /:[A-Za-z0-9+\/=]*:/
    BOOLEAN: /\?[01]/
"""


@lru_cache(maxsize=1)
def _dictionary_parser() -> Lark:
    return Lark(_DICTIONARY_GRAMMAR, parser="lalr", maybe_placeholders=False)


def parse_priority_urgency(value: str) -> int | None:
    """Read an explicit integer urgency from an RFC 9218 Priority field."""
    try:
        dictionary = _dictionary_parser().parse(value.lstrip(" "))
        for node in dictionary.iter_subtrees():
            if node.data in ("member", "parameter") and not re.fullmatch(
                r"[a-z*][a-z0-9_.*-]*", str(node.children[0])
            ):
                return None
        for binary in dictionary.scan_values(
            lambda node: isinstance(node, Token) and node.type == "BINARY"
        ):
            encoded = str(binary)[1:-1]
            base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True)
    except (UnexpectedInput, ValueError):
        return None

    urgency = None
    for member in dictionary.children:
        if not isinstance(member, Tree) or member.children[0] != "u":
            continue
        # RFC 8941: the last occurrence of a Dictionary key wins.
        urgency = None
        item = member.children[1]
        if isinstance(item, Tree) and item.data == "item":
            token = item.children[0]
            if isinstance(token, Token) and token.type == "INTEGER":
                number = int(token)
                if 0 <= number <= 7:
                    urgency = number
    return urgency


def get_header_priority(raw_request: Request) -> int | None:
    """Resolve opt-in ingress headers, not the legacy vendor override."""
    app = raw_request.scope.get("app")
    if app is None:
        return None
    args = getattr(app.state, "args", None)
    enabled = getattr(args, "allow_header_priority", None)
    if enabled is None:
        config = getattr(app.state, "vllm_config", None)
        scheduler = getattr(config, "scheduler_config", None)
        enabled = getattr(scheduler, "policy", None) == "priority"
    if not enabled:
        return None

    priorities = raw_request.headers.getlist("X-Priority")
    if len(priorities) == 1:
        priority = priorities[0].strip(" \t")
        if re.fullmatch(r"[+-]?[0-9]+", priority):
            try:
                number = int(priority)
                if -(1 << 63) <= number < (1 << 63):
                    return number
            except ValueError:
                pass

    fields = raw_request.headers.getlist("Priority")
    if fields:
        urgency = parse_priority_urgency(",".join(fields))
        if urgency is not None:
            return urgency * 10
    return None
