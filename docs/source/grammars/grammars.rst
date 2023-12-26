.. contents:: Table of Contents
    :depth: 3


.. _grammars:


Grammars
========

vLLM offers `Lark <https://lark-parser.readthedocs.io/en/stable/>`_ style EBNF grammars via ``vllm.grammar.GrammarLogitsProcessor``.

``GrammarLogitsProcessor`` ensures generated text follows the rules of a grammar. This provides the ability to guarantee your output is syntactically valid JSON, SQL, Python, RegEx, etc.

Sample Code for JSON
---------------------

.. code-block:: python

    json_grammar = r"""
    start: value
    value: WS* object WS*
    object: dict
          | list
          | string
          | signed_number      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : WS* string WS* ":" value

    string : "\"" escaped_string_char* "\""
    escaped_string_char: _STR_INNER_CHAR | _ESCAPED_CHAR
    _ESCAPED_CHAR: "\\" _ESCAPABLE_CHAR
    _STR_INNER_CHAR: /[^\\\"]/
    _ESCAPABLE_CHAR: /[\\\/bfnrtu]/

    signed_number: ["+"|"-"] number
    number: float | int
    float: int exp | decimal exp?
    decimal: int "." int? | "." int
    exp: ("e"|"E") signed_int
    signed_int: ["+"|"-"] int
    int: DIGIT+
    DIGIT: "0".."9"

    WS: /[ \t\f\r\n]/
    """
    grammar_logits_processor = GrammarLogitsProcessor(
        tokenizer,
        json_grammar,
        grammar_start="value"
    )
    SamplingParams(logits_processor=grammar_logits_processor)


Performance
-----------

For the provided JSON grammar in the subsection below, constrained to only keyboard characters, on the authors mid-end laptop using codeLlama-7b's vocabulary, generation occurred at the following rates:

- first 10 tokens: 3.47 tokens / second
- first 100 tokens: 8.61 tokens / second
- first 1,000 tokens: 14.41 tokens / second
- first 10,000 tokens: 23.80 tokens / second

There is a "warmup" period where token legality is cached based on parser state. The first generation and first tokens within that generation are the slowest.

**Design your EBNF grammar with minimal regexp**

Regexp processing is the most expensive task for GrammarLogitsProcessor. When designing your EBNF, it's better to keep your regexp short and simple if at all possible.

Breaking down the following expressions ESCAPE_STRING into an expression with many faster-terminating regex resulted in a dramatic speedup:

.. code-block::

    start: value
    ?value: dict
          | list
          | string
          | signed_number      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null
python parser test case
    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : string ":" value

    string : "\"" escaped_string_char* "\""
    escaped_string_char: STR_INNER_CHAR | ESCAPED_CHAR
    ESCAPED_CHAR: "\\" ANY_CHAR
    STR_INNER_CHAR: /[^\\\"]/
    ANY_CHAR: /[.]/

    signed_number: ["+"|"-"] number
    number: float | int
    float: int exp | decimal exp?
    decimal: int "." int? | "." int
    exp: ("e"|"E") signed_int
    signed_int: ["+"|"-"] int
    int: DIGIT+
    DIGIT: "0".."9"

    WS: /[ \t\f\r\n]/
    %ignore WS

    # old slow regex-based expressions:

    # %import common.ESCAPED_STRING
    # %import common.SIGNED_NUMBER
    # %import common.WS

**Constrain legal characters**

Every legal character in the alphabet must be checked against the parser by default. Mistral tokenizer, for example, has an alphabet of 3,298 characters, here are 40 random examples:

.. code-block::

    [ '堂', 'ู', 'ɔ', '🙌', 'Б', '레', '允', 'ả', '\ue934', '如', '試', 'K', '¯', '卷', '園', 'ए', '\\', '酒', 'थ', 'グ', '터', '연', 'Ș', 'ブ', '星', 'ြ', 'å', '軍', '案', '题', '银', '映', '표', '\x11', '級', '醒', 'ေ', '✭', '約', '😤']

Likely many of these characters aren't useful in your generation.

Expect increased performance if you constrain your generation to UTF-8, eliminating 3,042 unnecessary characters.

.. code-block::

    GrammarLogitsProcessor(
        tokenizer,
        grammar,
        legal_chars=set(map(chr, range(256))),,
    )

Example 2: constrain the grammar to the set of keyboard typeable characters:

.. code-block::

    def keyboard_chars():
        keyboard_chars = ""
        keyboard_chars += "abcdefghijklmnopqrstuvwxyz"
        keyboard_chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        keyboard_chars += "0123456789"
        keyboard_chars += "`~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/? "
        keyboard_chars += "\t\n"
        return keyboard_chars
    GrammarLogitsProcessor(
        tokenizer,
        grammar,
        legal_chars=set(keyboard_chars()),
    )


Resources
---------

- `How to write an EBNF grammar for Lark <https://lark-parser.readthedocs.io/en/latest/grammar.html>`_
- `Wikipedia - EBNF <https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form>`_
- `Wikipedia - LALR Parser <https://en.wikipedia.org/wiki/LALR_parser>`_

Example Lark Grammars
---------------------

Note: These grammars should

- `JSON <https://lark-parser.readthedocs.io/en/latest/examples/advanced/_json_parser.html>`_
- `Python3 <https://github.com/python-poetry/poetry-core/blob/main/src/poetry/core/_vendor/lark/grammars/python.lark>`_
- `Resource with many grammars including SQLite, TOML, YAML, Lua, and more <https://github.com/ligurio/lark-grammars>`_
