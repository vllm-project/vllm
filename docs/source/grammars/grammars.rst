.. _grammars:

Grammars
========

vLLM offers `Lark <https://lark-parser.readthedocs.io/en/stable/>`_ based EBNF grammars via ``vllm.grammar.GrammarLogitsProcessor``.

``GrammarLogitsProcessor`` ensures generated text follows the rules of a grammar. This provides the ability to guarantee your output is syntactically valid JSON, SQL, Python, RegEx, etc.

Sample Code for JSON
---------------------

.. code-block:: python

    json_grammar = """
    ?value: dict
          | list
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : string ":" value

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    """
    grammar_logits_processor = GrammarLogitsProcessor(
        tokenizer,
        json_grammar,
        grammar_start="value"
    )
    SamplingParams(logits_processor=grammar_logits_processor)

Resources
---------

- `How to write an EBNF grammar for Lark <https://lark-parser.readthedocs.io/en/latest/grammar.html>`_
- `Wikipedia - EBNF <https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form>`_
- `Wikipedia - LALR Parser <https://en.wikipedia.org/wiki/LALR_parser>`_

Example Lark Grammars
---------------------

- `JSON <https://lark-parser.readthedocs.io/en/latest/examples/advanced/_json_parser.html>`_
- `Python3 <https://github.com/python-poetry/poetry-core/blob/main/src/poetry/core/_vendor/lark/grammars/python.lark>`_
- `Resource with many grammars including SQLite, TOML, YAML, Lua, and more <https://github.com/ligurio/lark-grammars>`_

Performance
-----------

For a simple JSON grammar, on the authors mid-end laptop using codeLlama-7b's vocabulary, generation occurred at ~10 validated logit sets per second. However performance was improved dramatically from the baseline with a few tweaks to ~400/s. These tweaks include
- Optimizing the grammar increased performance by ~10x
- Constraining legal characters increased performance by ~4x

**Design your EBNF grammar with minimal regexp**

Regexp processing is the most expensive task for GrammarLogitsProcessor. When designing your EBNF, it's better to keep your regexp short and simple if at all possible.

Breaking down the following expressions ESCAPE_STRING into an expression with many faster-terminating regex resulted in an ~8x speedup:

.. code-block::
    start: value
    ?value: dict
          | list
          | string
          | signed_number      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

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

Expect an ~10x speedup if you constrain your generation to UTF-8, eliminating 3,042 unnecessary characters.

.. code-block::

    GrammarLogitsProcessor(
        ...,
        legal_chars=set([chr(i) for i in range(256)])
    )
