import pytest
import random

from transformers import AutoTokenizer

from vllm.grammar import TokenTrie


MODELS = ["codellama/CodeLlama-7b-hf"]


@pytest.fixture
def json_grammar():
    return """
    start: value
    value: dict
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
    %import common.WS
    %ignore WS
    """


@pytest.fixture
def json_example():
    return """
    {"widget": {
        "debug": "on",
        "window": {
            "title": "Sample Konfabulator Widget",
            "name": "main_window",
            "width": 500,
            "height": 500
        },
        "image": {
            "src": "Images/Sun.png",
            "name": "sun1",
            "hOffset": 250,
            "vOffset": 250,
            "alignment": "center"
        },
        "text": {
            "data": "Click Here",
            "size": 36,
            "style": "bold",
            "name": "text1",
            "hOffset": 250,
            "vOffset": 100,
            "alignment": "center",
            "onMouseUp": "sun1.opacity = (sun1.opacity / 100) * 90;"
        }
    }}""".strip()


@pytest.fixture
def yaml_grammar():
    return """
start		: yaml

yaml		: data
data		: ( scalar | sequence | mapping )

scalar		: ( number | string | date | BOOLEAN | NIL )
sequence	: ( inline_seq| indented_seq )
mapping		: ( inline_map | indented_map )

inline_seq	: "[" data ( "," data )* "]"
indented_seq	: OPTIONAL_TAB "-" data ( "\n" OPTIONAL_TAB "-" data )*
inline_map	: "{" key ":" data ( "," key ":" data )* "}"
indented_map	: TAB key ":" data ( "\n" TAB key ":" data )*

alpha		: LCASE_LETTER | UCASE_LETTER
alphanum	: alpha | DIGIT
string		: "\"" alphanum*  "\"" | alphanum+
key		: scalar
number		: ("+" | "-")? DIGIT+ ("." DIGIT+)?
date		: DIGIT~4 "-" DIGIT~2 "-" DIGIT~2 ( DIGIT~2 ":" DIGIT~2 ":" DIGIT~2 )?

LCASE_LETTER	: "a".."z"
UCASE_LETTER	: "A".."Z"
DIGIT		: "0".."9"
BOOLEAN		: "true" | "false"
NIL		: "~"
SPACE		: " "
OPTIONAL_TAB	: SPACE*
TAB		: SPACE+
""".strip()


@pytest.fixture
def yaml_example():
    return """
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.8"

sphinx:
   configuration: docs/source/conf.py

# If using Sphinx, optionally build your docs in additional formats such as PDF
formats:
   - pdf

# Optionally declare the Python requirements required to build your docs
python:
   install:
   - requirements: docs/requirements-docs.txt
""".strip()


@pytest.mark.parametrize("model_id", MODELS)
def test_next_token_validator_simple(
        model,
):
    hello_grammar = """
    ?start: "hello" | "world"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ntv = NextTokenValidator(tokenizer, hello_grammar)

    assert ntv.valid_token_str_set == {'wo', 'hell', 'h', 'he', 'hel', 'world', 'wor', 'w', 'hello'}
    assert ntv.valid_token_id_set == {265, 809, 107, 2805, 21558, 28727, 13436, 22493, 9471}


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("grammar_fixture, example_fixture", [
    ("json_grammar", "json_example"),
    ("yaml_grammar", "yaml_example")
])
def test_can_generate_with_grammar(
        model_id,
        grammar_fixture,
        example_fixture
):
    """Assert that example json file is legal to generate with GrammarLogitsProcessor"""
    grammar = request.getfixturevalue(grammar_fixture)
    example = request.getfixturevalue(example_fixture)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    next_token_validator = NextTokenValidator(
        tokenizer,
        grammar,
    )
    example_remainder = example
    while exampleo_remainder:
        legal_next_token_strs = list(next_token_validator.valid_token_str_set)
        random.shuffle(legal_next_token_strs)
        for tok in legal_next_token_strs:
            if example_remainder.startswith(tok):
                example_remainder = example_remainder[len(tok):]
        else:
            raise Exception("Couldn't find token to validate legal JSON given JSON grammar")


@pytest.mark.parametrize("model_id", MODELS)
def test_token_trie_sanity(
        model_id
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    toktrie = TokenTrie(tokenizer)

    all_prefixes = toktrie.get_next_level_token_prefixes("")

    # every token should be composable from a single unique char, so they will all be len of 1
    assert all([len(p) == 1 for p in all_prefixes])

    # every token should have one of these prefixes as a start character
    assert all([
        t[0] in all_prefixes
        for t in toktrie.norm_vocab
    ])

    # construct the set of next level prefixes
    all_subprefixes = set()
    for pfx in all_prefixes:
        all_subprefixes |= toktrie.get_next_level_token_prefixes(pfx)

    # these should have varying length because some tokens don't have level-2 prefixes
    assert len(set([len(spfx) for spfx in all_subprefixes])) > 1


def test_assert_ends_with_eos():
    assert False


def test_integration_with_vllm():
    assert False
