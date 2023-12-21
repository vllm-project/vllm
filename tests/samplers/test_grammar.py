import pytest
import random

from transformers import AutoTokenizer

from vllm.grammar import TokenTrie, NextTokenValidator


MODELS = ["codellama/CodeLlama-7b-hf"]


@pytest.fixture
def json_grammar():
    return r"""
    start: value
    value: dict
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
def csv_grammar():
    return """
    start: header _NL row+
    header: "#" " "? (WORD _SEPARATOR?)+
    row: (_anything _SEPARATOR?)+ _NL
    _anything: INT | WORD | NON_SEPARATOR_STRING | FLOAT | SIGNED_FLOAT
    NON_SEPARATOR_STRING: "/[a-zA-z.;\\\/]+/"
    _SEPARATOR: "\t"
              | ","

    # using these suboptimal common library terminals is a bad practice
    %import common.NEWLINE -> _NL
    %import common.WORD
    %import common.INT
    %import common.FLOAT
    %import common.SIGNED_FLOAT
    """


@pytest.fixture
def csv_example():
    return """
#foo\tbar\tbaz
1\t2\t3
bif\t\bif\tbif
""".strip()


@pytest.mark.parametrize("model_id", MODELS)
def test_next_token_validator_simple(
        model_id,
):
    hello_grammar = """
    ?start: "hello" | "world"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ntv = NextTokenValidator(tokenizer, hello_grammar)

    # tokens specific to codeLlama
    assert ntv.valid_token_str_set == {'wo', 'hell', 'h', 'he', 'hel', 'world', 'wor', 'w', 'hello'}
    assert sorted(ntv.valid_token_id_set) == [107, 122, 354, 827, 3952, 11526, 12199, 13762, 14181, 29882, 29893]


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("grammar_fixture, example_fixture", [
    ("json_grammar", "json_example"),
    ("csv_grammar", "csv_example")
])
def test_can_generate_with_grammar(
        model_id,
        request,
        grammar_fixture,
        example_fixture
):
    """Assert that example file is legal to generate with GrammarLogitsProcessor"""
    grammar = request.getfixturevalue(grammar_fixture)
    example = request.getfixturevalue(example_fixture)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    next_token_validator = NextTokenValidator(
        tokenizer,
        grammar,
        legal_chars=set([chr(i) for i in range(256)])
    )
    example_remainder = example
    while example_remainder:
        legal_next_token_strs = list(next_token_validator.valid_token_str_set)
        random.shuffle(legal_next_token_strs)
        for tok in legal_next_token_strs:
            if example_remainder.startswith(tok):
                next_token_validator.step_seq(tok)
                example_remainder = example_remainder[len(tok):]
                break
        else:
            raise Exception(f"Couldn't find token to create legal output given grammar: '{example_remainder}'")

    # EOS should be in the set of next legal tokens
    assert None in next_token_validator.valid_token_str_set


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


def test_assert_fails_for_invalid_examples():
    assert False


def test_integration_with_vllm():
    assert False
