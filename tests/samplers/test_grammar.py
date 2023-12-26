import pytest
import numpy as np
import json

from transformers import AutoTokenizer

from vllm.grammar import NextTokenValidator, GrammarLogitsProcessor
from vllm import LLM, SamplingParams


@pytest.fixture
def tokenizer():
    model_id = "codellama/CodeLlama-7b-hf"
    return AutoTokenizer.from_pretrained(model_id)


@pytest.fixture
def json_grammar():
    return r"""
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
    """.strip()


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
bif\tbif\tbif
""".strip() + "\n"  # grammar requires newline before eos


def sample_from_logits(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(logits), p=probs)


def test_next_token_validator_simple(tokenizer):
    hello_grammar = """
    ?start: "hello" | "world"
    """
    ntv = NextTokenValidator(tokenizer, hello_grammar)
    valid_next_str_set = set(ntv.get_valid_next_token_strs(""))
    valid_next_id_set = set(ntv.get_valid_next_token_ids(""))

    # tokens specific to codeLlama
    assert valid_next_str_set == {
        'wo', 'hell', 'h', 'he', 'hel', 'world', 'wor', 'w', 'hello'
    }
    assert sorted(valid_next_id_set) == [
        107, 122, 354, 827, 3952, 11526, 12199, 13762, 14181, 29882, 29893
    ]


def test_can_span_multiple_terminals(tokenizer):
    true_segmented_grammar = """
    ?start: "is" "t" "r" "u" "e"
    """
    ntv = NextTokenValidator(tokenizer, true_segmented_grammar)
    assert "true" in set(ntv.get_valid_next_token_strs("is"))


@pytest.mark.parametrize("grammar_fixture, example_fixture",
                         [("json_grammar", "json_example"),
                          ("csv_grammar", "csv_example")])
def test_can_generate_with_grammar(tokenizer, request, grammar_fixture,
                                   example_fixture):
    """Assert that example file is legal to generate with NextTokenValidator"""
    grammar = request.getfixturevalue(grammar_fixture)
    example = request.getfixturevalue(example_fixture)

    next_token_validator = NextTokenValidator(
        tokenizer,
        grammar,
        legal_chars=set(map(chr, range(256))),
    )
    example_remainder = example
    generation = ""
    while example_remainder:
        for tok in next_token_validator.get_valid_next_token_strs(generation):
            if tok is None:
                continue
            if example_remainder.startswith(tok):
                generation += tok
                example_remainder = example_remainder[len(tok):]
                break
        else:
            raise Exception(
                f"Couldn't find token to create legal output given grammar, remaining output: '{example_remainder}'"
            )

    # EOS should be in the set of next legal tokens
    assert None in next_token_validator.get_valid_next_token_strs(generation)


@pytest.mark.parametrize(
    "json_example",
    [
        "{\n    \"emptyObject\": {\n        \"innerEmptyObject\": {}\n    }\n}",  # empty obj
        "{\n    \"mixedArray\": [null, 123, \"text\", true, {\"key\": \"value\"}]\n}",  # mixed array
        "{\n    \"deepArray\": [[[[[\"deep\"]]]]]\n}",  # deeply nested list
        "{\n    \"\": true,\n    \"regularKey\": false\n}",  # empty keys
        "{\n    \"\\u043a\\u043b\\u044e\\u0447\": \"\\u0437\\u043d\\u0430\\u0447\\u0435\\u043d\\u0438\\u0435\",\n    \"emoji\\ud83d\\ude42\": \"value\\ud83d\\ude00\"\n}",  # unicode keys
    ])
def test_json_valid_with_edge_cases(tokenizer, json_grammar, json_example):
    next_token_validator = NextTokenValidator(
        tokenizer,
        json_grammar,
    )
    example_remainder = json_example
    generation = ""
    while example_remainder:
        for tok in next_token_validator.get_valid_next_token_strs(generation):
            if tok is None:
                continue
            if example_remainder.startswith(tok):
                generation += tok
                example_remainder = example_remainder[len(tok):]
                break
        else:
            raise Exception(
                f"Couldn't find token to create legal output given grammar, remaining output: '{example_remainder}'"
            )

    # EOS should be in the set of next legal tokens
    assert None in next_token_validator.get_valid_next_token_strs(generation)


@pytest.mark.parametrize(
    "json_example",
    [
        "{\n    \"key1\": \"value1\",\n    \"key2\": \"value2\",\n}",  # trailing comma
        "{\n    \"key\": \"value\" // This is a comment\n}\n",  # comment
        "{\n    \"number\": 1.2.3\n}",  # incorrect decimal format
        "{\n    \"key\": \"value\"unexpected\"\n}",  # incorrect str format
        "{\n    \"object\": {\"key\": \"value\", }\n}",  # trailing comma
        "{\n    \"array\": [1, 2,, 3]\n}\n",  # double comma
    ])
def test_json_fails_with_edge_cases(tokenizer, json_grammar, json_example):
    next_token_validator = NextTokenValidator(
        tokenizer,
        json_grammar,
    )
    example_remainder = json_example
    generation = ""
    while example_remainder:
        for tok in next_token_validator.get_valid_next_token_strs(generation):
            if tok is None:
                continue
            if example_remainder.startswith(tok):
                generation += tok
                example_remainder = example_remainder[len(tok):]
                break
        else:
            return

    raise Exception(f"Invalid json was accepted: '{json_example}'")


@pytest.mark.parametrize(
    "start_tok, validator",
    [
        ("5", float),  # 5 - float
        ("f", lambda s: bool(json.dumps(s))),  # f for false
        ("t", lambda s: bool(json.dumps(s))),  # t for false
        ('"', lambda s: str(json.dumps(s))),  #  " for string
    ])
def test_gen_primative(json_grammar, tokenizer, start_tok, validator):
    # Note: string may last a
    for _ in range(4):
        grammar_logits_processor = GrammarLogitsProcessor(
            tokenizer,
            json_grammar,
            legal_chars=set(map(chr, range(256))),
        )

        start_token_id = list(grammar_logits_processor.vocab[start_tok])[0]

        token_ids = [start_token_id]
        while True:
            logits = grammar_logits_processor(token_ids=token_ids,
                                              logits=np.random.uniform(
                                                  -10, 10,
                                                  len(tokenizer.vocab)))
            new_token_id = sample_from_logits(logits)
            if new_token_id == tokenizer.eos_token_id:
                break
            token_ids.append(new_token_id)

        validator(tokenizer.decode(token_ids))


def test_random_grammared_generation(json_grammar, tokenizer):
    # Generate JSON token-by-token with random logits until EOS is hit, then validate JSON
    # Bias logits so open syntax such that closing syntax such as ]}",
    # occur more frequently as time goes on so we don't get stuck in generation

    grammar_logits_processor = GrammarLogitsProcessor(
        tokenizer,
        json_grammar,
        legal_chars=set(map(chr, range(256))),
    )

    # bias closing tokens logits to prevent infinite generation
    closing_token_ids = set([
        tok_id for tok_str in ["]", "}", '"', ",", None]
        for tok_id in grammar_logits_processor.vocab[tok_str]
    ])
    closing_tokens_bias = -10

    # without this it mostly generates numbers since numbers represent far
    # more tokens than these, and numbers close much more quickly, are less
    # gramatically complicated and result in a less interesting test
    opening_token_ids = set([
        tok_id for tok_str in ["[", "{", '"', ","]
        for tok_id in grammar_logits_processor.vocab[tok_str]
    ])
    opening_tokens_bias = 5

    token_ids = []
    while True:
        logits = grammar_logits_processor(token_ids=token_ids,
                                          logits=np.random.uniform(
                                              -10, 10, len(tokenizer.vocab)))

        for closing_token_id in closing_token_ids:
            logits[closing_token_id] += closing_tokens_bias
        for opening_token_id in opening_token_ids:
            logits[opening_token_id] += opening_tokens_bias

        new_token_id = sample_from_logits(logits)
        if new_token_id == tokenizer.eos_token_id:
            break
        token_ids.append(new_token_id)
        closing_tokens_bias += 0.2
        opening_tokens_bias -= 0.1


def test_integration_with_vllm(vllm_runner, hf_runner):
    model_id = "facebook/opt-125m"
    dtype = "half"

    tokenizer = hf_runner(model_id, dtype=dtype).tokenizer
    grammar = """?start: "hello" | "world" """

    grammar_logits_processor = GrammarLogitsProcessor(tokenizer, grammar)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.1,
        max_tokens=256,
        logits_processors=[grammar_logits_processor])
    llm = LLM(model=model_id,
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)

    prompts = [
        "Random prompt unrelated to output",
        "Seriously, no matter what the prompt is..."
        "it will always follow the grammar"
    ]

    request_outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(request_outputs) == len(prompts)

    for request_output in llm.generate(prompts,
                                       sampling_params=sampling_params):
        assert len(request_output.outputs) == 1
        assert request_output.outputs[0].text in ("hello", "world")
