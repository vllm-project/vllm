# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import ClassVar

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    has_xgrammar_unsupported_json_features,
    validate_xgrammar_grammar,
)
from vllm.v1.structured_output.utils import choice_as_grammar

pytestmark = pytest.mark.cpu_test


def grammar_accepts(schema: dict, text: str) -> bool:
    return _is_grammar_accept_string(Grammar.from_json_schema(schema), text)


# ================================================
# Unsupported schemas
# ================================================


@pytest.fixture
def unsupported_multipleOf_schemas():
    return [
        {"type": "number", "multipleOf": 120},
        {"Even": {"type": "number", "multipleOf": 2}},
        {"type": "integer", "multipleOf": 120},
        {"Even": {"type": "integer", "multipleOf": 2}},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        # array + some constraints is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/968
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_string_schemas():
    return [
        # ========================================================
        # string + format is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/967
        {"type": "string", "format": "non_existing_format"},
        # ========================================================
        #
        # ========================================================
        # string + format/pattern + length constraint is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/966
        #
        # pattern/format is compiled but length bounds are silently dropped,
        # so the combination must be rejected instead of producing quietly
        # wrong output
        {"type": "string", "pattern": "^a+$", "maxLength": 2},
        {"type": "string", "pattern": "^a+$", "minLength": 3},
        {"type": "string", "format": "email", "maxLength": 10},
        # ========================================================
    ]


@pytest.fixture
def unsupported_propertyNames_combinations():
    return [
        # ========================================================
        # propertyNames + patternProperties is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/959
        #
        # xgrammar drops propertyNames whenever patternProperties is present,
        # so "grade_12" matches the pattern and escapes the name constraint
        {
            "type": "object",
            "propertyNames": {"pattern": "^grade_[0-9]$"},  # does NOT match "grade_12"
            "patternProperties": {"^grade_[0-9]+$": {"type": "integer"}},
        },
        # ========================================================
        #
        # ========================================================
        # propertyNames + maxLength is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/960
        {
            "type": "object",
            "propertyNames": {"pattern": "^a+$", "maxLength": 2},
        },
        # ========================================================
        #
        # ========================================================
        # propertyNames + properties is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/958
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "properties": {"Bad": {"type": "integer"}},
            "required": ["Bad"],
        },
        {
            "type": "object",
            "propertyNames": {"enum": ["good"]},
            "properties": {"Bad": {"type": "integer"}},
        },
        # ========================================================
        #
        # ========================================================
        # propertyNames + unevaluatedProperties is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/961
        #
        # xgrammar's propertyNames-only branch falls back to an unconstrained
        # value type, so it drops unevaluatedProperties whether it restricts
        # values (a schema) or forbids them (false).
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "unevaluatedProperties": False,
        },
        # also keep in mind that unevaluatedProperties can be a schema
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "unevaluatedProperties": {"type": "integer"},
        },
        # ========================================================
    ]


@pytest.fixture
def unsupported_patternProperties_combinations():
    return [
        # ========================================================
        # patternProperties + properties is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/964
        #
        # JSON Schema requires a property matched by both `properties` and
        # `patternProperties` to satisfy both (conjunction), but xgrammar
        # compiles them as alternative branches, so satisfying either one is
        # enough.
        {
            "type": "object",
            "patternProperties": {"^x$": {"type": "integer"}},
            "properties": {"x": {"type": "string"}},
        },
        # ========================================================
        #
        # ========================================================
        # patternProperties + patternProperties is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/965
        #
        # The same alternative-branches problem applies to overlapping
        # patternProperties patterns: a key matching both patterns only has to
        # satisfy one of them.
        {
            "type": "object",
            "patternProperties": {
                "^a[a-z]*$": {"type": "string"},
                "^[a-z]*z$": {"type": "integer"},
            },
        },
        # ========================================================
    ]


@pytest.fixture
def unsupported_multibranch_allof():
    """Regression tests for issue #56556
    [Bug]: JSON schema with multiple allOf branches is silently ignored
    by the xgrammar structured-output backend #56556

    Xgrammar currently does NOT support multi-branch allOf.
    Bug reported to xgrammar team for tracking progress on resolving:
    https://github.com/mlc-ai/xgrammar/issues/937
    """
    return [
        {
            "allOf": [
                {"type": "string"},
                {"enum": ["yes", "no"]},
            ]
        },
        # Multi-branch test case with > 2 branches:
        {
            "allOf": [
                {"type": "string"},
                {"minLength": 2},
                {"maxLength": 10},
            ]
        },
        {
            "allOf": [
                {"type": "integer"},
                {"minimum": 0},
            ]
        },
        # Non root multi-branch allOf:
        {
            "type": "object",
            "properties": {
                "is_this_sparta": {
                    "allOf": [
                        {"type": "string"},
                        {"enum": ["yes", "no"]},
                    ],
                },
            },
        },
    ]


@pytest.fixture
def unsupported_vllm_issue_56556_schema():
    """Bug repro from vLLM issue #56556"""
    return [
        {
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"x": {"type": "integer", "minimum": 10}},
                    "required": ["x"],
                }
            },
            "allOf": [
                {"$ref": "#/$defs/Base"},
                {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                    "required": ["y"],
                },
            ],
        },
    ]


@pytest.fixture
def unsupported_combinator_with_sibling_constraints():
    """A combinator beside constraint keywords on the same node.

    xgrammar silently drops the sibling keywords, so the schema goes
    unenforced: https://github.com/mlc-ai/xgrammar/issues/858
    """
    return [
        # xgrammar#858 repro
        {
            "type": "object",
            "properties": {"modifier": {"enum": ["", "dark"]}},
            "anyOf": [{"required": ["modifier"]}],
            "additionalProperties": False,
        },
        # single-branch allOf wrapping a constraint
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
            "allOf": [{"required": ["a"]}],
        },
        # oneOf beside type, branches do not declare a type
        {
            "type": "object",
            "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
        },
        # non-root: nested in properties
        {
            "type": "object",
            "properties": {
                "x": {"type": "string", "anyOf": [{"minLength": 1}]},
            },
        },
        # NEW: branch type is not within the sibling type, so dropping the
        # sibling `type` changes what xgrammar accepts
        {"type": "string", "anyOf": [{"type": "integer"}]},
    ]


# ================================================
# Supported schemas
# ================================================


@pytest.fixture
def supported_frankenstein_schema():
    # IMPORTANT(arpera):
    # Do NOT add more keywords here! This schema is overcrowded enough that a new
    # keyword can end up having no effect on the compiled grammar while the
    # test still passes. Give a new keyword its own fixture and named test.
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string"},
            "scores": {"type": "array", "items": {"type": "number"}},
            "car_type": {"type": "string", "enum": ["sedan", "suv", "truck"]},
            "car_brand": {"type": "string", "pattern": "^[a-zA-Z]+$"},
            "short_description": {"type": "string", "maxLength": 50},
            "mileage": {"type": "number", "minimum": 0, "maximum": 1000000},
            "model_year": {
                "type": "integer",
                "exclusiveMinimum": 1900,
                "exclusiveMaximum": 2100,
            },
            "long_description": {"type": "string", "minLength": 50, "maxLength": 2000},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
        "minProperties": 1,
        "maxProperties": 100,
    }


@pytest.fixture
def pattern_properties_schema():
    return {
        "type": "object",
        "patternProperties": {"^grade_[0-9]+$": {"type": "integer"}},
    }


@pytest.fixture
def property_names_schema():
    return {"type": "object", "propertyNames": {"pattern": "^[a-z_]+$"}}


@pytest.fixture
def property_names_with_additional_properties_schema():
    """This combination propertyNames + additionalProperties
    was unsupported some time ago by xgrammar.
    So, this test case was in unsupported category before.
    But this xgrammar's PR
    https://github.com/mlc-ai/xgrammar/pull/836
    added support of propertyNames + additionalProperties
    This fix was released in xgrammar v0.2.7
    which vLLM pins in PR #57272
    """
    return {
        "type": "object",
        "propertyNames": {"pattern": "^[a-z_]+$"},
        "additionalProperties": {"type": "integer"},
    }


@pytest.fixture
def supported_allof_anyof_and_oneof():
    """Xgrammar DO support some variants of anyOf, allOf, oneOf
    This test case was made during working on covering
    unsupported multi-branch allOf to make sure the bug
    is NOT present in other combinators.
    See for more context in this file test `unsupported_multibranch_allof`
    """
    return [
        # single-branch allOf:
        {
            "allOf": [
                {"type": "string"},
            ],
        },
        # Corner cases: empty allOf, this is a valid schema
        {"allOf": []},
        # multi-branch allOf:
        # is NOT supported yet, see vLLM issue #56556
        # single-branch anyOf:
        {
            "anyOf": [],
        },
        # multi-branch anyOf:
        {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ],
        },
        # single-branch oneOf:
        {
            "oneOf": [
                {"type": "integer", "minimum": 0, "maximum": 100},
            ],
        },
        # multi-branch oneOf:
        {
            "oneOf": [
                {"type": "integer", "minimum": 0, "maximum": 100},
                {"type": "string", "enum": ["auto", "none"]},
            ],
        },
        # Corner case:
        # "allOf" is a property name, not allOf combinator keyword
        {"type": "object", "properties": {"allOf": {"type": "string"}}},
    ]


@pytest.fixture
def supported_combinator_with_annotations_only():
    """Annotations beside a combinator are safe, so xgrammar keeps these.

    Pydantic emits these shapes for Optional fields, root Unions and
    discriminated unions.
    """
    return [
        {
            "title": "MaybeStr",
            "default": None,
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        {
            "discriminator": {"propertyName": "kind"},
            "oneOf": [
                {"type": "object", "properties": {"kind": {"const": "a"}}},
                {"type": "object", "properties": {"kind": {"const": "b"}}},
            ],
        },
        {
            "$defs": {"A": {"type": "string"}},
            "oneOf": [{"$ref": "#/$defs/A"}],
        },
        # NEW: sibling `type` is redundant because every branch already declares
        # a type within it, and xgrammar enforces these correctly.
        {
            "type": "object",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                    "required": ["b"],
                },
            ],
        },
        # NEW: list-valued sibling `type` covering every branch type
        {
            "type": ["string", "null"],
            "anyOf": [
                {"type": "string", "maxLength": 3},
                {"type": "null"},
            ],
        },
    ]


# ================================================
# Test has_xgrammar_unsupported_json_features functionality
# ================================================


class TestHasXGrammarUnsupportedJsonFeatures:
    @pytest.mark.parametrize(
        "schema_type",
        [
            "unsupported_string_schemas",
            "unsupported_multipleOf_schemas",
            "unsupported_array_schemas",
            "unsupported_propertyNames_combinations",
            "unsupported_patternProperties_combinations",
            # vLLM issue #56556
            "unsupported_multibranch_allof",
            "unsupported_vllm_issue_56556_schema",
            # xgrammar#858: combinator beside constraint keywords
            "unsupported_combinator_with_sibling_constraints",
        ],
    )
    def test_unsupported_json_features(self, schema_type, request):
        schemas = request.getfixturevalue(schema_type)
        for schema in schemas:
            assert has_xgrammar_unsupported_json_features(schema), (
                f"Schema should be unsupported: {schema}"
            )

    @pytest.mark.parametrize(
        "schema_type",
        [
            "supported_frankenstein_schema",
            "pattern_properties_schema",
            "property_names_schema",
            "property_names_with_additional_properties_schema",
            # Additional test cases implemented during work on
            # vLLM issue #56556
            "supported_allof_anyof_and_oneof",
            "supported_combinator_with_annotations_only",
        ],
    )
    def test_supported_json_features(self, schema_type, request):
        schemas = request.getfixturevalue(schema_type)
        if not isinstance(schemas, list):
            schemas = [schemas]
        for schema in schemas:
            assert not has_xgrammar_unsupported_json_features(schema), (
                f"Schema should be supported: {schema}"
            )

    class TestPR48416Regressions:
        @pytest.mark.parametrize(
            "schema",
            [
                {"type": ["number"], "multipleOf": 3},
                {
                    "type": ["string", "null"],
                    "pattern": "^a+$",
                    "maxLength": 2,
                },
                {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "uniqueItems": True,
                },
                {
                    "type": ["object", "null"],
                    "properties": {"Bad": {"type": "integer"}},
                    "propertyNames": {"pattern": "^[a-z]+$"},
                },
            ],
        )
        def test_list_type_does_not_bypass_unsupported_feature_check(self, schema):
            assert has_xgrammar_unsupported_json_features(schema)

        @pytest.mark.parametrize(
            "schema",
            [
                {"type": ["string", "null"]},
                {"type": ["integer", "string"]},
                {"type": ["array", "null"], "items": {"type": "string"}},
                {
                    "type": ["object", "null"],
                    "propertyNames": {"pattern": "^[a-z_]+$"},
                },
                {
                    "type": ["object", "null"],
                    "patternProperties": {"^S": {"type": "string"}},
                },
            ],
        )
        def test_supported_list_type_json_features(self, schema):
            assert not has_xgrammar_unsupported_json_features(schema)


class TestIsGrammarAcceptString:
    class TestPR42904Support:
        def test_property_names_constrains_keys(self, property_names_schema):
            assert grammar_accepts(property_names_schema, '{"score": 5}')
            assert grammar_accepts(property_names_schema, '{"score": "seven"}')
            assert not grammar_accepts(property_names_schema, '{"Score": 5}')
            assert not grammar_accepts(property_names_schema, '{"score_1": 5}')

        def test_property_names_keeps_additional_properties_value_schema(
            self, property_names_with_additional_properties_schema
        ):
            schema = property_names_with_additional_properties_schema
            assert grammar_accepts(schema, '{"score": 5}')
            assert not grammar_accepts(schema, '{"Score": 5}')
            assert not grammar_accepts(schema, '{"score": "seven"}')

        def test_pattern_properties_constrains_keys_and_values(
            self, pattern_properties_schema
        ):
            assert grammar_accepts(pattern_properties_schema, '{"grade_1": 5}')
            assert not grammar_accepts(pattern_properties_schema, '{"other": 5}')
            assert not grammar_accepts(pattern_properties_schema, '{"grade_1": "five"}')

    class TestPR48115Regressions:
        @pytest.mark.parametrize("codepoint", [*range(0x20), 0x7F])
        def test_choice_as_grammar_preserves_control_characters(self, codepoint):
            choice = f"a{chr(codepoint)}f"
            grammar = Grammar.from_ebnf(choice_as_grammar([choice]))

            assert _is_grammar_accept_string(grammar, choice)
            assert not _is_grammar_accept_string(grammar, "a")
            assert not _is_grammar_accept_string(grammar, "af")
            assert not _is_grammar_accept_string(grammar, f"a\\u{codepoint:04x}f")

        @pytest.mark.parametrize(
            ("choice", "other"),
            [
                (r"a\nf", "a\nf"),
                ('a quote " and a backslash \\', 'a quote " and a backslash '),
                ("café", "cafe"),
                ("日本語", "日本"),
                ("😀", "😁"),
            ],
        )
        def test_choice_as_grammar_preserves_literal_choices(self, choice, other):
            grammar = Grammar.from_ebnf(choice_as_grammar(["yes", choice]))

            assert _is_grammar_accept_string(grammar, "yes")
            assert _is_grammar_accept_string(grammar, choice)
            assert not _is_grammar_accept_string(grammar, other)
            assert not _is_grammar_accept_string(grammar, choice + "extra")

        def test_validate_xgrammar_preserves_multiline_choice(self):
            structured_outputs = StructuredOutputsParams(choice=["yes", "no\nplease"])
            validate_xgrammar_grammar(
                SamplingParams(structured_outputs=structured_outputs)
            )

            assert structured_outputs.choice is None
            grammar = Grammar.from_ebnf(structured_outputs.grammar)
            assert _is_grammar_accept_string(grammar, "yes")
            assert _is_grammar_accept_string(grammar, "no\nplease")
            assert not _is_grammar_accept_string(grammar, r"no\nplease")


# ================================================
# Test XgrammarBackend functionality
# ================================================


class TestXgrammarBackend:
    class TestPR58067Regressions:
        """Here we check behavior logic of our flag
        disable_any_whitespace when used with xgrammar backend.
        """

        _TOKENIZER = "openai-community/gpt2"
        _VOCAB_SIZE = 50257
        tokenizer: ClassVar[PreTrainedTokenizerBase]

        @classmethod
        @pytest.fixture(scope="class", autouse=True)
        def _shared_tokenizer(cls):
            cls.tokenizer = AutoTokenizer.from_pretrained(cls._TOKENIZER)

        @classmethod
        def _backend(cls, *, disable_any_whitespace: bool) -> XgrammarBackend:
            vllm_config = VllmConfig(
                structured_outputs_config=StructuredOutputsConfig(
                    backend="xgrammar",
                    disable_any_whitespace=disable_any_whitespace,
                )
            )
            return XgrammarBackend(
                vllm_config, tokenizer=cls.tokenizer, vocab_size=cls._VOCAB_SIZE
            )

        @classmethod
        def _accepts_json(cls, backend: XgrammarBackend, json_text: str) -> bool:
            grammar = backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")
            return grammar.accept_tokens("req", cls.tokenizer.encode(json_text))

        def test_disable_any_whitespace_is_false(self):
            """Verify expected behavior of disable_any_whitespace=False"""
            backend = self._backend(disable_any_whitespace=False)

            # Check that grammar accepts json with NO space after colon
            assert self._accepts_json(backend, '{"no_space_after_me":"yes_ofcourse"}')

            # Check that grammar accepts json with space after colon
            assert self._accepts_json(backend, '{"yes_space_after_me": true}')

            # Check that grammar accepts json with NO space after comma
            assert self._accepts_json(
                backend,
                '{"never_space_after_comma":"accepted","good_boy":":3"}',
            )

            # Check that grammar accepts json with space after comma
            assert self._accepts_json(
                backend,
                '{"where_space_after_comma":42, "wow_thats_cool":4242}',
            )

        def test_disable_any_whitespace_is_true(self):
            """Verify expected behavior of disable_any_whitespace=True"""
            backend = self._backend(disable_any_whitespace=True)

            # Check that grammar accepts json with NO space after colon
            assert self._accepts_json(backend, '{"no_space_after_me":"yes_ofcourse"}')

            # Check that grammar rejects json with space after colon
            assert not self._accepts_json(backend, '{"yes_space_after_me": true}')

            # FIXME(arpera):
            # Comma + separators=(",", ":") is wrong on xgrammar <=0.2.8
            # Seehttps://github.com/mlc-ai/xgrammar/issues/945
            # Re-enable both tests once vLLM pins xgrammar with the fix (>=0.2.9).
            #
            # # Check that grammar accepts json with NO space after comma
            # assert self._accepts_json(
            #     backend,
            #     '{"never_space_after_comma":"accepted","good_boy":":3"}',
            # )
            #
            # # Check that grammar rejects json with space after comma
            # assert not self._accepts_json(
            #     backend,
            #     '{"where_space_after_comma":42, "wow_thats_cool":4242}',
            # )
