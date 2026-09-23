# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
    validate_xgrammar_grammar,
)
from vllm.v1.structured_output.utils import choice_as_grammar

pytestmark = pytest.mark.cpu_test


def grammar_accepts(schema: dict, text: str) -> bool:
    return _is_grammar_accept_string(Grammar.from_json_schema(schema), text)


@pytest.fixture
def unsupported_string_schemas():
    return [
        {"type": "string", "format": "non_existing_format"},
        # TODO(arpera):
        # pattern/format is compiled but length bounds are silently dropped,
        # so the combination must be rejected instead of producing quietly
        # wrong output
        # https://github.com/mlc-ai/xgrammar/issues/749
        {"type": "string", "pattern": "^a+$", "maxLength": 2},
        {"type": "string", "pattern": "^a+$", "minLength": 3},
        {"type": "string", "format": "email", "maxLength": 10},
    ]


@pytest.fixture
def unsupported_integer_schemas():
    return [
        {"type": "integer", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_number_schemas():
    return [
        {"type": "number", "multipleOf": 120},
    ]


@pytest.fixture
def unsupported_array_schemas():
    return [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 5},
    ]


@pytest.fixture
def unsupported_property_names_combinations():
    return [
        # TODO(arpera):
        # this case is not covered by xgrammar, so we should report this bug to xgrammar
        # xgrammar drops propertyNames whenever patternProperties is present,
        # so "grade_12" matches the pattern and escapes the name constraint
        {
            "type": "object",
            "patternProperties": {"^grade_[0-9]+$": {"type": "integer"}},
            "propertyNames": {"pattern": "^grade_[0-9]$"},  # does NOT match "grade_12"
        },
        # propertyNames makes xgrammar discard the sibling additionalProperties
        # value schema: https://github.com/mlc-ai/xgrammar/issues/826
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "additionalProperties": {"type": "integer"},
        },
        # propertyNames is a string schema that conventionally omits "type", so
        # it escapes the string check while xgrammar drops its length bound all
        # the same: https://github.com/mlc-ai/xgrammar/issues/749
        {
            "type": "object",
            "propertyNames": {"pattern": "^a+$", "maxLength": 2},
        },
        # xgrammar emits named `properties` separately from `propertyNames` and
        # only applies `propertyNames` in its additional-properties branch, so a
        # key declared in `properties` escapes the name constraint regardless of
        # what `propertyNames` contains.
        {
            "type": "object",
            "properties": {"Bad": {"type": "integer"}},
            "required": ["Bad"],
            "propertyNames": {"pattern": "^[a-z]+$"},
        },
        {
            "type": "object",
            "properties": {"Bad": {"type": "integer"}},
            "propertyNames": {"enum": ["good"]},
        },
        # xgrammar's propertyNames-only branch falls back to an unconstrained
        # value type, so it drops unevaluatedProperties whether it restricts
        # values (a schema) or forbids them (false).
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "unevaluatedProperties": {"type": "integer"},
        },
        {
            "type": "object",
            "propertyNames": {"pattern": "^[a-z]+$"},
            "unevaluatedProperties": False,
        },
    ]


@pytest.fixture
def unsupported_pattern_properties_combinations():
    return [
        # JSON Schema requires a property matched by both `properties` and
        # `patternProperties` to satisfy both (conjunction), but xgrammar
        # compiles them as alternative branches, so satisfying either one is
        # enough.
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "patternProperties": {"^x$": {"type": "integer"}},
        },
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
    ]


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
def property_names_schema():
    return {"type": "object", "propertyNames": {"pattern": "^[a-z_]+$"}}


@pytest.fixture
def pattern_properties_schema():
    return {
        "type": "object",
        "patternProperties": {"^grade_[0-9]+$": {"type": "integer"}},
    }


class TestHasXGrammarUnsupportedJsonFeatures:
    @pytest.mark.parametrize(
        "schema_type",
        [
            "unsupported_string_schemas",
            "unsupported_integer_schemas",
            "unsupported_number_schemas",
            "unsupported_array_schemas",
            "unsupported_property_names_combinations",
            "unsupported_pattern_properties_combinations",
        ],
    )
    def test_unsupported_json_features_by_type(self, schema_type, request):
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
        ],
    )
    def test_supported_json_features(self, schema_type, request):
        schema = request.getfixturevalue(schema_type)
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
