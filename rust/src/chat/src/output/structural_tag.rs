// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies parser-built xgrammar structural-tag constraints.

use serde::Deserialize;
use serde_json::{Value, json};
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputConstraint, StructuredOutputsParams,
};
use vllm_parser::output_grammar::{BuiltOutputGrammar, GrammarCoverage};
use vllm_text::TextRequest;
use xgrammar_structural_tag::StructuralTag;
use xgrammar_structural_tag::format::{Format, GrammarFormat, TagFormat};

use crate::{Error, Result};

/// Normalize the user's structured-output constraint to the answer format a
/// parser may compose into its output grammar.
///
/// Like Python's `structured_outputs_to_format`, this ignores the request's
/// options: no backend reads them per request, only from the engine config.
///
/// Returns `None` when the constraint must stay on its own wire field, so the
/// request keeps today's behavior: a grammar that is not XGrammar EBNF (such
/// as Lark), or a structural tag that does not parse.
pub(crate) fn answer_format(params: &StructuredOutputsParams) -> Option<Format> {
    Some(match &params.constraint {
        StructuredOutputConstraint::Json(Value::String(schema)) => {
            Format::json_schema(serde_json::from_str(schema).ok()?)
        }
        StructuredOutputConstraint::Json(schema) => Format::json_schema(schema.clone()),
        StructuredOutputConstraint::JsonObject => Format::json_schema(json!({ "type": "object" })),
        StructuredOutputConstraint::Regex(pattern) => Format::regex(pattern),
        StructuredOutputConstraint::Choice(choices) => {
            Format::or(choices.iter().map(Format::const_string).collect())
        }
        // Lark rules use `:`; XGrammar EBNF rules use `::=`.
        StructuredOutputConstraint::Grammar(grammar) if grammar.contains("::=") => {
            Format::Grammar(GrammarFormat {
                grammar: grammar.clone(),
            })
        }
        StructuredOutputConstraint::Grammar(_) => return None,
        StructuredOutputConstraint::StructuralTag(tag) => structural_tag_format(tag)?,
    })
}

/// The legacy structural tag shape: free text with JSON tags behind triggers.
#[derive(Deserialize)]
struct LegacyStructuralTag {
    structures: Vec<LegacyStructure>,
    triggers: Vec<String>,
}

#[derive(Deserialize)]
struct LegacyStructure {
    begin: String,
    schema: Value,
    end: String,
}

/// Read a structural tag in its current (`format`) or legacy
/// (`structures`/`triggers`) shape.
fn structural_tag_format(tag: &str) -> Option<Format> {
    let value: Value = serde_json::from_str(tag).ok()?;
    if value.get("structures").is_none() {
        return Some(serde_json::from_value::<StructuralTag>(value).ok()?.format);
    }
    let legacy: LegacyStructuralTag = serde_json::from_value(value).ok()?;
    let triggers: Vec<&str> = legacy.triggers.iter().map(String::as_str).collect();
    let tags = legacy
        .structures
        .into_iter()
        .map(|structure| {
            TagFormat::new(
                structure.begin,
                Format::json_schema(structure.schema),
                structure.end,
            )
        })
        .collect();
    Some(Format::triggered_tags(&triggers, tags))
}

/// Apply one parser-built output grammar to the prepared text request.
///
/// The parser has already composed the user's answer constraint into the
/// grammar where its protocol allows, so the grammar replaces the request's
/// structured outputs. `None` leaves the request untouched.
pub(crate) fn apply_output_grammar(
    request: &mut TextRequest,
    built: Option<BuiltOutputGrammar>,
) -> Result<()> {
    let Some(built) = built else {
        return Ok(());
    };
    let structural_tag = StructuralTag::new(built.format).to_json_string().map_err(|error| {
        Error::OutputGrammar {
            error: Box::new(error),
        }
    })?;

    // Overwrite any existing structured output settings with the structural tag constraint.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });
    request.reasoning_ended = match built.coverage {
        GrammarCoverage::FromTokenZero => Some(true),
        GrammarCoverage::FinalOutputOnly => None,
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputOptions;

    use super::*;

    #[test]
    fn answer_format_normalizes_composable_constraints() {
        let schema = json!({ "type": "object", "properties": { "a": { "type": "integer" } } });
        let cases = [
            (
                StructuredOutputsParams::json(schema.clone()),
                Format::json_schema(schema.clone()),
            ),
            (
                StructuredOutputsParams::json(Value::String(schema.to_string())),
                Format::json_schema(schema),
            ),
            (
                StructuredOutputsParams::json_object(),
                Format::json_schema(json!({ "type": "object" })),
            ),
            (
                StructuredOutputsParams::regex("[a-z]+"),
                Format::regex("[a-z]+"),
            ),
            (
                StructuredOutputsParams::choice(vec!["yes".to_string(), "no".to_string()]),
                Format::or(vec![
                    Format::const_string("yes"),
                    Format::const_string("no"),
                ]),
            ),
            (
                StructuredOutputsParams::grammar("root ::= \"a\""),
                Format::Grammar(GrammarFormat {
                    grammar: "root ::= \"a\"".to_string(),
                }),
            ),
            (
                StructuredOutputsParams::structural_tag(
                    r#"{"type": "structural_tag", "format": {"type": "const_string", "value": "x"}}"#,
                ),
                Format::const_string("x"),
            ),
            (
                StructuredOutputsParams::structural_tag(
                    r#"{"type": "structural_tag", "structures": [{"begin": "<a>", "schema": {"type": "object"}, "end": "</a>"}], "triggers": ["<a"]}"#,
                ),
                Format::triggered_tags(
                    &["<a"],
                    vec![TagFormat::new(
                        "<a>",
                        Format::json_schema(json!({ "type": "object" })),
                        "</a>",
                    )],
                ),
            ),
            (
                StructuredOutputsParams {
                    options: StructuredOutputOptions {
                        disable_any_whitespace: true,
                        disable_additional_properties: true,
                        whitespace_pattern: Some(" ?".to_string()),
                    },
                    ..StructuredOutputsParams::json_object()
                },
                Format::json_schema(json!({ "type": "object" })),
            ),
        ];
        for (params, expected) in cases {
            assert_eq!(answer_format(&params), Some(expected), "{params:?}");
        }
    }

    #[test]
    fn answer_format_leaves_other_constraints_on_their_wire_field() {
        let cases = [
            StructuredOutputsParams::json(Value::String("{".to_string())),
            StructuredOutputsParams::grammar("start: \"a\""),
            StructuredOutputsParams::structural_tag(r#"{"structures": [{"begin": "<a>"}]}"#),
            StructuredOutputsParams::structural_tag(r#"{"format": {"type": "unknown"}}"#),
        ];
        for params in cases {
            assert_eq!(answer_format(&params), None, "{params:?}");
        }
    }

    #[test]
    fn output_grammar_overwrites_answer_constraint() {
        let mut request = TextRequest::for_test();
        request.sampling_params.structured_outputs =
            Some(StructuredOutputsParams::json(json!({ "type": "object" })));

        let built = BuiltOutputGrammar::from_token_zero(Format::const_string("answer"));
        apply_output_grammar(&mut request, Some(built)).unwrap();

        assert_eq!(request.reasoning_ended, Some(true));
        let params = request.sampling_params.structured_outputs.unwrap();
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        let serialized = params.constraint.as_structural_tag().unwrap();
        let value: serde_json::Value = serde_json::from_str(serialized).unwrap();
        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "const_string");
        assert_eq!(value["format"]["value"], "answer");
    }
}
