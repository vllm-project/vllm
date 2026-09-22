// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output grammar construction shared by reasoning and tool parsers.

use std::{fmt, str::FromStr};

use serde_with::{DeserializeFromStr, SerializeDisplay};
use thiserror::Error;
use xgrammar_structural_tag::builders::{StructuralTagBuilder, StructuralTagOptions};
use xgrammar_structural_tag::format::Format;
use xgrammar_structural_tag::tool::ToolChoiceValue;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
};

use crate::tool::Tool;

/// Result alias for output grammar construction.
pub type Result<T> = std::result::Result<T, OutputGrammarError>;

/// How much of the generated stream an output grammar covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarCoverage {
    /// The grammar covers reasoning and advances from the first generated token.
    FromTokenZero,
    /// The grammar covers only output emitted after reasoning ends.
    FinalOutputOnly,
}

/// One parser-built output grammar and its generated-stream coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltOutputGrammar {
    /// Typed XGrammar structural-tag format.
    pub format: Format,
    /// Portion of the generated stream covered by `format`.
    pub coverage: GrammarCoverage,
}

impl BuiltOutputGrammar {
    /// A grammar that covers reasoning and everything after it.
    pub fn from_token_zero(format: Format) -> Self {
        Self {
            format,
            coverage: GrammarCoverage::FromTokenZero,
        }
    }

    /// A grammar that covers only the output emitted after reasoning ends.
    pub fn final_output_only(format: Format) -> Self {
        Self {
            format,
            coverage: GrammarCoverage::FinalOutputOnly,
        }
    }
}

/// Server-side floor for tool-call structural tags (`--tool-strict-level`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    DeserializeFromStr,
    SerializeDisplay,
)]
pub enum ToolStrictLevel {
    /// Derive tool constraints from the request's tool choice and per-tool strictness.
    #[default]
    Auto,
    /// Constrain the tool-call envelope for every request with tools.
    Function,
    /// Additionally pin argument schemas, as if every tool were `strict: true`.
    Parameter,
}

impl ToolStrictLevel {
    pub const AUTO_LITERAL: &str = "auto";
    pub const FUNCTION_LITERAL: &str = "function";
    pub const PARAMETER_LITERAL: &str = "parameter";
}

impl FromStr for ToolStrictLevel {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::FUNCTION_LITERAL) {
            Ok(Self::Function)
        } else if value.eq_ignore_ascii_case(Self::PARAMETER_LITERAL) {
            Ok(Self::Parameter)
        } else {
            Err(format!(
                "unknown tool strict level {value:?}; expected one of {}, {}, {}",
                Self::AUTO_LITERAL,
                Self::FUNCTION_LITERAL,
                Self::PARAMETER_LITERAL
            ))
        }
    }
}

impl fmt::Display for ToolStrictLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => Self::AUTO_LITERAL,
            Self::Function => Self::FUNCTION_LITERAL,
            Self::Parameter => Self::PARAMETER_LITERAL,
        })
    }
}

/// Request facts available while an initialized parser builds an output grammar.
pub struct OutputGrammarContext<'a> {
    /// Effective tools available for this generation.
    pub tools: &'a [Tool],
    /// Effective tool-choice policy.
    pub tool_choice: &'a ToolChoice,
    /// Server-side floor for tool-call constraints.
    pub tool_strict_level: ToolStrictLevel,
    /// Whether the request permits more than one tool call.
    pub parallel_tool_calls: bool,
}

/// Errors produced while building an output grammar.
#[derive(Debug, Error)]
pub enum OutputGrammarError {
    /// A model-specific structural-tag builder rejected its inputs.
    #[error("failed to build output grammar")]
    Build(#[from] xgrammar_structural_tag::Error),
    /// A model-specific parser could not project a builder format to its
    /// visible-output language.
    #[error("unexpected output grammar shape from `{builder}` builder")]
    UnexpectedBuilderFormat {
        /// Builder whose output did not match the parser's known shape.
        builder: &'static str,
    },
}

/// Build the visible (post-reasoning) language from a crate structural-tag
/// builder, or `None` when the request does not ask for strict tool calling.
///
/// This is the pre-existing strict-tool-calling grammar: the builder runs with
/// `reasoning = false`, so the result describes only what follows the reasoning
/// phase. It backs the default `ToolParser::build_visible_format` and the
/// native unified parsers that have not yet grown their own builder.
pub(crate) fn visible_format_from_builder(
    builder: Option<&dyn StructuralTagBuilder>,
    ctx: &OutputGrammarContext<'_>,
) -> Result<Option<Format>> {
    let Some(builder) = builder else {
        return Ok(None);
    };

    format_from_builder(
        builder,
        ctx,
        StructuralTagOptions::default().with_reasoning(false),
    )
}

fn format_from_builder(
    builder: &dyn StructuralTagBuilder,
    ctx: &OutputGrammarContext<'_>,
    options: StructuralTagOptions,
) -> Result<Option<Format>> {
    if !tool_grammar_applies(ctx) {
        return Ok(None);
    }

    let structural_tag = build_structural_tag(
        builder,
        &tool_params(ctx.tools, ctx.tool_strict_level),
        ctx.tool_choice.clone(),
        options.with_parallel_tool_calls(ctx.parallel_tool_calls),
    )?;

    Ok(Some(structural_tag.format))
}

#[cfg(test)]
pub(crate) fn full_format_from_builder_for_test(
    builder: &dyn StructuralTagBuilder,
    ctx: &OutputGrammarContext<'_>,
    reasoning: xgrammar_structural_tag::builders::ReasoningMode,
) -> Result<Option<Format>> {
    format_from_builder(
        builder,
        ctx,
        StructuralTagOptions::default().with_reasoning(reasoning),
    )
}

fn tool_params(tools: &[Tool], strict_level: ToolStrictLevel) -> Vec<ToolParam> {
    tools
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: Some(
                    strict_level >= ToolStrictLevel::Parameter || tool.strict == Some(true),
                ),
            }))
        })
        .collect()
}

/// Whether the request asks for a tool grammar at all.
fn tool_grammar_applies(ctx: &OutputGrammarContext<'_>) -> bool {
    if ctx.tools.is_empty() {
        return false;
    }

    match ctx.tool_choice {
        ToolChoice::Value(ToolChoiceValue::None) => false,
        ToolChoice::Value(ToolChoiceValue::Auto) => {
            ctx.tool_strict_level >= ToolStrictLevel::Function
                || ctx.tools.iter().any(|tool| tool.strict == Some(true))
        }
        ToolChoice::Value(ToolChoiceValue::Required)
        | ToolChoice::NamedFunction(_)
        | ToolChoice::AllowedTools(_)
        | ToolChoice::FlatAllowedTools(_)
        | ToolChoice::Builtin(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use xgrammar_structural_tag::builders::Qwen3Builder;

    use super::*;

    fn tool(name: &str, strict: Option<bool>) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
            strict,
        }
    }

    fn build(tools: &[Tool], tool_choice: &ToolChoice) -> Option<Format> {
        visible_format_from_builder(
            Some(&Qwen3Builder),
            &OutputGrammarContext {
                tools,
                tool_choice,
                tool_strict_level: ToolStrictLevel::Auto,
                parallel_tool_calls: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn tool_grammar_requires_tools_and_a_constraining_choice() {
        let non_strict = [tool("search", None)];
        assert!(build(&non_strict, &ToolChoice::none()).is_none());
        assert!(build(&non_strict, &ToolChoice::auto()).is_none());
        assert!(build(&non_strict, &ToolChoice::required()).is_some());
        assert!(build(&non_strict, &ToolChoice::function("search")).is_some());

        // `auto` only constrains once at least one tool is strict.
        let strict = [tool("search", Some(true)), tool("lookup", None)];
        assert!(build(&strict, &ToolChoice::auto()).is_some());
    }

    fn build_at_level(tool_choice: ToolChoice, tools: Vec<Tool>, level: ToolStrictLevel) -> Format {
        visible_format_from_builder(
            Some(&Qwen3Builder),
            &OutputGrammarContext {
                tools: &tools,
                tool_choice: &tool_choice,
                tool_strict_level: level,
                parallel_tool_calls: true,
            },
        )
        .unwrap()
        .unwrap()
    }

    fn tag_string(format: &Format) -> String {
        serde_json::to_string(format).unwrap()
    }

    #[test]
    fn function_level_constrains_auto_without_strict_tools() {
        let request = build_at_level(
            ToolChoice::auto(),
            vec![tool("search", None)],
            ToolStrictLevel::Function,
        );

        assert!(tag_string(&request).contains("search"));
    }

    #[test]
    fn absent_strict_is_non_strict() {
        let unset = build_at_level(
            ToolChoice::required(),
            vec![tool("search", None)],
            ToolStrictLevel::Auto,
        );
        let non_strict = build_at_level(
            ToolChoice::required(),
            vec![tool("search", Some(false))],
            ToolStrictLevel::Auto,
        );
        let strict = build_at_level(
            ToolChoice::required(),
            vec![tool("search", Some(true))],
            ToolStrictLevel::Auto,
        );

        assert_eq!(tag_string(&unset), tag_string(&non_strict));
        assert_ne!(tag_string(&unset), tag_string(&strict));
    }

    #[test]
    fn strict_tool_does_not_pin_its_neighbour() {
        let mixed = build_at_level(
            ToolChoice::auto(),
            vec![tool("weather", Some(true)), tool("search", None)],
            ToolStrictLevel::Auto,
        );
        let explicit = build_at_level(
            ToolChoice::auto(),
            vec![tool("weather", Some(true)), tool("search", Some(false))],
            ToolStrictLevel::Auto,
        );

        assert_eq!(tag_string(&mixed), tag_string(&explicit));
    }

    #[test]
    fn parameter_level_pins_argument_schemas() {
        let pinned = build_at_level(
            ToolChoice::auto(),
            vec![tool("search", None)],
            ToolStrictLevel::Parameter,
        );
        let strict = build_at_level(
            ToolChoice::auto(),
            vec![tool("search", Some(true))],
            ToolStrictLevel::Auto,
        );
        let envelope_only = build_at_level(
            ToolChoice::auto(),
            vec![tool("search", None)],
            ToolStrictLevel::Function,
        );

        assert_eq!(tag_string(&pinned), tag_string(&strict));
        assert_ne!(tag_string(&pinned), tag_string(&envelope_only));
    }

    #[test]
    fn tool_strict_level_parses_case_insensitively() {
        assert_eq!("AUTO".parse::<ToolStrictLevel>(), Ok(ToolStrictLevel::Auto));
        assert_eq!(
            "PARAMETER".parse::<ToolStrictLevel>(),
            Ok(ToolStrictLevel::Parameter)
        );
        assert_eq!(
            "function".parse::<ToolStrictLevel>(),
            Ok(ToolStrictLevel::Function)
        );
        assert_eq!(ToolStrictLevel::default(), ToolStrictLevel::Auto);
        assert_eq!(
            serde_json::to_value(ToolStrictLevel::Auto).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::from_value::<ToolStrictLevel>(json!("auto")).unwrap(),
            ToolStrictLevel::Auto
        );
        assert!("off".parse::<ToolStrictLevel>().is_err());
        assert!("strict".parse::<ToolStrictLevel>().is_err());
    }
}
