// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output grammar construction shared by reasoning and tool parsers.

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

/// Request facts available while an initialized parser builds an output grammar.
pub struct OutputGrammarContext<'a> {
    /// Effective tools available for this generation.
    pub tools: &'a [Tool],
    /// Effective tool-choice policy.
    pub tool_choice: &'a ToolChoice,
}

/// Errors produced while building an output grammar.
#[derive(Debug, Error)]
pub enum OutputGrammarError {
    /// A model-specific structural-tag builder rejected its inputs.
    #[error("failed to build output grammar")]
    Build(#[from] xgrammar_structural_tag::Error),
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
    if !tool_grammar_applies(ctx) {
        return Ok(None);
    }

    let structural_tag = build_structural_tag(
        builder,
        &tool_params(ctx.tools),
        ctx.tool_choice.clone(),
        StructuralTagOptions::default().with_reasoning(false),
    )?;

    Ok(Some(structural_tag.format))
}

fn tool_params(tools: &[Tool]) -> Vec<ToolParam> {
    tools
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: tool.strict,
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
            ctx.tools.iter().any(|tool| tool.strict == Some(true))
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
            &OutputGrammarContext { tools, tool_choice },
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
}
