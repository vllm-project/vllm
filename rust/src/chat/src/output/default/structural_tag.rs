// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies xgrammar structural-tag constraints for strict tool calling.

use serde_json::{Value, json};
use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputConstraint, StructuredOutputsParams,
};
use vllm_parser::tool::StructuralTagBuilder;
use vllm_parser::unified::{ScopedCallerConstraint, ScopedStructuralTagBuilder, ScopedToolChoice};
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

use crate::error::{bail_unsupported_structured_outputs, unsupported_structured_outputs};
use crate::parser::ToolStrictLevel;
use crate::request::{ChatRequest, ChatTool, ChatToolChoice};
use crate::{Error, Result as ChatResult};

/// Apply structural tag constraints to the request based on the tool parser's structural tag
/// support, the request's tool choice, and the server-side strictness floor.
///
/// A [`ScopedStructuralTagBuilder`] covers the whole generation and folds the
/// caller's constraint into the answer channel, while a legacy
/// [`StructuralTagBuilder`] constrains tool channels only and, as it always
/// has, overwrites any caller constraint once the tool choice triggers a tag.
pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: Option<&dyn StructuralTagBuilder>,
    scoped_builder: Option<&dyn ScopedStructuralTagBuilder>,
    strict_level: ToolStrictLevel,
) -> ChatResult<()> {
    if let Some(scoped_builder) = scoped_builder {
        return apply_scoped_structural_tag_constraint(request, scoped_builder, strict_level);
    }

    let Some(builder) = builder else {
        return Ok(());
    };
    let Some(tool_choice) = structural_tag_tool_choice(request, strict_level) else {
        return Ok(());
    };

    let tools = request
        .tools()
        .iter()
        .map(|tool| {
            // A tool without an explicit `strict` is non-strict: its call envelope is
            // constrained, but its arguments stay free unless the server level is `parameter`.
            let strict = strict_level >= ToolStrictLevel::Parameter || tool.strict == Some(true);
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: Some(strict),
            }))
        })
        .collect::<Vec<_>>();

    let structural_tag = build_structural_tag(
        builder,
        &tools,
        tool_choice,
        StructuralTagOptions::default().with_reasoning(false),
    )
    .and_then(|tag| tag.to_json_string())
    .map_err(|error| Error::StructuralTag {
        message: error.to_report_string(),
    })?;

    // Overwrite any existing structured output settings with the structural tag constraint.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });

    Ok(())
}

/// Apply a whole-generation structural tag from the parser's scoped builder,
/// folding any caller-provided constraint into the answer channel.
fn apply_scoped_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: &dyn ScopedStructuralTagBuilder,
    strict_level: ToolStrictLevel,
) -> ChatResult<()> {
    let constraint = request
        .sampling_params
        .structured_outputs
        .as_ref()
        .map(|params| &params.constraint);
    // A caller-supplied structural tag is never wrapped in a parser-built one;
    // unlike with the legacy builders, it also wins over the tool choice.
    if constraint.is_some_and(StructuredOutputConstraint::is_structural_tag) {
        return Ok(());
    }

    let tool_choice = scoped_tool_choice(request);
    let forced_tool_choice = matches!(
        tool_choice,
        Some(ScopedToolChoice::Required | ScopedToolChoice::Function(_))
    );
    // A forced tool-call turn has no answer channel for the constraint to
    // constrain; reject the combination instead of silently dropping it.
    if forced_tool_choice && constraint.is_some() {
        bail_unsupported_structured_outputs!(
            "the constraint cannot be combined with tool_choice \"required\" or a named tool choice for this parser"
        );
    }
    // The whole-generation grammar is anchored at a fresh turn's bare first
    // channel header, which only exists when the prompt ends with the
    // generation prompt `<|start|>assistant`. Without one
    // (`continue_final_message`, `add_generation_prompt: false`) the grammar
    // would force a spurious header, so no parser tag is built: a forced tool
    // choice cannot be honored, and a caller constraint is left as sent,
    // applying from the continuation point.
    if !request.chat_options.add_generation_prompt() {
        if forced_tool_choice {
            bail_unsupported_structured_outputs!(
                "tool_choice \"required\" or a named tool choice cannot be enforced without a generation prompt for this parser"
            );
        }
        return Ok(());
    }
    // `required` and named choices always generate tool channels; `auto` does
    // when the server raises the strictness floor or a tool opts in with
    // `strict: true` (the same gating as the legacy path).
    let forces_tool_channels =
        forced_tool_choice || (tool_choice.is_some() && strict_tool_floor(request, strict_level));
    if constraint.is_none() && !forces_tool_channels {
        return Ok(());
    }

    // The Json constraint also carries JSON-encoded schema *strings*
    // (validated upstream); the tag needs the schema value itself.
    let string_schema = match constraint {
        Some(StructuredOutputConstraint::Json(Value::String(raw))) => {
            Some(serde_json::from_str::<Value>(raw).map_err(|error| {
                unsupported_structured_outputs!("invalid JSON schema: {}", error.as_report())
            })?)
        }
        _ => None,
    };
    let generic_object = json!({"type": "object"});
    let caller = match constraint {
        None | Some(StructuredOutputConstraint::StructuralTag(_)) => None,
        Some(StructuredOutputConstraint::Json(Value::String(_))) => {
            string_schema.as_ref().map(ScopedCallerConstraint::JsonSchema)
        }
        Some(StructuredOutputConstraint::Json(schema)) => {
            Some(ScopedCallerConstraint::JsonSchema(schema))
        }
        Some(StructuredOutputConstraint::JsonObject) => {
            Some(ScopedCallerConstraint::JsonSchema(&generic_object))
        }
        Some(StructuredOutputConstraint::Regex(pattern)) => {
            Some(ScopedCallerConstraint::Regex(pattern))
        }
        Some(StructuredOutputConstraint::Choice(choices)) => {
            Some(ScopedCallerConstraint::Choice(choices))
        }
        Some(StructuredOutputConstraint::Grammar(grammar)) => {
            Some(ScopedCallerConstraint::Grammar(grammar))
        }
    };
    // A tool without an explicit `strict` is non-strict: its call envelope is
    // constrained, but its arguments stay free unless the server level is
    // `parameter` (the legacy path's rule).
    let tools: Vec<ChatTool> = request
        .tools()
        .iter()
        .map(|tool| ChatTool {
            strict: Some(strict_level >= ToolStrictLevel::Parameter || tool.strict == Some(true)),
            ..tool.clone()
        })
        .collect();

    let structural_tag = builder
        .build_scoped(
            &tools,
            tool_choice,
            caller,
            &StructuralTagOptions::default().with_reasoning(false),
        )
        .and_then(|tag| tag.to_json_string())
        .map_err(|error| match error {
            // Serialization is the only server-side failure; every other
            // builder error rejects request data the grammar cannot express.
            xgrammar_structural_tag::Error::Serialize(_) => Error::StructuralTag {
                message: error.to_report_string(),
            },
            _ => Error::UnsupportedStructuredOutputs {
                message: error.to_report_string(),
            },
        })?;

    // Overwrite any existing structured output settings with the structural tag
    // constraint. The caller's `StructuredOutputOptions` are not carried over:
    // the v1 engine reads those knobs only from the server-level
    // structured_outputs_config, which the frontend does not see, so the
    // folded JSON schema keeps xgrammar's default whitespace handling.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });
    // The grammar covers the reasoning span itself, so the engine applies it
    // from the first generated token instead of waiting for an engine-side
    // reasoning parser to report a reasoning end (which the legacy Muse
    // Glimmer reasoner never does for a plain answer).
    request.reasoning_ended = Some(true);

    Ok(())
}

/// Whether `auto` tool choice constrains tool calls: the server raised the
/// strictness floor, or at least one tool opted in with `strict: true`.
fn strict_tool_floor(request: &ChatRequest, strict_level: ToolStrictLevel) -> bool {
    strict_level >= ToolStrictLevel::Function
        || request.tools().iter().any(|tool| tool.strict == Some(true))
}

/// Resolve the tool choice used for a whole-generation structural tag based on
/// the request.
///
/// Returns `None` if no tool channels should be generated (no tools, or tool
/// choice `none`).
fn scoped_tool_choice(request: &ChatRequest) -> Option<ScopedToolChoice> {
    if request.tools().is_empty() {
        return None;
    }

    match request.tool_choice() {
        ChatToolChoice::None => None,
        ChatToolChoice::Auto => Some(ScopedToolChoice::Auto),
        ChatToolChoice::Required => Some(ScopedToolChoice::Required),
        ChatToolChoice::Function { name } => Some(ScopedToolChoice::Function(name.clone())),
    }
}

/// Resolve the tool choice used for [`xgrammar_structural_tag`] based on the request.
///
/// Returns `None` if no structural tag constraints should be applied.
fn structural_tag_tool_choice(
    request: &ChatRequest,
    strict_level: ToolStrictLevel,
) -> Option<StructuralTagToolChoice> {
    if request.tools().is_empty() {
        return None;
    }

    match request.tool_choice() {
        ChatToolChoice::None => None,
        ChatToolChoice::Auto => {
            strict_tool_floor(request, strict_level).then(StructuralTagToolChoice::auto)
        }

        ChatToolChoice::Required => Some(StructuralTagToolChoice::required()),
        ChatToolChoice::Function { name } => Some(StructuralTagToolChoice::function(name.clone())),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{Value, json};
    use vllm_engine_core_client::protocol::structured_outputs::{
        StructuredOutputBackend, StructuredOutputsParams,
    };
    use vllm_parser::tool::{Qwen3CoderToolParser, Tool, ToolParser};
    use vllm_parser::unified::{MuseGlimmerUnifiedParser, UnifiedParser};
    use vllm_tokenizer::test_utils::TestTokenizer;
    use xgrammar_structural_tag::format::{Format, StructuralTag};

    use super::*;
    use crate::request::{ChatMessage, GenerationPromptMode, ResolvedToolContext};

    fn chat_tool(name: &str, strict: Option<bool>) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            strict,
        }
    }

    fn qwen3_coder_parser(tools: &[Tool]) -> Box<dyn ToolParser> {
        Qwen3CoderToolParser::create(tools).expect("Qwen3 Coder parser should build")
    }

    fn muse_glimmer_parser() -> Box<dyn UnifiedParser> {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|start|>", 1001)
            .with_regular_token("<|message|>", 1002)
            .with_regular_token("<|eom|>", 1003)
            .with_regular_token("<|eot|>", 1004);
        MuseGlimmerUnifiedParser::create(&[], Arc::new(tokenizer))
            .expect("Muse Glimmer parser should build")
    }

    /// Records `build_scoped` arguments and returns a minimal valid tag.
    #[derive(Default)]
    struct MockScopedBuilder {
        calls: Mutex<Vec<ScopedBuilderCall>>,
    }

    #[derive(Debug, PartialEq)]
    enum RecordedConstraint {
        JsonSchema(Value),
        Regex(String),
        Choice(Vec<String>),
        Grammar(String),
    }

    #[derive(Debug)]
    struct ScopedBuilderCall {
        tool_names: Vec<String>,
        tool_stricts: Vec<Option<bool>>,
        tool_choice: Option<ScopedToolChoice>,
        caller: Option<RecordedConstraint>,
    }

    impl MockScopedBuilder {
        fn calls(&self) -> std::sync::MutexGuard<'_, Vec<ScopedBuilderCall>> {
            self.calls.lock().unwrap()
        }
    }

    impl ScopedStructuralTagBuilder for MockScopedBuilder {
        fn build_scoped(
            &self,
            tools: &[Tool],
            tool_choice: Option<ScopedToolChoice>,
            caller: Option<ScopedCallerConstraint<'_>>,
            _options: &StructuralTagOptions,
        ) -> xgrammar_structural_tag::Result<StructuralTag> {
            self.calls.lock().unwrap().push(ScopedBuilderCall {
                tool_names: tools.iter().map(|tool| tool.name.clone()).collect(),
                tool_stricts: tools.iter().map(|tool| tool.strict).collect(),
                tool_choice,
                caller: caller.map(|caller| match caller {
                    ScopedCallerConstraint::JsonSchema(schema) => {
                        RecordedConstraint::JsonSchema(schema.clone())
                    }
                    ScopedCallerConstraint::Regex(pattern) => {
                        RecordedConstraint::Regex(pattern.to_string())
                    }
                    ScopedCallerConstraint::Choice(choices) => {
                        RecordedConstraint::Choice(choices.to_vec())
                    }
                    ScopedCallerConstraint::Grammar(grammar) => {
                        RecordedConstraint::Grammar(grammar.to_string())
                    }
                }),
            });
            Ok(StructuralTag::new(Format::any_text()))
        }
    }

    fn request(tool_choice: ChatToolChoice, tools: Vec<Tool>) -> ChatRequest {
        ChatRequest {
            tool_context: ResolvedToolContext::new(&[], tools, Some(tool_choice), true)
                .expect("tool context should resolve"),
            ..ChatRequest::for_test()
        }
    }

    fn structural_tag_value(request: &ChatRequest) -> Value {
        let params = request
            .sampling_params
            .structured_outputs
            .as_ref()
            .expect("structured outputs should be set");
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        let structural_tag = params
            .constraint
            .as_structural_tag()
            .expect("structured output constraint should be structural_tag");
        serde_json::from_str(structural_tag).expect("structural_tag should be valid JSON")
    }

    fn structured_outputs(request: &ChatRequest) -> &StructuredOutputsParams {
        request
            .sampling_params
            .structured_outputs
            .as_ref()
            .expect("structured outputs should be set")
    }

    #[test]
    fn auto_strict_tool_choice_builds_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn required_dynamic_tool_builds_structural_tag_from_effective_tools() {
        let messages = vec![ChatMessage::developer(
            "",
            Some(vec![chat_tool("lookup", None)]),
        )];
        let tool_context =
            ResolvedToolContext::new(&messages, Vec::new(), Some(ChatToolChoice::Required), true)
                .expect("dynamic tool context should resolve");
        let mut request = ChatRequest {
            messages,
            tool_context,
            ..ChatRequest::for_test()
        };
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert!(tag.to_string().contains("lookup"));
    }

    #[test]
    fn required_initial_and_dynamic_tools_build_one_structural_tag() {
        let messages = vec![ChatMessage::developer(
            "",
            Some(vec![chat_tool("lookup", None)]),
        )];
        let tool_context = ResolvedToolContext::new(
            &messages,
            vec![chat_tool("search", None)],
            Some(ChatToolChoice::Required),
            true,
        )
        .expect("tool context should resolve");
        let mut request = ChatRequest {
            messages,
            tool_context,
            ..ChatRequest::for_test()
        };
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("search"));
        assert!(tag.contains("lookup"));
    }

    #[test]
    fn auto_non_strict_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn auto_strict_tool_choice_overwrites_existing_json_guidance() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(json!({"type": "object"}))
        });
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let params = structured_outputs(&request);
        assert!(params.constraint.is_structural_tag());
        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn required_tool_choice_builds_structural_tag_without_strict_tools() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn required_tool_choice_overwrites_existing_json_object_guidance() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json_object()
        });
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let params = structured_outputs(&request);
        assert!(params.constraint.is_structural_tag());
        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn named_tool_choice_builds_structural_tag_for_named_tool_only() {
        let mut request = request(
            ChatToolChoice::Function {
                name: "lookup".to_string(),
            },
            vec![chat_tool("search", None), chat_tool("lookup", None)],
        );
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn none_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn none_tool_choice_preserves_existing_json_object_guidance() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json_object()
        });
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag decision should succeed");

        let params = structured_outputs(&request);
        assert!(params.constraint.is_json_object());
    }

    #[test]
    fn scoped_builder_folds_caller_json_schema_without_tools() {
        let mut request = request(ChatToolChoice::None, vec![]);
        let schema = json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"]
        });
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(schema.clone())
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        // The grammar covers the reasoning span, so the engine applies it from
        // the first generated token.
        assert_eq!(request.reasoning_ended, Some(true));
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].tool_names.is_empty());
        assert_eq!(calls[0].tool_choice, None);
        assert_eq!(
            calls[0].caller,
            Some(RecordedConstraint::JsonSchema(schema))
        );
    }

    #[test]
    fn scoped_builder_folds_caller_json_object_as_generic_object_schema() {
        let mut request = request(ChatToolChoice::None, vec![]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json_object()
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].caller,
            Some(RecordedConstraint::JsonSchema(json!({"type": "object"})))
        );
    }

    #[test]
    fn scoped_builder_folds_caller_json_schema_with_none_tool_choice() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", None)]);
        let schema = json!({"type": "object"});
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(schema.clone())
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_names, vec!["search".to_string()]);
        assert_eq!(calls[0].tool_choice, None);
        assert_eq!(
            calls[0].caller,
            Some(RecordedConstraint::JsonSchema(schema))
        );
    }

    #[test]
    fn scoped_builder_builds_for_auto_strict_tool_with_caller_schema() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        let schema = json!({"type": "object"});
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(schema.clone())
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Auto));
        assert_eq!(
            calls[0].caller,
            Some(RecordedConstraint::JsonSchema(schema))
        );
    }

    #[test]
    fn scoped_builder_builds_for_auto_strict_tool_without_caller_schema() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Auto));
        assert_eq!(calls[0].caller, None);
    }

    #[test]
    fn scoped_builder_builds_for_required_tool_choice_without_caller_schema() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Required));
        assert_eq!(calls[0].caller, None);
    }

    #[test]
    fn scoped_builder_skips_auto_non_strict_tools_without_caller_schema() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
        assert_eq!(request.reasoning_ended, None);
        assert!(builder.calls().is_empty());
    }

    #[test]
    fn scoped_builder_rejects_regex_constraint_with_forced_tool_choice() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::regex("^[a-z]+$")
        });
        let builder = MockScopedBuilder::default();

        let error = apply_structural_tag_constraint(
            &mut request,
            None,
            Some(&builder),
            ToolStrictLevel::Auto,
        )
        .expect_err("regex constraint with forced tool choice should fail");

        assert!(matches!(error, Error::UnsupportedStructuredOutputs { .. }));
        assert!(error.is_request_validation_error());
        assert!(error.to_report_string().contains("tool_choice \"required\""));
        // The caller's constraint is left untouched.
        assert!(structured_outputs(&request).constraint.is_regex());
        assert!(builder.calls().is_empty());
    }

    #[test]
    fn scoped_builder_rejects_caller_schema_with_forced_tool_choice() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(json!({"type": "object"}))
        });
        let builder = MockScopedBuilder::default();

        let error = apply_structural_tag_constraint(
            &mut request,
            None,
            Some(&builder),
            ToolStrictLevel::Auto,
        )
        .expect_err("caller schema with required tool choice should fail");

        assert!(matches!(error, Error::UnsupportedStructuredOutputs { .. }));
        assert!(error.is_request_validation_error());
        assert!(error.to_report_string().contains("tool_choice \"required\""));
        // The caller's constraint is left untouched.
        assert!(matches!(
            structured_outputs(&request).constraint,
            StructuredOutputConstraint::Json(_)
        ));
        assert!(builder.calls().is_empty());
    }

    #[test]
    fn scoped_builder_folds_regex_choice_and_grammar_into_answer_channel() {
        // Left raw, these would apply from token 0 over the channel framing
        // (the engine has no reasoning parser to withhold them).
        let cases = [
            (
                StructuredOutputsParams::regex("^[a-z]+$"),
                RecordedConstraint::Regex("^[a-z]+$".to_string()),
            ),
            (
                StructuredOutputsParams::choice(vec!["yes".to_string(), "no".to_string()]),
                RecordedConstraint::Choice(vec!["yes".to_string(), "no".to_string()]),
            ),
            (
                StructuredOutputsParams::grammar("root ::= \"ok\""),
                RecordedConstraint::Grammar("root ::= \"ok\"".to_string()),
            ),
        ];
        for (params, recorded) in cases {
            let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
            request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
                backend: StructuredOutputBackend::Xgrammar,
                ..params
            });
            let builder = MockScopedBuilder::default();

            apply_structural_tag_constraint(
                &mut request,
                None,
                Some(&builder),
                ToolStrictLevel::Auto,
            )
            .expect("scoped structural tag should build");

            assert!(structured_outputs(&request).constraint.is_structural_tag());
            assert_eq!(request.reasoning_ended, Some(true));
            let calls = builder.calls();
            assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Auto));
            assert_eq!(calls[0].caller, Some(recorded));
        }
    }

    #[test]
    fn scoped_builder_skips_none_tool_choice_without_caller_constraint() {
        // A server strictness floor must not override `tool_choice: none`.
        for strict in [Some(true), None] {
            for level in [
                ToolStrictLevel::Auto,
                ToolStrictLevel::Function,
                ToolStrictLevel::Parameter,
            ] {
                let mut request = request(ChatToolChoice::None, vec![chat_tool("search", strict)]);
                let builder = MockScopedBuilder::default();

                apply_structural_tag_constraint(&mut request, None, Some(&builder), level)
                    .expect("structural tag decision should succeed");

                assert!(
                    request.sampling_params.structured_outputs.is_none(),
                    "{level:?}"
                );
                assert_eq!(request.reasoning_ended, None, "{level:?}");
                assert!(builder.calls().is_empty(), "{level:?}");
            }
        }
    }

    #[test]
    fn scoped_builder_resolves_the_server_strict_floor_per_tool() {
        // `auto` with non-strict tools constrains only from the `function`
        // floor; arguments are pinned only from `parameter` or `strict: true`.
        let cases = [
            (ChatToolChoice::Auto, None, ToolStrictLevel::Auto, None),
            (
                ChatToolChoice::Auto,
                None,
                ToolStrictLevel::Function,
                Some(Some(false)),
            ),
            (
                ChatToolChoice::Auto,
                None,
                ToolStrictLevel::Parameter,
                Some(Some(true)),
            ),
            (
                ChatToolChoice::Required,
                None,
                ToolStrictLevel::Auto,
                Some(Some(false)),
            ),
            (
                ChatToolChoice::Required,
                Some(true),
                ToolStrictLevel::Auto,
                Some(Some(true)),
            ),
        ];
        for (tool_choice, strict, level, expected) in cases {
            let mut request = request(tool_choice.clone(), vec![chat_tool("search", strict)]);
            let builder = MockScopedBuilder::default();

            apply_structural_tag_constraint(&mut request, None, Some(&builder), level)
                .expect("structural tag decision should succeed");

            let calls = builder.calls();
            assert_eq!(
                calls.first().map(|call| call.tool_stricts[0]),
                expected,
                "{tool_choice:?} strict={strict:?} at {level:?}"
            );
        }
    }

    #[test]
    fn caller_structural_tag_is_never_double_wrapped() {
        let caller_tag = r#"{"type":"structural_tag","format":{"type":"any_text","excludes":[]}}"#;
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::structural_tag(caller_tag)
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("caller structural tag should be preserved");

        assert_eq!(
            structured_outputs(&request).constraint.as_structural_tag().map(String::as_str),
            Some(caller_tag)
        );
        assert!(builder.calls().is_empty());
    }

    #[test]
    fn legacy_builder_still_overwrites_caller_structural_tag() {
        let caller_tag = r#"{"type":"structural_tag","format":{"type":"any_text","excludes":[]}}"#;
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::structural_tag(caller_tag)
        });
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag should build");

        // Legacy parsers keep their pre-scoped behavior: `required` wins.
        assert!(structural_tag_value(&request).to_string().contains("search"));
    }

    #[test]
    fn legacy_builder_preserves_caller_json_constraint_without_tag_trigger() {
        let schema = json!({"type": "object"});
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(schema.clone())
        });
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
            None,
            ToolStrictLevel::Auto,
        )
        .expect("structural tag decision should succeed");

        assert_eq!(
            structured_outputs(&request).constraint.as_json(),
            Some(&schema)
        );
    }

    #[test]
    fn scoped_builder_parses_string_form_caller_schema() {
        let mut request = request(ChatToolChoice::None, vec![]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json(json!(r#"{"type":"object"}"#))
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder), ToolStrictLevel::Auto)
            .expect("scoped structural tag should build");

        assert_eq!(
            builder.calls()[0].caller,
            Some(RecordedConstraint::JsonSchema(json!({"type": "object"})))
        );
    }

    #[test]
    fn scoped_builder_builds_no_tag_without_generation_prompt() {
        // The whole-generation grammar needs the prompt to end with
        // `<|start|>assistant`; a caller constraint is left as sent.
        for mode in [
            GenerationPromptMode::ContinueFinalAssistant,
            GenerationPromptMode::NoGenerationPrompt,
        ] {
            let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
            request.chat_options.generation_prompt_mode = mode;
            request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
                backend: StructuredOutputBackend::Xgrammar,
                ..StructuredOutputsParams::json(json!({"type": "object"}))
            });
            let builder = MockScopedBuilder::default();

            apply_structural_tag_constraint(
                &mut request,
                None,
                Some(&builder),
                ToolStrictLevel::Auto,
            )
            .expect("structural tag decision should succeed");

            assert!(
                structured_outputs(&request).constraint.is_json(),
                "{mode:?}"
            );
            assert_eq!(request.reasoning_ended, None, "{mode:?}");
            assert!(builder.calls().is_empty(), "{mode:?}");
        }
    }

    #[test]
    fn scoped_builder_rejects_forced_tool_choice_without_generation_prompt() {
        for mode in [
            GenerationPromptMode::ContinueFinalAssistant,
            GenerationPromptMode::NoGenerationPrompt,
        ] {
            let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
            request.chat_options.generation_prompt_mode = mode;
            let builder = MockScopedBuilder::default();

            let error = apply_structural_tag_constraint(
                &mut request,
                None,
                Some(&builder),
                ToolStrictLevel::Auto,
            )
            .expect_err("required tool choice without a generation prompt should fail");

            assert!(
                matches!(error, Error::UnsupportedStructuredOutputs { .. }),
                "{mode:?}"
            );
            assert!(error.is_request_validation_error());
            assert!(error.to_report_string().contains("without a generation prompt"));
            assert!(builder.calls().is_empty(), "{mode:?}");
        }
    }

    #[test]
    fn scoped_builder_tool_name_rejection_is_request_validation_error() {
        let mut request = request(
            ChatToolChoice::Required,
            vec![chat_tool("get weather", None)],
        );
        let parser = muse_glimmer_parser();

        let error = apply_structural_tag_constraint(
            &mut request,
            None,
            parser.scoped_structural_tag_builder(),
            ToolStrictLevel::Auto,
        )
        .expect_err("a tool name outside the recipient charset should fail");

        assert!(matches!(error, Error::UnsupportedStructuredOutputs { .. }));
        assert!(error.is_request_validation_error());
        assert!(error.to_report_string().contains("get weather"));
    }

    fn build(tool_choice: ChatToolChoice, tools: Vec<Tool>, level: ToolStrictLevel) -> ChatRequest {
        let mut request = request(tool_choice, tools);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None, level)
            .expect("structural tag decision should succeed");
        request
    }

    fn tag_string(request: &ChatRequest) -> String {
        structural_tag_value(request).to_string()
    }

    #[test]
    fn function_level_constrains_auto_without_strict_tools() {
        let request = build(
            ChatToolChoice::Auto,
            vec![chat_tool("search", None)],
            ToolStrictLevel::Function,
        );

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn absent_strict_is_non_strict() {
        let unset = build(
            ChatToolChoice::Required,
            vec![chat_tool("search", None)],
            ToolStrictLevel::Auto,
        );
        let non_strict = build(
            ChatToolChoice::Required,
            vec![chat_tool("search", Some(false))],
            ToolStrictLevel::Auto,
        );
        let strict = build(
            ChatToolChoice::Required,
            vec![chat_tool("search", Some(true))],
            ToolStrictLevel::Auto,
        );

        assert_eq!(tag_string(&unset), tag_string(&non_strict));
        assert_ne!(tag_string(&unset), tag_string(&strict));
    }

    #[test]
    fn strict_tool_does_not_pin_its_neighbour() {
        let mixed = build(
            ChatToolChoice::Auto,
            vec![chat_tool("weather", Some(true)), chat_tool("search", None)],
            ToolStrictLevel::Auto,
        );
        let explicit = build(
            ChatToolChoice::Auto,
            vec![
                chat_tool("weather", Some(true)),
                chat_tool("search", Some(false)),
            ],
            ToolStrictLevel::Auto,
        );

        assert_eq!(tag_string(&mixed), tag_string(&explicit));
    }

    #[test]
    fn parameter_level_pins_argument_schemas() {
        let pinned = build(
            ChatToolChoice::Auto,
            vec![chat_tool("search", None)],
            ToolStrictLevel::Parameter,
        );
        let strict = build(
            ChatToolChoice::Auto,
            vec![chat_tool("search", Some(true))],
            ToolStrictLevel::Auto,
        );
        let envelope_only = build(
            ChatToolChoice::Auto,
            vec![chat_tool("search", None)],
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
