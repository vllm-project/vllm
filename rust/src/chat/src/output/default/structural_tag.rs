// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies xgrammar structural-tag constraints for strict tool calling.

use std::borrow::Cow;

use serde_json::{Value, json};
use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputConstraint, StructuredOutputsParams,
};
use vllm_parser::tool::StructuralTagBuilder;
use vllm_parser::unified::{ScopedStructuralTagBuilder, ScopedToolChoice};
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

use crate::error::bail_unsupported_structured_outputs;
use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

/// Apply structural tag constraints to the request based on the tool parser's structural tag
/// support and the request's tool choice.
///
/// A [`ScopedStructuralTagBuilder`] covers the whole generation and folds the
/// caller's response schema into the answer channel, while a legacy
/// [`StructuralTagBuilder`] constrains tool channels only and, as it always
/// has, overwrites any caller constraint once the tool choice triggers a tag.
pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: Option<&dyn StructuralTagBuilder>,
    scoped_builder: Option<&dyn ScopedStructuralTagBuilder>,
) -> ChatResult<()> {
    if let Some(scoped_builder) = scoped_builder {
        return apply_scoped_structural_tag_constraint(request, scoped_builder);
    }

    let Some(builder) = builder else {
        return Ok(());
    };
    let Some(tool_choice) = structural_tag_tool_choice(request) else {
        return Ok(());
    };

    let tools = request
        .tools()
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: tool.strict,
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
/// folding any caller-provided response schema into the answer channel.
fn apply_scoped_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: &dyn ScopedStructuralTagBuilder,
) -> ChatResult<()> {
    // A whole-generation grammar is anchored at a fresh turn's first channel
    // header; with `continue_final_message` generation resumes mid-channel,
    // where the grammar would force a spurious header. Leave the request
    // unconstrained (the parser still handles the prefilled channel).
    if request.chat_options.continue_final_message() {
        return Ok(());
    }
    // A caller-supplied structural tag is never wrapped in a parser-built one.
    if let Some(params) = &request.sampling_params.structured_outputs
        && params.constraint.is_structural_tag()
    {
        return Ok(());
    }

    let tool_choice = scoped_tool_choice(request);
    // `required` and named choices always generate tool channels; `auto` does
    // only with at least one strict tool (the same gating as the legacy path).
    let forces_tool_channels = match &tool_choice {
        Some(ScopedToolChoice::Required | ScopedToolChoice::Function(_)) => true,
        Some(ScopedToolChoice::Auto) => {
            request.tools().iter().any(|tool| tool.strict == Some(true))
        }
        None => false,
    };

    // The caller's response schema, if the constraint can be scoped into the
    // answer channel. Borrowed where possible: the builder only reads it.
    let constraint = request
        .sampling_params
        .structured_outputs
        .as_ref()
        .map(|params| &params.constraint);
    let caller_schema = match constraint {
        // The Json constraint also carries JSON-encoded schema *strings*
        // (validated upstream); the tag needs the schema value itself.
        Some(StructuredOutputConstraint::Json(Value::String(raw))) => {
            serde_json::from_str(raw).ok().map(Cow::Owned)
        }
        Some(StructuredOutputConstraint::Json(schema)) => Some(Cow::Borrowed(schema)),
        Some(StructuredOutputConstraint::JsonObject) => Some(Cow::Owned(json!({"type": "object"}))),
        // Regex, grammar, and choice constraints cannot fold into a
        // structural tag.
        _ => None,
    };

    match &caller_schema {
        // A non-scopable constraint cannot fold into the answer channel, so it
        // cannot be combined with tool calls; reject it honestly instead of
        // silently dropping the caller's constraint.
        None if constraint.is_some() && forces_tool_channels => {
            bail_unsupported_structured_outputs!(
                "the constraint cannot be combined with tool calls for this parser"
            );
        }
        None if !forces_tool_channels => return Ok(()),
        // A forced tool-call turn has no answer channel for the schema to
        // constrain; reject the combination instead of silently dropping it.
        Some(_)
            if matches!(
                tool_choice,
                Some(ScopedToolChoice::Required | ScopedToolChoice::Function(_))
            ) =>
        {
            bail_unsupported_structured_outputs!(
                "the constraint cannot be combined with tool_choice \"required\" or a named tool choice for this parser"
            );
        }
        _ => {}
    }

    let structural_tag = builder
        .build_scoped(
            request.tools(),
            tool_choice,
            caller_schema.as_deref(),
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
    // structured_outputs_config, and a structural tag has no whitespace or
    // additionalProperties switch to map them onto anyway.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });

    Ok(())
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
fn structural_tag_tool_choice(request: &ChatRequest) -> Option<StructuralTagToolChoice> {
    if request.tools().is_empty() {
        return None;
    }

    match request.tool_choice() {
        // For `Auto`, only apply the structural tag if there's at least one strict tool.
        ChatToolChoice::Auto if request.tools().iter().any(|tool| tool.strict == Some(true)) => {
            Some(StructuralTagToolChoice::auto())
        }
        ChatToolChoice::Auto | ChatToolChoice::None => None,

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

    #[derive(Debug)]
    struct ScopedBuilderCall {
        tool_names: Vec<String>,
        tool_choice: Option<ScopedToolChoice>,
        caller_schema: Option<Value>,
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
            caller_schema: Option<&Value>,
            _options: &StructuralTagOptions,
        ) -> xgrammar_structural_tag::Result<StructuralTag> {
            self.calls.lock().unwrap().push(ScopedBuilderCall {
                tool_names: tools.iter().map(|tool| tool.name.clone()).collect(),
                tool_choice,
                caller_schema: caller_schema.cloned(),
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("search"));
        assert!(tag.contains("lookup"));
    }

    #[test]
    fn auto_non_strict_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn none_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].tool_names.is_empty());
        assert_eq!(calls[0].tool_choice, None);
        assert_eq!(calls[0].caller_schema, Some(schema));
    }

    #[test]
    fn scoped_builder_folds_caller_json_object_as_generic_object_schema() {
        let mut request = request(ChatToolChoice::None, vec![]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::json_object()
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].caller_schema, Some(json!({"type": "object"})));
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

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_names, vec!["search".to_string()]);
        assert_eq!(calls[0].tool_choice, None);
        assert_eq!(calls[0].caller_schema, Some(schema));
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

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Auto));
        assert_eq!(calls[0].caller_schema, Some(schema));
    }

    #[test]
    fn scoped_builder_builds_for_auto_strict_tool_without_caller_schema() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Auto));
        assert_eq!(calls[0].caller_schema, None);
    }

    #[test]
    fn scoped_builder_builds_for_required_tool_choice_without_caller_schema() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        let calls = builder.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_choice, Some(ScopedToolChoice::Required));
        assert_eq!(calls[0].caller_schema, None);
    }

    #[test]
    fn scoped_builder_skips_auto_non_strict_tools_without_caller_schema() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
        assert!(builder.calls().is_empty());
    }

    #[test]
    fn scoped_builder_rejects_non_scopable_constraint_with_forced_tool_choice() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::regex("^[a-z]+$")
        });
        let builder = MockScopedBuilder::default();

        let error = apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect_err("non-scopable constraint with forced tool choice should fail");

        assert!(matches!(error, Error::UnsupportedStructuredOutputs { .. }));
        assert!(error.is_request_validation_error());
        assert!(error.to_report_string().contains("cannot be combined with tool calls"));
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

        let error = apply_structural_tag_constraint(&mut request, None, Some(&builder))
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
    fn scoped_builder_preserves_non_scopable_constraint_without_forced_tool_choice() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            backend: StructuredOutputBackend::Xgrammar,
            ..StructuredOutputsParams::regex("^[a-z]+$")
        });
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("structural tag decision should succeed");

        assert!(structured_outputs(&request).constraint.is_regex());
        assert!(builder.calls().is_empty());
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

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), None)
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

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("scoped structural tag should build");

        assert_eq!(
            builder.calls()[0].caller_schema,
            Some(json!({"type": "object"}))
        );
    }

    #[test]
    fn scoped_builder_leaves_continued_final_message_unconstrained() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
        let builder = MockScopedBuilder::default();

        apply_structural_tag_constraint(&mut request, None, Some(&builder))
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
        assert!(builder.calls().is_empty());
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
        )
        .expect_err("a tool name outside the recipient charset should fail");

        assert!(matches!(error, Error::UnsupportedStructuredOutputs { .. }));
        assert!(error.is_request_validation_error());
        assert!(error.to_report_string().contains("get weather"));
    }
}
