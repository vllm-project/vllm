// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies xgrammar structural-tag constraints for strict tool calling.

use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputsParams,
};
use vllm_parser::tool::StructuralTagBuilder;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

use crate::parser::ToolStrictLevel;
use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

/// Apply structural tag constraints to the request based on the tool parser's structural tag
/// support, the request's tool choice, and the server-side strictness floor.
pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    builder: Option<&dyn StructuralTagBuilder>,
    strict_level: ToolStrictLevel,
) -> ChatResult<()> {
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
        StructuralTagOptions::default()
            .with_reasoning(false)
            .with_parallel_tool_calls(request.parallel_tool_calls()),
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
        // For `Auto`, apply the structural tag only when the server raises the floor or at
        // least one tool opts in with `strict: true`.
        ChatToolChoice::Auto => (strict_level >= ToolStrictLevel::Function
            || request.tools().iter().any(|tool| tool.strict == Some(true)))
        .then(StructuralTagToolChoice::auto),

        ChatToolChoice::Required => Some(StructuralTagToolChoice::required()),
        ChatToolChoice::Function { name } => Some(StructuralTagToolChoice::function(name.clone())),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use vllm_engine_core_client::protocol::structured_outputs::{
        StructuredOutputBackend, StructuredOutputsParams,
    };
    use vllm_parser::tool::{Qwen3CoderToolParser, Tool, ToolParser};

    use super::*;
    use crate::request::{ChatMessage, ResolvedToolContext};

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
    fn tool_grammar_honors_parallel_call_policy() {
        for parallel in [false, true] {
            let tools = vec![chat_tool("lookup", Some(true))];
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(
                    &[],
                    tools,
                    Some(ChatToolChoice::Auto),
                    parallel,
                )
                .unwrap(),
                ..ChatRequest::for_test()
            };
            let parser = qwen3_coder_parser(request.tools());
            apply_structural_tag_constraint(
                &mut request,
                parser.structural_tag_builder(),
                ToolStrictLevel::Auto,
            )
            .unwrap();
            let tag = structural_tag_value(&request);
            assert_eq!(tag["format"]["type"], "triggered_tags");
            assert_eq!(tag["format"]["stop_after_first"], !parallel);
        }
    }

    #[test]
    fn auto_strict_tool_choice_builds_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(
            &mut request,
            parser.structural_tag_builder(),
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
            ToolStrictLevel::Auto,
        )
        .expect("structural tag decision should succeed");

        let params = structured_outputs(&request);
        assert!(params.constraint.is_json_object());
    }

    fn build(tool_choice: ChatToolChoice, tools: Vec<Tool>, level: ToolStrictLevel) -> ChatRequest {
        let mut request = request(tool_choice, tools);
        let parser = qwen3_coder_parser(request.tools());

        apply_structural_tag_constraint(&mut request, parser.structural_tag_builder(), level)
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
