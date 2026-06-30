//! Applies xgrammar structural-tag constraints for strict tool calling.

use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::{StructuredOutputBackend, StructuredOutputsParams};
use vllm_parser::tool::StructuralTagModel;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

/// Apply structural tag constraints to the request based on the tool parser's structural tag
/// support and the request's tool choice.
pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    model: Option<StructuralTagModel>,
) -> ChatResult<()> {
    let Some(model) = model else {
        return Ok(());
    };
    let Some(tool_choice) = structural_tag_tool_choice(request) else {
        return Ok(());
    };

    let tools = request
        .tools
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

    let structural_tag = build_structural_tag(model, &tools, tool_choice, false)
        .and_then(|tag| tag.to_json_string())
        .map_err(|error| Error::StructuralTag {
            message: error.to_report_string(),
        })?;

    // Overwrite any existing structured output settings with the structural tag constraint.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        structural_tag: Some(structural_tag),
        backend: StructuredOutputBackend::Xgrammar,
        ..Default::default()
    });

    Ok(())
}

/// Resolve the tool choice used for [`xgrammar_structural_tag`] based on the request.
///
/// Returns `None` if no structural tag constraints should be applied.
fn structural_tag_tool_choice(request: &ChatRequest) -> Option<StructuralTagToolChoice> {
    if request.tools.is_empty() {
        return None;
    }

    match &request.tool_choice {
        // For `Auto`, only apply the structural tag if there's at least one strict tool.
        ChatToolChoice::Auto if request.tools.iter().any(|tool| tool.strict == Some(true)) => {
            Some(StructuralTagToolChoice::auto())
        }
        ChatToolChoice::Auto | ChatToolChoice::None => None,

        ChatToolChoice::Required => Some(StructuralTagToolChoice::required()),
        ChatToolChoice::Function { name } => Some(StructuralTagToolChoice::function(name.clone())),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use vllm_engine_core_client::protocol::{StructuredOutputBackend, StructuredOutputsParams};
    use vllm_parser::tool::{Qwen3CoderToolParser, Tool, ToolParser};

    use super::*;

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
            tool_choice,
            tools,
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
        serde_json::from_str(
            params.structural_tag.as_deref().expect("structural_tag should be set"),
        )
        .expect("structural_tag should be valid JSON")
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
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn auto_non_strict_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn auto_strict_tool_choice_overwrites_existing_json_guidance() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            json: Some(json!({"type": "object"})),
            backend: StructuredOutputBackend::Xgrammar,
            ..Default::default()
        });
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag should build");

        let params = structured_outputs(&request);
        assert!(params.json.is_none());
        assert!(params.structural_tag.is_some());
        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn required_tool_choice_builds_structural_tag_without_strict_tools() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn required_tool_choice_overwrites_existing_json_object_guidance() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            json_object: Some(true),
            backend: StructuredOutputBackend::Xgrammar,
            ..Default::default()
        });
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag should build");

        let params = structured_outputs(&request);
        assert!(params.json_object.is_none());
        assert!(params.structural_tag.is_some());
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
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn none_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn none_tool_choice_preserves_existing_json_object_guidance() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);
        request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
            json_object: Some(true),
            backend: StructuredOutputBackend::Xgrammar,
            ..Default::default()
        });
        let parser = qwen3_coder_parser(&request.tools);

        apply_structural_tag_constraint(&mut request, parser.structural_tag_model())
            .expect("structural tag decision should succeed");

        let params = structured_outputs(&request);
        assert_eq!(params.json_object, Some(true));
        assert!(params.structural_tag.is_none());
    }
}
