//! Applies xgrammar structural-tag constraints for strict tool calling.

use thiserror_ext::AsReport;
use vllm_engine_core_client::protocol::{StructuredOutputBackend, StructuredOutputsParams};
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_optional_structural_tag,
};

use crate::parser::tool::ToolParser;
use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

pub(super) fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    parser: &dyn ToolParser,
) -> ChatResult<()> {
    let Some(model) = parser.structural_tag_model() else {
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

    let tag =
        build_optional_structural_tag(model, &tools, tool_choice, false).map_err(|error| {
            Error::StructuralTag {
                message: error.to_report_string(),
            }
        })?;
    let Some(tag) = tag else {
        return Ok(());
    };
    let structural_tag = tag.to_json_string().map_err(|error| Error::StructuralTag {
        message: error.to_report_string(),
    })?;

    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        structural_tag: Some(structural_tag),
        backend: StructuredOutputBackend::Xgrammar,
        ..Default::default()
    });
    Ok(())
}

fn structural_tag_tool_choice(request: &ChatRequest) -> Option<StructuralTagToolChoice> {
    match &request.tool_choice {
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
    use vllm_engine_core_client::protocol::StructuredOutputBackend;
    use vllm_tool_parser::{
        Result as ToolParserResult, StructuralTagModel, Tool, ToolParserOutput,
    };

    use super::*;

    struct StructuralTagParser;

    impl ToolParser for StructuralTagParser {
        fn create(_tools: &[Tool]) -> ToolParserResult<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn structural_tag_model(&self) -> Option<StructuralTagModel> {
            Some(StructuralTagModel::Qwen3Coder)
        }

        fn parse_into(
            &mut self,
            _chunk: &str,
            _output: &mut ToolParserOutput,
        ) -> ToolParserResult<()> {
            Ok(())
        }

        fn finish(&mut self) -> ToolParserResult<ToolParserOutput> {
            Ok(ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

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

    #[test]
    fn auto_strict_tool_choice_builds_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn auto_non_strict_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn required_tool_choice_builds_structural_tag_without_strict_tools() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

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

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn none_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }
}
