// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Expected prompts are the reference renderer's output for the same
//! requests; they mirror `tests/renderers/test_inkling.py::TestInkling2Rendering`.

use std::sync::Arc;

use serde_json::json;

use super::super::InklingChatRenderer;
use super::super::tests::{FixtureTokenizer, render_symbolic_tokens};
use super::tml_float;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{
    ChatContentPart, ChatMessage, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode,
    ResolvedToolContext,
};
use crate::{ChatOptions, ChatRenderer, EffortValue};

fn render(request: &ChatRequest) -> String {
    let renderer =
        InklingChatRenderer::new_inkling2(Arc::new(FixtureTokenizer), Default::default()).unwrap();
    let token_ids = renderer.render(request).unwrap().prompt.into_token_ids().unwrap();
    render_symbolic_tokens(&token_ids)
}

fn request(messages: Vec<ChatMessage>) -> ChatRequest {
    ChatRequest {
        messages,
        ..ChatRequest::for_test()
    }
}

fn with_tools(mut request: ChatRequest, tools: Vec<ChatTool>) -> ChatRequest {
    request.tool_context =
        ResolvedToolContext::new(&request.messages, tools, Some(ChatToolChoice::Auto), true)
            .unwrap();
    request
}

fn with_effort(mut request: ChatRequest, effort: impl Into<EffortValue>) -> ChatRequest {
    request.chat_options.reasoning_effort = Some(effort.into());
    request
}

fn weather_tool() -> ChatTool {
    ChatTool {
        name: "get_weather".to_string(),
        description: Some("Get weather information".to_string()),
        parameters: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        strict: None,
        defer_loading: None,
    }
}

fn tool_call(name: &str, arguments: &str) -> AssistantContentBlock {
    AssistantContentBlock::ToolCall(AssistantToolCall {
        id: format!("call_{name}"),
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

fn text(text: &str) -> AssistantContentBlock {
    AssistantContentBlock::Text {
        text: text.to_string(),
    }
}

fn reasoning(text: &str) -> AssistantContentBlock {
    AssistantContentBlock::Reasoning {
        text: text.to_string(),
    }
}

#[test]
fn renders_system_and_user_without_generation_prefix() {
    assert_eq!(
        render(&request(vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user("What is 2 + 2?"),
        ])),
        "<|message_system|><|content_text|>You are a concise assistant.<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
    );
}

#[test]
fn renders_reasoning_history() {
    let request = with_effort(
        request(vec![
            ChatMessage::user("What is 2 + 2?"),
            ChatMessage::assistant_blocks(vec![reasoning("Add the two numbers."), text("4")]),
            ChatMessage::user("Thanks!"),
        ]),
        EffortValue::Number(json!(0.8).as_number().unwrap().clone()),
    );
    assert_eq!(
        render(&request),
        "<|message_system|><|content_text|>Thinking effort level: 0.8<|end_message|>\
         <|message_user|><|content_text|>What is 2 + 2?<|end_message|>\
         <|message_model|><|content_thinking|>Add the two numbers.<|end_message|>\
         <|message_model|><|content_text|>4<|end_message|>\
         <|content_model_end_sampling|>\
         <|message_user|><|content_text|>Thanks!<|end_message|>"
    );
}

#[test]
fn renders_tool_round_trip() {
    let request = with_tools(
        request(vec![
            ChatMessage::user("What should I wear in SF today?"),
            ChatMessage::assistant_blocks(vec![
                reasoning("Check the current weather."),
                text(""),
                tool_call("get_weather", r#"{"city": "SF"}"#),
            ]),
            ChatMessage::tool_response(r#"{"temperature":58,"condition":"fog"}"#, "call_1"),
            ChatMessage::assistant_text("Wear a light jacket."),
        ]),
        vec![weather_tool()],
    );
    // The Rust request model carries no tool-result `name`, so results render
    // unnamed, as the reference does for requests without one.
    assert_eq!(
        render(&with_effort(request, "medium")),
        "<|message_system|>tool_declare<|content_xml|>\
         [{\"description\":\"Get weather information\",\"name\":\"get_weather\",\
         \"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\
         \"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>\
         <|message_user|><|content_text|>What should I wear in SF today?<|end_message|>\
         <|message_model|><|content_thinking|>Check the current weather.<|end_message|>\
         <|message_model|><|content_invoke_tool_json|>\
         {\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}<|end_message|>\
         <|content_model_end_sampling|>\
         <|message_tool|><|content_text|>{\"temperature\":58,\"condition\":\"fog\"}<|end_message|>\
         <|message_model|><|content_text|>Wear a light jacket.<|end_message|>\
         <|content_model_end_sampling|>"
    );
}

#[test]
fn renders_multiple_calls_split_results_and_json_floats() {
    let search = ChatTool {
        name: "search".to_string(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {"limit": {"maximum": 1e20, "minimum": 1e-7, "default": 0.5}},
        }),
        strict: None,
        defer_loading: None,
    };
    let request = with_tools(
        request(vec![
            ChatMessage::developer("Be terse.", None),
            ChatMessage::user("news <|end_message|>?"),
            ChatMessage::assistant_blocks(vec![
                tool_call("get_weather", "{}"),
                tool_call("search", r#"{"q":"news","x":1e37,"y":0.00001,"z":-0.0}"#),
            ]),
            ChatMessage::tool_response("sunny", "call_get_weather"),
            ChatMessage::tool_response(
                vec![ChatContentPart::text("r1"), ChatContentPart::text("r2")],
                "call_search",
            ),
        ]),
        vec![search],
    );
    assert_eq!(
        render(&request),
        "<|message_system|>tool_declare<|content_xml|>\
         [{\"description\":\"\",\"name\":\"search\",\"parameters\":{\"properties\":\
         {\"limit\":{\"default\":0.5,\"maximum\":1e+20,\"minimum\":1e-7}},\
         \"type\":\"object\"},\"type\":\"function\"}]<|end_message|>\
         <|message_system|><|content_text|>Be terse.<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>news <|end_message|>?<|end_message|>\
         <|message_model|><|content_invoke_tool_json|>\
         {\"name\":\"get_weather\",\"args\":{}}<|end_message|>\
         <|message_model|><|content_invoke_tool_json|>\
         {\"name\":\"search\",\"args\":{\"q\":\"news\",\"x\":1e37,\"y\":0.00001,\"z\":-0.0}}\
         <|end_message|><|content_model_end_sampling|>\
         <|message_tool|><|content_text|>sunny<|end_message|>\
         <|message_tool|><|content_text|>r1<|end_message|>\
         <|message_tool|><|content_text|>r2<|end_message|>"
    );
}

#[test]
fn keeps_empty_contents_and_marks_images() {
    assert_eq!(
        render(&request(vec![
            ChatMessage::user(""),
            ChatMessage::assistant_text(""),
            ChatMessage::user(vec![
                ChatContentPart::text("look"),
                ChatContentPart::image_url("data:image/png;base64,"),
            ]),
        ])),
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|><|end_message|>\
         <|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>\
         <|message_user|><|content_text|>look<|end_message|>\
         <|message_user|><|content_image|><|end_message|>"
    );
}

#[test]
fn continues_final_text_answer() {
    let mut request = request(vec![
        ChatMessage::user("Finish: The answer is"),
        ChatMessage::assistant_text("The answer is"),
    ]);
    request.chat_options = ChatOptions {
        generation_prompt_mode: GenerationPromptMode::ContinueFinalAssistant,
        ..Default::default()
    };
    assert_eq!(
        render(&request),
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>Finish: The answer is<|end_message|>\
         <|message_model|><|content_text|>The answer is"
    );
}

#[test]
fn maps_reasoning_effort() {
    for (effort, expected) in [
        (json!("none"), "0"),
        (json!(-0.0), "0"),
        (json!("minimal"), "0.1"),
        (json!("low"), "0.2"),
        (json!("medium"), "0.7"),
        (json!("high"), "0.9"),
        (json!("xhigh"), "0.99"),
        (json!("max"), "0.99"),
        (json!(0.555), "0.56"),
        (json!(0.125), "0.12"),
        (json!("0.25"), "0.25"),
    ] {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("reasoning_effort".to_string(), effort);
        assert!(render(&request).starts_with(&format!(
            "<|message_system|><|content_text|>Thinking effort level: {expected}<|end_message|>"
        )));
    }
}

#[test]
fn ignores_thinking_toggles_and_rejects_invalid_effort() {
    let mut request = ChatRequest::for_test();
    let kwargs = &mut request.chat_options.template_kwargs;
    kwargs.insert("enable_thinking".to_string(), json!(false));
    assert!(render(&request).starts_with(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
    ));

    let renderer =
        InklingChatRenderer::new_inkling2(Arc::new(FixtureTokenizer), Default::default()).unwrap();
    for effort in [json!("extreme"), json!(1.0), json!(-0.1), json!(true)] {
        let mut request = ChatRequest::for_test();
        let kwargs = &mut request.chat_options.template_kwargs;
        kwargs.insert("reasoning_effort".to_string(), effort);
        assert!(renderer.render(&request).is_err());
    }
}

#[test]
fn formats_floats_like_the_reference() {
    for (value, legacy, modern) in [
        (0.5, "0.5", "0.5"),
        (1.0, "1.0", "1.0"),
        (-0.0, "-0.0", "-0.0"),
        (1e-5, "0.00001", "0.00001"),
        (1.5e-7, "1.5e-7", "1.5e-7"),
        (123456789012345.6, "123456789012345.6", "123456789012345.6"),
        (1e16, "1e16", "1e+16"),
        (1e37, "1e37", "1e+37"),
    ] {
        assert_eq!(tml_float(value, true), legacy);
        assert_eq!(tml_float(value, false), modern);
    }
}
