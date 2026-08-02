// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;
use std::sync::Arc;

use expect_test::{ExpectFile, expect_file};
use serde_json::json;
use thiserror_ext::AsReport;
use vllm_text::tokenizer::Tokenizer;

use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};

use super::{
    AUDIO_END, CONTENT_AUDIO_INPUT, CONTENT_IMAGE, CONTENT_INVOKE_TOOL_JSON, CONTENT_TEXT,
    CONTENT_THINKING, CONTENT_XML, END_MESSAGE, InklingChatRenderer, MESSAGE_MODEL, MESSAGE_SYSTEM,
    MESSAGE_TOOL, MESSAGE_USER,
};
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{ChatMessage, ChatRequest, GenerationPromptMode, ReasoningEffort};
use crate::{ChatRenderer, Error};

struct FixtureTokenizer;

impl Tokenizer for FixtureTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<String> {
        Ok(token_ids
            .iter()
            .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|message_user|>" => Some(200000),
            "<|message_model|>" => Some(200001),
            "<|message_system|>" => Some(200002),
            "<|message_tool|>" => Some(200003),
            "<|content_text|>" => Some(200004),
            "<|content_image|>" => Some(200005),
            "<|content_model_end_sampling|>" => Some(200006),
            "<|content_thinking|>" => Some(200008),
            "<|end_message|>" => Some(200010),
            "<|content_audio_input|>" => Some(200020),
            CONTENT_XML => Some(200024),
            "<|audio_end|>" => Some(200043),
            "<|content_invoke_tool_json|>" => Some(200049),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        let token = match id {
            200000 => "<|message_user|>",
            200001 => "<|message_model|>",
            200002 => "<|message_system|>",
            200003 => "<|message_tool|>",
            200004 => "<|content_text|>",
            200005 => "<|content_image|>",
            200006 => "<|content_model_end_sampling|>",
            200008 => "<|content_thinking|>",
            200010 => "<|end_message|>",
            200020 => "<|content_audio_input|>",
            200024 => CONTENT_XML,
            200043 => "<|audio_end|>",
            200049 => "<|content_invoke_tool_json|>",
            _ => return None,
        };
        Some(token.to_string())
    }
}

fn renderer() -> InklingChatRenderer {
    InklingChatRenderer::new(Arc::new(FixtureTokenizer)).unwrap()
}

fn render_token_ids(request: &ChatRequest) -> Vec<u32> {
    renderer()
        .render(request)
        .unwrap()
        .prompt
        .into_token_ids()
        .expect("Inkling renderer returns token IDs")
}

fn fixture_request(name: &str) -> ChatRequest {
    fixture_chat_request(&fixture_path(name), inkling_fixture_options())
}

fn inkling_fixture_options() -> FixtureRequestOptions {
    FixtureRequestOptions {
        enable_thinking: false,
        no_generation_prompt_when_last_assistant: false,
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderer/inkling")
        .join("fixtures")
        .join(name)
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = format!("{}\n", render_symbolic_tokens(&render_token_ids(&request)));

    expected.assert_eq(&rendered);
}

fn render_symbolic_tokens(token_ids: &[u32]) -> String {
    let mut out = String::new();
    let mut text_bytes = Vec::new();
    for token_id in token_ids {
        if let Some(marker) = symbolic_tokens()
            .iter()
            .find_map(|(marker, marker_id)| (token_id == marker_id).then_some(*marker))
        {
            flush_text_bytes(&mut out, &mut text_bytes);
            out.push_str(marker);
            continue;
        }

        let byte = u8::try_from(*token_id)
            .unwrap_or_else(|_| panic!("unexpected non-byte fixture text token {token_id}"));
        text_bytes.push(byte);
    }
    flush_text_bytes(&mut out, &mut text_bytes);
    out
}

fn flush_text_bytes(out: &mut String, text_bytes: &mut Vec<u8>) {
    if text_bytes.is_empty() {
        return;
    }
    out.push_str(std::str::from_utf8(text_bytes).unwrap());
    text_bytes.clear();
}

fn symbolic_tokens() -> [(&'static str, u32); 13] {
    [
        (MESSAGE_USER, 200000),
        (MESSAGE_MODEL, 200001),
        (MESSAGE_SYSTEM, 200002),
        (MESSAGE_TOOL, 200003),
        (CONTENT_TEXT, 200004),
        (CONTENT_IMAGE, 200005),
        (super::CONTENT_MODEL_END_SAMPLING, 200006),
        (CONTENT_THINKING, 200008),
        (END_MESSAGE, 200010),
        (CONTENT_AUDIO_INPUT, 200020),
        (CONTENT_XML, 200024),
        (AUDIO_END, 200043),
        (CONTENT_INVOKE_TOOL_JSON, 200049),
    ]
}

#[test]
fn renders_text_and_image_as_inkling_tokens() {
    assert_fixture(
        "text_image_input.json",
        expect_file!["fixtures/text_image_output.txt"],
    );
}

#[test]
fn renders_audio_marker_only_blocks() {
    assert_fixture(
        "text_audio_input.json",
        expect_file!["fixtures/text_audio_output.txt"],
    );
}

#[test]
fn renders_reasoning_text_tool_call_and_tool_response() {
    assert_fixture(
        "tool_round_trip_input.json",
        expect_file!["fixtures/tool_round_trip_output.txt"],
    );
}

#[test]
fn renders_request_and_developer_tools_as_tool_declare() {
    assert_fixture(
        "tool_declare_input.json",
        expect_file!["fixtures/tool_declare_output.txt"],
    );
}

#[test]
fn renders_named_reasoning_effort_after_tool_declarations() {
    for (effort, expected) in [
        (ReasoningEffort::None, "0.0"),
        (ReasoningEffort::Minimal, "0.1"),
        (ReasoningEffort::Low, "0.2"),
        (ReasoningEffort::Medium, "0.7"),
        (ReasoningEffort::High, "0.9"),
        (ReasoningEffort::XHigh, "0.99"),
        (ReasoningEffort::Max, "0.99"),
    ] {
        let mut request = fixture_request("tool_declare_input.json");
        request.chat_options.reasoning_effort = Some(effort);

        let rendered = render_symbolic_tokens(&render_token_ids(&request));
        let tool_end = rendered.find("<|end_message|>").unwrap();
        let effort_block = format!(
            "<|message_system|><|content_text|>Thinking effort level: \
             {expected}<|end_message|>"
        );
        let effort_start = rendered.find(&effort_block).unwrap();
        let system_end = rendered.find("rules<|end_message|>").unwrap();

        assert!(effort_start > tool_end);
        assert!(effort_start > system_end);
    }
}

#[test]
fn emits_one_reasoning_effort_for_multi_turn_conversation() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage::system("rules"),
            ChatMessage::user("user1"),
            ChatMessage::assistant_text("assistant1"),
            ChatMessage::user("user2"),
        ],
        ..ChatRequest::for_test()
    };

    let rendered = render_symbolic_tokens(&render_token_ids(&request));
    let effort = "<|message_system|><|content_text|>Thinking effort level: 0.9\
                  <|end_message|>";

    assert_eq!(rendered.matches(effort).count(), 1);
    assert!(rendered.find("rules").unwrap() < rendered.find(effort).unwrap());
    assert!(rendered.find(effort).unwrap() < rendered.find("user1").unwrap());
    assert!(rendered.find("user1").unwrap() < rendered.find("assistant1").unwrap());
    assert!(rendered.find("assistant1").unwrap() < rendered.find("user2").unwrap());
}

#[test]
fn canonicalizes_numeric_zero_reasoning_effort() {
    for value in [0.0, -0.0] {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("reasoning_effort".to_string(), json!(value));

        assert!(
            render_symbolic_tokens(&render_token_ids(&request)).starts_with(
                "<|message_system|><|content_text|>Thinking effort level: 0.0\
                 <|end_message|>"
            )
        );
    }
}

#[test]
fn defaults_reasoning_effort_to_high() {
    let request = ChatRequest::for_test();

    assert!(
        render_symbolic_tokens(&render_token_ids(&request)).starts_with(
            "<|message_system|><|content_text|>Thinking effort level: 0.9\
             <|end_message|>"
        )
    );
}

#[test]
fn renders_numeric_reasoning_effort_template_kwarg() {
    let mut request = ChatRequest::for_test();
    request
        .chat_options
        .template_kwargs
        .insert("reasoning_effort".to_string(), json!(0.8));

    assert_eq!(
        render_symbolic_tokens(&render_token_ids(&request)),
        "<|message_system|><|content_text|>Thinking effort level: 0.8\
         <|end_message|><|message_user|><|content_text|>test\
         <|end_message|><|message_model|>"
    );
}

#[test]
fn ignores_unsupported_reasoning_effort_values() {
    for value in [json!(true), json!("invalid"), json!(null)] {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("reasoning_effort".to_string(), value);

        assert_eq!(
            render_symbolic_tokens(&render_token_ids(&request)),
            "<|message_user|><|content_text|>test<|end_message|><|message_model|>"
        );
    }
}

#[test]
fn rejects_out_of_range_reasoning_effort() {
    for value in [0.990_000_1, 1.0, 1.5, -0.1] {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("reasoning_effort".to_string(), json!(value));

        let error = renderer().render(&request).unwrap_err();
        assert!(error.as_report().to_string().contains("must be in [0.0, 0.99]"));
    }
}

#[test]
fn rejects_continue_final_message() {
    let request = ChatRequest {
        messages: vec![ChatMessage::assistant_text("partial")],
        chat_options: crate::ChatOptions {
            generation_prompt_mode: GenerationPromptMode::ContinueFinalAssistant,
            ..Default::default()
        },
        ..ChatRequest::for_test()
    };

    let error = renderer().render(&request).unwrap_err();
    assert!(matches!(error, Error::ChatTemplate(_)));
    assert!(error.as_report().to_string().contains("continue_final_message"));
}

#[test]
fn renders_developer_messages_as_system() {
    let request = ChatRequest {
        messages: vec![ChatMessage::developer("rules", None)],
        ..ChatRequest::for_test()
    };

    assert_eq!(
        render_symbolic_tokens(&render_token_ids(&request)),
        "<|message_system|><|content_text|>rules<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9\
         <|end_message|><|message_model|>"
    );
}

#[test]
fn rejects_non_object_tool_call_arguments() {
    let request = ChatRequest {
        messages: vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: "[]".to_string(),
            }),
        ])],
        ..ChatRequest::for_test()
    };

    let error = renderer().render(&request).unwrap_err();
    assert!(error.as_report().to_string().contains("JSON object"));
}
