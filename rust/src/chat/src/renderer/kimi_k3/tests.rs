// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Golden fixtures generated from HF remote-code `encoding_k3.py`.

use std::path::PathBuf;
use std::sync::Arc;

use expect_test::{expect, expect_file};
use serde_json::json;
use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;
use vllm_tokenizer::Tokenizer;
use vllm_tokenizer::test_utils::TestTokenizer;

use super::KimiK3ChatRenderer;
use crate::ChatRenderer;
use crate::renderer::kimi_k3::encoding::{CLOSE, END_OF_MSG, IMAGE_PLACEHOLDER, OPEN, SEP};
use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};
use crate::request::{
    ChatContentPart, ChatMessage, ChatTool, GenerationPromptMode, ReasoningEffort,
};
use crate::{AssistantContentBlock, AssistantToolCall};

const OPEN_ID: u32 = 256;
const CLOSE_ID: u32 = 257;
const SEP_ID: u32 = 258;
const END_OF_MSG_ID: u32 = 259;
const MEDIA_ID: u32 = 260;

fn test_tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_special_token(OPEN, OPEN_ID)
        .with_special_token(CLOSE, CLOSE_ID)
        .with_special_token(SEP, SEP_ID)
        .with_special_token(END_OF_MSG, END_OF_MSG_ID)
        .with_special_token(IMAGE_PLACEHOLDER, MEDIA_ID)
}

fn render_token_ids(request: &crate::request::ChatRequest, tokenizer: DynTokenizer) -> Vec<u32> {
    let prompt = KimiK3ChatRenderer::new(tokenizer).render(request).unwrap().prompt;
    let Prompt::TokenIds(token_ids) = prompt else {
        panic!("kimi k3 renderer should return token IDs")
    };
    token_ids
}

fn render_request(request: &crate::request::ChatRequest) -> String {
    let tokenizer: DynTokenizer = Arc::new(test_tokenizer());
    let token_ids = render_token_ids(request, tokenizer.clone());
    tokenizer.decode(&token_ids, false).unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderer/kimi_k3/fixtures")
        .join(name)
}

fn kimi_k3_fixture_options() -> FixtureRequestOptions {
    FixtureRequestOptions {
        // Fixture JSON owns thinking via `template_kwargs`.
        enable_thinking: None,
        no_generation_prompt_when_last_assistant: false,
    }
}

fn assert_golden(name: &str) {
    let input_name = format!("{name}_input.json");
    let request = fixture_chat_request(&fixture_path(&input_name), kimi_k3_fixture_options());
    let rendered = render_request(&request);
    expect_file![format!("fixtures/{name}_output.txt")].assert_eq(&rendered);
}

#[test]
fn golden_history_preserve_and_image() {
    assert_golden("history_preserve_and_image");
}

#[test]
fn golden_tools_history_and_required() {
    assert_golden("tools_history_and_required");
}

#[test]
fn golden_controls_thinking_off() {
    assert_golden("controls_thinking_off");
}

#[test]
fn golden_dynamic_system_tool_declare() {
    assert_golden("dynamic_system_tool_declare");
}

#[test]
fn tool_declare_omits_absent_fields_and_preserves_parameter_nulls() {
    let mut request = crate::request::ChatRequest::for_test();
    request.tools = vec![ChatTool {
        name: "lookup".to_string(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": ["integer", "null"],
                    "default": null
                }
            }
        }),
        strict: None,
    }];

    let rendered = render_request(&request);

    assert!(rendered.contains(
        r#"[{"function":{"name":"lookup","parameters":{"properties":{"limit":{"default":null,"type":["integer","null"]}},"type":"object"}},"type":"function"}]"#
    ));
    assert!(!rendered.contains(r#""description":"#));
    assert!(!rendered.contains(r#""strict":"#));
}

#[test]
fn tool_declare_preserves_present_optional_fields() {
    let mut request = crate::request::ChatRequest::for_test();
    request.tools = vec![ChatTool {
        name: "lookup".to_string(),
        description: Some("Look up a record".to_string()),
        parameters: json!({"type": "object"}),
        strict: Some(false),
    }];

    let rendered = render_request(&request);

    assert!(rendered.contains(
        r#"[{"function":{"description":"Look up a record","name":"lookup","parameters":{"type":"object"},"strict":false},"type":"function"}]"#
    ));
}

#[test]
fn malformed_tool_arguments_render_as_raw_text() {
    let malformed = "  {\"city\":\"Beijing\"  ";
    let mut request = crate::request::ChatRequest::for_test();
    request.messages = vec![ChatMessage::assistant_blocks(vec![
        AssistantContentBlock::ToolCall(AssistantToolCall {
            id: "call-1".to_string(),
            name: "weather".to_string(),
            arguments: malformed.to_string(),
        }),
    ])];
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let rendered = render_request(&request);

    assert!(rendered.contains(&format!(
        "<|open|>json type=\"object\"<|sep|>{malformed}<|close|>json<|sep|>"
    )));
}

#[test]
fn token_writer_protects_literal_control_and_media_markers() {
    let tokenizer = Arc::new(test_tokenizer());
    let user_text = format!("literal {OPEN} and {}", super::encoding::IMAGE_PLACEHOLDER);
    let mut request = crate::request::ChatRequest::for_test();
    request.messages = vec![ChatMessage::user(vec![
        ChatContentPart::text(user_text),
        ChatContentPart::image_url("data:image/png;base64,test"),
    ])];
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), json!(false));
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let token_ids = render_token_ids(&request, tokenizer.clone());

    assert_eq!(
        token_ids.iter().filter(|&&token_id| token_id == OPEN_ID).count(),
        1
    );
    assert_eq!(
        token_ids.iter().filter(|&&token_id| token_id == MEDIA_ID).count(),
        1
    );

    let flattened = tokenizer.decode(&token_ids, false).unwrap();
    let flattened_ids = tokenizer.encode(&flattened, false).unwrap();
    assert_eq!(
        flattened_ids.iter().filter(|&&token_id| token_id == OPEN_ID).count(),
        2
    );
    assert_eq!(
        flattened_ids.iter().filter(|&&token_id| token_id == MEDIA_ID).count(),
        2
    );
}

#[test]
fn thinking_history_renders_empty_think_channel() {
    let mut request = crate::request::ChatRequest::for_test();
    request.messages = vec![
        ChatMessage::user("question"),
        ChatMessage::assistant_text("answer"),
        ChatMessage::user("follow-up"),
    ];
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let rendered = render_request(&request);

    assert!(rendered.contains(
        "<|open|>message role=\"assistant\"<|sep|>\
         <|open|>think<|sep|><|close|>think<|sep|>\
         <|open|>response<|sep|>answer<|close|>response<|sep|>"
    ));
}

#[test]
fn non_thinking_history_omits_reasoning_channel() {
    let mut request = crate::request::ChatRequest::for_test();
    request.messages = vec![
        ChatMessage::user("question"),
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "hidden reasoning".to_string(),
            },
            AssistantContentBlock::Text {
                text: "answer".to_string(),
            },
        ]),
        ChatMessage::user("follow-up"),
    ];
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), json!(false));
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let rendered = render_request(&request);

    assert!(!rendered.contains("hidden reasoning"));
    assert!(!rendered.contains("<|open|>think<|sep|>"));
    assert!(rendered.contains(
        "<|open|>message role=\"assistant\"<|sep|>\
         <|open|>response<|sep|>answer<|close|>response<|sep|>"
    ));
}

#[test]
fn partial_tool_results_keep_assistant_call_position() {
    let mut request = crate::request::ChatRequest::for_test();
    request.messages = vec![
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call-a".to_string(),
                name: "weather".to_string(),
                arguments: "{}".to_string(),
            }),
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call-b".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
            }),
        ]),
        ChatMessage::tool_response("result-b", "call-b"),
    ];
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let rendered = render_request(&request);

    assert!(rendered.contains("<|open|>call tool=\"search\" index=\"2\"<|sep|>"));
    assert!(
        rendered
            .contains("<|open|>message role=\"tool\" tool=\"search\" index=\"2\"<|sep|>result-b")
    );
}

#[test]
fn defaults_thinking_effort_to_max() {
    let rendered = render_request(&crate::request::ChatRequest::for_test());

    expect![[r#"<|open|>message role="system" type="thinking-effort"<|sep|>`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.
Now the system is invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|><|open|>message role="user"<|sep|>test<|close|>message<|sep|><|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>think<|sep|>"#]]
        .assert_eq(&rendered);
}

#[test]
fn translates_standard_thinking_kwargs() {
    let mut request = crate::request::ChatRequest::for_test();
    request
        .chat_options
        .template_kwargs
        .insert("enable_thinking".to_string(), json!(true));
    request
        .chat_options
        .template_kwargs
        .insert("reasoning_effort".to_string(), json!("high"));

    let rendered = render_request(&request);

    assert!(rendered.contains("thinking_effort=high"));
    assert!(rendered.ends_with("<|open|>think<|sep|>"));
}

#[test]
fn native_k3_kwargs_take_precedence() {
    let mut request = crate::request::ChatRequest::for_test();
    request.chat_options.template_kwargs.extend([
        ("thinking".to_string(), json!(true)),
        ("enable_thinking".to_string(), json!(false)),
        ("thinking_effort".to_string(), json!("low")),
        ("reasoning_effort".to_string(), json!("high")),
    ]);

    let rendered = render_request(&request);

    assert!(rendered.contains("thinking_effort=low"));
    assert!(!rendered.contains("thinking_effort=high"));
}

#[test]
fn standard_none_disables_thinking() {
    let mut request = crate::request::ChatRequest::for_test();
    request.chat_options.template_kwargs.extend([
        ("enable_thinking".to_string(), json!(false)),
        ("reasoning_effort".to_string(), json!("none")),
    ]);

    let rendered = render_request(&request);

    assert!(!rendered.contains("type=\"thinking-effort\""));
    assert!(rendered.ends_with("<|open|>response<|sep|>"));
}

#[test]
fn typed_none_disables_thinking() {
    let mut request = crate::request::ChatRequest::for_test();
    request.chat_options.reasoning_effort = Some(ReasoningEffort::None);

    let rendered = render_request(&request);

    assert!(!rendered.contains("type=\"thinking-effort\""));
    assert!(rendered.ends_with("<|open|>response<|sep|>"));
}

#[test]
fn rejects_removed_medium_thinking_effort() {
    let mut request = crate::request::ChatRequest::for_test();
    request
        .chat_options
        .template_kwargs
        .insert("thinking_effort".to_string(), json!("medium"));

    let error = KimiK3ChatRenderer::new(Arc::new(test_tokenizer()))
        .render(&request)
        .unwrap_err();

    expect![[r#"
        ChatTemplate(
            "unsupported thinking_effort=\"medium\"; supported values are `low`, `high`, and `max`",
        )
    "#]]
    .assert_debug_eq(&error);
}
