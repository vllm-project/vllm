// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Golden fixtures generated from HF remote-code `encoding_k3.py`.

use std::path::PathBuf;

use expect_test::{expect, expect_file};
use serde_json::json;

use super::KimiK3ChatRenderer;
use crate::AssistantContentBlock;
use crate::ChatRenderer;
use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};
use crate::request::{ChatMessage, GenerationPromptMode, ReasoningEffort};

fn render_request(request: &crate::request::ChatRequest) -> String {
    KimiK3ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
        .into_text()
        .expect("kimi k3 renderer should return text prompt")
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

    let error = KimiK3ChatRenderer::new().render(&request).unwrap_err();

    expect![[r#"
        ChatTemplate(
            "unsupported thinking_effort=\"medium\"; supported values are `low`, `high`, and `max`",
        )
    "#]]
    .assert_debug_eq(&error);
}
