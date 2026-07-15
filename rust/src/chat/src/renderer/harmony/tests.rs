// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;

use expect_test::{ExpectFile, expect, expect_file};
use thiserror_ext::AsReport as _;

use super::HarmonyChatRenderer;
use super::encoding::harmony_encoding;
use crate::ChatRenderer;
use crate::error::Error;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};
use crate::request::{
    ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, ReasoningEffort,
};

const PINNED_DATE: &str = "2025-06-28";

fn fixture_request(input_name: &str) -> ChatRequest {
    fixture_chat_request(
        &fixture_path(input_name),
        FixtureRequestOptions {
            enable_thinking: false,
            no_generation_prompt_when_last_assistant: false,
        },
    )
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderer/harmony")
        .join("fixtures")
        .join(name)
}

fn test_renderer(use_system_instructions: bool) -> HarmonyChatRenderer {
    HarmonyChatRenderer::with_options(PINNED_DATE, use_system_instructions).unwrap()
}

fn render_token_ids(request: &ChatRequest) -> Vec<u32> {
    render_token_ids_with(&test_renderer(false), request)
}

fn render_token_ids_with(renderer: &HarmonyChatRenderer, request: &ChatRequest) -> Vec<u32> {
    renderer
        .render(request)
        .unwrap()
        .prompt
        .into_token_ids()
        .expect("Harmony renderer returns token IDs")
}

fn render_prompt_text(request: &ChatRequest) -> String {
    render_prompt_text_with(&test_renderer(false), request)
}

fn render_prompt_text_with(renderer: &HarmonyChatRenderer, request: &ChatRequest) -> String {
    let token_ids = render_token_ids_with(renderer, request);
    harmony_encoding().unwrap().tokenizer().decode_utf8(&token_ids).unwrap()
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = format!("{}\n", render_prompt_text(&request));
    expected.assert_eq(&rendered);
}

#[test]
fn renders_token_ids() {
    let request = fixture_request("simple_user.json");

    assert!(!render_token_ids(&request).is_empty());
}

#[test]
fn renders_simple_user_fixture() {
    assert_fixture("simple_user.json", expect_file!["fixtures/simple_user.txt"]);
}

#[test]
fn renders_leading_system_fixture() {
    assert_fixture(
        "leading_system.json",
        expect_file!["fixtures/leading_system.txt"],
    );
}

#[test]
fn renders_system_instructions_env_fixture() {
    let renderer = test_renderer(true);
    let request = fixture_request("leading_system.json");
    let rendered = format!("{}\n", render_prompt_text_with(&renderer, &request));
    expect_file!["fixtures/system_instructions_env.txt"].assert_eq(&rendered);
}

#[test]
fn renders_request_tools_fixture() {
    assert_fixture(
        "request_tools.json",
        expect_file!["fixtures/request_tools.txt"],
    );
}

#[test]
fn renders_developer_tools_fixture() {
    assert_fixture(
        "developer_tools.json",
        expect_file!["fixtures/developer_tools.txt"],
    );
}

#[test]
fn renders_assistant_history_fixture() {
    assert_fixture(
        "assistant_history.json",
        expect_file!["fixtures/assistant_history.txt"],
    );
}

#[test]
fn renders_tool_roundtrip_fixture() {
    assert_fixture(
        "tool_roundtrip.json",
        expect_file!["fixtures/tool_roundtrip.txt"],
    );
}

#[test]
fn drops_stale_analysis_fixture() {
    assert_fixture(
        "drop_analysis.json",
        expect_file!["fixtures/drop_analysis.txt"],
    );
}

#[test]
fn rejects_invalid_reasoning_effort() {
    let mut request = ChatRequest::for_test();
    request.chat_options.reasoning_effort = Some(ReasoningEffort::None);

    let error = test_renderer(false).render(&request).unwrap_err();

    expect![[r#"chat template error: reasoning_effort="none" is not supported by Harmony. Supported values are: low, medium, high."#]]
        .assert_eq(&error.to_report_string());
}

#[test]
fn rejects_unknown_tool_response_id() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(
                AssistantToolCall {
                    id: "call-known".to_string(),
                    name: "lookup".to_string(),
                    arguments: "{}".to_string(),
                },
            )]),
            ChatMessage::tool_response("{}", "call-unknown"),
        ],
        ..ChatRequest::for_test()
    };

    let error = test_renderer(false).render(&request).unwrap_err();

    expect![
        "chat template error: invalid Harmony tool message: unknown tool_call_id `call-unknown`"
    ]
    .assert_eq(&error.to_report_string());
}

#[test]
fn rejects_multimodal_input() {
    let request = ChatRequest {
        messages: vec![ChatMessage::user(vec![ChatContentPart::image_url(
            "data:image/png;base64,test",
        )])],
        ..ChatRequest::for_test()
    };

    let error = test_renderer(false).render(&request).unwrap_err();

    assert!(matches!(
        error,
        Error::UnsupportedMultimodalContent("image_url")
    ));
}

#[test]
fn rejects_continue_final_assistant() {
    let mut request = ChatRequest {
        messages: vec![
            ChatMessage::user("write"),
            ChatMessage::assistant_text("partial"),
        ],
        ..ChatRequest::for_test()
    };
    request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

    let error = test_renderer(false).render(&request).unwrap_err();

    expect!["chat template error: Harmony renderer does not support continue_final_message"]
        .assert_eq(&error.to_report_string());
}

#[test]
fn no_generation_prompt_omits_trailing_assistant_start() {
    let mut request = fixture_request("simple_user.json");
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

    let rendered = render_prompt_text(&request);

    assert!(!rendered.ends_with("<|start|>assistant"));
}
