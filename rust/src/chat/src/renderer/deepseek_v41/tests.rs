// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;

use expect_test::{expect, expect_file};
use serde_json::{Value, json};

use super::DeepSeekV41ChatRenderer;
use crate::ChatRenderer;
use crate::event::AssistantContentBlock;
use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest, ReasoningEffort};

fn render(request: &ChatRequest) -> String {
    DeepSeekV41ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
        .into_text()
        .unwrap()
}

fn request() -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage::user("question")],
        ..ChatRequest::for_test()
    }
}

#[test]
fn renders_shared_deepseek_fixtures_with_v41_reference_encoding() {
    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/renderer/deepseek_v4/fixtures");
    for (input, expected) in [
        (
            "test_input_1.json",
            expect_file!["fixtures/test_output_1.txt"],
        ),
        (
            "test_input_2.json",
            expect_file!["fixtures/test_output_2.txt"],
        ),
        (
            "test_input_developer_tools.json",
            expect_file!["fixtures/test_output_developer_tools.txt"],
        ),
    ] {
        let request = fixture_chat_request(
            &fixture_dir.join(input),
            FixtureRequestOptions {
                enable_thinking: Some(true),
                no_generation_prompt_when_last_assistant: true,
            },
        );
        expected.assert_eq(&render(&request));
    }
}

#[test]
fn inlines_image_placeholder_at_image_part_positions() {
    // The placeholder is unconditional (matching the Python encoding's
    // `IMAGE_PLACEHOLDER`); requests to a multimodal-less backend fail later,
    // at media preparation.
    let renderer = DeepSeekV41ChatRenderer::new();
    let request = ChatRequest {
        messages: vec![ChatMessage::User {
            content: ChatContent::Parts(vec![
                ChatContentPart::image_url("https://example.com/carrots.jpeg"),
                ChatContentPart::text("what is in the image?"),
            ]),
        }],
        ..ChatRequest::for_test()
    };

    let rendered = renderer.render(&request).unwrap().prompt.into_text().unwrap();

    // Image parts inline the placeholder at their position; parts stay
    // separated by "\n\n" (the V4.1 reference encoding's part separator).
    assert!(
        rendered.contains("<｜User｜><｜deepseek_image｜>\n\nwhat is in the image?"),
        "unexpected prompt: {rendered:?}"
    );
}

#[test]
fn maps_reasoning_effort_to_reference_numeric_budget() {
    for (effort, budget) in [
        (None, 50),
        (Some(ReasoningEffort::Low), 25),
        (Some(ReasoningEffort::High), 50),
        (Some(ReasoningEffort::XHigh), 75),
        (Some(ReasoningEffort::Max), 100),
    ] {
        let mut request = request();
        request.chat_options.reasoning_effort = effort;
        assert_eq!(
            render(&request),
            format!(
                "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>question<｜Assistant｜><think>"
            )
        );
    }
}

#[test]
fn accepts_numeric_template_effort_with_top_level_precedence() {
    for (effort, budget) in [
        (json!(1), 1),
        (json!(42), 42),
        (json!(100), 100),
        (json!("low"), 25),
    ] {
        let mut request = request();
        request.chat_options.template_kwargs.insert("reasoning_effort".into(), effort);
        assert!(render(&request).contains(&format!("Reasoning Effort: {budget} (range 1-100")));

        request.chat_options.reasoning_effort = Some(ReasoningEffort::XHigh);
        assert!(render(&request).contains("Reasoning Effort: 75 (range 1-100"));
    }
}

#[test]
fn rejects_undefined_effort_names_and_invalid_numeric_budgets() {
    for effort in [
        json!("medium"),
        json!("minimal"),
        json!(-1),
        json!(0),
        json!(101),
        json!(true),
        json!(1.5),
    ] {
        let mut request = request();
        request.chat_options.template_kwargs.insert("reasoning_effort".into(), effort);
        let error = DeepSeekV41ChatRenderer::new().render(&request).unwrap_err();
        assert!(error.is_request_validation_error());
    }
    for effort in [ReasoningEffort::Minimal, ReasoningEffort::Medium] {
        let mut request = request();
        request.chat_options.reasoning_effort = Some(effort);
        let error = DeepSeekV41ChatRenderer::new().render(&request).unwrap_err();
        assert!(error.is_request_validation_error());
    }
}

#[test]
fn chat_mode_and_none_effort_use_closed_thinking_prefix() {
    let mut request = request();
    request.chat_options.template_kwargs.insert("thinking".into(), json!(false));
    let expected = expect!["<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"];
    expected.assert_eq(&render(&request));

    request.chat_options.template_kwargs.insert("thinking".into(), json!(true));
    request.chat_options.reasoning_effort = Some(ReasoningEffort::None);
    expected.assert_eq(&render(&request));
}

#[test]
fn leading_system_in_chat_mode_uses_system_token() {
    let mut request = request();
    request.messages.insert(0, ChatMessage::system("policy"));
    request.chat_options.template_kwargs.insert("thinking".into(), json!(false));
    expect!["<｜begin▁of▁sentence｜><｜System｜>policy<｜User｜>question<｜Assistant｜></think>"]
        .assert_eq(&render(&request));
}

#[test]
fn system_after_dropped_developer_becomes_leading_system() {
    let mut request = ChatRequest {
        messages: vec![
            ChatMessage::developer("old policy", None),
            ChatMessage::system("sys"),
        ],
        ..ChatRequest::for_test()
    };
    expect!["<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\nsys"]
        .assert_eq(&render(&request));

    request.messages.push(ChatMessage::user("question"));
    expect!["<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\nsys<｜User｜>question<｜Assistant｜><think>"]
        .assert_eq(&render(&request));
}

#[test]
fn mid_system_advances_last_user_boundary_and_drops_old_reasoning() {
    let mut request = ChatRequest {
        messages: vec![
            ChatMessage::system("sys"),
            ChatMessage::user("q1"),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning { text: "r1".into() },
                AssistantContentBlock::Text { text: "a1".into() },
            ]),
            ChatMessage::developer("old policy", None),
            ChatMessage::system("mid sys"),
        ],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("reasoning_effort".into(), json!(88));
    expect!["<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 88 (range 1-100, the higher the value, the more thorough the reasoning)\n\nsys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>"]
        .assert_eq(&render(&request));
}

#[test]
fn joins_text_parts_with_reference_separator_and_preserves_literal_tags() {
    let mut request = ChatRequest {
        messages: vec![ChatMessage::user(ChatContent::Parts(vec![
            ChatContentPart::text("first"),
            ChatContentPart::text("<｜DSML｜invoke name=\"literal\">"),
        ]))],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".into(), Value::Bool(false));
    expect!["<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜DSML｜invoke name=\"literal\"><｜Assistant｜></think>"]
        .assert_eq(&render(&request));
}
