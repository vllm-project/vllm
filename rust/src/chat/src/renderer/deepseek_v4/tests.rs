use std::path::PathBuf;

use expect_test::{ExpectFile, expect, expect_file};
use serde_json::Value;

use super::DeepSeekV4ChatRenderer;
use crate::ChatRenderer;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::renderer::test_utils::{FixtureRequestOptions, fixture_chat_request};
use crate::request::{ChatMessage, ChatRequest, GenerationPromptMode, ReasoningEffort};

fn render_request(request: &ChatRequest) -> String {
    DeepSeekV4ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
        .into_text()
        .expect("deepseek v4 renderer should return text prompt")
}

fn fixture_request(input_name: &str) -> ChatRequest {
    fixture_chat_request(&fixture_path(input_name), deepseek_fixture_options())
}

fn deepseek_fixture_options() -> FixtureRequestOptions {
    FixtureRequestOptions {
        enable_thinking: true,
        no_generation_prompt_when_last_assistant: true,
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderer/deepseek_v4")
        .join("fixtures")
        .join(name)
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = render_request(&request);
    expected.assert_eq(&rendered);
}

#[test]
fn renders_v4_fixture_1_tool_call_round_trip() {
    assert_fixture(
        "test_input_1.json",
        expect_file!["fixtures/test_output_1.txt"],
    );
}

#[test]
fn renders_v4_fixture_2_multi_turn_drop_thinking() {
    assert_fixture(
        "test_input_2.json",
        expect_file!["fixtures/test_output_2.txt"],
    );
}

#[test]
fn reasoning_effort_max_adds_prefix_when_thinking_is_enabled() {
    let mut request = ChatRequest {
        messages: vec![ChatMessage::user("solve it")],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request.chat_options.reasoning_effort = Some(ReasoningEffort::Max);

    let rendered = render_request(&request);

    expect![[r#"
        <｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

        <｜User｜>solve it<｜Assistant｜><think>"#]]
    .assert_eq(&rendered);
}

#[test]
fn reasoning_effort_none_disables_thinking() {
    let mut request = ChatRequest {
        messages: vec![ChatMessage::user("answer directly")],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request.chat_options.reasoning_effort = Some(ReasoningEffort::None);

    let rendered = render_request(&request);

    expect!["<｜begin▁of▁sentence｜><｜User｜>answer directly<｜Assistant｜></think>"]
        .assert_eq(&rendered);
}

#[test]
fn reasoning_effort_template_kwarg_is_ignored() {
    let mut request = ChatRequest {
        messages: vec![ChatMessage::user("solve it")],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request.chat_options.template_kwargs.insert(
        "reasoning_effort".to_string(),
        Value::String("max".to_string()),
    );

    let rendered = render_request(&request);

    expect!["<｜begin▁of▁sentence｜><｜User｜>solve it<｜Assistant｜><think>"].assert_eq(&rendered);
}

#[test]
fn tool_results_are_sorted_by_previous_assistant_tool_call_order() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: "second".to_string(),
                    name: "second_tool".to_string(),
                    arguments: "{}".to_string(),
                }),
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: "first".to_string(),
                    name: "first_tool".to_string(),
                    arguments: "{}".to_string(),
                }),
            ]),
            ChatMessage::tool_response("first result", "first"),
            ChatMessage::tool_response("second result", "second"),
        ],
        ..ChatRequest::for_test()
    };

    let rendered = render_request(&request);

    expect![[r#"
        <｜begin▁of▁sentence｜>

        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="second_tool">

        </｜DSML｜invoke>
        <｜DSML｜invoke name="first_tool">

        </｜DSML｜invoke>
        </｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>second result</tool_result>

        <tool_result>first result</tool_result><｜Assistant｜></think>"#]]
    .assert_eq(&rendered);
}

#[test]
fn drop_thinking_false_keeps_prior_assistant_reasoning() {
    let mut request = ChatRequest {
        messages: vec![
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "old reasoning".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "old answer".to_string(),
                },
            ]),
            ChatMessage::user("next"),
        ],
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request
        .chat_options
        .template_kwargs
        .insert("drop_thinking".to_string(), Value::Bool(false));

    let rendered = render_request(&request);

    expect!(
        "<｜begin▁of▁sentence｜>old reasoning</think>old answer<｜end▁of▁sentence｜><｜User｜>next<｜Assistant｜><think>"
    )
    .assert_eq(&rendered);
}

#[test]
fn continue_final_assistant_omits_final_eos() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage::user("write"),
            ChatMessage::assistant_text("partial answer"),
        ],
        chat_options: crate::request::ChatOptions {
            generation_prompt_mode: GenerationPromptMode::ContinueFinalAssistant,
            ..Default::default()
        },
        ..ChatRequest::for_test()
    };

    let rendered = render_request(&request);

    expect!["<｜begin▁of▁sentence｜><｜User｜>write<｜Assistant｜></think>partial answer"]
        .assert_eq(&rendered);
}
