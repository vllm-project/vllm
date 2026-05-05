use std::fs;
use std::path::PathBuf;

use expect_test::{ExpectFile, expect, expect_file};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror_ext::AsReport;

use super::DeepSeekV32ChatRenderer;
use crate::error::Error;
use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{ChatMessage, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode};
use crate::{ChatRenderer, ChatRole};

#[derive(Debug, Deserialize)]
struct FixtureRequest {
    #[serde(default)]
    tools: Vec<FixtureTool>,
    messages: Vec<FixtureMessage>,
}

#[derive(Debug, Deserialize)]
struct FixtureTool {
    function: FixtureToolFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolFunction {
    name: String,
    description: Option<String>,
    parameters: Value,
    #[serde(default)]
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
enum FixtureMessage {
    System {
        content: String,
    },
    Developer {
        content: String,
        #[serde(default)]
        tools: Vec<FixtureTool>,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default)]
        reasoning_content: String,
        #[serde(default)]
        tool_calls: Vec<FixtureToolCall>,
    },
    Tool {
        content: String,
        #[serde(default)]
        tool_call_id: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct FixtureToolCall {
    #[serde(default)]
    id: Option<String>,
    function: FixtureToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolCallFunction {
    name: String,
    arguments: String,
}

fn render_request(request: &ChatRequest) -> String {
    DeepSeekV32ChatRenderer::new()
        .render(request)
        .unwrap()
        .prompt
        .into_text()
        .expect("deepseek renderer should return text prompt")
}

fn render_result(request: &ChatRequest) -> Result<String, Error> {
    DeepSeekV32ChatRenderer::new().render(request).map(|rendered| {
        rendered
            .prompt
            .into_text()
            .expect("deepseek renderer should return text prompt")
    })
}

fn thinking_request(messages: Vec<ChatMessage>) -> ChatRequest {
    let mut request = ChatRequest {
        request_id: "deepseek-v32-small-test".to_string(),
        messages,
        ..ChatRequest::for_test()
    };
    if matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::Assistant)
    ) {
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
    }
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request
}

fn fixture_request(input_name: &str) -> ChatRequest {
    let fixture = fs::read_to_string(fixture_path(input_name)).unwrap();
    let fixture: FixtureRequest = serde_json::from_str(&fixture).unwrap();
    let mut request = ChatRequest {
        request_id: "deepseek-v32-fixture".to_string(),
        messages: fixture
            .messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| match message {
                FixtureMessage::System { content } => ChatMessage::system(content),
                FixtureMessage::Developer { content, tools } => ChatMessage::developer(
                    content,
                    (!tools.is_empty()).then(|| to_chat_tools(&tools)),
                ),
                FixtureMessage::User { content } => ChatMessage::user(content),
                FixtureMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();
                    if !reasoning_content.is_empty() {
                        blocks.push(AssistantContentBlock::Reasoning {
                            text: reasoning_content,
                        });
                    }
                    if !content.is_empty() {
                        blocks.push(AssistantContentBlock::Text { text: content });
                    }
                    blocks.extend(tool_calls.into_iter().enumerate().map(
                        |(tool_index, tool_call)| {
                            AssistantContentBlock::ToolCall(AssistantToolCall {
                                id: tool_call.id.unwrap_or_else(|| {
                                    format!("fixture-tool-call-{index}-{tool_index}")
                                }),
                                name: tool_call.function.name,
                                arguments: tool_call.function.arguments,
                            })
                        },
                    ));
                    ChatMessage::assistant_blocks(blocks)
                }
                FixtureMessage::Tool {
                    content,
                    tool_call_id,
                } => ChatMessage::tool_response(
                    content,
                    tool_call_id.unwrap_or_else(|| format!("fixture-tool-response-{index}")),
                ),
            })
            .collect(),
        tools: to_chat_tools(&fixture.tools),
        tool_choice: if fixture.tools.is_empty() {
            ChatToolChoice::None
        } else {
            ChatToolChoice::Auto
        },
        ..ChatRequest::for_test()
    };
    if matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::Assistant)
    ) {
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
    }
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));
    request
}

fn to_chat_tools(tools: &[FixtureTool]) -> Vec<ChatTool> {
    tools
        .iter()
        .map(|tool| ChatTool {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/renderer/deepseek_v32")
        .join("fixtures")
        .join(name)
}

fn assert_fixture(input_name: &str, expected: ExpectFile) {
    let request = fixture_request(input_name);
    let rendered = render_request(&request);
    expected.assert_eq(&rendered);
}

#[test]
fn renders_vllm_parity_prompt_for_request_level_tools_fixture() {
    assert_fixture(
        "test_input.json",
        expect_file!["fixtures/test_output_vllm_parity.txt"],
    );
}

#[test]
fn renders_official_search_fixture_without_date() {
    assert_fixture(
        "test_input_search_wo_date.json",
        expect_file!["fixtures/test_output_search_wo_date.txt"],
    );
}

#[test]
fn renders_official_search_fixture_with_date() {
    assert_fixture(
        "test_input_search_w_date.json",
        expect_file!["fixtures/test_output_search_w_date.txt"],
    );
}

#[test]
fn request_level_tools_are_lowered_as_synthetic_leading_system_message() {
    let mut request = ChatRequest {
        request_id: "deepseek-v32-tools".to_string(),
        messages: vec![
            ChatMessage::system("System prompt."),
            ChatMessage::text(ChatRole::User, "Hello"),
        ],
        tools: vec![ChatTool {
            name: "lookup".to_string(),
            description: Some("Look things up".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }),
            strict: None,
        }],
        tool_choice: ChatToolChoice::Auto,
        ..ChatRequest::for_test()
    };
    request
        .chat_options
        .template_kwargs
        .insert("thinking".to_string(), Value::Bool(true));

    let rendered = render_request(&request);

    assert!(rendered.starts_with("<｜begin▁of▁sentence｜>\n\n## Tools\n"));
    assert!(rendered.contains("</functions>\nSystem prompt."));
    assert!(rendered.ends_with("<｜User｜>Hello<｜Assistant｜><think>"));
}

#[test]
fn developer_turn_is_treated_as_last_user_like_turn() {
    let request = thinking_request(vec![ChatMessage::developer("Follow policy.", None)]);

    let rendered = render_request(&request);

    assert!(rendered.contains("# The user's message is: Follow policy."));
    assert!(rendered.ends_with("<｜Assistant｜><think>"));
}

#[test]
fn historical_assistant_reasoning_is_dropped_before_final_user_turn() {
    let request = thinking_request(vec![
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "internal reasoning".to_string(),
            },
            AssistantContentBlock::Text {
                text: "Visible answer.".to_string(),
            },
        ]),
        ChatMessage::user("What about the next one?"),
    ]);

    let rendered = render_request(&request);

    assert!(!rendered.contains("internal reasoning"));
    assert!(rendered.contains("Visible answer.<｜end▁of▁sentence｜>"));
    assert!(rendered.ends_with("<｜User｜>What about the next one?<｜Assistant｜><think>"));
}

#[test]
fn historical_assistant_reasoning_is_dropped_before_final_developer_turn() {
    let request = thinking_request(vec![
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "internal reasoning".to_string(),
            },
            AssistantContentBlock::Text {
                text: "Visible answer.".to_string(),
            },
        ]),
        ChatMessage::developer("Follow the rubric.", None),
    ]);

    let rendered = render_request(&request);

    assert!(!rendered.contains("internal reasoning"));
    assert!(rendered.contains("Visible answer.<｜end▁of▁sentence｜>"));
    assert!(rendered.ends_with(
        "<｜User｜>\n\n# The user's message is: Follow the rubric.<｜Assistant｜><think>"
    ));
}

#[test]
fn tool_results_after_last_user_resume_thinking() {
    let request = thinking_request(vec![
        ChatMessage::user("Check the weather."),
        ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(AssistantToolCall {
            id: "call-weather".to_string(),
            name: "weather".to_string(),
            arguments: "{\"city\":\"Hangzhou\"}".to_string(),
        })]),
        ChatMessage::tool_response("{\"ok\":true}", "call-weather"),
    ]);

    let rendered = render_request(&request);

    assert!(rendered.contains(
        "<｜User｜>Check the weather.<｜Assistant｜><think></think>\n\n<｜DSML｜function_calls>"
    ));
    assert!(rendered.ends_with("</function_results>\n\n<think>"));
}

#[test]
fn tool_results_follow_assistant_tool_call_id_order() {
    let request = thinking_request(vec![
        ChatMessage::user("Check two cities."),
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call-hangzhou".to_string(),
                name: "weather".to_string(),
                arguments: "{\"city\":\"Hangzhou\"}".to_string(),
            }),
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call-beijing".to_string(),
                name: "weather".to_string(),
                arguments: "{\"city\":\"Beijing\"}".to_string(),
            }),
        ]),
        ChatMessage::tool_response("{\"city\":\"Beijing\"}", "call-beijing"),
        ChatMessage::tool_response("{\"city\":\"Hangzhou\"}", "call-hangzhou"),
    ]);

    let rendered = render_request(&request);

    assert!(rendered.contains(
        "<function_results>\n<result>{\"city\":\"Hangzhou\"}</result>\n<result>{\"city\":\"Beijing\"}</result>\n</function_results>"
    ));
}

#[test]
fn tool_results_require_matching_tool_call_ids() {
    let request = thinking_request(vec![
        ChatMessage::user("Check the weather."),
        ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(AssistantToolCall {
            id: "call-weather".to_string(),
            name: "weather".to_string(),
            arguments: "{\"city\":\"Hangzhou\"}".to_string(),
        })]),
        ChatMessage::tool_response("{\"ok\":true}", "call-unknown"),
    ]);

    let error = render_result(&request).unwrap_err();

    expect!["chat template error: invalid DeepSeek V3.2 tool message: unknown tool_call_id `call-unknown`"]
        .assert_eq(&error.to_report_string());
}

#[test]
fn assistant_after_last_user_requires_reasoning_or_tool_calls() {
    let request = thinking_request(vec![
        ChatMessage::user("Hello"),
        ChatMessage::assistant_text("Hi there."),
    ]);

    let error = render_result(&request).unwrap_err();

    expect!["chat template error: invalid DeepSeek V3.2 assistant message after last user message: expected reasoning or tool calls"]
        .assert_eq(&error.to_report_string());
}
