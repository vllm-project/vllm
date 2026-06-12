//! Harmony output tests share the upstream `openai-harmony` tiktoken cache.
//!
//! Use a file lock for tests that load the encoding so `cargo nextest` cannot
//! start multiple processes that concurrently populate the same cache file.

use std::sync::Arc;

use futures::executor::block_on;
use futures::{TryStreamExt as _, stream};
use openai_harmony::chat::{Message, Role};
use serial_test::file_serial;
use vllm_text::output::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, Finished};

use super::*;
use crate::output::ChatOutputProcessor;
use crate::request::{ChatRequest, ChatTool, ChatToolChoice};
use crate::{AssistantMessageExt, ChatEvent, FinishReason};

fn assistant_prefix() -> Vec<u32> {
    harmony_encoding()
        .unwrap()
        .render_conversation_for_completion(std::iter::empty::<&Message>(), Role::Assistant, None)
        .unwrap()
}

fn completion_tokens(messages: &[Message]) -> Vec<u32> {
    let encoding = harmony_encoding().unwrap();
    let prefix = assistant_prefix();
    let rendered = encoding.render_conversation(messages.iter(), None).unwrap();
    assert!(rendered.starts_with(&prefix));
    rendered[prefix.len()..].to_vec()
}

fn text_message(channel: &str, text: &str) -> Message {
    Message::from_role_and_content(Role::Assistant, text).with_channel(channel)
}

fn tool_message(name: &str, arguments: &str, channel: &str) -> Message {
    Message::from_role_and_content(Role::Assistant, arguments)
        .with_channel(channel)
        .with_recipient(format!("functions.{name}"))
        .with_content_type("json")
}

fn decoded_start() -> DecodedTextEvent {
    DecodedTextEvent::Start {
        prompt_token_ids: Arc::<[u32]>::from([]),
        prompt_logprobs: None,
    }
}

fn finished() -> Finished {
    Finished {
        prompt_token_count: 0,
        output_token_count: 0,
        finish_reason: FinishReason::stop_eos(),
        kv_transfer_params: None,
    }
}

async fn collect_events(
    processor: HarmonyChatOutputProcessor,
    events: Vec<DecodedTextEvent>,
) -> Vec<ChatEvent> {
    Box::new(processor)
        .process(Box::pin(stream::iter(events.into_iter().map(Ok))))
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

fn request_with_tools() -> ChatRequest {
    ChatRequest {
        tool_choice: ChatToolChoice::Auto,
        tools: vec![ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
            strict: None,
        }],
        ..ChatRequest::for_test()
    }
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn interrupted_final_message_is_preserved() {
    let tokens = completion_tokens(&[text_message("final", "hello")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens[..tokens.len() - 1].to_vec(),
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    assert_eq!(
        events.last(),
        Some(&ChatEvent::Done {
            message: crate::AssistantMessage {
                content: vec![crate::AssistantContentBlock::Text {
                    text: "hello".to_string(),
                }],
            },
            prompt_token_count: 0,
            output_token_count: 0,
            finish_reason: FinishReason::stop_eos(),
            kv_transfer_params: None,
        })
    );
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn eos_flush_preserves_trailing_replacement_text() {
    let mut tokens = completion_tokens(&[text_message("final", "Hi")]);
    tokens.pop();
    tokens.push(u32::MAX);

    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens,
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    let ChatEvent::Done { message, .. } = events.last().unwrap() else {
        panic!("expected done");
    };
    assert_eq!(message.text(), format!("Hi{}", char::REPLACEMENT_CHARACTER));
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn interrupted_analysis_message_is_preserved() {
    let tokens = completion_tokens(&[text_message("analysis", "think")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens[..tokens.len() - 1].to_vec(),
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    assert_eq!(
        events.last(),
        Some(&ChatEvent::Done {
            message: crate::AssistantMessage {
                content: vec![crate::AssistantContentBlock::Reasoning {
                    text: "think".to_string(),
                }],
            },
            prompt_token_count: 0,
            output_token_count: 0,
            finish_reason: FinishReason::stop_eos(),
            kv_transfer_params: None,
        })
    );
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn commentary_preamble_is_visible_but_commentary_tool_payload_is_not() {
    let tokens = completion_tokens(&[
        text_message("commentary", "Let me check."),
        tool_message("get_weather", r#"{"city":"Paris"}"#, "commentary"),
    ]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens,
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    let done = events.last().unwrap();
    let ChatEvent::Done { message, .. } = done else {
        panic!("expected done");
    };
    assert_eq!(message.text(), "Let me check.");
    assert_eq!(message.tool_calls().count(), 1);
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn multiple_messages_get_newline_separators() {
    let tokens = completion_tokens(&[
        text_message("analysis", "first think"),
        text_message("analysis", "second think"),
        text_message("final", "first answer"),
        text_message("final", "second answer"),
    ]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens,
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    let ChatEvent::Done { message, .. } = events.last().unwrap() else {
        panic!("expected done");
    };
    assert_eq!(
        message.reasoning().as_deref(),
        Some("first think\nsecond think")
    );
    assert_eq!(message.text(), "first answer\nsecond answer");
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn tool_calls_stream_arguments_and_finish_with_local_id_shape() {
    let tokens = completion_tokens(&[tool_message(
        "get_weather",
        r#"{"city":"Paris"}"#,
        "commentary",
    )]);
    let midpoint = tokens.len() / 2;
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens[..midpoint].to_vec(),
                logprobs: None,
                finished: None,
            },
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens[midpoint..].to_vec(),
                logprobs: None,
                finished: Some(finished()),
            },
        ],
    ));

    let mut saw_start = None;
    let mut saw_args = String::new();
    let mut saw_end = None;
    for event in &events {
        match event {
            ChatEvent::ToolCallStart { id, name, .. } => {
                assert!(id.starts_with("call_"));
                assert_eq!(name, "get_weather");
                saw_start = Some(id.clone());
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => saw_args.push_str(delta),
            ChatEvent::ToolCallEnd { call, .. } => {
                saw_end = Some(call.clone());
            }
            _ => {}
        }
    }

    let start_id = saw_start.expect("tool start");
    assert_eq!(saw_args, r#"{"city":"Paris"}"#);
    let end = saw_end.expect("tool end");
    assert_eq!(end.id, start_id);
    assert_eq!(end.arguments, r#"{"city":"Paris"}"#);
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn semantic_events_precede_same_update_logprobs() {
    let tokens = completion_tokens(&[text_message("final", "hello")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: tokens,
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs { entries: vec![] }],
                }),
                finished: Some(finished()),
            },
        ],
    ));

    let block_delta_index = events
        .iter()
        .position(|event| matches!(event, ChatEvent::BlockDelta { .. }))
        .unwrap();
    let logprobs_index = events
        .iter()
        .position(|event| matches!(event, ChatEvent::LogprobsDelta { .. }))
        .unwrap();
    assert!(block_delta_index < logprobs_index);
}

#[test]
fn rejects_generic_parser_overrides() {
    let reasoning_error =
        validate_harmony_parser_overrides(&ParserSelection::Auto, &ParserSelection::None)
            .unwrap_err();
    assert_eq!(
        reasoning_error.to_string(),
        "gpt_oss uses native Harmony output parsing; generic reasoning parser override `none` is not supported"
    );

    let tool_error = validate_harmony_parser_overrides(
        &ParserSelection::Explicit("json".to_string()),
        &ParserSelection::Auto,
    )
    .unwrap_err();
    assert_eq!(
        tool_error.to_string(),
        "gpt_oss uses native Harmony output parsing; generic tool parser override `json` is not supported"
    );
}

#[test]
#[file_serial(harmony_tiktoken_cache)]
fn allows_auto_auto_only() {
    validate_harmony_parser_overrides(&ParserSelection::Auto, &ParserSelection::Auto).unwrap();
    let _ = HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap();
}
