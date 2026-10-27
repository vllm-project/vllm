// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use futures::executor::block_on;
use futures::{TryStreamExt as _, stream};
use openai_harmony::chat::{Message, Role};
use vllm_text::output::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedText, DecodedTextEvent, Finished, SampledDelta,
};

use super::*;
use crate::event::AssistantBlockKind;
use crate::output::ChatOutputProcessor;
use crate::request::{ChatRequest, ChatTool, ChatToolChoice, ResolvedToolContext};
use crate::{AssistantMessageExt, ChatEvent, ChatTokenUsage, FinishReason};

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
        usage: vllm_llm::TokenUsage {
            prompt_token_count: 0,
            output_token_count: 0,
            cached_token_count: 0,
        },
        finish_reason: FinishReason::stop_eos(),
        kv_transfer_params: None,
        ec_transfer_params: None,
        sampling_mask: None,
        timestamps: Default::default(),
    }
}

fn decoded_tokens(
    token_ids: Vec<u32>,
    logprobs: Option<DecodedLogprobs>,
    finished: Option<Finished>,
) -> DecodedTextEvent {
    DecodedTextEvent::TextDelta {
        decoded: DecodedText::default(),
        sampled: SampledDelta {
            token_ids,
            logprobs,
        },
        finished: finished.map(Box::new),
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
    let tools = vec![ChatTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
        strict: None,
        defer_loading: None,
    }];
    ChatRequest {
        tool_context: ResolvedToolContext::new(&[], tools, Some(ChatToolChoice::Auto), true)
            .expect("tool context should resolve"),
        ..ChatRequest::for_test()
    }
}

#[test]
fn interrupted_final_message_is_preserved() {
    let tokens = completion_tokens(&[text_message("final", "hello")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens[..tokens.len() - 1].to_vec(), None, Some(finished())),
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
            usage: ChatTokenUsage::from(vllm_llm::TokenUsage {
                prompt_token_count: 0,
                output_token_count: 0,
                cached_token_count: 0,
            }),
            timestamps: Default::default(),
            finish_reason: FinishReason::stop_eos(),
            kv_transfer_params: None,
            ec_transfer_params: None,
        })
    );
}

#[test]
fn eos_flush_preserves_trailing_replacement_text() {
    let mut tokens = completion_tokens(&[text_message("final", "Hi")]);
    tokens.pop();
    tokens.push(u32::MAX);

    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens, None, Some(finished())),
        ],
    ));

    let ChatEvent::Done { message, .. } = events.last().unwrap() else {
        panic!("expected done");
    };
    assert_eq!(message.text(), format!("Hi{}", char::REPLACEMENT_CHARACTER));
}

#[test]
fn interrupted_analysis_message_is_preserved() {
    let tokens = completion_tokens(&[text_message("analysis", "think")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens[..tokens.len() - 1].to_vec(), None, Some(finished())),
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
            usage: ChatTokenUsage {
                engine: vllm_llm::TokenUsage {
                    prompt_token_count: 0,
                    output_token_count: 0,
                    cached_token_count: 0,
                },
                reasoning_tokens: 1,
            },
            timestamps: Default::default(),
            finish_reason: FinishReason::stop_eos(),
            kv_transfer_params: None,
            ec_transfer_params: None,
        })
    );
}

#[test]
fn commentary_preamble_is_visible_but_commentary_tool_payload_is_not() {
    let tokens = completion_tokens(&[
        text_message("commentary", "Let me check."),
        tool_message("get_weather", r#"{"city":"Paris"}"#, "commentary"),
    ]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens, None, Some(finished())),
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
            decoded_tokens(tokens, None, Some(finished())),
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
            decoded_tokens(tokens[..midpoint].to_vec(), None, None),
            decoded_tokens(tokens[midpoint..].to_vec(), None, Some(finished())),
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

fn content_token_count(text: &str) -> usize {
    harmony_encoding().unwrap().tokenizer().encode_ordinary(text).len()
}

fn reasoning_token_counts(events: &[ChatEvent]) -> (Vec<usize>, usize) {
    let deltas = events
        .iter()
        .filter_map(|event| match event {
            ChatEvent::BlockDelta {
                kind: AssistantBlockKind::Reasoning,
                token_count,
                ..
            // The newline separating two reasoning messages has no tokens.
            } => Some(token_count.unwrap_or_default()),
            _ => None,
        })
        .collect();
    let ChatEvent::Done { usage, .. } = events.last().unwrap() else {
        panic!("expected done");
    };
    (deltas, usage.reasoning_tokens)
}

#[test]
fn reasoning_tokens_count_analysis_and_tool_call_content() {
    let analysis = "Let me check the weather.";
    let arguments = r#"{"city":"Paris"}"#;
    let tokens = completion_tokens(&[
        text_message("analysis", analysis),
        tool_message("get_weather", arguments, "commentary"),
    ]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&request_with_tools()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens, None, Some(finished())),
        ],
    ));

    // Like Python, content of `analysis` and of `commentary` with a recipient
    // counts, Harmony control and header tokens do not. Tool-call tokens are
    // only in the final usage.
    let (deltas, total) = reasoning_token_counts(&events);
    assert_eq!(deltas.iter().sum::<usize>(), content_token_count(analysis));
    assert_eq!(
        total,
        content_token_count(analysis) + content_token_count(arguments)
    );
}

#[test]
fn reasoning_tokens_exclude_final_and_preamble_content() {
    let tokens = completion_tokens(&[
        text_message("commentary", "Let me check."),
        text_message("final", "Hello"),
    ]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(tokens, None, Some(finished())),
        ],
    ));

    assert_eq!(reasoning_token_counts(&events), (vec![], 0));
}

fn stream_one_token_per_update(messages: &[Message]) -> Vec<ChatEvent> {
    let tokens = completion_tokens(messages);
    let last = tokens.len() - 1;
    let mut decoded = vec![decoded_start()];
    for (i, token) in tokens.into_iter().enumerate() {
        decoded.push(decoded_tokens(
            vec![token],
            None,
            (i == last).then(finished),
        ));
    }
    block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        decoded,
    ))
}

#[test]
fn reasoning_tokens_are_attributed_to_their_streamed_deltas() {
    let first = "Let me think about this.";
    let second = "Then answer.";
    let events = stream_one_token_per_update(&[
        text_message("analysis", first),
        text_message("analysis", second),
        text_message("final", "Hi"),
    ]);

    // One delta per content token, and the separating newline carries none.
    let mut expected = vec![1; content_token_count(first)];
    expected.push(0);
    expected.extend(vec![1; content_token_count(second)]);
    let total = expected.iter().sum();
    assert_eq!(reasoning_token_counts(&events), (expected, total));
}

#[test]
fn reasoning_tokens_of_a_split_character_join_its_delta() {
    // Each of these characters spans several byte-level tokens, so the first
    // tokens of a character produce no text delta of their own.
    let text = "\u{1D518}\u{1D52B}";
    let tokens = content_token_count(text);
    assert!(tokens > text.chars().count());
    let events =
        stream_one_token_per_update(&[text_message("analysis", text), text_message("final", "Hi")]);

    let (deltas, total) = reasoning_token_counts(&events);
    assert_eq!(deltas.len(), text.chars().count());
    assert_eq!(deltas.iter().sum::<usize>(), tokens);
    assert_eq!(total, tokens);
}

#[test]
fn semantic_events_precede_same_update_logprobs() {
    let tokens = completion_tokens(&[text_message("final", "hello")]);
    let events = block_on(collect_events(
        HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap(),
        vec![
            decoded_start(),
            decoded_tokens(
                tokens,
                Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs { entries: vec![] }],
                }),
                Some(finished()),
            ),
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
fn allows_auto_auto_only() {
    validate_harmony_parser_overrides(&ParserSelection::Auto, &ParserSelection::Auto).unwrap();
    let _ = HarmonyChatOutputProcessor::new(&ChatRequest::for_test()).unwrap();
}
