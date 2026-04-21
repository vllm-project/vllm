//! Adapts decoded text updates into reasoning-aware assistant deltas.
//!
//! This stage sits between low-level token decoding and final block assembly.
//! It is the only place in the new pipeline that understands reasoning
//! separation: `decoded.rs` still only produces plain text deltas, while later
//! stages consume the semantic `Text` / `Reasoning` split emitted here.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use thiserror_ext::AsReport;
use tracing::warn;
use vllm_text::output::DecodedTextEvent;

use super::ContentEvent;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::DecodedTextEventStream;
use crate::parser::reasoning::{ReasoningDelta, ReasoningParser};

/// Per-stream reasoning parsing state.
struct ReasoningState {
    /// Reasoning parser for the current model family.
    parser: Box<dyn ReasoningParser>,
    /// Whether reasoning parsing has already failed for this stream.
    parser_failed: bool,
}

impl ReasoningState {
    /// Create one fresh reasoning-adaptation state for a new streamed response.
    fn new(parser: Box<dyn ReasoningParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
        }
    }

    /// Convert one decoded text delta into zero or more semantic assistant deltas.
    fn process_delta(&mut self, delta: String) -> Vec<ContentEvent> {
        // If the parser has already failed, skip parsing and return plain text deltas.
        if self.parser_failed {
            return vec![ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            }];
        }

        let mut events = Vec::new();

        match self.parser.push(&delta) {
            Ok(result) => {
                push_reasoning_delta(&mut events, result);
            }
            Err(error) => {
                if !self.parser_failed {
                    warn!(
                        error = %error.as_report(),
                        "reasoning parser failed; falling back to plain text deltas"
                    );
                    self.parser_failed = true;
                }
                push_text_delta(&mut events, AssistantBlockKind::Text, delta);
            }
        }

        events
    }

    /// Initialize parser state once prompt token IDs are available.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        if self.parser_failed {
            return;
        }

        match self.parser.initialize(prompt_token_ids) {
            Ok(()) => {}
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "failed to initialize reasoning parser; falling back to plain text deltas"
                );
                self.parser_failed = true;
            }
        }
    }

    /// Flush any parser-held partial delimiter state at end of stream.
    fn finish(&mut self) -> Vec<ContentEvent> {
        if self.parser_failed {
            return Vec::new();
        }

        match self.parser.finish() {
            Ok(result) => {
                let mut events = Vec::new();
                push_reasoning_delta(&mut events, result);
                events
            }
            Err(error) => {
                warn!(error = %error.as_report(), "failed to flush reasoning parser state");
                Vec::new()
            }
        }
    }
}

/// Push one semantic text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<ContentEvent>, kind: AssistantBlockKind, delta: String) {
    if delta.is_empty() {
        return;
    }
    events.push(ContentEvent::TextDelta { kind, delta });
}

/// Convert one parsed reasoning delta into zero or more content events.
fn push_reasoning_delta(events: &mut Vec<ContentEvent>, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        push_text_delta(events, AssistantBlockKind::Reasoning, reasoning);
    }
    if let Some(content) = delta.content {
        push_text_delta(events, AssistantBlockKind::Text, content);
    }
}

/// Wrap one decoded-text stream into the internal reasoning event stream.
#[try_stream(ok = ContentEvent, error = Error)]
pub(crate) async fn reasoning_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
) {
    pin_mut!(decoded_stream);

    // Without a parser, pass through as plain text deltas.
    let Some(reasoning_parser) = reasoning_parser else {
        while let Some(event) = decoded_stream.next().await.transpose()? {
            for next in ContentEvent::from_decoded_plain_text(event) {
                yield next;
            }
        }
        return Ok(());
    };

    let mut state = ReasoningState::new(reasoning_parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                state.initialize(&prompt_token_ids);
                yield ContentEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                }
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                for next in state.process_delta(delta) {
                    yield next;
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    yield ContentEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    };
                }
                if let Some(finished) = finished {
                    for next in state.finish() {
                        yield next;
                    }
                    yield ContentEvent::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use futures::{StreamExt as _, stream};
    use vllm_llm::FinishReason;
    use vllm_text::output::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, DecodedTokenLogprob,
    };
    use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::super::ContentEvent;
    use super::reasoning_event_stream;
    use crate::event::AssistantBlockKind;
    use crate::parser::reasoning::{
        ReasoningDelta, ReasoningError, ReasoningParser, ReasoningParserFactory, names,
    };

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_text::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<think>" => Some(1),
                "</think>" => Some(2),
                _ => None,
            }
        }
    }

    struct FailingReasoningParser {
        fail_next: bool,
    }

    impl ReasoningParser for FailingReasoningParser {
        fn create(_tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>, ReasoningError>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self { fail_next: true }))
        }

        fn push(&mut self, _text: &str) -> Result<ReasoningDelta, ReasoningError> {
            if self.fail_next {
                self.fail_next = false;
                return Err(ReasoningError::MissingToken {
                    token: "<think>".to_string(),
                });
            }
            Ok(ReasoningDelta::default())
        }
    }

    fn test_reasoning_parser(factory: &mut ReasoningParserFactory) -> Box<dyn ReasoningParser> {
        factory.register_parser::<FailingReasoningParser>("failing");

        factory.create("failing", Arc::new(FakeTokenizer)).unwrap()
    }

    #[tokio::test]
    async fn reasoning_parser_failure_falls_back_to_plain_text() {
        let mut factory = ReasoningParserFactory::new();
        let events = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2, 3].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "abc".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "def".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: Some(vllm_text::Finished {
                    prompt_token_count: 3,
                    output_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let collected = reasoning_event_stream(events, Some(test_reasoning_parser(&mut factory)))
            .collect::<Vec<_>>()
            .await;

        let events = collected
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .expect("reasoning stream should not fail");

        assert_eq!(
            events,
            vec![
                ContentEvent::Start {
                    prompt_token_ids: vec![1, 2, 3].into(),
                    prompt_logprobs: None,
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "abc".to_string(),
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "def".to_string(),
                },
                ContentEvent::Done {
                    prompt_token_count: 3,
                    output_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                },
            ]
        );
    }

    #[tokio::test]
    async fn reasoning_stream_preserves_logprobs_delta() {
        let events = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "abc".to_string(),
                token_ids: vec![],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                finished: None,
            }),
        ]);

        let collected = reasoning_event_stream(events, None)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert_eq!(
            collected,
            vec![
                ContentEvent::Start {
                    prompt_token_ids: vec![1].into(),
                    prompt_logprobs: None,
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "abc".to_string(),
                },
                ContentEvent::LogprobsDelta {
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "a".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            }],
                        }],
                    }),
                    token_ids: vec![],
                },
            ]
        );
    }

    #[tokio::test]
    async fn qwen3_parser_uses_prompt_end_marker_to_switch_to_content() {
        let tokenizer = Arc::new(FakeTokenizer);
        let events = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![2].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "thought ".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "done</think>OK".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: None,
            }),
        ]);

        let factory = ReasoningParserFactory::new();
        let collected = reasoning_event_stream(
            events,
            Some(factory.create(names::QWEN3, tokenizer).unwrap()),
        )
        .collect::<Vec<_>>()
        .await;

        let events = collected
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .expect("reasoning stream should not fail");

        assert_eq!(
            events,
            vec![
                ContentEvent::Start {
                    prompt_token_ids: vec![2].into(),
                    prompt_logprobs: None,
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "thought ".to_string(),
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "done</think>OK".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn qwen3_parser_tolerates_prompt_prefill_reasoning() {
        let tokenizer = Arc::new(FakeTokenizer);
        let events = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "thought ".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "done</think>OK".to_string(),
                token_ids: vec![],
                logprobs: None,
                finished: None,
            }),
        ]);

        let factory = ReasoningParserFactory::new();
        let collected = reasoning_event_stream(
            events,
            Some(factory.create(names::QWEN3, tokenizer).unwrap()),
        )
        .collect::<Vec<_>>()
        .await;

        let events = collected
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .expect("reasoning stream should not fail");

        assert_eq!(
            events,
            vec![
                ContentEvent::Start {
                    prompt_token_ids: vec![1].into(),
                    prompt_logprobs: None,
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Reasoning,
                    delta: "thought ".to_string(),
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Reasoning,
                    delta: "done".to_string(),
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "OK".to_string(),
                },
            ]
        );
    }
}
