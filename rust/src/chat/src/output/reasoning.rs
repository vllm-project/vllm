//! Adapts decoded text updates into reasoning-aware assistant deltas.
//!
//! This stage sits between low-level token decoding and final block assembly.
//! It is the only place in the new pipeline that understands reasoning
//! separation: `decoded.rs` still only produces plain text deltas, while later
//! stages consume the semantic `Text` / `Reasoning` split emitted here.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use reasoning_parser::ReasoningParser;
use thiserror_ext::AsReport;
use tracing::warn;
use vllm_text::output::DecodedTextEvent;

use super::ContentEvent;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::DecodedTextEventStream;

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

        match self.parser.parse_reasoning_streaming_incremental(&delta) {
            Ok(result) => {
                push_text_delta(
                    &mut events,
                    AssistantBlockKind::Reasoning,
                    result.reasoning_text,
                );
                push_text_delta(&mut events, AssistantBlockKind::Text, result.normal_text);
            }
            Err(error) => {
                if !self.parser_failed {
                    warn!(
                        parser = self.parser.model_type(),
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
}

/// Push one semantic text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<ContentEvent>, kind: AssistantBlockKind, delta: String) {
    if delta.is_empty() {
        return;
    }
    events.push(ContentEvent::TextDelta { kind, delta });
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
                prompt_token_count,
                prompt_logprobs,
            } => {
                yield ContentEvent::Start {
                    prompt_token_count,
                    prompt_logprobs,
                }
            }
            DecodedTextEvent::TextDelta {
                delta,
                logprobs,
                finished,
                ..
            } => {
                for next in state.process_delta(delta) {
                    yield next;
                }
                if let Some(logprobs) = logprobs {
                    yield ContentEvent::LogprobsDelta { logprobs };
                }
                if let Some(finished) = finished {
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
    use futures::{StreamExt as _, stream};
    use reasoning_parser::{ParseError, ParserResult, ReasoningParser};
    use vllm_llm::FinishReason;
    use vllm_text::output::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, DecodedTokenLogprob,
    };

    use super::super::ContentEvent;
    use super::reasoning_event_stream;
    use crate::event::AssistantBlockKind;

    struct FailingReasoningParser {
        fail_next: bool,
    }

    impl ReasoningParser for FailingReasoningParser {
        fn detect_and_parse_reasoning(&mut self, _text: &str) -> Result<ParserResult, ParseError> {
            Ok(ParserResult::default())
        }

        fn parse_reasoning_streaming_incremental(
            &mut self,
            _text: &str,
        ) -> Result<ParserResult, ParseError> {
            if self.fail_next {
                self.fail_next = false;
                return Err(ParseError::ConfigError("boom".to_string()));
            }

            Ok(ParserResult::default())
        }

        fn reset(&mut self) {
            self.fail_next = false;
        }

        fn model_type(&self) -> &str {
            "test"
        }

        fn is_in_reasoning(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn reasoning_parser_failure_falls_back_to_plain_text() {
        let events = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_count: 3,
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

        let collected = reasoning_event_stream(
            events,
            Some(Box::new(FailingReasoningParser { fail_next: true })),
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
                    prompt_token_count: 3,
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
                prompt_token_count: 1,
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "abc".to_string(),
                token_ids: vec![],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
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
                    prompt_token_count: 1,
                    prompt_logprobs: None,
                },
                ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "abc".to_string(),
                },
                ContentEvent::LogprobsDelta {
                    logprobs: DecodedLogprobs {
                        positions: vec![DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "a".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            }],
                        }],
                    },
                },
            ]
        );
    }
}
