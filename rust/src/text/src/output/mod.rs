//! Output processing helpers shared by text and chat layers.

pub use decoded::{DecodedTextEvent, TextDecodeOptions, decoded_text_event_stream};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

mod decoded;
mod logprobs;

use futures::{StreamExt as _, pin_mut};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::{Error, Result, TextOutputStream};

/// Final decoded text plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedTextOutput {
    pub text: String,
    pub prompt_token_count: usize,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub stop_reason: Option<StopReason>,
}

#[allow(clippy::manual_async_fn, reason = "specify `Send` bound")]
#[easy_ext::ext(TextOutputStreamExt)]
impl<T: TextOutputStream> T {
    /// Collect the stream to completion and return the final decoded text plus terminal metadata.
    pub fn collect_output(self) -> impl Future<Output = Result<CollectedTextOutput>> + Send {
        async move {
            let stream = self;
            pin_mut!(stream);
            let mut prompt_logprobs = None;
            let mut logprob_positions: Vec<DecodedPositionLogprobs> = Vec::new();

            while let Some(event) = stream.next().await.transpose()? {
                match event {
                    DecodedTextEvent::Start {
                        prompt_logprobs: start_prompt_logprobs,
                        prompt_token_count: _,
                    } => {
                        prompt_logprobs = start_prompt_logprobs;
                    }
                    DecodedTextEvent::TextDelta { logprobs, .. } => {
                        if let Some(logprobs) = logprobs {
                            logprob_positions.extend(logprobs.positions);
                        }
                    }
                    DecodedTextEvent::Done {
                        text,
                        prompt_token_count,
                        token_ids,
                        finish_reason,
                        stop_reason,
                    } => {
                        return Ok(CollectedTextOutput {
                            text,
                            prompt_token_count,
                            prompt_logprobs,
                            logprobs: (!logprob_positions.is_empty()).then_some(DecodedLogprobs {
                                positions: logprob_positions,
                            }),
                            token_ids,
                            finish_reason,
                            stop_reason,
                        });
                    }
                }
            }

            // Note: this is actually unreachable, as the underlying stream always emit an error on
            // unexpected close.
            Err(Error::StreamClosedBeforeTerminalOutput {
                request_id: "unknown".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use vllm_engine_core_client::protocol::FinishReason;

    use super::*;

    #[tokio::test]
    async fn collect_output_retains_prompt_and_sample_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_count: 2,
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: String::new(),
                text: String::new(),
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "bc".to_string(),
                text: "bc".to_string(),
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "bc".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::Done {
                text: "bc".to_string(),
                prompt_token_count: 2,
                token_ids: vec![1, 2],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
            }),
        ]);

        let collected = stream.collect_output().await.unwrap();
        assert_eq!(collected.text, "bc");
        assert_eq!(collected.prompt_token_count, 2);
        assert_eq!(
            collected.prompt_logprobs,
            Some(DecodedPromptLogprobs {
                first_token: "o".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token: "p".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    }],
                }],
            })
        );
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "bc".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }
}
