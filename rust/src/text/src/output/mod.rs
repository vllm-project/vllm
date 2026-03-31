//! Output processing helpers shared by text and chat layers.

pub use decoded::{DecodedTextEvent, Finished, TextDecodeOptions, decoded_text_event_stream};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};

mod decoded;
mod logprobs;

use futures::{StreamExt as _, pin_mut};

use crate::{Error, FinishReason, Result, TextOutputStream};

/// Final decoded text plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedTextOutput {
    pub text: String,
    pub prompt_token_count: usize,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
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
            let mut collected: Option<CollectedTextOutput> = None;

            while let Some(event) = stream.next().await.transpose()? {
                match event {
                    DecodedTextEvent::Start {
                        prompt_logprobs: start_prompt_logprobs,
                        ..
                    } => {
                        prompt_logprobs = start_prompt_logprobs;
                    }
                    DecodedTextEvent::TextDelta {
                        delta,
                        token_ids: delta_token_ids,
                        logprobs: mut delta_logprobs,
                        finished,
                    } => {
                        if let Some(c) = collected.as_mut() {
                            c.text.push_str(&delta);
                            c.token_ids.extend(delta_token_ids);
                            if let Some(dlp) = delta_logprobs.as_mut() {
                                if let Some(lp) = c.logprobs.as_mut() {
                                    lp.positions.extend_from_slice(&dlp.positions);
                                } else {
                                    c.logprobs = delta_logprobs;
                                }
                            }
                        } else {
                            collected = Some(CollectedTextOutput {
                                text: delta,
                                prompt_logprobs: prompt_logprobs.take(),
                                logprobs: delta_logprobs,
                                token_ids: delta_token_ids,
                                // These are updated below.
                                prompt_token_count: 0,
                                finish_reason: FinishReason::Error,
                            })
                        };

                        if let Some(finished) = finished {
                            let mut collected = collected.unwrap();
                            collected.prompt_token_count = finished.prompt_token_count;
                            collected.finish_reason = finished.finish_reason;
                            return Ok(collected);
                        }
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
    use vllm_llm::FinishReason;

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
                delta: "bc".to_string(),
                token_ids: vec![1, 2],
                logprobs: Some(DecodedLogprobs {
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
                }),
                finished: Some(Finished {
                    prompt_token_count: 2,
                    output_token_count: 2,
                    finish_reason: FinishReason::stop_eos(),
                }),
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

    #[tokio::test]
    async fn collect_output_accumulates_intermediate_deltas() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_count: 2,
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "he".to_string(),
                token_ids: vec![1, 2],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "h".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "e".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            }],
                        },
                    ],
                }),
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "llo".to_string(),
                token_ids: vec![3, 4, 5],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "l".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "l".to_string(),
                                logprob: -0.4,
                                rank: 1,
                            }],
                        },
                        DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token: "o".to_string(),
                                logprob: -0.5,
                                rank: 1,
                            }],
                        },
                    ],
                }),
                finished: Some(Finished {
                    prompt_token_count: 2,
                    output_token_count: 5,
                    finish_reason: FinishReason::stop_eos(),
                }),
            }),
        ]);

        let collected = stream.collect_output().await.unwrap();
        assert_eq!(collected.text, "hello");
        assert_eq!(collected.prompt_token_count, 2);
        assert_eq!(collected.prompt_logprobs, None);
        assert_eq!(collected.token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "h".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "e".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "l".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "l".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "o".to_string(),
                            logprob: -0.5,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }
}
