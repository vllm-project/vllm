use serde::{Deserialize, Serialize};
use vllm_llm::{Logprobs, PositionLogprobs};

use crate::backend::TextBackend;
use crate::error::Error;

/// One decoded token candidate and its logprob metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedTokenLogprob {
    /// Best-effort decoded token string for this candidate.
    pub token: String,
    /// Log probability of this token candidate.
    pub logprob: f32,
    /// Vocabulary rank of this token candidate.
    pub rank: u32,
}

/// One position's decoded token candidates and their logprobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPositionLogprobs {
    /// Candidate tokens for this position.
    pub entries: Vec<DecodedTokenLogprob>,
}

/// Decoded sample logprobs for generated token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedLogprobs {
    /// Generated token positions covered by this payload.
    pub positions: Vec<DecodedPositionLogprobs>,
}

/// Decoded prompt logprobs for prompt token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPromptLogprobs {
    /// Prompt positions covered by this payload.
    ///
    /// The first prompt position is always `None`, matching vLLM's prompt-logprobs semantics:
    /// the first prompt token has no left context to score against.
    pub positions: Vec<Option<DecodedPositionLogprobs>>,
}

/// Decode generated-token logprobs from the raw `llm` token-ID shape into the text-layer
/// decoded-token representation.
///
/// Each returned position corresponds to one generated token position from the same `llm` update.
pub(super) fn decode_logprobs<B: TextBackend + ?Sized>(
    backend: &B,
    logprobs: &Logprobs,
    skip_special_tokens: bool,
) -> Result<DecodedLogprobs, Error> {
    Ok(DecodedLogprobs {
        positions: logprobs
            .positions
            .iter()
            .map(|position| decode_position_logprobs(backend, position, skip_special_tokens))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

/// Decode prompt logprobs from the raw `llm` token-ID shape into the text-layer decoded-token
/// representation.
///
/// The returned payload always prepends one leading `None` entry so the first prompt token remains
/// unscored, matching vLLM's northbound prompt-logprobs semantics.
pub(super) fn decode_prompt_logprobs<B: TextBackend + ?Sized>(
    backend: &B,
    logprobs: &Logprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPromptLogprobs, Error> {
    let mut positions = Vec::with_capacity(logprobs.positions.len() + 1);
    positions.push(None);
    positions.extend(
        logprobs
            .positions
            .iter()
            .map(|position| {
                decode_position_logprobs(backend, position, skip_special_tokens).map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(DecodedPromptLogprobs { positions })
}

/// Decode one token position's raw candidate set into decoded token strings plus logprob metadata.
///
/// This decodes every candidate token ID independently through the active text backend.
fn decode_position_logprobs<B: TextBackend + ?Sized>(
    backend: &B,
    position: &PositionLogprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPositionLogprobs, Error> {
    Ok(DecodedPositionLogprobs {
        entries: position
            .entries
            .iter()
            .map(|entry| {
                Ok(DecodedTokenLogprob {
                    token: backend.decode(&[entry.token_id], skip_special_tokens)?,
                    logprob: entry.logprob,
                    rank: entry.rank,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?,
    })
}

#[cfg(test)]
mod tests {
    use vllm_llm::{Logprobs, PositionLogprobs, TokenLogprob};

    use super::*;

    #[derive(Debug)]
    struct ByteBackend;

    impl TextBackend for ByteBackend {
        fn encode(&self, _text: &str) -> crate::Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> crate::Result<String> {
            Ok(String::from_utf8_lossy(
                &token_ids
                    .iter()
                    .map(|token_id| *token_id as u8)
                    .collect::<Vec<_>>(),
            )
            .into_owned())
        }
    }

    #[test]
    fn decode_logprobs_decodes_every_candidate_token() {
        let backend = ByteBackend;
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'a' as u32,
                        logprob: -0.1,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: b'b' as u32,
                        logprob: -0.2,
                        rank: 1,
                    },
                ],
            }],
        };

        assert_eq!(
            decode_logprobs(&backend, &logprobs, false).unwrap(),
            DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token: "a".to_string(),
                            logprob: -0.1,
                            rank: 3,
                        },
                        DecodedTokenLogprob {
                            token: "b".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                }],
            }
        );
    }

    #[test]
    fn decode_prompt_logprobs_prepends_leading_none() {
        let backend = ByteBackend;
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![TokenLogprob {
                    token_id: b'x' as u32,
                    logprob: -0.4,
                    rank: 1,
                }],
            }],
        };

        assert_eq!(
            decode_prompt_logprobs(&backend, &logprobs, false).unwrap(),
            DecodedPromptLogprobs {
                positions: vec![
                    None,
                    Some(DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "x".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    }),
                ],
            }
        );
    }
}
