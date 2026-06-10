use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use vllm_llm::{Logprobs, PositionLogprobs};
use vllm_tokenizer::Tokenizer;

use crate::error::Error;

/// One decoded token candidate and its logprob metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedTokenLogprob {
    /// Original vocabulary token ID for this candidate.
    pub token_id: u32,
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
    /// Original vocabulary token ID for the first prompt token.
    pub first_token_id: u32,
    /// Best-effort decoded string for the first prompt token.
    ///
    /// The first prompt token has no left context to score against, so it is
    /// stored separately instead of appearing in `scored_positions`.
    pub first_token: String,
    /// Scored prompt positions after the first prompt token.
    ///
    /// `scored_positions[i]` corresponds to the prompt token at position `i +
    /// 1`.
    pub scored_positions: Vec<DecodedPositionLogprobs>,
}

/// Decode generated-token logprobs from the raw `llm` token-ID shape into the
/// text-layer decoded-token representation.
///
/// Each returned position corresponds to one generated token position from the
/// same `llm` update.
pub(super) fn decode_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    logprobs: &Logprobs,
    skip_special_tokens: bool,
) -> Result<DecodedLogprobs, Error> {
    Ok(DecodedLogprobs {
        positions: logprobs
            .positions
            .iter()
            .map(|position| decode_position_logprobs(tokenizer, position, skip_special_tokens))
            .try_collect()?,
    })
}

/// Decode prompt logprobs from the raw `llm` token-ID shape into the text-layer
/// decoded-token representation.
///
/// The returned payload stores the first prompt token separately and decodes
/// the remaining scored prompt positions into `scored_positions`, matching
/// vLLM's prompt-logprobs semantics.
pub(super) fn decode_prompt_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    prompt_token_ids: &[u32],
    logprobs: &Logprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPromptLogprobs, Error> {
    let first_token_id = prompt_token_ids
        .first()
        .copied()
        .expect("prompt logprobs require at least one prompt token");
    let first_token = tokenizer.decode(&[first_token_id], skip_special_tokens)?;
    let scored_positions = logprobs
        .positions
        .iter()
        .map(|position| decode_position_logprobs(tokenizer, position, skip_special_tokens))
        .try_collect()?;

    Ok(DecodedPromptLogprobs {
        first_token_id,
        first_token,
        scored_positions,
    })
}

/// Decode one token position's raw candidate set into decoded token strings
/// plus logprob metadata.
///
/// This decodes every candidate token ID independently through the active text
/// backend.
fn decode_position_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    position: &PositionLogprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPositionLogprobs, Error> {
    Ok(DecodedPositionLogprobs {
        entries: position
            .entries
            .iter()
            .map(|entry| {
                tokenizer.decode(&[entry.token_id], skip_special_tokens).map(|token| {
                    DecodedTokenLogprob {
                        token_id: entry.token_id,
                        token,
                        logprob: entry.logprob,
                        rank: entry.rank,
                    }
                })
            })
            .try_collect()?,
    })
}

#[cfg(test)]
mod tests {
    use vllm_llm::{Logprobs, PositionLogprobs, TokenLogprob};

    use super::*;

    #[derive(Debug)]
    struct ByteTokenizer;

    impl vllm_tokenizer::Tokenizer for ByteTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(String::from_utf8_lossy(
                &token_ids.iter().map(|token_id| *token_id as u8).collect::<Vec<_>>(),
            )
            .into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }
    }

    #[test]
    fn decode_logprobs_decodes_every_candidate_token() {
        let tokenizer = ByteTokenizer;
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
            decode_logprobs(&tokenizer, &logprobs, false).unwrap(),
            DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token_id: b'a' as u32,
                            token: "a".to_string(),
                            logprob: -0.1,
                            rank: 3,
                        },
                        DecodedTokenLogprob {
                            token_id: b'b' as u32,
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
    fn decode_prompt_logprobs_separates_first_prompt_token() {
        let tokenizer = ByteTokenizer;
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
            decode_prompt_logprobs(&tokenizer, &[b'p' as u32, b'x' as u32], &logprobs, false)
                .unwrap(),
            DecodedPromptLogprobs {
                first_token_id: b'p' as u32,
                first_token: "p".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: b'x' as u32,
                        token: "x".to_string(),
                        logprob: -0.4,
                        rank: 1,
                    }],
                }],
            }
        );
    }
}
