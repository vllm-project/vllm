// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::VecDeque;

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
/// same `llm` update. `context` carries the sampled tokens across the
/// request's updates.
pub(super) fn decode_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    logprobs: &Logprobs,
    context: &mut SampledContext,
) -> Result<DecodedLogprobs, Error> {
    Ok(DecodedLogprobs {
        positions: decode_positions(tokenizer, &logprobs.positions, context)?,
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
) -> Result<DecodedPromptLogprobs, Error> {
    let first_token_id = prompt_token_ids
        .first()
        .copied()
        .expect("prompt logprobs require at least one prompt token");
    let first_token = decode_logprob_token(tokenizer, first_token_id)?;
    // Like the Python frontend, the context starts after the unscored first
    // prompt token.
    let scored_positions = decode_positions(
        tokenizer,
        &logprobs.positions,
        &mut SampledContext::default(),
    )?;

    Ok(DecodedPromptLogprobs {
        first_token_id,
        first_token,
        scored_positions,
    })
}

/// The most recent sampled (or prompt) tokens before the next logprob
/// position, oldest first.
///
/// Four tokens cover any UTF-8 sequence split across byte-fallback tokens.
#[derive(Debug, Default)]
pub(super) struct SampledContext {
    token_ids: VecDeque<u32>,
}

impl SampledContext {
    const MAX_TOKENS: usize = 4;

    fn push(&mut self, token_id: u32) {
        if self.token_ids.len() == Self::MAX_TOKENS {
            self.token_ids.pop_front();
        }
        self.token_ids.push_back(token_id);
    }
}

/// Decode consecutive positions, advancing `context` by each position's
/// sampled token (its first entry).
fn decode_positions<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    positions: &[PositionLogprobs],
    context: &mut SampledContext,
) -> Result<Vec<DecodedPositionLogprobs>, Error> {
    positions
        .iter()
        .map(|position| {
            let decoded = decode_position_logprobs(tokenizer, position, context)?;
            if let Some(sampled) = position.entries.first() {
                context.push(sampled.token_id);
            }
            Ok(decoded)
        })
        .try_collect()
}

/// Decode one token position's raw candidate set into decoded token strings
/// plus logprob metadata.
///
/// This decodes every candidate token ID independently through the active text
/// backend.
fn decode_position_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    position: &PositionLogprobs,
    context: &SampledContext,
) -> Result<DecodedPositionLogprobs, Error> {
    Ok(DecodedPositionLogprobs {
        entries: position
            .entries
            .iter()
            .map(|entry| {
                let mut token = decode_logprob_token(tokenizer, entry.token_id)?;
                if token.ends_with(char::REPLACEMENT_CHARACTER) {
                    token = repair_split_character(tokenizer, entry.token_id, context)?;
                }
                Ok::<_, Error>(DecodedTokenLogprob {
                    token_id: entry.token_id,
                    token,
                    logprob: entry.logprob,
                    rank: entry.rank,
                })
            })
            .try_collect()?,
    })
}

/// Text of a candidate whose own decode ends in an incomplete UTF-8 sequence,
/// completed with the sampled tokens before it, like the Python frontend
/// (`_correct_decoded_token`).
///
/// Returns the text the candidate completes, counting bytes of context tokens
/// that are themselves incomplete, or `""` when the sequence is still
/// incomplete with up to four context tokens.
fn repair_split_character<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    token_id: u32,
    context: &SampledContext,
) -> Result<String, Error> {
    let context = context.token_ids.iter().copied().collect_vec();
    for start in (0..context.len()).rev() {
        let window = &context[start..];
        let full = tokenizer.decode(&[window, &[token_id]].concat(), false)?;
        if full.ends_with(char::REPLACEMENT_CHARACTER) {
            continue;
        }
        // Trailing context tokens that are incomplete on their own were reported
        // as such, so their bytes belong to this candidate's text.
        let mut clean_end = window.len();
        while clean_end > 0
            && decode_logprob_token(tokenizer, window[clean_end - 1])?
                .ends_with(char::REPLACEMENT_CHARACTER)
        {
            clean_end -= 1;
        }
        let clean_prefix = match clean_end {
            0 => String::new(),
            _ => tokenizer.decode(&window[..clean_end], false)?,
        };
        if let Some(rest) = full.strip_prefix(clean_prefix.as_str()) {
            return Ok(rest.to_owned());
        }
        // Tokenizer normalization can change the prefix; drop the longest common
        // prefix instead.
        let common = full
            .char_indices()
            .zip(clean_prefix.chars())
            .take_while(|((_, a), b)| a == b)
            .last()
            .map_or(0, |((i, a), _)| i + a.len_utf8());
        return Ok(full[common..].to_owned());
    }
    Ok(String::new())
}

/// Decode one logprob token on its own.
///
/// Special tokens keep their text regardless of the request's
/// `skip_special_tokens`, like the Python frontend
/// (`convert_ids_list_to_tokens`): they are real candidates, and completions
/// `text_offset` advances by these strings.
fn decode_logprob_token<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    token_id: u32,
) -> Result<String, Error> {
    Ok(tokenizer.decode(&[token_id], false)?)
}

#[cfg(test)]
mod tests {
    use vllm_llm::{Logprobs, PositionLogprobs, TokenLogprob};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    #[test]
    fn decode_logprobs_decodes_every_candidate_token() {
        let tokenizer = TestTokenizer::new();
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
            decode_logprobs(&tokenizer, &logprobs, &mut SampledContext::default()).unwrap(),
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
        let tokenizer = TestTokenizer::new();
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
            decode_prompt_logprobs(&tokenizer, &[b'p' as u32, b'x' as u32], &logprobs).unwrap(),
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

    #[test]
    fn logprob_tokens_keep_special_tokens() {
        let tokenizer = TestTokenizer::new()
            .with_special_token("<|im_start|>", 300)
            .with_special_token("<|im_end|>", 301);
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'.' as u32,
                        logprob: -0.1,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 301,
                        logprob: -2.0,
                        rank: 2,
                    },
                ],
            }],
        };

        let tokens = |positions: &[DecodedPositionLogprobs]| {
            positions
                .iter()
                .flat_map(|position| position.entries.iter().map(|entry| entry.token.clone()))
                .collect::<Vec<_>>()
        };
        let decoded =
            decode_logprobs(&tokenizer, &logprobs, &mut SampledContext::default()).unwrap();
        assert_eq!(tokens(&decoded.positions), [".", "<|im_end|>"]);

        let prompt = decode_prompt_logprobs(&tokenizer, &[300, b'.' as u32], &logprobs).unwrap();
        assert_eq!(prompt.first_token, "<|im_start|>");
        assert_eq!(tokens(&prompt.scored_positions), [".", "<|im_end|>"]);
    }

    /// One position per token, each sampling that token.
    fn sampled_positions(token_ids: &[u32]) -> Logprobs {
        Logprobs {
            positions: token_ids
                .iter()
                .map(|&token_id| PositionLogprobs {
                    entries: vec![TokenLogprob {
                        token_id,
                        logprob: -0.1,
                        rank: 1,
                    }],
                })
                .collect(),
        }
    }

    fn sampled_tokens(positions: &[DecodedPositionLogprobs]) -> Vec<String> {
        positions.iter().map(|position| position.entries[0].token.clone()).collect()
    }

    #[test]
    fn logprob_tokens_repair_characters_split_across_updates() {
        let tokenizer = TestTokenizer::new();
        // "你" is E4 BD A0; it arrives over two updates, then "!".
        let mut context = SampledContext::default();
        let first =
            decode_logprobs(&tokenizer, &sampled_positions(&[0xE4, 0xBD]), &mut context).unwrap();
        let second = decode_logprobs(
            &tokenizer,
            &sampled_positions(&[0xA0, b'!' as u32]),
            &mut context,
        )
        .unwrap();

        assert_eq!(sampled_tokens(&first.positions), ["", ""]);
        assert_eq!(sampled_tokens(&second.positions), ["你", "!"]);
    }

    #[test]
    fn prompt_logprob_tokens_repair_split_characters() {
        let tokenizer = TestTokenizer::new();
        let prompt = [b'a' as u32, 0xE4, 0xBD, 0xA0];
        let decoded =
            decode_prompt_logprobs(&tokenizer, &prompt, &sampled_positions(&prompt[1..])).unwrap();

        assert_eq!(sampled_tokens(&decoded.scored_positions), ["", "", "你"]);
    }
}
