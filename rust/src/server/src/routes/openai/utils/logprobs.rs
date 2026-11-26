// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use indexmap::IndexMap;
use itertools::Itertools as _;
use vllm_text::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTokenLogprob,
};

use super::types::{
    ChatLogProbs, ChatLogProbsContent, LogProbs, PromptLogprob, PromptLogprobs, TopLogProb,
};
use crate::error::{ApiError, bail_server_error, server_error};

/// Convert decoded token-position logprobs into the OpenAI completions
/// `logprobs` shape.
pub fn decoded_logprobs_to_openai(
    logprobs: &DecodedLogprobs,
    initial_text_offset: u32,
    return_tokens_as_token_ids: bool,
) -> Result<LogProbs, ApiError> {
    let mut text_offset = Vec::with_capacity(logprobs.positions.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut tokens = Vec::with_capacity(logprobs.positions.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut current_offset = initial_text_offset;

    for position in &logprobs.positions {
        let chosen = position.entries.first().ok_or_else(|| {
            server_error!("decoded logprobs position unexpectedly had no token candidates")
        })?;

        let token_str = format_token(chosen, return_tokens_as_token_ids);
        text_offset.push(current_offset);
        token_logprobs.push(Some(clamp_logprob(chosen.logprob)));
        current_offset = current_offset.saturating_add(text_len(&token_str));
        tokens.push(token_str);
        top_logprobs.push(Some(position_top_logprobs_map(
            position,
            return_tokens_as_token_ids,
        )));
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    })
}

/// Convert decoded prompt logprobs into the OpenAI completions `logprobs`
/// shape.
///
/// The first prompt token is included with `None` logprob metadata, matching
/// Python vLLM's echoed completions behavior.
pub fn decoded_prompt_logprobs_to_openai(
    prompt_logprobs: &DecodedPromptLogprobs,
    initial_text_offset: u32,
    return_tokens_as_token_ids: bool,
) -> Result<LogProbs, ApiError> {
    let mut text_offset = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut token_logprobs = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut tokens = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut top_logprobs = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut current_offset = initial_text_offset;

    let first_token_str = if return_tokens_as_token_ids {
        format!("token_id:{}", prompt_logprobs.first_token_id)
    } else {
        prompt_logprobs.first_token.clone()
    };
    text_offset.push(current_offset);
    token_logprobs.push(None);
    current_offset = current_offset.saturating_add(text_len(&first_token_str));
    tokens.push(first_token_str);
    top_logprobs.push(None);

    for position in &prompt_logprobs.scored_positions {
        let chosen = position.entries.first().ok_or_else(|| {
            server_error!("decoded prompt logprobs position unexpectedly had no token candidates")
        })?;

        let token_str = format_token(chosen, return_tokens_as_token_ids);
        text_offset.push(current_offset);
        token_logprobs.push(Some(clamp_logprob(chosen.logprob)));
        current_offset = current_offset.saturating_add(text_len(&token_str));
        tokens.push(token_str);
        top_logprobs.push(Some(position_top_logprobs_map(
            position,
            return_tokens_as_token_ids,
        )));
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    })
}

/// Map decoded prompt logprobs into vLLM-style per-position maps.
///
/// Like Python's list, which starts with the first token's `None` and gets the
/// scored positions once prefill completes, this is `[None]` when no prompt
/// logprobs arrived: a one-token prompt has none to score, and a request that
/// ended during prefill got none.
pub fn prompt_logprobs_to_maps(prompt_logprobs: Option<&DecodedPromptLogprobs>) -> PromptLogprobs {
    let scored_positions =
        prompt_logprobs.map_or(&[][..], |prompt_logprobs| &prompt_logprobs.scored_positions);
    std::iter::once(None)
        .chain(scored_positions.iter().map(|position| {
            Some(
                position
                    .entries
                    .iter()
                    .map(|entry| {
                        (
                            entry.token_id,
                            PromptLogprob {
                                logprob: clamp_logprob(entry.logprob),
                                rank: entry.rank,
                                decoded_token: entry.token.clone(),
                            },
                        )
                    })
                    .collect(),
            )
        }))
        .collect()
}

/// Convert decoded token-position logprobs into the OpenAI chat `logprobs`
/// shape.
pub fn decoded_logprobs_to_openai_chat(
    logprobs: &DecodedLogprobs,
    top_logprobs: i32,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(|pos| position_to_chat_logprobs_content(pos, top_logprobs, return_tokens_as_token_ids))
        .try_collect()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

/// Count visible text positions using OpenAI completions' character-offset
/// convention.
pub fn text_len(text: &str) -> u32 {
    u32::try_from(text.chars().count()).unwrap_or(u32::MAX)
}

/// The text offset just past the last token of a completions `logprobs`
/// payload, where the next token's offset starts.
pub fn text_end_offset(logprobs: &LogProbs) -> Option<u32> {
    logprobs
        .text_offset
        .last()
        .zip(logprobs.tokens.last())
        .map(|(&offset, token)| offset.saturating_add(text_len(token)))
}

/// Concatenate two OpenAI-style completion logprobs payloads in token order.
pub fn append_openai_logprobs(mut prefix: LogProbs, suffix: LogProbs) -> LogProbs {
    prefix.tokens.extend(suffix.tokens);
    prefix.token_logprobs.extend(suffix.token_logprobs);
    prefix.top_logprobs.extend(suffix.top_logprobs);
    prefix.text_offset.extend(suffix.text_offset);
    prefix
}

/// Build the non-stream completions `logprobs` payload from collected text
/// output.
///
/// With `echoed_prompt_logprobs`, the returned payload matches Python vLLM's
/// echoed completions behavior by concatenating prompt and completion logprobs
/// into one OpenAI `LogProbs` object.
pub fn collected_logprobs_to_openai(
    collected: &CollectedTextOutput,
    echoed_prompt_logprobs: Option<LogProbs>,
    return_tokens_as_token_ids: bool,
) -> Result<LogProbs, ApiError> {
    let completion_start = echoed_prompt_logprobs.as_ref().and_then(text_end_offset).unwrap_or(0);
    let completion_logprobs = match collected.logprobs.as_ref() {
        Some(completion_logprobs) => decoded_logprobs_to_openai(
            completion_logprobs,
            completion_start,
            return_tokens_as_token_ids,
        )?,
        // A request that generated nothing, such as one aborted during
        // prefill, has no positions to report.
        None if collected.token_ids.is_empty() => LogProbs::default(),
        None => bail_server_error!(
            "completion response requested logprobs but generation returned none"
        ),
    };
    Ok(match echoed_prompt_logprobs {
        Some(prompt_logprobs) => append_openai_logprobs(prompt_logprobs, completion_logprobs),
        None => completion_logprobs,
    })
}

/// Format a token entry as either its decoded string or `token_id:{id}`.
fn format_token(entry: &DecodedTokenLogprob, as_token_id: bool) -> String {
    if as_token_id {
        format!("token_id:{}", entry.token_id)
    } else {
        entry.token.clone()
    }
}

/// A sampled token repeated among the top-k candidates, or several tokens
/// decoding to the same string (byte fragments, which decode to "\u{fffd}"),
/// share one entry. The entry stays where its first candidate is, so the token
/// at this position, which the engine row puts first, keeps the first entry.
/// It holds the highest logprob of its candidates; on a tie the earlier one
/// stays. The Python frontend does the same after #65741.
fn position_top_logprobs_map(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
) -> IndexMap<String, f32> {
    let mut map = IndexMap::with_capacity(position.entries.len());
    for entry in &position.entries {
        let logprob = clamp_logprob(entry.logprob);
        let kept = map.entry(format_token(entry, return_tokens_as_token_ids)).or_insert(logprob);
        *kept = kept.max(logprob);
    }
    map
}

fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
    top_logprobs: i32,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        server_error!("decoded chat logprobs position unexpectedly had no token candidates")
    })?;

    let token_str = format_token(chosen, return_tokens_as_token_ids);
    Ok(ChatLogProbsContent {
        token: token_str.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token_bytes(&token_str)),
        top_logprobs: chat_top_logprob_entries(position, top_logprobs)
            .map(|entry| {
                let t = format_token(entry, return_tokens_as_token_ids);
                TopLogProb {
                    logprob: clamp_logprob(entry.logprob),
                    bytes: Some(token_bytes(&t)),
                    token: t,
                }
            })
            .collect(),
    })
}

fn chat_top_logprob_entries(
    position: &DecodedPositionLogprobs,
    top_logprobs: i32,
) -> impl Iterator<Item = &DecodedTokenLogprob> {
    let limit = if top_logprobs == -1 {
        position.entries.len()
    } else {
        usize::try_from(top_logprobs).unwrap_or(0)
    };

    // The engine row is the sampled token followed by the top-k or requested
    // tokens, so the sampled token may repeat. Keep each token's first entry.
    position.entries.iter().unique_by(|entry| entry.token_id).take(limit)
}

fn token_bytes(token: &str) -> Vec<u8> {
    token.as_bytes().to_vec()
}

pub fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}

#[cfg(test)]
mod tests {
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{
        decoded_logprobs_to_openai, decoded_logprobs_to_openai_chat,
        decoded_prompt_logprobs_to_openai, prompt_logprobs_to_maps,
    };

    #[test]
    fn prompt_logprobs_maps_keep_only_the_first_position_without_payload() {
        assert_eq!(prompt_logprobs_to_maps(None), vec![None]);
    }

    #[test]
    fn prompt_logprobs_maps_preserve_token_ids_and_metadata() {
        let prompt_logprobs = vllm_text::DecodedPromptLogprobs {
            first_token_id: 10,
            first_token: "first".to_string(),
            scored_positions: sample_logprobs().positions,
        };

        let maps = prompt_logprobs_to_maps(Some(&prompt_logprobs));
        assert_eq!(
            serde_json::to_value(maps).expect("serialize prompt logprobs"),
            serde_json::json!([
                null,
                {
                    "1": {"logprob": -0.10000000149011612_f64, "rank": 1, "decoded_token": "A"},
                    "2": {"logprob": -1.0, "rank": 2, "decoded_token": "B"},
                    "3": {"logprob": -2.0, "rank": 3, "decoded_token": "C"},
                }
            ])
        );
    }

    #[test]
    fn prompt_logprobs_maps_keep_engine_order() {
        // Prompt token `C` is the rank-3 token: the engine row is `[C, A, B, C]`.
        let mut scored_positions = sample_logprobs().positions;
        let prompt_token = scored_positions[0].entries[2].clone();
        scored_positions[0].entries.insert(0, prompt_token);
        let prompt_logprobs = vllm_text::DecodedPromptLogprobs {
            first_token_id: 10,
            first_token: "first".to_string(),
            scored_positions,
        };

        let maps = prompt_logprobs_to_maps(Some(&prompt_logprobs));
        let token_ids = maps[1].as_ref().unwrap().keys().copied().collect::<Vec<_>>();
        assert_eq!(token_ids, [3, 1, 2]);
    }

    fn sample_logprobs() -> DecodedLogprobs {
        DecodedLogprobs {
            positions: vec![DecodedPositionLogprobs {
                entries: vec![
                    DecodedTokenLogprob {
                        token_id: 1,
                        token: "A".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token_id: 2,
                        token: "B".to_string(),
                        logprob: -1.0,
                        rank: 2,
                    },
                    DecodedTokenLogprob {
                        token_id: 3,
                        token: "C".to_string(),
                        logprob: -2.0,
                        rank: 3,
                    },
                ],
            }],
        }
    }

    #[test]
    fn completion_top_logprobs_keep_engine_order() {
        // Sampled `C` is the rank-3 token: the engine row is `[C, A, B, C]`.
        // Python's dict keeps `C` first and once, then the rest in rank order.
        let mut logprobs = sample_logprobs();
        let sampled = logprobs.positions[0].entries[2].clone();
        logprobs.positions[0].entries.insert(0, sampled);

        let top_logprobs = |as_token_ids| {
            let logprobs = decoded_logprobs_to_openai(&logprobs, 0, as_token_ids).unwrap();
            serde_json::to_string(&logprobs.top_logprobs).unwrap()
        };
        expect_test::expect![[r#"[{"C":-2.0,"A":-0.1,"B":-1.0}]"#]].assert_eq(&top_logprobs(false));
        expect_test::expect![[r#"[{"token_id:3":-2.0,"token_id:1":-0.1,"token_id:2":-1.0}]"#]]
            .assert_eq(&top_logprobs(true));
    }

    /// Byte fragments: sampled `C` (rank 3), `A` and `B` all decode to the same
    /// string, here "" (the logic does not depend on which string; the Rust
    /// frontend decodes them to "\u{fffd}" until #63926). The engine row is
    /// `[C, A, B, C, D]`.
    fn colliding_logprobs() -> DecodedLogprobs {
        let mut logprobs = sample_logprobs();
        for entry in &mut logprobs.positions[0].entries {
            entry.token = String::new();
        }
        let sampled = logprobs.positions[0].entries[2].clone();
        logprobs.positions[0].entries.insert(0, sampled);
        logprobs.positions[0].entries.push(DecodedTokenLogprob {
            token_id: 4,
            token: "D".to_string(),
            logprob: -3.0,
            rank: 4,
        });
        logprobs
    }

    #[test]
    fn completion_top_logprobs_keep_highest_on_string_collision() {
        let logprobs = colliding_logprobs();
        let top_logprobs = |as_token_ids| {
            let logprobs = decoded_logprobs_to_openai(&logprobs, 0, as_token_ids).unwrap();
            serde_json::to_string(&logprobs.top_logprobs).unwrap()
        };
        expect_test::expect![[r#"[{"":-0.1,"D":-3.0}]"#]].assert_eq(&top_logprobs(false));
        // Token IDs keep the candidates apart.
        expect_test::expect![[
            r#"[{"token_id:3":-2.0,"token_id:1":-0.1,"token_id:2":-1.0,"token_id:4":-3.0}]"#
        ]]
        .assert_eq(&top_logprobs(true));
    }

    #[test]
    fn completion_top_logprobs_keep_sampled_entry_on_string_collision() {
        // Sampled `A` (rank 1) and `B` decode to the same string, and `A` is the
        // most likely: the engine row is `[A, A, B, C]` and `B` must not take
        // over the shared entry.
        let mut logprobs = sample_logprobs();
        let entries = &mut logprobs.positions[0].entries;
        entries[0].token = String::new();
        entries[1].token = String::new();
        let sampled = entries[0].clone();
        entries.insert(0, sampled);

        let logprobs = decoded_logprobs_to_openai(&logprobs, 0, false).unwrap();
        assert_eq!(
            logprobs.top_logprobs[0].as_ref().unwrap()[""],
            logprobs.token_logprobs[0].unwrap()
        );
        expect_test::expect![[r#"[{"":-0.1,"C":-2.0}]"#]]
            .assert_eq(&serde_json::to_string(&logprobs.top_logprobs).unwrap());
    }

    #[test]
    fn echoed_prompt_top_logprobs_keep_highest_on_string_collision() {
        let prompt_logprobs = vllm_text::DecodedPromptLogprobs {
            first_token_id: 10,
            first_token: "first".to_string(),
            scored_positions: colliding_logprobs().positions,
        };

        let logprobs = decoded_prompt_logprobs_to_openai(&prompt_logprobs, 0, false).unwrap();
        expect_test::expect![[r#"[null,{"":-0.1,"D":-3.0}]"#]]
            .assert_eq(&serde_json::to_string(&logprobs.top_logprobs).unwrap());
    }

    fn chat_top_logprobs_len(top_logprobs: i32) -> usize {
        let chat_logprobs =
            decoded_logprobs_to_openai_chat(&sample_logprobs(), top_logprobs, false)
                .expect("chat logprobs");
        chat_logprobs.content.expect("content")[0].top_logprobs.len()
    }

    #[test]
    fn chat_logprobs_respects_requested_top_logprobs_count() {
        assert_eq!(chat_top_logprobs_len(0), 0);
        assert_eq!(chat_top_logprobs_len(1), 1);
        assert_eq!(chat_top_logprobs_len(-1), 3);
    }

    #[test]
    fn chat_logprobs_dedupes_repeated_tokens_before_top_logprobs_cut() {
        // Sampled `A` is also the top-1 token: the engine row is `[A, A, B, C]`.
        let mut logprobs = sample_logprobs();
        let sampled = logprobs.positions[0].entries[0].clone();
        logprobs.positions[0].entries.insert(0, sampled);

        let top_tokens = |top_logprobs| {
            let chat_logprobs =
                decoded_logprobs_to_openai_chat(&logprobs, top_logprobs, false).unwrap();
            chat_logprobs.content.unwrap()[0]
                .top_logprobs
                .iter()
                .map(|entry| entry.token.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(top_tokens(2), ["A", "B"]);
        assert_eq!(top_tokens(-1), ["A", "B", "C"]);
    }

    #[test]
    fn chat_logprobs_clamps_nan_and_negative_infinity_for_json() {
        let mut logprobs = sample_logprobs();
        logprobs.positions[0].entries[0].logprob = f32::NAN;
        logprobs.positions[0].entries[0].rank = 0;
        logprobs.positions[0].entries[1].logprob = f32::NEG_INFINITY;

        let response = decoded_logprobs_to_openai_chat(&logprobs, -1, false).unwrap();
        let json = serde_json::to_string_pretty(&response).unwrap();
        expect_test::expect![[r#"
            {
              "content": [
                {
                  "token": "A",
                  "logprob": -9999.0,
                  "bytes": [
                    65
                  ],
                  "top_logprobs": [
                    {
                      "token": "A",
                      "logprob": -9999.0,
                      "bytes": [
                        65
                      ]
                    },
                    {
                      "token": "B",
                      "logprob": -9999.0,
                      "bytes": [
                        66
                      ]
                    },
                    {
                      "token": "C",
                      "logprob": -2.0,
                      "bytes": [
                        67
                      ]
                    }
                  ]
                }
              ]
            }"#]]
        .assert_eq(&json);
    }
}
