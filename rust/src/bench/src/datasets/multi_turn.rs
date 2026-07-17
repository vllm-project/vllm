// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::{ConversationTurn, MultiTurnConversation};
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Configuration for generating random multi-turn conversations.
#[derive(Debug, Clone)]
pub struct MultiTurnRandomConfig {
    pub num_conversations: usize,
    pub min_turns: usize,
    pub max_turns: usize,
    /// Shared prefix length prepended to the conversation.
    ///
    /// In normal accumulated-history mode this is added to turn 0, so all
    /// later turns inherit it through history. In no-history prefix-sharing
    /// mode it is added to every independent turn.
    pub prefix_len: usize,
    /// Input length for turn 0.
    pub input_len: usize,
    /// Input length for turns 1+. 0 = fallback to input_len.
    pub per_turn_input_len: usize,
    pub output_len: usize,
    pub seed: u64,
    pub request_id_prefix: String,
    pub prefix_sharing_config: Option<PrefixSharingConfig>,
}

/// Configuration for 3-tier prefix sharing in multi-turn user messages.
#[derive(Debug, Clone)]
pub struct PrefixSharingConfig {
    /// Fraction of per-turn input tokens shared across ALL conversations.
    pub global_ratio: f64,
    /// Fraction of per-turn input tokens shared within each conversation.
    pub conversation_ratio: f64,
}

/// Generate a deterministic token sequence from allowed tokens using offset+modulo.
fn make_token_seq(allowed_tokens: &[u32], offset: usize, len: usize) -> Vec<u32> {
    let at_len = allowed_tokens.len();
    (0..len).map(|i| allowed_tokens[(offset + i) % at_len]).collect()
}

/// Generate synthetic multi-turn conversations with random user messages.
///
/// Each conversation has `num_turns` turns, each with a random user prompt
/// of `input_len` tokens and `output_len` expected output tokens.
pub fn generate_multi_turn_random(
    tokenizer: &TokenizerKind,
    cfg: &MultiTurnRandomConfig,
) -> Result<Vec<MultiTurnConversation>> {
    let num_conversations = cfg.num_conversations;
    let min_turns = cfg.min_turns;
    let max_turns = cfg.max_turns;
    let prefix_len = cfg.prefix_len;
    let input_len = cfg.input_len;
    let output_len = cfg.output_len;
    let seed = cfg.seed;
    let request_id_prefix = &cfg.request_id_prefix;
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }

    let vocab_size = tokenizer.vocab_size() as usize;
    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = input_len.saturating_sub(num_special);
    let real_per_turn_len = if cfg.per_turn_input_len > 0 {
        cfg.per_turn_input_len.saturating_sub(num_special)
    } else {
        real_input_len
    };

    if real_input_len < 1 {
        return Err(BenchError::Config(format!(
            "--random-input-len too small: with {num_special} special tokens, \
             effective input length is {real_input_len}"
        )));
    }
    if real_per_turn_len < 1 {
        return Err(BenchError::Config(format!(
            "--per-turn-input-len too small: with {num_special} special tokens, \
             effective per-turn input length is {real_per_turn_len}"
        )));
    }

    // Prefix sharing mode: generate 3-tier prefixed messages
    let mut rng = StdRng::seed_from_u64(seed);
    if let Some(ref ps_cfg) = cfg.prefix_sharing_config {
        return generate_prefix_sharing_conversations(
            tokenizer,
            cfg,
            ps_cfg,
            &allowed_tokens,
            &mut rng,
        );
    }
    let shared_prefix_text =
        generate_shared_prefix_text(tokenizer, &allowed_tokens, prefix_len, seed)?;

    // Pre-generate per-conversation turn counts and per-turn offsets deterministically.
    // Turn counts are drawn first so the RNG sequence is stable regardless of vocab_size.
    let conv_turn_counts: Vec<usize> = (0..num_conversations)
        .map(|_| {
            if min_turns == max_turns {
                min_turns
            } else {
                rng.random_range(min_turns..=max_turns)
            }
        })
        .collect();

    let offsets: Vec<Vec<usize>> = conv_turn_counts
        .iter()
        .map(|&n| (0..n).map(|_| rng.random_range(0..vocab_size)).collect())
        .collect();

    // Parallel generation across conversations
    offsets
        .par_iter()
        .enumerate()
        .map(|(conv_idx, conv_offsets)| {
            let mut turns = Vec::with_capacity(conv_offsets.len());
            for (turn_idx, &offset) in conv_offsets.iter().enumerate() {
                let target_len = if turn_idx == 0 {
                    real_input_len
                } else {
                    real_per_turn_len
                };
                // Use max_turns stride to keep offsets unique across variable-length convs
                let inner_seq = make_token_seq(
                    &allowed_tokens,
                    offset + conv_idx * max_turns + turn_idx,
                    target_len,
                );

                let (prompt, adjusted) =
                    gen_prompt_to_target_len(tokenizer, &inner_seq, target_len)?;
                let (prompt, token_len) = if turn_idx == 0 && !shared_prefix_text.is_empty() {
                    let combined = format!("{}{}", &*shared_prefix_text, prompt);
                    let token_len = tokenizer.encode(&combined, false)?.len();
                    (combined, token_len)
                } else {
                    (prompt, adjusted.len())
                };

                turns.push(ConversationTurn {
                    user_message: Arc::from(prompt),
                    user_message_len: token_len,
                    expected_output_len: output_len,
                });
            }

            Ok(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            })
        })
        .collect()
}

/// Generate conversations with 3-tier prefix sharing.
///
/// Each turn's user message = [global_prefix][conversation_prefix][unique_suffix].
/// No history accumulation — each turn sends only its own fixed-length message.
fn generate_prefix_sharing_conversations(
    tokenizer: &TokenizerKind,
    cfg: &MultiTurnRandomConfig,
    ps_cfg: &PrefixSharingConfig,
    allowed_tokens: &[u32],
    rng: &mut StdRng,
) -> Result<Vec<MultiTurnConversation>> {
    let num_conversations = cfg.num_conversations;
    let min_turns = cfg.min_turns;
    let max_turns = cfg.max_turns;
    let prefix_len = cfg.prefix_len;
    let output_len = cfg.output_len;
    let request_id_prefix = &cfg.request_id_prefix;

    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = cfg.input_len.saturating_sub(num_special);
    let real_per_turn_len = if cfg.per_turn_input_len > 0 {
        cfg.per_turn_input_len.saturating_sub(num_special)
    } else {
        real_input_len
    };

    // Compute segment lengths from turn-0 (real_input_len) so the shared prefix
    // bytes stay byte-identical across all turns regardless of per_turn_input_len.
    let global_len = (real_input_len as f64 * ps_cfg.global_ratio).floor() as usize;
    let conv_len = (real_input_len as f64 * ps_cfg.conversation_ratio).floor() as usize;
    let unique_len = real_input_len.saturating_sub(global_len + conv_len);

    // Validate that turns 1+ still have room for a non-empty unique suffix
    if real_per_turn_len <= global_len + conv_len {
        return Err(BenchError::Config(format!(
            "--per-turn-input-len ({real_per_turn_len} after special tokens) is too small: \
             global_len={global_len} + conv_len={conv_len} already fills the budget. \
             Increase --per-turn-input-len or reduce prefix ratios."
        )));
    }

    let at_len = allowed_tokens.len();
    let shared_prefix_text =
        generate_shared_prefix_text(tokenizer, allowed_tokens, prefix_len, cfg.seed)?;

    // Generate global prefix text once
    let global_text: Arc<str> = if global_len > 0 {
        let offset: usize = rng.random_range(0..at_len);
        let seq = make_token_seq(allowed_tokens, offset, global_len);
        let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, global_len)?;
        Arc::from(text)
    } else {
        Arc::from("")
    };

    // Generate per-conversation prefix texts
    let conv_texts: Vec<Arc<str>> = if conv_len > 0 {
        let mut texts = Vec::with_capacity(num_conversations);
        for conv_idx in 0..num_conversations {
            let offset: usize = rng.random_range(0..at_len);
            let seq = make_token_seq(allowed_tokens, offset + conv_idx, conv_len);
            let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, conv_len)?;
            texts.push(Arc::from(text));
        }
        texts
    } else {
        vec![Arc::from(""); num_conversations]
    };

    // Pre-generate per-conversation turn counts and unique offsets deterministically.
    let vocab_size = tokenizer.vocab_size() as usize;
    let conv_turn_counts: Vec<usize> = (0..num_conversations)
        .map(|_| {
            if min_turns == max_turns {
                min_turns
            } else {
                rng.random_range(min_turns..=max_turns)
            }
        })
        .collect();

    let unique_offsets: Vec<Vec<usize>> = conv_turn_counts
        .iter()
        .map(|&n| (0..n).map(|_| rng.random_range(0..vocab_size)).collect())
        .collect();

    // Parallel generation across conversations
    unique_offsets
        .par_iter()
        .enumerate()
        .map(|(conv_idx, conv_offsets)| {
            let mut turns = Vec::with_capacity(conv_offsets.len());
            for (turn_idx, &offset) in conv_offsets.iter().enumerate() {
                // Turn 0 uses unique_len derived from real_input_len;
                // turns 1+ use per-turn unique_len (prefix bytes stay identical).
                let turn_unique_len = if turn_idx == 0 {
                    unique_len
                } else {
                    real_per_turn_len.saturating_sub(global_len + conv_len)
                };

                // Generate unique suffix
                let unique_text = if turn_unique_len > 0 {
                    let seq = make_token_seq(
                        allowed_tokens,
                        offset + conv_idx * max_turns + turn_idx,
                        turn_unique_len,
                    );
                    let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, turn_unique_len)?;
                    text
                } else {
                    String::new()
                };

                // Concatenate: optional random prefix + global + conversation + unique.
                // Prefix-sharing mode sends each turn independently, so the random
                // prefix must be included on every turn to be present in every request.
                let combined = format!(
                    "{}{}{}{}",
                    &*shared_prefix_text, &*global_text, &*conv_texts[conv_idx], unique_text
                );
                // Re-encode to get actual token count (BPE boundary effects)
                let token_len = tokenizer.encode(&combined, false)?.len();

                turns.push(ConversationTurn {
                    user_message: Arc::from(combined),
                    user_message_len: token_len,
                    expected_output_len: output_len,
                });
            }

            Ok(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            })
        })
        .collect()
}

fn generate_shared_prefix_text(
    tokenizer: &TokenizerKind,
    allowed_tokens: &[u32],
    prefix_len: usize,
    seed: u64,
) -> Result<Arc<str>> {
    if prefix_len == 0 {
        return Ok(Arc::from(""));
    }

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD));
    let tokens: Vec<u32> = (0..prefix_len)
        .map(|_| allowed_tokens[rng.random_range(0..allowed_tokens.len())])
        .collect();
    let (text, _) = gen_prompt_to_target_len(tokenizer, &tokens, prefix_len)?;
    Ok(Arc::from(text))
}

/// Load multi-turn conversations from a ShareGPT dataset.
///
/// Walks ALL turns in each entry (not just first 2). Filters entries
/// with at least 4 messages (2 user + 2 assistant = 2 real turns).
pub fn load_sharegpt_multi_turn(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_conversations: usize,
    output_len_override: Option<usize>,
    max_turns: Option<usize>,
    seed: u64,
    request_id_prefix: &str,
) -> Result<Vec<MultiTurnConversation>> {
    let content = std::fs::read_to_string(dataset_path).map_err(|e| {
        BenchError::Config(format!(
            "Failed to read ShareGPT file '{dataset_path}': {e}"
        ))
    })?;

    let data: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| BenchError::Config(format!("Invalid JSON in ShareGPT file: {e}")))?;

    let entries = data
        .as_array()
        .ok_or_else(|| BenchError::Config("ShareGPT file must contain a JSON array".into()))?;

    // Filter entries with at least 4 messages (2 turns: user+assistant+user+assistant)
    let mut filtered: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            entry
                .get("conversations")
                .and_then(|c| c.as_array())
                .map(|a| a.len() >= 4)
                .unwrap_or(false)
        })
        .collect();

    if filtered.is_empty() {
        return Err(BenchError::Config(
            "No valid multi-turn entries in ShareGPT file (need at least 4 messages per entry)"
                .into(),
        ));
    }

    // Shuffle
    let mut rng = StdRng::seed_from_u64(seed);
    filtered.shuffle(&mut rng);

    let mut conversations = Vec::new();

    for entry in &filtered {
        if conversations.len() >= num_conversations {
            break;
        }

        let msgs = entry["conversations"].as_array().unwrap();
        let mut turns = Vec::new();

        // Walk alternating human/gpt pairs, stopping early once max_turns reached
        // to avoid tokenizing turns that would be discarded by truncate().
        let mut i = 0;
        while i + 1 < msgs.len() {
            if let Some(m) = max_turns
                && turns.len() >= m
            {
                break;
            }
            let from = msgs[i].get("from").and_then(|f| f.as_str()).unwrap_or("");
            let user_text = msgs[i].get("value").and_then(|v| v.as_str()).unwrap_or("");
            let assistant_text = msgs[i + 1].get("value").and_then(|v| v.as_str()).unwrap_or("");

            // Expect human then gpt
            if from != "human" || user_text.is_empty() {
                i += 1;
                continue;
            }

            let user_ids = tokenizer.encode(user_text, false)?;
            let user_len = user_ids.len();

            let expected_output_len = if let Some(override_len) = output_len_override {
                override_len
            } else {
                let assistant_ids = tokenizer.encode(assistant_text, false)?;
                assistant_ids.len().max(1)
            };

            turns.push(ConversationTurn {
                user_message: Arc::from(user_text),
                user_message_len: user_len,
                expected_output_len,
            });

            i += 2;
        }

        if turns.len() >= 2 {
            let conv_idx = conversations.len();
            conversations.push(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            });
        }
    }

    if conversations.is_empty() {
        return Err(BenchError::Config(
            "No valid multi-turn conversations after filtering ShareGPT dataset.".into(),
        ));
    }

    // Oversample if needed
    if conversations.len() < num_conversations {
        let original_len = conversations.len();
        let needed = num_conversations - original_len;
        for i in 0..needed {
            let mut conv = conversations[rng.random_range(0..original_len)].clone();
            conv.conversation_id = format!("{request_id_prefix}conv-{}", original_len + i);
            conversations.push(conv);
        }
        println!(
            "Oversampled multi-turn conversations from {original_len} to {} total.",
            conversations.len()
        );
    }

    Ok(conversations)
}

/// Ensure decoded-then-encoded prompt length matches the target.
fn gen_prompt_to_target_len(
    tokenizer: &TokenizerKind,
    token_sequence: &[u32],
    target_len: usize,
) -> Result<(String, Vec<u32>)> {
    let max_retry = 20;
    let mut tokens = token_sequence.to_vec();

    for retry in 0..=max_retry {
        let prompt = tokenizer.decode(&tokens, true)?;
        tokens = tokenizer.encode(&prompt, false)?;

        if retry >= max_retry {
            // BPE tokenizers can oscillate by ±1 on certain boundaries.
            // For benchmark random content, accept close-enough and truncate/pad.
            if tokens.len() > target_len {
                tokens.truncate(target_len);
            }
            // If still short by 1-2 tokens, accept as-is — negligible for benchmarks.
            // Re-decode after truncation to ensure prompt string matches token vector.
            let prompt = tokenizer.decode(&tokens, true)?;
            return Ok((prompt, tokens));
        }

        if tokens.len() == target_len {
            return Ok((prompt, tokens));
        } else if tokens.len() < target_len {
            let allowed = tokenizer.get_allowed_tokens();
            let needed = target_len - tokens.len();
            if allowed.is_empty() {
                let vocab_size = tokenizer.vocab_size() as usize;
                for j in 0..needed {
                    tokens.push(((tokens.len() + j) % vocab_size) as u32);
                }
            } else {
                for j in 0..needed {
                    tokens.push(allowed[(tokens.len() + j) % allowed.len()]);
                }
            }
        } else {
            tokens.truncate(target_len);
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common_prefix_bytes(strings: &[&str]) -> usize {
        if strings.is_empty() {
            return 0;
        }
        let first = strings[0].as_bytes();
        let mut len = first.len();
        for s in &strings[1..] {
            let b = s.as_bytes();
            len = len.min(b.len());
            for i in 0..len {
                if first[i] != b[i] {
                    len = i;
                    break;
                }
            }
        }
        len
    }

    #[test]
    #[ignore]
    fn test_prefix_sharing_structure() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 5,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 0,
            output_len: 100,
            seed: 42,
            request_id_prefix: "test-".to_string(),
            prefix_sharing_config: Some(PrefixSharingConfig {
                global_ratio: 0.1,
                conversation_ratio: 0.8,
            }),
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 5);

        let messages: Vec<Vec<&str>> = conversations
            .iter()
            .map(|c| c.turns.iter().map(|t| &*t.user_message).collect())
            .collect();

        // 1. Global prefix: all messages share a common prefix
        let all_msgs: Vec<&str> = messages.iter().flat_map(|v| v.iter().copied()).collect();
        let global_prefix = common_prefix_bytes(&all_msgs);
        println!("Global prefix bytes: {global_prefix}");
        assert!(global_prefix > 0, "Global prefix must be non-empty");

        // 2. Conversation prefix: turns within same conversation share more
        for (i, conv_msgs) in messages.iter().enumerate() {
            let conv_prefix = common_prefix_bytes(conv_msgs);
            println!("Conv {i} prefix bytes: {conv_prefix} (global: {global_prefix})");
            assert!(
                conv_prefix > global_prefix,
                "Conv prefix ({conv_prefix}) must exceed global prefix ({global_prefix})"
            );
        }

        // 3. Different conversations diverge after global prefix
        let cross = common_prefix_bytes(&[messages[0][0], messages[1][0]]);
        let within = common_prefix_bytes(&messages[0]);
        println!("Cross-conv prefix: {cross}, within-conv prefix: {within}");
        assert!(
            cross < within,
            "Cross-conv ({cross}) must be < within-conv ({within})"
        );

        // 4. Turns within same conversation are not identical (unique suffix)
        for (i, conv_msgs) in messages.iter().enumerate() {
            for a in 0..conv_msgs.len() {
                for b in (a + 1)..conv_msgs.len() {
                    assert_ne!(
                        conv_msgs[a], conv_msgs[b],
                        "Conv {i} turn {a} and {b} must differ"
                    );
                }
            }
        }

        // 5. Token lengths approximately match target
        for (i, conv) in conversations.iter().enumerate() {
            for (j, turn) in conv.turns.iter().enumerate() {
                let diff = (turn.user_message_len as i64 - 1000).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (diff {diff})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 10,
                    "Token len {} too far from 1000",
                    turn.user_message_len
                );
            }
        }

        println!("All prefix sharing checks passed!");
    }

    #[test]
    #[ignore]
    fn test_per_turn_input_len_default_mode() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 4,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 512,
            per_turn_input_len: 128,
            output_len: 64,
            seed: 1,
            request_id_prefix: "test-".to_string(),
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 4);

        for (i, conv) in conversations.iter().enumerate() {
            assert_eq!(conv.turns.len(), 3);
            for (j, turn) in conv.turns.iter().enumerate() {
                let expected = if j == 0 { 512usize } else { 128usize };
                let diff = (turn.user_message_len as i64 - expected as i64).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (expected ~{expected}, diff {diff})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 5,
                    "Conv {i} turn {j}: token len {} too far from {expected}",
                    turn.user_message_len
                );
            }
        }
        println!("per_turn_input_len default-mode checks passed!");
    }

    #[test]
    #[ignore]
    fn test_variable_turns_range() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 50,
            min_turns: 2,
            max_turns: 5,
            prefix_len: 0,
            input_len: 256,
            per_turn_input_len: 0,
            output_len: 32,
            seed: 7,
            request_id_prefix: "test-".to_string(),
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 50);

        let mut distinct_counts = std::collections::HashSet::new();
        for conv in &conversations {
            let n = conv.turns.len();
            assert!((2..=5).contains(&n), "turn count {n} out of [2,5]");
            distinct_counts.insert(n);
        }
        assert!(
            distinct_counts.len() >= 2,
            "expected at least 2 distinct turn counts, got {distinct_counts:?}"
        );
        println!("variable_turns_range checks passed! counts: {distinct_counts:?}");
    }

    #[test]
    #[ignore]
    fn test_variable_turns_fixed() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 10,
            min_turns: 4,
            max_turns: 4,
            prefix_len: 0,
            input_len: 256,
            per_turn_input_len: 0,
            output_len: 32,
            seed: 42,
            request_id_prefix: "test-".to_string(),
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        for conv in &conversations {
            assert_eq!(conv.turns.len(), 4, "expected exactly 4 turns");
        }
        println!("variable_turns_fixed checks passed!");
    }

    #[test]
    #[ignore]
    fn test_per_turn_input_len_prefix_sharing() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        // Turn 0 input_len=1000, turns 1+ per_turn_input_len=600
        // global_len ≈ 100 (10%), conv_len ≈ 800 (80%), unique ≈ 100
        // per-turn unique ≈ 600 - 900 = negative → would error; use smaller ratios
        // global=0.05 (50), conv=0.5 (500), unique_t0=450, unique_t1=600-550=50
        let cfg = MultiTurnRandomConfig {
            num_conversations: 4,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 600,
            output_len: 64,
            seed: 3,
            request_id_prefix: "test-".to_string(),
            prefix_sharing_config: Some(PrefixSharingConfig {
                global_ratio: 0.05,
                conversation_ratio: 0.50,
            }),
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 4);

        let messages: Vec<Vec<&str>> = conversations
            .iter()
            .map(|c| c.turns.iter().map(|t| &*t.user_message).collect())
            .collect();

        // Global prefix bytes shared across all turns of all conversations
        let all_msgs: Vec<&str> = messages.iter().flat_map(|v| v.iter().copied()).collect();
        let global_prefix = common_prefix_bytes(&all_msgs);
        assert!(global_prefix > 0, "Global prefix must be non-empty");

        // Within each conversation, prefix grows (conv prefix longer than global)
        for (i, conv_msgs) in messages.iter().enumerate() {
            let conv_prefix = common_prefix_bytes(conv_msgs);
            assert!(
                conv_prefix > global_prefix,
                "Conv {i}: conv_prefix ({conv_prefix}) must exceed global ({global_prefix})"
            );
        }

        // Turn 0 length ≈ 1000, turns 1+ ≈ 600
        for (i, conv) in conversations.iter().enumerate() {
            for (j, turn) in conv.turns.iter().enumerate() {
                let expected = if j == 0 { 1000usize } else { 600usize };
                let diff = (turn.user_message_len as i64 - expected as i64).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (expected ~{expected})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 10,
                    "Conv {i} turn {j}: token len {} too far from {expected}",
                    turn.user_message_len
                );
            }
        }
        println!("per_turn_input_len prefix-sharing checks passed!");
    }
}
