// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rayon::prelude::*;

use super::SampleRequest;
use crate::config::RangeRatio;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Generate random dataset with rayon parallelism.
///
/// This is the key performance win — Python does sequential tokenizer calls
/// while Rust parallelizes across CPU cores with native tokenizer speed.
///
/// Mirrors Python's RandomDataset.sample() from datasets.py:470-560.
pub fn generate_random_dataset(
    tokenizer: &TokenizerKind,
    num_requests: usize,
    input_len: usize,
    output_len: usize,
    prefix_len: usize,
    range_ratio: RangeRatio,
    cache_hit_fraction: f64,
    cache_ratio: f64,
    seed: u64,
    request_id_prefix: &str,
    use_token_ids: bool,
    batch_size: usize,
) -> Result<Vec<SampleRequest>> {
    let vocab_size = tokenizer.vocab_size();
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }

    if batch_size > 1 && use_token_ids {
        return Err(BenchError::Config(
            "--random-batch-size > 1 is not supported with --prompt-token-ids".into(),
        ));
    }

    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = input_len.saturating_sub(num_special);

    // Python semantics: sample uniformly from [len*(1-r), len*(1+r)].
    let (input_low, input_high) = range_ratio.input_bounds(real_input_len);
    let (output_low, output_high) = range_ratio.output_bounds(output_len);
    if !range_ratio.is_fixed() {
        println!(
            "Sampling input_len from [{input_low}, {input_high}] and \
             output_len from [{output_low}, {output_high}]"
        );
    }

    // Bimodal prefix-cache mode: a fraction of prompts (warm) reuse a shared cached
    // prefix covering `cache_ratio` of their length; the rest (cold) are fully unique.
    // Models e.g. "80% of prompts have 95% of input cached" with
    // --random-cache-hit-fraction 0.8 --random-cache-ratio 0.95. In this mode
    // --random-input-len is the TOTAL prompt length L (the cached prefix is part of L),
    // and --random-prefix-len is ignored.
    let bimodal = cache_hit_fraction > 0.0 && cache_ratio > 0.0;
    if bimodal {
        if !use_token_ids {
            return Err(BenchError::Config(
                "bimodal prefix-cache (--random-cache-hit-fraction) requires --prompt-token-ids \
                 so warm prompts send identical token IDs and actually hit the prefix cache"
                    .into(),
            ));
        }
        if cache_hit_fraction > 1.0 || cache_ratio > 1.0 {
            return Err(BenchError::Config(
                "--random-cache-hit-fraction and --random-cache-ratio must be in [0, 1]".into(),
            ));
        }
    }

    // Length of the shared cached base prefix.
    let base_len = if bimodal {
        ((input_high as f64) * cache_ratio).ceil() as usize
    } else {
        prefix_len
    };

    // Validate (non-bimodal keeps the original check)
    if !bimodal {
        let min_total = prefix_len + input_low;
        if min_total < 1 {
            return Err(BenchError::Config(format!(
                "--random-input-len too small: with {num_special} special tokens and \
                 range_ratio={:?}, minimum total input is {min_total}",
                range_ratio
            )));
        }
    }

    // Generate the shared base prefix once (sequential, only happens once).
    let prefix_token_ids = if base_len > 0 {
        generate_prefix(tokenizer, &allowed_tokens, base_len, seed)?
    } else {
        Vec::new()
    };

    // Pre-generate per-request sampling params using deterministic RNG
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);

    struct RequestParams {
        cached_len: usize, // tokens taken from the shared base (cache-hittable)
        suffix_len: usize, // unique tokens appended after the cached prefix
        output_len: usize,
        offset: usize,
    }

    let params: Vec<RequestParams> = (0..num_requests)
        .map(|_| {
            let ol = if output_low == output_high {
                output_low
            } else {
                rng.random_range(output_low..=output_high)
            };
            let off = rng.random_range(0..vocab_size as usize);
            if bimodal {
                // Total length L from the input distribution; prefix is part of L.
                let l = if input_low == input_high {
                    input_low
                } else {
                    rng.random_range(input_low..=input_high)
                };
                let warm = rng.random::<f64>() < cache_hit_fraction;
                let cached = if warm {
                    (((l as f64) * cache_ratio).round() as usize).min(base_len).min(l)
                } else {
                    0
                };
                RequestParams {
                    cached_len: cached,
                    suffix_len: l - cached,
                    output_len: ol,
                    offset: off,
                }
            } else {
                // Original behavior: full shared prefix + variable unique input.
                let il = if input_low == input_high {
                    input_low
                } else {
                    rng.random_range(input_low..=input_high)
                };
                RequestParams {
                    cached_len: prefix_len,
                    suffix_len: il,
                    output_len: ol,
                    offset: off,
                }
            }
        })
        .collect();

    // Phase 1: Generate all token sequences (parallel, fast — just array ops)
    let prefix_ref = &prefix_token_ids;
    let allowed_ref = &allowed_tokens;
    let rid_prefix = request_id_prefix.to_string();

    let token_sequences: Vec<Vec<u32>> = params
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let at_len = allowed_ref.len();
            let mut seq = Vec::with_capacity(p.cached_len + p.suffix_len);
            seq.extend_from_slice(&prefix_ref[..p.cached_len]);
            for j in 0..p.suffix_len {
                seq.push(allowed_ref[(p.offset + i + j) % at_len]);
            }
            seq
        })
        .collect();

    let target_lens: Vec<usize> = params.iter().map(|p| p.cached_len + p.suffix_len).collect();

    if use_token_ids {
        // Fast path: store token IDs directly. The completions backend sends
        // them as `"prompt": [id1, id2, ...]`, bypassing both client-side decode
        // and server-side tokenization. Token counts are exact by construction.
        let result: Vec<SampleRequest> = token_sequences
            .into_par_iter()
            .enumerate()
            .map(|(i, tokens)| SampleRequest {
                prompt: Arc::from(""),
                prompt_len: target_lens[i],
                expected_output_len: params[i].output_len,
                request_id: Some(format!("{rid_prefix}{i}")),
                prompt_token_ids: Some(Arc::from(tokens)),
                ..Default::default()
            })
            .collect();
        Ok(result)
    } else {
        // Default path: decode tokens to text, re-encode,
        // truncate to target length, decode again. Sends text prompts for maximum
        let result: Vec<SampleRequest> = token_sequences
            .into_par_iter()
            .enumerate()
            .map(|(i, tokens)| {
                let target = target_lens[i];
                // decode → encode → truncate → decode
                let prompt_text = tokenizer.decode(&tokens, true)?;
                let mut re_encoded = tokenizer.encode(&prompt_text, false)?;
                re_encoded.truncate(target);
                let prompt = tokenizer.decode(&re_encoded, true)?;
                let prompt_len = re_encoded.len();
                Ok(SampleRequest {
                    prompt: Arc::from(prompt),
                    prompt_len,
                    expected_output_len: params[i].output_len,
                    request_id: Some(format!("{rid_prefix}{i}")),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(super::batch_requests(result, batch_size, request_id_prefix))
    }
}

fn generate_prefix(
    tokenizer: &TokenizerKind,
    allowed_tokens: &[u32],
    prefix_len: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD));
    let tokens: Vec<u32> = (0..prefix_len)
        .map(|_| allowed_tokens[rng.random_range(0..allowed_tokens.len())])
        .collect();

    let (_, adjusted) =
        gen_prompt_decode_to_target_len(tokenizer, &tokens, prefix_len, false, allowed_tokens)?;
    Ok(adjusted)
}

/// Ensure decoded-then-encoded prompt length matches the target.
///
/// Mirrors Python's `gen_prompt_decode_to_target_len` from datasets.py:381-435.
pub(crate) fn gen_prompt_decode_to_target_len(
    tokenizer: &TokenizerKind,
    token_sequence: &[u32],
    target_len: usize,
    add_special_tokens: bool,
    allowed_tokens: &[u32],
) -> Result<(String, Vec<u32>)> {
    let max_retry = 20;
    let mut tokens = token_sequence.to_vec();

    for retry in 0..=max_retry {
        let prompt = tokenizer.decode(&tokens, true)?;
        tokens = tokenizer.encode(&prompt, add_special_tokens)?;

        if retry >= max_retry {
            if tokens.len() != target_len {
                return Err(BenchError::Tokenizer(format!(
                    "Token length mismatch after {max_retry} retries: \
                     target={target_len}, actual={}. \
                     encode/decode roundtrip cannot converge.",
                    tokens.len()
                )));
            }
            return Ok((prompt, tokens));
        }

        if tokens.len() == target_len {
            return Ok((prompt, tokens));
        } else if tokens.len() < target_len {
            // Pad with tokens from the allowed set (UTF-8-safe for tiktoken)
            let needed = target_len - tokens.len();
            if allowed_tokens.is_empty() {
                let vocab_size = tokenizer.vocab_size() as usize;
                for j in 0..needed {
                    tokens.push(((tokens.len() + j) % vocab_size) as u32);
                }
            } else {
                for j in 0..needed {
                    tokens.push(allowed_tokens[(tokens.len() + j) % allowed_tokens.len()]);
                }
            }
        } else {
            // Truncate
            tokens.truncate(target_len);
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer;

    // Integration test requires a tokenizer, so only run with --ignored
    #[test]
    #[ignore]
    fn test_generate_random_dataset_token_ids() {
        let tokenizer = tokenizer::load_tokenizer("gpt2", false, None).unwrap();
        let requests = generate_random_dataset(
            &tokenizer,
            10,  // num_requests
            128, // input_len
            32,  // output_len
            0,   // prefix_len
            RangeRatio {
                input: 0.0,
                output: 0.0,
            }, // range_ratio (0.0 = fixed length)
            0.0, // cache_hit_fraction (0 = bimodal off)
            0.0, // cache_ratio
            42,  // seed
            "test-",
            true, // use_token_ids
            1,    // batch_size
        )
        .unwrap();

        assert_eq!(requests.len(), 10);
        for req in &requests {
            assert!(req.prompt_token_ids.is_some());
            assert_eq!(req.prompt_token_ids.as_ref().unwrap().len(), req.prompt_len);
            assert!(req.prompt_len > 0);
            assert_eq!(req.expected_output_len, 32);
        }
    }

    #[test]
    #[ignore]
    fn test_generate_random_dataset_text() {
        let tokenizer = tokenizer::load_tokenizer("gpt2", false, None).unwrap();
        let requests = generate_random_dataset(
            &tokenizer,
            10,  // num_requests
            128, // input_len
            32,  // output_len
            0,   // prefix_len
            RangeRatio {
                input: 0.0,
                output: 0.0,
            }, // range_ratio (0.0 = fixed length)
            0.0, // cache_hit_fraction (0 = bimodal off)
            0.0, // cache_ratio
            42,  // seed
            "test-",
            false, // use_token_ids = false → text prompts
            1,     // batch_size
        )
        .unwrap();

        assert_eq!(requests.len(), 10);
        for req in &requests {
            assert!(req.prompt_token_ids.is_none());
            assert!(!req.prompt.is_empty());
            assert!(req.prompt_len > 0);
            assert!(req.prompt_len <= 128);
            assert_eq!(req.expected_output_len, 32);
        }
    }

    /// Test that generated prompts have EXACT target token length (token ID mode).
    #[test]
    #[ignore]
    fn test_token_length_exact_local() {
        let tokenizer = tokenizer::load_tokenizer("gpt2", false, None).unwrap();
        let target_len = 512;
        let requests = generate_random_dataset(
            &tokenizer,
            50,
            target_len,
            64,
            0,
            RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            0.0,
            0.0,
            123,
            "len-test-",
            true,
            1,
        )
        .unwrap();

        for (i, req) in requests.iter().enumerate() {
            let token_ids = req.prompt_token_ids.as_ref().expect("should have token IDs");
            assert_eq!(
                token_ids.len(),
                target_len,
                "Request {i}: expected {target_len} token IDs, got {}",
                token_ids.len()
            );
            assert_eq!(req.prompt_len, target_len);
        }
    }

    /// Test that tiktoken tokenizer produces exact target token lengths (token ID mode).
    #[test]
    #[ignore]
    fn test_token_length_exact_tiktoken() {
        // Use Qwen2.5 which has a tiktoken-format tokenizer
        let tokenizer = tokenizer::load_tokenizer("Qwen/Qwen2.5-0.5B", false, None);
        let tokenizer = match tokenizer {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Skipping tiktoken test (tokenizer unavailable): {e}");
                return;
            }
        };

        // Verify it's actually a tiktoken tokenizer or local — either way test convergence
        let target_len = 256;
        let requests = generate_random_dataset(
            &tokenizer,
            20,
            target_len,
            32,
            0,
            RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            0.0,
            0.0,
            42,
            "tiktoken-test-",
            true,
            1,
        )
        .unwrap();

        for (i, req) in requests.iter().enumerate() {
            let token_ids = req.prompt_token_ids.as_ref().expect("should have token IDs");
            assert_eq!(
                token_ids.len(),
                target_len,
                "Request {i}: expected {target_len} token IDs, got {}",
                token_ids.len()
            );
            assert_eq!(req.prompt_len, target_len);
        }
    }

    /// Test encode/decode roundtrip stability for tiktoken.
    /// After one decode→encode cycle with UTF-8-safe tokens, length must not drift.
    #[test]
    #[ignore]
    fn test_tiktoken_roundtrip_stability() {
        let tokenizer = tokenizer::load_tokenizer("Qwen/Qwen2.5-0.5B", false, None);
        let tokenizer = match tokenizer {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Skipping roundtrip test (tokenizer unavailable): {e}");
                return;
            }
        };

        let allowed = tokenizer.get_allowed_tokens();
        assert!(!allowed.is_empty(), "allowed tokens should not be empty");

        // Build a sequence from allowed tokens only
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(99);
        let seq: Vec<u32> = (0..512).map(|_| allowed[rng.random_range(0..allowed.len())]).collect();

        let decoded = tokenizer.decode(&seq, true).unwrap();
        let re_encoded = tokenizer.encode(&decoded, false).unwrap();
        let re_decoded = tokenizer.decode(&re_encoded, true).unwrap();
        let re_re_encoded = tokenizer.encode(&re_decoded, false).unwrap();

        // After first cycle, length should stabilize
        assert_eq!(
            re_encoded.len(),
            re_re_encoded.len(),
            "Roundtrip should stabilize: first re-encode={}, second re-encode={}",
            re_encoded.len(),
            re_re_encoded.len()
        );
    }
}
