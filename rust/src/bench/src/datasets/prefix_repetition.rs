// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prefix repetition dataset: N distinct shared prefixes, each reused by
//! `num_prompts / num_prefixes` requests with a fresh random suffix.
//! The standard prefix-cache stress workload; mirrors Python's
//! `PrefixRepetitionRandomDataset`.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::SampleRequest;
use super::random::gen_prompt_decode_to_target_len;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Generate the prefix repetition dataset.
///
/// Like Python, `num_requests % num_prefixes` remainder requests are dropped:
/// the total is `(num_requests / num_prefixes) * num_prefixes`.
pub fn generate_prefix_repetition_dataset(
    tokenizer: &TokenizerKind,
    num_requests: usize,
    prefix_len: usize,
    suffix_len: usize,
    num_prefixes: usize,
    output_len: usize,
    seed: u64,
    request_id_prefix: &str,
    disable_shuffle: bool,
) -> Result<Vec<SampleRequest>> {
    let prompts_per_prefix = num_requests / num_prefixes;
    if prompts_per_prefix == 0 {
        return Err(BenchError::Config(format!(
            "num_prompts ({num_requests}) must be >= num_prefixes ({num_prefixes})"
        )));
    }
    let total = prompts_per_prefix * num_prefixes;
    if total != num_requests {
        println!(
            "prefix_repetition: generating {total} requests \
             ({num_prefixes} prefixes x {prompts_per_prefix} prompts each; \
             {} dropped to divide evenly)",
            num_requests - total
        );
    }

    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }
    let allowed_ref = &allowed_tokens;

    // Exact-length random token block: decode -> re-encode -> converge to target.
    let gen_block = |target_len: usize, item_seed: u64| -> Result<Vec<u32>> {
        let mut rng = StdRng::seed_from_u64(item_seed);
        let tokens: Vec<u32> = (0..target_len)
            .map(|_| allowed_ref[rng.random_range(0..allowed_ref.len())])
            .collect();
        let (_, adjusted) =
            gen_prompt_decode_to_target_len(tokenizer, &tokens, target_len, false, allowed_ref)?;
        Ok(adjusted)
    };

    // Generate the shared prefixes (one per group), then suffixes in parallel.
    let prefixes: Vec<Vec<u32>> = (0..num_prefixes)
        .map(|p| gen_block(prefix_len, seed.wrapping_add(0xF1F0).wrapping_add(p as u64)))
        .collect::<Result<Vec<_>>>()?;

    let rid_prefix = request_id_prefix.to_string();
    let mut requests: Vec<SampleRequest> = (0..total)
        .into_par_iter()
        .map(|i| {
            let prefix_tokens = &prefixes[i / prompts_per_prefix];
            let suffix_tokens = gen_block(suffix_len, seed.wrapping_add(0xBEEF + i as u64))?;

            let mut combined = Vec::with_capacity(prefix_tokens.len() + suffix_tokens.len());
            combined.extend_from_slice(prefix_tokens);
            combined.extend_from_slice(&suffix_tokens);
            let prompt = tokenizer.decode(&combined, true)?;

            Ok(SampleRequest {
                prompt: Arc::from(prompt),
                prompt_len: combined.len(),
                expected_output_len: output_len,
                request_id: Some(format!("{rid_prefix}{i}")),
                ..Default::default()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Interleave prefixes (Python shuffles too) so one prefix group isn't sent
    // as a contiguous burst.
    if !disable_shuffle {
        let mut rng = StdRng::seed_from_u64(seed);
        requests.shuffle(&mut rng);
    }

    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gpt2 via built-in tiktoken encoding — loads without network access.
    fn test_tokenizer() -> TokenizerKind {
        crate::tokenizer::load_tokenizer("gpt2", false, None)
            .expect("gpt2 built-in tiktoken should always load without network")
    }

    #[test]
    fn test_prefix_repetition_structure() {
        let tok = test_tokenizer();
        // 7 requests / 3 prefixes -> 2 per prefix, 6 total (remainder dropped like Python)
        let reqs = generate_prefix_repetition_dataset(&tok, 7, 32, 16, 3, 64, 0, "t-", true)
            .expect("generation should succeed");
        assert_eq!(reqs.len(), 6);
        assert!(reqs.iter().all(|r| r.expected_output_len == 64));
        // Exact-length blocks: prompt_len == prefix + suffix
        assert!(
            reqs.iter().all(|r| r.prompt_len == 32 + 16),
            "lens: {:?}",
            reqs.iter().map(|r| r.prompt_len).collect::<Vec<_>>()
        );
        // Consecutive pairs (shuffle disabled) share a common prefix; requests
        // from different groups don't.
        let common = |a: &str, b: &str| -> usize {
            a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
        };
        let same_group = common(&reqs[0].prompt, &reqs[1].prompt);
        let diff_group = common(&reqs[0].prompt, &reqs[2].prompt);
        assert!(
            same_group > diff_group,
            "same-group shared prefix ({same_group}) should exceed cross-group ({diff_group})"
        );
    }

    #[test]
    fn test_prefix_repetition_too_few_requests_errors() {
        let tok = test_tokenizer();
        assert!(generate_prefix_repetition_dataset(&tok, 2, 32, 16, 3, 64, 0, "t-", true).is_err());
    }
}
