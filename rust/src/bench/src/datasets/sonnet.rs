// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Default values mirror Python's SonnetDataset defaults (datasets.py).
pub const DEFAULT_PREFIX_LEN: usize = 200;
pub const DEFAULT_INPUT_LEN: usize = 550;
pub const DEFAULT_OUTPUT_LEN: usize = 150;

const BASE_PROMPT: &str = "Pick as many lines as you can from these poem lines:\n";

/// Shakespeare's sonnets, public domain. Bundled so `--dataset-name sonnet` works
/// out of the box without `--dataset-path`. Source:
/// https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt
const BUILTIN_SONNET: &str = include_str!("sonnet.txt");

/// Load the sonnet dataset and generate `num_requests` prompts targeting `input_len`
/// total prompt tokens.
///
/// Mirrors Python's `SonnetDataset.sample()` from `vllm/benchmarks/datasets/datasets.py`.
/// The Rust port skips `apply_chat_template` (no Jinja runtime here): `base_offset`
/// and `prompt_len` are computed from the raw text. The resulting prompts are slightly
/// shorter than Python's chat-template-formatted version (off by the chat scaffolding
/// tokens, typically <20).
pub fn load_sonnet_dataset(
    tokenizer: &TokenizerKind,
    dataset_path: Option<&str>,
    num_requests: usize,
    input_len: usize,
    output_len: usize,
    prefix_len: usize,
    seed: u64,
    request_id_prefix: &str,
) -> Result<Vec<SampleRequest>> {
    let content = match dataset_path {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| BenchError::Config(format!("Failed to read sonnet file '{path}': {e}")))?,
        None => BUILTIN_SONNET.to_string(),
    };

    // Match Python's f.readlines(): keep trailing newlines so that joining lines
    // reconstructs the original text without inserting extra separators.
    let lines: Vec<String> = content.split_inclusive('\n').map(|s| s.to_string()).collect();

    if lines.is_empty() {
        let src = dataset_path.unwrap_or("<built-in>");
        return Err(BenchError::Config(format!("Sonnet file '{src}' is empty")));
    }

    // Average tokens per line (used to estimate how many lines to draw).
    let mut total_tokens: usize = 0;
    for line in &lines {
        let ids = tokenizer.encode(line, false)?;
        total_tokens += ids.len();
    }
    let avg_len = total_tokens as f64 / lines.len() as f64;
    if avg_len <= 0.0 {
        return Err(BenchError::Config(
            "Sonnet lines tokenized to zero tokens on average".into(),
        ));
    }

    let base_ids = tokenizer.encode(BASE_PROMPT, false)?;
    let base_offset = base_ids.len();
    if input_len <= base_offset {
        return Err(BenchError::Config(format!(
            "--sonnet-input-len ({input_len}) must be larger than the base prompt length ({base_offset})"
        )));
    }

    let num_input_lines = ((input_len - base_offset) as f64 / avg_len).round() as i64;
    let num_prefix_lines =
        (((prefix_len as i64 - base_offset as i64) as f64) / avg_len).round().max(0.0) as i64;
    let num_input_lines = num_input_lines.max(0) as usize;
    let num_prefix_lines = (num_prefix_lines as usize).min(lines.len());
    let num_input_lines = num_input_lines.max(num_prefix_lines);

    let prefix_lines: &[String] = &lines[..num_prefix_lines];
    let extras_per_request = num_input_lines - num_prefix_lines;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(num_requests);
    let mut ind = 0usize;
    let mut attempts = 0usize;
    let max_attempts = num_requests.saturating_mul(20).max(1000);

    while samples.len() < num_requests {
        if attempts >= max_attempts {
            return Err(BenchError::Config(format!(
                "Could not assemble {num_requests} sonnet prompts under input_len={input_len} \
                 after {attempts} attempts. Try increasing --sonnet-input-len."
            )));
        }
        attempts += 1;

        let mut prompt = String::with_capacity(BASE_PROMPT.len() + 256 * num_input_lines);
        prompt.push_str(BASE_PROMPT);
        for line in prefix_lines {
            prompt.push_str(line);
        }
        for _ in 0..extras_per_request {
            // random.choices with replacement — duplicates are allowed.
            let line = lines.choose(&mut rng).unwrap();
            prompt.push_str(line);
        }

        let prompt_ids = tokenizer.encode(&prompt, false)?;
        let prompt_len = prompt_ids.len();
        if prompt_len <= input_len {
            samples.push(SampleRequest {
                prompt: Arc::from(prompt),
                prompt_len,
                expected_output_len: output_len,
                request_id: Some(format!("{request_id_prefix}{ind}")),
                ..Default::default()
            });
            ind += 1;
        }
    }

    Ok(samples)
}
