// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Custom dataset: JSONL file with one request per line.
//!
//! ```jsonl
//! {"prompt": "What is the capital of India?", "output_tokens": 10}
//! {"prompt": "What is the capital of Iran?", "output_tokens": 1520}
//! ```
//!
//! Mirrors Python's `CustomDataset`. `output_tokens` is optional unless
//! `--custom-output-len -1` is passed. Unlike Python, prompts are always sent
//! raw (no client-side chat template; see `--skip-chat-template`).

use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Deserialize;

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

#[derive(Deserialize)]
struct CustomLine {
    prompt: String,
    output_tokens: Option<serde_json::Value>,
}

/// Load the custom JSONL dataset.
///
/// `output_len < 0` means "use the per-line output_tokens field" (Python's
/// `--custom-output-len -1`); otherwise `output_len` applies to every request.
pub fn load_custom_dataset(
    tokenizer: &TokenizerKind,
    path: &str,
    num_requests: usize,
    output_len: i64,
    seed: u64,
    request_id_prefix: &str,
    no_oversample: bool,
    disable_shuffle: bool,
) -> Result<Vec<SampleRequest>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| BenchError::Config(format!("Failed to read custom dataset '{path}': {e}")))?;

    let mut lines: Vec<CustomLine> = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: CustomLine = serde_json::from_str(line).map_err(|e| {
            BenchError::Config(format!(
                "Invalid JSONL at {path}:{}: {e} (each line must be an object \
                 with a 'prompt' field)",
                lineno + 1
            ))
        })?;
        lines.push(parsed);
    }
    if lines.is_empty() {
        return Err(BenchError::Config(format!(
            "Custom dataset '{path}' contains no entries"
        )));
    }

    // Python shuffles the loaded data (seeded) before taking num_requests.
    if !disable_shuffle {
        let mut rng = StdRng::seed_from_u64(seed);
        lines.shuffle(&mut rng);
    }

    let mut requests: Vec<SampleRequest> = Vec::with_capacity(num_requests.min(lines.len()));
    for (i, item) in lines.iter().enumerate() {
        if requests.len() >= num_requests {
            break;
        }

        let expected_output_len = if output_len < 0 {
            let raw = item.output_tokens.as_ref().ok_or_else(|| {
                BenchError::Config(
                    "custom dataset: --custom-output-len -1 requires an \
                     'output_tokens' field on every line"
                        .into(),
                )
            })?;
            raw.as_i64().filter(|v| *v > 0).ok_or_else(|| {
                BenchError::Config(format!(
                    "custom dataset: invalid 'output_tokens' value {raw}: \
                         must be a positive integer"
                ))
            })? as usize
        } else {
            output_len as usize
        };

        let prompt_len = tokenizer.encode(&item.prompt, true)?.len();
        requests.push(SampleRequest {
            prompt: Arc::from(item.prompt.as_str()),
            prompt_len,
            expected_output_len,
            request_id: Some(format!("{request_id_prefix}{i}")),
            ..Default::default()
        });
    }

    super::oversample_requests(
        &mut requests,
        num_requests,
        request_id_prefix,
        no_oversample,
    );
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_jsonl(name: &str, content: &str) -> String {
        let path = std::env::temp_dir().join(format!("vllm-bench-custom-{name}.jsonl"));
        std::fs::write(&path, content).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// gpt2 via built-in tiktoken encoding — loads without network access.
    fn test_tokenizer() -> TokenizerKind {
        crate::tokenizer::load_tokenizer("gpt2", false, None)
            .expect("gpt2 built-in tiktoken should always load without network")
    }

    #[test]
    fn test_load_custom_dataset_basic() {
        let path = write_temp_jsonl(
            "basic",
            r#"{"prompt": "hello world", "output_tokens": 10}
{"prompt": "foo bar baz", "output_tokens": 20}
"#,
        );
        let reqs = load_custom_dataset(&test_tokenizer(), &path, 2, 256, 0, "t-", true, true)
            .expect("load should succeed");
        assert_eq!(reqs.len(), 2);
        // Fixed output_len (256) wins over per-line output_tokens by default
        assert!(reqs.iter().all(|r| r.expected_output_len == 256));
        assert_eq!(&*reqs[0].prompt, "hello world");
        assert!(reqs[0].prompt_len > 0);
    }

    #[test]
    fn test_load_custom_dataset_per_line_output_tokens() {
        let path = write_temp_jsonl(
            "perline",
            r#"{"prompt": "hello", "output_tokens": 10}
{"prompt": "world", "output_tokens": 20}
"#,
        );
        let reqs = load_custom_dataset(&test_tokenizer(), &path, 2, -1, 0, "t-", true, true)
            .expect("load should succeed");
        assert_eq!(reqs[0].expected_output_len, 10);
        assert_eq!(reqs[1].expected_output_len, 20);
    }

    #[test]
    fn test_load_custom_dataset_missing_output_tokens_errors() {
        let path = write_temp_jsonl("missing", r#"{"prompt": "hello"}"#);
        let err = load_custom_dataset(&test_tokenizer(), &path, 1, -1, 0, "t-", true, true)
            .expect_err("should fail without output_tokens");
        assert!(err.to_string().contains("output_tokens"));
    }

    #[test]
    fn test_load_custom_dataset_missing_prompt_errors() {
        let path = write_temp_jsonl("noprompt", r#"{"text": "hello"}"#);
        assert!(
            load_custom_dataset(&test_tokenizer(), &path, 1, 256, 0, "t-", true, true).is_err()
        );
    }
}
