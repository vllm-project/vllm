// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Timed-trace dataset: JSONL replay trace with per-request arrival timestamps.
//!
//! ```jsonl
//! {"timestamp": 12.4, "input_length": 42, "output_length": 8, "hash_ids": [981, 4412, 7]}
//! ```
//!
//! Mirrors Python's `TimedTrace`. Prompts carry no text: each `hash_id` expands
//! to a pseudo-random token chunk of `chunk_size` tokens (the last chunk may be
//! partial so the prompt length equals `input_length` exactly). Chunks are
//! memoized per (hash_id, size), so requests sharing a hash prefix get
//! byte-identical token prefixes and prefix-cache hits reproduce. Field names
//! are remappable via `--timed-trace-label-*`; timestamps are scaled by
//! `--timed-trace-sec-multiplier`.
//!
//! Deviation from Python: chunk contents are seeded from (hash_id, size) with a
//! fixed mixing function, so they are bit-reproducible across runs. Python
//! seeds via `hash(f"{h}:{size}")`, which is randomized per process unless
//! PYTHONHASHSEED is pinned.

use std::collections::HashMap;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

pub struct TimedTraceOptions<'a> {
    pub chunk_size: usize,
    pub sec_multiplier: f64,
    pub label_timestamp: &'a str,
    pub label_input_length: &'a str,
    pub label_output_length: &'a str,
    pub label_hash_ids: &'a str,
}

/// Load a timed-trace JSONL dataset. Row order (and thus the timestamp
/// schedule) is preserved: no shuffle, no oversampling.
pub fn load_timed_trace_dataset(
    tokenizer: &TokenizerKind,
    path: &str,
    num_requests: usize,
    request_id_prefix: &str,
    opts: &TimedTraceOptions<'_>,
) -> Result<Vec<SampleRequest>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| BenchError::Config(format!("Failed to read timed trace '{path}': {e}")))?;

    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Config(
            "timed_trace: tokenizer reports an empty vocabulary".into(),
        ));
    }

    let mut chunk_cache: HashMap<(i64, usize), Arc<[u32]>> = HashMap::new();
    let mut requests: Vec<SampleRequest> = Vec::new();

    for (lineno, line) in content.lines().enumerate() {
        if requests.len() >= num_requests {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = serde_json::from_str(line).map_err(|e| {
            BenchError::Config(format!("Invalid JSONL at {path}:{}: {e}", lineno + 1))
        })?;

        let input_length = require_u64(
            &entry,
            opts.label_input_length,
            "--timed-trace-label-input-length",
        )? as usize;
        let output_length = require_u64(
            &entry,
            opts.label_output_length,
            "--timed-trace-label-output-length",
        )? as usize;
        let timestamp = require_f64(
            &entry,
            opts.label_timestamp,
            "--timed-trace-label-timestamp",
        )? * opts.sec_multiplier;
        let hash_ids = parse_hash_ids(&entry, opts.label_hash_ids)?;

        let prompt_ids = expand_prompt(
            &hash_ids,
            input_length,
            opts.chunk_size,
            &allowed_tokens,
            &mut chunk_cache,
        );

        requests.push(SampleRequest {
            prompt_len: prompt_ids.len(),
            prompt_token_ids: Some(Arc::from(prompt_ids)),
            expected_output_len: output_length,
            request_id: Some(format!("{request_id_prefix}{lineno}")),
            timestamp: Some(timestamp),
            ..Default::default()
        });
    }

    if requests.is_empty() {
        return Err(BenchError::Config(format!(
            "Timed trace '{path}' contains no entries"
        )));
    }
    Ok(requests)
}

/// Expand prefix-hash ids into token ids, consuming `chunk_size` tokens per
/// hash until `target_len` is reached (the final chunk may be partial). A
/// repeated (hash_id, size) yields the identical chunk via the cache. If the
/// trace supplies fewer hashes than `target_len / chunk_size`, the prompt
/// comes out shorter — same as Python.
fn expand_prompt(
    hash_ids: &[i64],
    target_len: usize,
    chunk_size: usize,
    allowed_tokens: &[u32],
    cache: &mut HashMap<(i64, usize), Arc<[u32]>>,
) -> Vec<u32> {
    let mut prompt = Vec::with_capacity(target_len);
    let mut remaining = target_len;
    for &h in hash_ids {
        let size = remaining.min(chunk_size);
        let chunk = cache.entry((h, size)).or_insert_with(|| {
            let mut rng = StdRng::seed_from_u64(chunk_seed(h, size));
            (0..size)
                .map(|_| allowed_tokens[rng.random_range(0..allowed_tokens.len())])
                .collect()
        });
        prompt.extend_from_slice(chunk);
        remaining -= size;
        if remaining == 0 {
            break;
        }
    }
    prompt
}

/// Stable seed for a chunk. The golden-ratio multiply is bijective on u64, so
/// distinct (hash_id, size) pairs collide only by astronomical accident.
fn chunk_seed(hash_id: i64, size: usize) -> u64 {
    (hash_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (size as u64)
}

fn require_field<'v>(entry: &'v Value, label: &str, flag: &str) -> Result<&'v Value> {
    entry.get(label).ok_or_else(|| {
        let available: Vec<&str> = entry
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        BenchError::Config(format!(
            "Field '{label}' not found in trace entry. Available fields: \
             {available:?}. Use {flag} to specify the correct field name."
        ))
    })
}

fn require_u64(entry: &Value, label: &str, flag: &str) -> Result<u64> {
    let v = require_field(entry, label, flag)?;
    v.as_u64().ok_or_else(|| {
        BenchError::Config(format!(
            "Field '{label}' must be a non-negative integer, got: {v}"
        ))
    })
}

fn require_f64(entry: &Value, label: &str, flag: &str) -> Result<f64> {
    let v = require_field(entry, label, flag)?;
    v.as_f64()
        .ok_or_else(|| BenchError::Config(format!("Field '{label}' must be a number, got: {v}")))
}

/// Missing hash_ids field means an empty prompt (mirrors Python's
/// `entry.get(label, [])`); a present field must be an array of integers.
fn parse_hash_ids(entry: &Value, label: &str) -> Result<Vec<i64>> {
    let Some(v) = entry.get(label) else {
        return Ok(Vec::new());
    };
    let arr = v
        .as_array()
        .ok_or_else(|| BenchError::Config(format!("Field '{label}' must be an array, got: {v}")))?;
    arr.iter()
        .map(|x| {
            x.as_i64().ok_or_else(|| {
                BenchError::Config(format!(
                    "Field '{label}' must contain only integers, got: {x}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_jsonl(name: &str, content: &str) -> String {
        let path = std::env::temp_dir().join(format!("vllm-bench-timed-trace-{name}.jsonl"));
        std::fs::write(&path, content).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// gpt2 via built-in tiktoken encoding — loads without network access.
    fn test_tokenizer() -> TokenizerKind {
        TokenizerKind::Tiktoken(
            crate::tiktoken::load_builtin_tiktoken("gpt2")
                .expect("gpt2 built-in tiktoken should always load without network"),
        )
    }

    fn opts(chunk_size: usize, sec_multiplier: f64) -> TimedTraceOptions<'static> {
        TimedTraceOptions {
            chunk_size,
            sec_multiplier,
            label_timestamp: "timestamp",
            label_input_length: "input_length",
            label_output_length: "output_length",
            label_hash_ids: "hash_ids",
        }
    }

    #[test]
    fn test_prompt_len_matches_input_length_and_fields_map() {
        let path = write_temp_jsonl(
            "basic",
            r#"{"timestamp": 1000, "input_length": 40, "output_length": 7, "hash_ids": [1, 2, 3]}
{"timestamp": 2500, "input_length": 32, "output_length": 9, "hash_ids": [1, 4]}
"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 0.001))
            .expect("load should succeed");
        assert_eq!(reqs.len(), 2);
        // 40 = 16 + 16 + partial 8 from three hashes
        assert_eq!(reqs[0].prompt_len, 40);
        assert_eq!(reqs[0].prompt_token_ids.as_ref().unwrap().len(), 40);
        assert_eq!(reqs[0].expected_output_len, 7);
        assert_eq!(reqs[0].timestamp, Some(1.0));
        assert_eq!(reqs[1].timestamp, Some(2.5));
        assert_eq!(reqs[0].request_id.as_deref(), Some("t-0"));
    }

    #[test]
    fn test_shared_hash_prefix_gives_identical_tokens() {
        let path = write_temp_jsonl(
            "prefix",
            r#"{"timestamp": 0, "input_length": 32, "output_length": 1, "hash_ids": [7, 8]}
{"timestamp": 1, "input_length": 32, "output_length": 1, "hash_ids": [7, 9]}
"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        let a = reqs[0].prompt_token_ids.as_ref().unwrap();
        let b = reqs[1].prompt_token_ids.as_ref().unwrap();
        assert_eq!(a[..16], b[..16], "shared hash 7 must expand identically");
        assert_ne!(a[16..], b[16..], "distinct hashes 8/9 must differ");
    }

    #[test]
    fn test_reload_is_bit_identical() {
        let path = write_temp_jsonl(
            "determinism",
            r#"{"timestamp": 0, "input_length": 40, "output_length": 1, "hash_ids": [11, 12, 13]}"#,
        );
        let load = || {
            load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 1.0))
                .expect("load should succeed")
        };
        assert_eq!(load()[0].prompt_token_ids, load()[0].prompt_token_ids);
    }

    #[test]
    fn test_too_few_hashes_yields_short_prompt() {
        let path = write_temp_jsonl(
            "short",
            r#"{"timestamp": 0, "input_length": 100, "output_length": 1, "hash_ids": [1]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].prompt_len, 16, "one hash covers only one chunk");
    }

    #[test]
    fn test_num_requests_caps_rows() {
        let path = write_temp_jsonl(
            "cap",
            r#"{"timestamp": 0, "input_length": 8, "output_length": 1, "hash_ids": [1]}
{"timestamp": 1, "input_length": 8, "output_length": 1, "hash_ids": [2]}
{"timestamp": 2, "input_length": 8, "output_length": 1, "hash_ids": [3]}
"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 2, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs.len(), 2);
    }

    #[test]
    fn test_missing_field_error_names_label_and_flag() {
        let path = write_temp_jsonl(
            "missing",
            r#"{"timestamp": 0, "input_length": 8, "output_length": 1, "hash_ids": [1]}"#,
        );
        let mut o = opts(16, 1.0);
        o.label_input_length = "input_len";
        let err = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &o)
            .expect_err("should fail on missing field");
        let msg = err.to_string();
        assert!(msg.contains("input_len") && msg.contains("--timed-trace-label-input-length"));
    }
}
