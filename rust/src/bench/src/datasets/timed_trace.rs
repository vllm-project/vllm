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
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }

    let entries: Vec<(usize, Value)> = super::read_jsonl(path, "timed trace", Some(num_requests))?;

    let mut chunk_cache: HashMap<(u64, usize), Arc<[u32]>> = HashMap::new();
    let mut requests: Vec<SampleRequest> = Vec::with_capacity(entries.len());

    for (lineno, entry) in &entries {
        let input_length = require_u64(
            entry,
            opts.label_input_length,
            "--timed-trace-label-input-length",
        )? as usize;
        let output_length = require_u64(
            entry,
            opts.label_output_length,
            "--timed-trace-label-output-length",
        )? as usize;
        let timestamp = require_f64(entry, opts.label_timestamp, "--timed-trace-label-timestamp")?
            * opts.sec_multiplier;
        // A non-finite offset panics in `Duration::from_secs_f64` once the
        // scheduler consumes it; reject here, naming the offending row.
        if !timestamp.is_finite() {
            return Err(BenchError::Config(format!(
                "Field '{}' at {path}:{} scales to a non-finite offset ({timestamp}); \
                 check --timed-trace-sec-multiplier",
                opts.label_timestamp,
                lineno + 1,
            )));
        }
        let hash_ids = parse_hash_ids(entry, opts.label_hash_ids)?;

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

    Ok(requests)
}

/// Expand prefix-hash ids into token ids, consuming `chunk_size` tokens per
/// hash until `target_len` is reached (the final chunk may be partial). A
/// repeated (hash_id, size) yields the identical chunk via the cache. If the
/// trace supplies fewer hashes than `target_len / chunk_size`, the prompt
/// comes out shorter — same as Python.
fn expand_prompt(
    hash_ids: &[u64],
    target_len: usize,
    chunk_size: usize,
    allowed_tokens: &[u32],
    cache: &mut HashMap<(u64, usize), Arc<[u32]>>,
) -> Vec<u32> {
    // `target_len` is raw trace input, so reserve only what the hashes can
    // actually produce: a corrupt or mis-unit row would otherwise ask the
    // allocator for terabytes before any length filter runs.
    let capacity = target_len.min(hash_ids.len().saturating_mul(chunk_size));
    let mut prompt = Vec::with_capacity(capacity);
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
/// distinct `hash_id`s never collide at a fixed `size`. The trailing xor is not
/// injective across sizes: two unrelated hashes seen at different sizes can
/// alias and share a token prefix, at ~2^-64 odds per pair.
fn chunk_seed(hash_id: u64, size: usize) -> u64 {
    hash_id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (size as u64)
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

/// Float-encoded integers (`512.0`) are accepted, matching Python's `int()`.
fn require_u64(entry: &Value, label: &str, flag: &str) -> Result<u64> {
    let v = require_field(entry, label, flag)?;
    v.as_u64()
        .or_else(|| {
            v.as_f64()
                .filter(|f| f.fract() == 0.0 && (0.0..u64::MAX as f64).contains(f))
                .map(|f| f as u64)
        })
        .ok_or_else(|| {
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
///
/// Ids are kept as `u64` because traces carry raw unsigned 64-bit block hashes;
/// negative ids keep the two's-complement value `chunk_seed` used before.
fn parse_hash_ids(entry: &Value, label: &str) -> Result<Vec<u64>> {
    let Some(v) = entry.get(label) else {
        return Ok(Vec::new());
    };
    let arr = v
        .as_array()
        .ok_or_else(|| BenchError::Config(format!("Field '{label}' must be an array, got: {v}")))?;
    arr.iter()
        .map(|x| {
            x.as_u64().or_else(|| x.as_i64().map(|i| i as u64)).ok_or_else(|| {
                BenchError::Config(format!(
                    "Field '{label}' must contain only integers, got: {x}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{test_tokenizer, write_temp_jsonl};
    use super::*;

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
            "timed-trace-basic",
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
            "timed-trace-prefix",
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
            "timed-trace-determinism",
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
            "timed-trace-short",
            r#"{"timestamp": 0, "input_length": 100, "output_length": 1, "hash_ids": [1]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].prompt_len, 16, "one hash covers only one chunk");
    }

    /// `input_length` is untrusted: the reservation must follow what the hashes
    /// can produce, not the raw field, or the allocator aborts the process.
    #[test]
    fn test_absurd_input_length_does_not_overallocate() {
        let path = write_temp_jsonl(
            "timed-trace-absurd-len",
            r#"{"timestamp": 0, "input_length": 10000000000000, "output_length": 1, "hash_ids": [1]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 10, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].prompt_len, 16);
    }

    #[test]
    fn test_num_requests_caps_rows() {
        let path = write_temp_jsonl(
            "timed-trace-cap",
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
            "timed-trace-missing",
            r#"{"timestamp": 0, "input_length": 8, "output_length": 1, "hash_ids": [1]}"#,
        );
        let mut o = opts(16, 1.0);
        o.label_input_length = "input_len";
        let err = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &o)
            .expect_err("should fail on missing field");
        let msg = err.to_string();
        assert!(msg.contains("input_len") && msg.contains("--timed-trace-label-input-length"));
    }

    /// Hashes above `i64::MAX` are ordinary 64-bit block hashes, and Python
    /// takes them as unbounded ints.
    #[test]
    fn test_accepts_unsigned_64bit_hash_ids() {
        let path = write_temp_jsonl(
            "timed-trace-u64-hash",
            r#"{"timestamp": 0, "input_length": 16, "output_length": 1, "hash_ids": [18446744073709551615]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].prompt_len, 16);
    }

    /// Exporters that round-trip lengths through floats emit `512.0`; Python's
    /// `int()` accepts it.
    #[test]
    fn test_accepts_float_encoded_lengths() {
        let path = write_temp_jsonl(
            "timed-trace-float-len",
            r#"{"timestamp": 0, "input_length": 16.0, "output_length": 4.0, "hash_ids": [1]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].prompt_len, 16);
        assert_eq!(reqs[0].expected_output_len, 4);
    }

    #[test]
    fn test_non_finite_scaled_timestamp_is_rejected() {
        let path = write_temp_jsonl(
            "timed-trace-overflow-ts",
            r#"{"timestamp": 1e308, "input_length": 16, "output_length": 1, "hash_ids": [1]}"#,
        );
        let err = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &opts(16, 1e10))
            .expect_err("should reject an offset that overflows to inf");
        assert!(err.to_string().contains("non-finite offset"));
    }

    /// Negative offsets load unchanged; the scheduler clamps them to fire
    /// immediately (Python's `get_request` does the same).
    #[test]
    fn test_negative_timestamp_loads() {
        let path = write_temp_jsonl(
            "timed-trace-negative-ts",
            r#"{"timestamp": -5.0, "input_length": 16, "output_length": 1, "hash_ids": [1]}"#,
        );
        let reqs = load_timed_trace_dataset(&test_tokenizer(), &path, 1, "t-", &opts(16, 1.0))
            .expect("load should succeed");
        assert_eq!(reqs[0].timestamp, Some(-5.0));
    }
}
