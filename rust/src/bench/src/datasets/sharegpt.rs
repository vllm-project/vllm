// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Default validation bounds matching Python's is_valid_sequence() defaults.
const MIN_LEN: usize = 4;
const MAX_PROMPT_LEN: usize = 1024;
const MAX_TOTAL_LEN: usize = 2048;

/// Default HuggingFace dataset repo and filename for ShareGPT.
const DEFAULT_SHAREGPT_REPO: &str = "anon8231489123/ShareGPT_Vicuna_unfiltered";
const DEFAULT_SHAREGPT_FILE: &str = "ShareGPT_V3_unfiltered_cleaned_split.json";

/// Download the default ShareGPT dataset from HuggingFace Hub.
/// Uses hf-hub's built-in cache — subsequent calls return the cached path instantly.
pub async fn download_sharegpt_dataset() -> Result<String> {
    tracing::info!(
        repository = DEFAULT_SHAREGPT_REPO,
        file = DEFAULT_SHAREGPT_FILE,
        "downloading ShareGPT dataset"
    );
    let repo = crate::hub::HubRepo::dataset(DEFAULT_SHAREGPT_REPO.to_string())
        .map_err(BenchError::Config)?;
    let path = repo.get(DEFAULT_SHAREGPT_FILE).await.map_err(|e| {
        BenchError::Config(format!(
            "Failed to download ShareGPT dataset from '{DEFAULT_SHAREGPT_REPO}': {e}"
        ))
    })?;
    let path_str = path.to_string_lossy().to_string();
    tracing::info!(dataset = "sharegpt", path = %path_str, "dataset is ready");
    Ok(path_str)
}

/// Load and sample from a ShareGPT-format JSON dataset.
///
/// Mirrors Python's ShareGPTDataset from datasets.py:1230-1313.
pub fn load_sharegpt_dataset(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_requests: usize,
    output_len_override: Option<usize>,
    seed: u64,
    request_id_prefix: &str,
    no_oversample: bool,
    disable_shuffle: bool,
) -> Result<Vec<SampleRequest>> {
    // Load JSON file
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

    // Filter entries with at least 2 conversation turns
    let mut filtered: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            entry
                .get("conversations")
                .and_then(|c| c.as_array())
                .map(|a| a.len() >= 2)
                .unwrap_or(false)
        })
        .collect();

    if filtered.is_empty() {
        return Err(BenchError::Config(
            "No valid entries in ShareGPT file (need at least 2 conversation turns)".into(),
        ));
    }

    // Shuffle (unless disabled)
    let mut rng = StdRng::seed_from_u64(seed);
    if !disable_shuffle {
        filtered.shuffle(&mut rng);
    }

    // Sample requests
    let mut samples = Vec::new();
    let mut ind = 0;

    for entry in &filtered {
        if samples.len() >= num_requests {
            break;
        }

        let conversations = entry["conversations"].as_array().unwrap();
        let prompt = conversations[0]["value"].as_str().unwrap_or("");
        let completion = conversations[1]["value"].as_str().unwrap_or("");

        if prompt.is_empty() {
            continue;
        }

        // Tokenize prompt and completion
        let prompt_ids = tokenizer.encode(prompt, false)?;
        let prompt_len = prompt_ids.len();

        let new_output_len = if let Some(override_len) = output_len_override {
            override_len
        } else {
            let completion_ids = tokenizer.encode(completion, false)?;
            completion_ids.len()
        };

        // Validate sequence lengths (matching Python's is_valid_sequence)
        let skip_min_output = output_len_override.is_some();
        if !is_valid_sequence(prompt_len, new_output_len, skip_min_output) {
            continue;
        }

        samples.push(SampleRequest {
            prompt: Arc::from(prompt),
            prompt_len,
            expected_output_len: new_output_len,
            request_id: Some(format!("{request_id_prefix}{ind}")),
            ..Default::default()
        });
        ind += 1;
    }

    // Oversample if dataset is smaller than requested
    if samples.len() < num_requests {
        if no_oversample {
            tracing::info!(
                dataset = "sharegpt",
                samples = samples.len(),
                requested = num_requests,
                "skipping dataset oversampling"
            );
        } else if !samples.is_empty() {
            let needed = num_requests - samples.len();
            let original_len = samples.len();
            for i in 0..needed {
                let mut req = samples[rng.random_range(0..original_len)].clone();
                req.request_id = Some(format!("{request_id_prefix}{}", original_len + i));
                samples.push(req);
            }
            tracing::info!(
                dataset = "sharegpt",
                original_samples = original_len,
                samples = samples.len(),
                "oversampled dataset"
            );
        }
    }

    if samples.is_empty() {
        return Err(BenchError::Config(
            "No valid samples after filtering ShareGPT dataset. \
             Try relaxing constraints or using a larger dataset."
                .into(),
        ));
    }

    Ok(samples)
}

/// Validate a sequence based on prompt and output lengths.
/// Mirrors Python's is_valid_sequence() from datasets.py:260-284.
fn is_valid_sequence(
    prompt_len: usize,
    output_len: usize,
    skip_min_output_len_check: bool,
) -> bool {
    if prompt_len < MIN_LEN {
        return false;
    }
    if !skip_min_output_len_check && output_len < MIN_LEN {
        return false;
    }
    if prompt_len > MAX_PROMPT_LEN {
        return false;
    }
    if prompt_len + output_len > MAX_TOTAL_LEN {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_sequence() {
        // Valid
        assert!(is_valid_sequence(100, 50, false));
        // Prompt too short
        assert!(!is_valid_sequence(3, 50, false));
        // Output too short
        assert!(!is_valid_sequence(100, 3, false));
        // Output too short but skip check
        assert!(is_valid_sequence(100, 1, true));
        // Prompt too long
        assert!(!is_valid_sequence(1025, 50, false));
        // Combined too long
        assert!(!is_valid_sequence(1024, 1025, false));
    }
}
