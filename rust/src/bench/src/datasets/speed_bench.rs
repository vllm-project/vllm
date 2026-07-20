// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::SampleRequest;
use crate::cli::SpeedBenchConfig;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Marker text for masked entries that need external fetch.
const MASKED_PREFIX: &str = "FULL BENCHMARK DATA SHOULD BE FETCHED";

/// Cache directory for downloaded SPEED-Bench datasets.
fn cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("vllm-bench")
        .join("datasets")
}

/// Download SPEED-Bench dataset from HuggingFace datasets-server API.
/// Results are cached as JSON locally for subsequent runs.
pub fn download_speed_bench(config: SpeedBenchConfig) -> Result<String> {
    let config_name = config.as_str();

    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let cache_path = dir.join(format!("speed-bench-{config_name}.json"));

    // Return cached file if it exists
    if cache_path.exists() {
        let path_str = cache_path.to_string_lossy().to_string();
        println!("SPEED-Bench ({config_name}) cached: {path_str}");
        return Ok(path_str);
    }

    println!("Downloading SPEED-Bench ({config_name}) from HuggingFace datasets-server...");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| BenchError::Config(format!("Failed to build HTTP client: {e}")))?;

    let mut all_rows: Vec<serde_json::Value> = Vec::new();
    let mut offset = 0usize;
    let page_size = 100usize;

    loop {
        let url = format!(
            "https://datasets-server.huggingface.co/rows\
             ?dataset=nvidia/SPEED-Bench\
             &config={config_name}\
             &split=test\
             &offset={offset}\
             &length={page_size}"
        );

        // Retry on transient errors (502, 503, timeouts)
        let max_retries = 3;
        let mut data: Option<serde_json::Value> = None;
        for attempt in 0..=max_retries {
            let resp = match client.get(&url).send() {
                Ok(r) => r,
                Err(e) => {
                    if attempt < max_retries {
                        std::thread::sleep(std::time::Duration::from_secs(
                            2 * (attempt as u64 + 1),
                        ));
                        continue;
                    }
                    return Err(BenchError::Config(format!(
                        "SPEED-Bench download failed after {max_retries} retries: {e}"
                    )));
                }
            };

            if resp.status().is_server_error() && attempt < max_retries {
                std::thread::sleep(std::time::Duration::from_secs(2 * (attempt as u64 + 1)));
                continue;
            }

            if !resp.status().is_success() {
                return Err(BenchError::Config(format!(
                    "SPEED-Bench API returned HTTP {}",
                    resp.status()
                )));
            }

            data = Some(resp.json().map_err(|e| {
                BenchError::Config(format!("Failed to parse SPEED-Bench API response: {e}"))
            })?);
            break;
        }

        let data = data.unwrap();

        let rows = data["rows"]
            .as_array()
            .ok_or_else(|| BenchError::Config("No 'rows' in API response".into()))?;

        if rows.is_empty() {
            break;
        }

        for row in rows {
            if let Some(row_data) = row.get("row") {
                all_rows.push(row_data.clone());
            }
        }

        let fetched = rows.len();
        offset += fetched;

        // Print progress
        let total = data["num_rows_total"].as_u64().unwrap_or(0);
        eprint!("\r  Fetched {offset}/{total} rows...");

        if fetched < page_size {
            break;
        }
    }
    eprintln!(); // newline after progress

    if all_rows.is_empty() {
        return Err(BenchError::Config(
            "SPEED-Bench download returned no rows".into(),
        ));
    }

    // Save to cache
    let json_str = serde_json::to_string(&all_rows)?;
    std::fs::write(&cache_path, &json_str)?;

    let path_str = cache_path.to_string_lossy().to_string();
    println!(
        "SPEED-Bench ({config_name}): {} rows saved to {path_str}",
        all_rows.len()
    );
    Ok(path_str)
}

/// Load SPEED-Bench dataset and convert to SampleRequests.
///
/// Filters out masked entries and optionally filters by category.
/// Requires an output length override since SPEED-Bench has no reference outputs.
pub fn load_speed_bench_dataset(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_requests: usize,
    output_len: usize,
    seed: u64,
    request_id_prefix: &str,
    category_filter: Option<&str>,
    no_oversample: bool,
    disable_shuffle: bool,
    max_input_len: Option<usize>,
) -> Result<Vec<SampleRequest>> {
    let content = std::fs::read_to_string(dataset_path).map_err(|e| {
        BenchError::Config(format!(
            "Failed to read SPEED-Bench file '{dataset_path}': {e}"
        ))
    })?;

    let entries: Vec<serde_json::Value> = serde_json::from_str(&content)
        .map_err(|e| BenchError::Config(format!("Invalid JSON in SPEED-Bench file: {e}")))?;

    // Filter entries
    let mut filtered: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            // Must have turns array with at least one non-empty entry
            let turns = match entry.get("turns").and_then(|t| t.as_array()) {
                Some(t) if !t.is_empty() => t,
                _ => return false,
            };

            // Skip masked entries
            let first_turn = turns[0].as_str().unwrap_or("");
            if first_turn.starts_with(MASKED_PREFIX) || first_turn.is_empty() {
                return false;
            }

            // Single-turn only: skip multi-turn entries
            let is_multiturn = entry.get("multiturn").and_then(|m| m.as_bool()).unwrap_or(false);
            if is_multiturn {
                return false;
            }

            // Category filter
            if let Some(cat) = category_filter {
                let entry_cat = entry.get("category").and_then(|c| c.as_str()).unwrap_or("");
                if entry_cat != cat {
                    return false;
                }
            }

            true
        })
        .collect();

    if filtered.is_empty() {
        let cat_msg = category_filter.map(|c| format!(" with category '{c}'")).unwrap_or_default();
        return Err(BenchError::Config(format!(
            "No valid single-turn entries in SPEED-Bench{cat_msg}. \
             Try a different --speed-bench-config or remove --speed-bench-category filter."
        )));
    }

    // Shuffle
    let mut rng = StdRng::seed_from_u64(seed);
    if !disable_shuffle {
        filtered.shuffle(&mut rng);
    }

    // Build SampleRequests
    let mut samples = Vec::new();
    let mut idx = 0;

    for entry in &filtered {
        if samples.len() >= num_requests {
            break;
        }

        let turns = entry["turns"].as_array().unwrap();
        let prompt = turns[0].as_str().unwrap_or("");

        // Tokenize to get prompt length
        let prompt_ids = tokenizer.encode(prompt, false)?;
        let prompt_len = prompt_ids.len();

        if prompt_len < 4 {
            continue;
        }

        // Truncate if max_input_len is set
        let (final_prompt, final_len) = if let Some(max_len) = max_input_len {
            if prompt_len > max_len {
                let truncated_ids = &prompt_ids[..max_len];
                let truncated_text = tokenizer.decode(truncated_ids, true)?;
                (Arc::from(truncated_text.as_str()), max_len)
            } else {
                (Arc::from(prompt), prompt_len)
            }
        } else {
            (Arc::from(prompt), prompt_len)
        };

        samples.push(SampleRequest {
            prompt: final_prompt,
            prompt_len: final_len,
            expected_output_len: output_len,
            request_id: Some(format!("{request_id_prefix}{idx}")),
            ..Default::default()
        });
        idx += 1;
    }

    // Oversample if needed
    if samples.len() < num_requests {
        if no_oversample {
            println!(
                "Skipping oversampling. Total samples: {} (requested: {num_requests})",
                samples.len()
            );
        } else if !samples.is_empty() {
            let original_len = samples.len();
            let needed = num_requests - original_len;
            for i in 0..needed {
                let mut req = samples[rng.random_range(0..original_len)].clone();
                req.request_id = Some(format!("{request_id_prefix}{}", original_len + i));
                samples.push(req);
            }
            println!(
                "Oversampled SPEED-Bench from {original_len} to {} total samples.",
                samples.len()
            );
        }
    }

    if samples.is_empty() {
        return Err(BenchError::Config(
            "No valid samples after filtering SPEED-Bench dataset.".into(),
        ));
    }

    // Print category distribution
    let mut cat_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for entry in &filtered[..filtered.len().min(samples.len())] {
        let cat = entry.get("category").and_then(|c| c.as_str()).unwrap_or("unknown");
        *cat_counts.entry(cat).or_insert(0) += 1;
    }
    let mut cats: Vec<_> = cat_counts.into_iter().collect();
    cats.sort_by_key(|b| std::cmp::Reverse(b.1));
    let cat_str: Vec<String> = cats.iter().map(|(k, v)| format!("{k}:{v}")).collect();
    println!("SPEED-Bench categories: {}", cat_str.join(", "));

    Ok(samples)
}
