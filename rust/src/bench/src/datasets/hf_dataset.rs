// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Cache directory for downloaded HF datasets.
fn cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("vllm-bench")
        .join("datasets")
}

/// Sanitize a dataset name for use in filenames (replace `/` and other unsafe chars).
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Detected column format for extracting prompts from HF dataset rows.
enum ColumnFormat {
    /// Column contains chat messages (array of {role, content} or {from, value}).
    Chat(String),
    /// Single text column for prompt, optional output column.
    Text {
        prompt_col: String,
        output_col: Option<String>,
    },
    /// Multiple columns combined (e.g., context + input for LongBench).
    Combined {
        cols: Vec<String>,
        output_col: Option<String>,
    },
}

/// Make a GET request with retry logic (3 retries with exponential backoff).
/// Returns the parsed JSON response.
fn get_with_retry(
    client: &reqwest::blocking::Client,
    url: &str,
    label: &str,
) -> Result<serde_json::Value> {
    let max_retries = 3;
    for attempt in 0..=max_retries {
        let resp = match client.get(url).send() {
            Ok(r) => r,
            Err(e) => {
                if attempt < max_retries {
                    std::thread::sleep(std::time::Duration::from_secs(2 * (attempt as u64 + 1)));
                    continue;
                }
                return Err(BenchError::Config(format!(
                    "{label} failed after {max_retries} retries: {e}"
                )));
            }
        };

        // Detect gated/private dataset errors
        let status = resp.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(BenchError::Config(format!(
                "{label} returned HTTP {status}. This dataset may be gated or private. \
                 Try setting the HF_TOKEN environment variable with a valid HuggingFace token."
            )));
        }

        if status.is_server_error() && attempt < max_retries {
            std::thread::sleep(std::time::Duration::from_secs(2 * (attempt as u64 + 1)));
            continue;
        }

        if !status.is_success() {
            return Err(BenchError::Config(format!(
                "{label} returned HTTP {status}"
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .map_err(|e| BenchError::Config(format!("Failed to parse {label} response: {e}")))?;
        return Ok(data);
    }
    unreachable!()
}

/// Download an arbitrary HuggingFace dataset via the datasets-server REST API.
///
/// Returns `(cache_path, resolved_config, resolved_split)`.
///
/// If both `subset` and `split` are provided, the `/info` call is skipped as an optimization.
/// Paginated download fetches rows in pages of 100 until `num_rows_needed` are collected
/// or the dataset is exhausted.
pub fn download_hf_dataset(
    dataset: &str,
    subset: Option<&str>,
    split: Option<&str>,
    num_rows_needed: usize,
) -> Result<(String, String, String)> {
    let encoded_dataset: String =
        url::form_urlencoded::byte_serialize(dataset.as_bytes()).collect();

    let mut client_builder =
        reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(120));

    // Add HF_TOKEN auth header if available
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let mut headers = reqwest::header::HeaderMap::new();
        let header_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| BenchError::Config(format!("Invalid HF_TOKEN: {e}")))?;
        headers.insert(reqwest::header::AUTHORIZATION, header_value);
        client_builder = client_builder.default_headers(headers);
    }

    let client = client_builder
        .build()
        .map_err(|e| BenchError::Config(format!("Failed to build HTTP client: {e}")))?;

    // Resolve config and split
    let (resolved_config, resolved_split) = if let (Some(cfg), Some(spl)) = (subset, split) {
        // Both provided — skip /info call
        (cfg.to_string(), spl.to_string())
    } else {
        // Call /info to discover available configs and splits
        let info_url =
            format!("https://datasets-server.huggingface.co/info?dataset={encoded_dataset}");
        let info = get_with_retry(&client, &info_url, "HF dataset /info")?;

        let dataset_info =
            info.get("dataset_info").and_then(|d| d.as_object()).ok_or_else(|| {
                BenchError::Config(format!(
                    "No 'dataset_info' in /info response for '{dataset}'. \
                     The dataset may not exist or may not be accessible."
                ))
            })?;

        // Resolve config
        let resolved_config = if let Some(user_cfg) = subset {
            if !dataset_info.contains_key(user_cfg) {
                let available: Vec<&String> = dataset_info.keys().collect();
                return Err(BenchError::Config(format!(
                    "Config '{user_cfg}' not found in dataset '{dataset}'. Available: {available:?}"
                )));
            }
            user_cfg.to_string()
        } else if dataset_info.contains_key("default") {
            "default".to_string()
        } else {
            dataset_info.keys().next().cloned().ok_or_else(|| {
                BenchError::Config(format!("No configs found in dataset '{dataset}'"))
            })?
        };

        // Resolve split from the config's splits
        let splits_info = dataset_info
            .get(&resolved_config)
            .and_then(|c| c.get("splits"))
            .and_then(|s| s.as_object());

        let resolved_split = if let Some(user_spl) = split {
            if let Some(si) = splits_info
                && !si.contains_key(user_spl)
            {
                let available: Vec<&String> = si.keys().collect();
                return Err(BenchError::Config(format!(
                    "Split '{user_spl}' not found in config '{resolved_config}' \
                         of dataset '{dataset}'. Available: {available:?}"
                )));
            }
            user_spl.to_string()
        } else if let Some(si) = splits_info {
            // Priority: train > test > validation > first
            let priority = ["train", "test", "validation"];
            let mut found = None;
            for p in &priority {
                if si.contains_key(*p) {
                    found = Some(p.to_string());
                    break;
                }
            }
            found
                .unwrap_or_else(|| si.keys().next().cloned().unwrap_or_else(|| "train".to_string()))
        } else {
            "train".to_string()
        };

        (resolved_config, resolved_split)
    };

    println!("HF dataset: {dataset} (config={resolved_config}, split={resolved_split})");

    // Check cache
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let cache_path = dir.join(format!(
        "hf-{}-{}-{}.json",
        sanitize_name(dataset),
        sanitize_name(&resolved_config),
        sanitize_name(&resolved_split)
    ));

    if cache_path.exists() {
        let path_str = cache_path.to_string_lossy().to_string();
        println!("HF dataset cached: {path_str}");
        return Ok((path_str, resolved_config, resolved_split));
    }

    println!("Downloading HF dataset '{dataset}' from datasets-server...");

    let encoded_config: String =
        url::form_urlencoded::byte_serialize(resolved_config.as_bytes()).collect();
    let encoded_split: String =
        url::form_urlencoded::byte_serialize(resolved_split.as_bytes()).collect();

    let mut all_rows: Vec<serde_json::Value> = Vec::new();
    let mut offset = 0usize;
    let page_size = 100usize;

    loop {
        let url = format!(
            "https://datasets-server.huggingface.co/rows\
             ?dataset={encoded_dataset}\
             &config={encoded_config}\
             &split={encoded_split}\
             &offset={offset}\
             &length={page_size}"
        );

        let data = get_with_retry(&client, &url, "HF dataset /rows")?;

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

        let total = data["num_rows_total"].as_u64().unwrap_or(0);
        eprint!("\r  Fetched {offset}/{total} rows...");

        // Stop if we have enough rows or reached end of dataset
        if all_rows.len() >= num_rows_needed || fetched < page_size {
            break;
        }
    }
    eprintln!(); // newline after progress

    if all_rows.is_empty() {
        return Err(BenchError::Config(format!(
            "HF dataset '{dataset}' download returned no rows"
        )));
    }

    // Save to cache
    let json_str = serde_json::to_string(&all_rows)?;
    std::fs::write(&cache_path, &json_str)?;

    let path_str = cache_path.to_string_lossy().to_string();
    println!("HF dataset: {} rows saved to {path_str}", all_rows.len());
    Ok((path_str, resolved_config, resolved_split))
}

/// Check if a JSON value looks like a chat message (has role+content or from+value).
fn is_chat_message(val: &serde_json::Value) -> bool {
    if let Some(obj) = val.as_object() {
        (obj.contains_key("role") && obj.contains_key("content"))
            || (obj.contains_key("from") && obj.contains_key("value"))
    } else {
        false
    }
}

/// Check if a JSON value is an array of chat messages.
fn is_chat_array(val: &serde_json::Value) -> bool {
    val.as_array()
        .map(|arr| !arr.is_empty() && arr.iter().all(is_chat_message))
        .unwrap_or(false)
}

/// Known column names for chat-format data.
const CHAT_COLUMNS: &[&str] = &["conversation", "conversations", "messages"];

/// Known column names for single text prompts (in priority order).
const TEXT_COLUMNS: &[&str] = &[
    "prompt",
    "question",
    "problem",
    "input",
    "text",
    "content",
    "instruction",
];

/// Known column names for output/completion data.
const OUTPUT_COLUMNS: &[&str] = &[
    "completion",
    "response",
    "answer",
    "output",
    "solution",
    "answers",
];

/// Detect the column format from the first row of the dataset.
fn detect_column_format(
    row: &serde_json::Value,
    text_column_override: Option<&str>,
) -> Result<ColumnFormat> {
    let obj = row
        .as_object()
        .ok_or_else(|| BenchError::Config("HF dataset row is not a JSON object".into()))?;

    // Helper: find first matching output column
    let find_output_col = || -> Option<String> {
        for col in OUTPUT_COLUMNS {
            if obj.contains_key(*col) {
                return Some(col.to_string());
            }
        }
        None
    };

    // 1. User override via --hf-text-column
    if let Some(col_name) = text_column_override {
        if !obj.contains_key(col_name) {
            let available: Vec<&String> = obj.keys().collect();
            return Err(BenchError::Config(format!(
                "Column '{col_name}' not found in dataset. Available columns: {available:?}"
            )));
        }
        let val = &obj[col_name];
        if is_chat_array(val) {
            return Ok(ColumnFormat::Chat(col_name.to_string()));
        }
        return Ok(ColumnFormat::Text {
            prompt_col: col_name.to_string(),
            output_col: find_output_col(),
        });
    }

    // 2. Chat columns
    for col in CHAT_COLUMNS {
        if let Some(val) = obj.get(*col)
            && is_chat_array(val)
        {
            return Ok(ColumnFormat::Chat(col.to_string()));
        }
    }

    // 3. "turns" column — array of strings
    if let Some(val) = obj.get("turns")
        && let Some(arr) = val.as_array()
        && !arr.is_empty()
        && arr[0].is_string()
    {
        return Ok(ColumnFormat::Text {
            prompt_col: "turns".to_string(),
            output_col: find_output_col(),
        });
    }

    // 4. Combined: context + input
    if obj.contains_key("context") && obj.contains_key("input") {
        return Ok(ColumnFormat::Combined {
            cols: vec!["context".to_string(), "input".to_string()],
            output_col: find_output_col(),
        });
    }

    // 5. Single text columns
    for col in TEXT_COLUMNS {
        if obj.contains_key(*col) {
            return Ok(ColumnFormat::Text {
                prompt_col: col.to_string(),
                output_col: find_output_col(),
            });
        }
    }

    // Fallback: list available columns
    let available: Vec<&String> = obj.keys().collect();
    Err(BenchError::Config(format!(
        "Could not auto-detect prompt column in HF dataset. \
         Available columns: {available:?}. \
         Use --hf-text-column to specify the column containing prompts."
    )))
}

/// Extract the first user message from a chat message array.
/// Supports both {role, content} and {from, value} formats.
fn extract_chat_prompt(messages: &[serde_json::Value]) -> Option<String> {
    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .or_else(|| msg.get("from").and_then(|f| f.as_str()));
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .or_else(|| msg.get("value").and_then(|v| v.as_str()));

        if let (Some(role), Some(content)) = (role, content)
            && (role == "user" || role == "human")
        {
            return Some(content.to_string());
        }
    }
    None
}

/// Extract the first assistant message from a chat message array.
fn extract_chat_completion(messages: &[serde_json::Value]) -> Option<String> {
    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .or_else(|| msg.get("from").and_then(|f| f.as_str()));
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .or_else(|| msg.get("value").and_then(|v| v.as_str()));

        if let (Some(role), Some(content)) = (role, content)
            && (role == "assistant" || role == "gpt")
        {
            return Some(content.to_string());
        }
    }
    None
}

/// Load an HF dataset from the cached JSON file and convert to SampleRequests.
pub fn load_hf_dataset(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_requests: usize,
    hf_output_len: Option<usize>,
    seed: u64,
    request_id_prefix: &str,
    text_column_override: Option<&str>,
    no_oversample: bool,
    disable_shuffle: bool,
) -> Result<Vec<SampleRequest>> {
    let content = std::fs::read_to_string(dataset_path).map_err(|e| {
        BenchError::Config(format!(
            "Failed to read HF dataset file '{dataset_path}': {e}"
        ))
    })?;

    let entries: Vec<serde_json::Value> = serde_json::from_str(&content)
        .map_err(|e| BenchError::Config(format!("Invalid JSON in HF dataset file: {e}")))?;

    if entries.is_empty() {
        return Err(BenchError::Config(
            "HF dataset file contains no rows".into(),
        ));
    }

    // Detect column format from first row
    let format = detect_column_format(&entries[0], text_column_override)?;

    // Print detected format
    match &format {
        ColumnFormat::Chat(col) => println!("HF dataset: detected chat column '{col}'"),
        ColumnFormat::Text {
            prompt_col,
            output_col,
        } => {
            let out_msg = output_col.as_deref().unwrap_or("none");
            println!("HF dataset: detected text column '{prompt_col}', output column: {out_msg}");
        }
        ColumnFormat::Combined { cols, output_col } => {
            let out_msg = output_col.as_deref().unwrap_or("none");
            println!(
                "HF dataset: detected combined columns {:?}, output column: {out_msg}",
                cols
            );
        }
    }

    // Build shuffled indices
    let mut indices: Vec<usize> = (0..entries.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    if !disable_shuffle {
        indices.shuffle(&mut rng);
    }

    let mut samples = Vec::new();
    let mut idx = 0;
    let mut warned_no_output = false;

    for &entry_idx in &indices {
        if samples.len() >= num_requests {
            break;
        }

        let row = &entries[entry_idx];

        // Extract prompt and optional completion based on format
        let (prompt, completion) = match &format {
            ColumnFormat::Chat(col) => {
                let messages =
                    row.get(col.as_str()).and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let prompt = match extract_chat_prompt(&messages) {
                    Some(p) => p,
                    None => continue,
                };
                let completion = extract_chat_completion(&messages);
                (prompt, completion)
            }
            ColumnFormat::Text {
                prompt_col,
                output_col,
            } => {
                let prompt_val = match row.get(prompt_col.as_str()) {
                    Some(v) => v,
                    None => continue,
                };

                // Handle "turns" column (array of strings — take first element)
                let prompt = if prompt_col == "turns" {
                    match prompt_val.as_array().and_then(|arr| arr.first()) {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => continue,
                    }
                } else {
                    prompt_val.as_str().unwrap_or("").to_string()
                };

                let completion = output_col.as_ref().and_then(|col| {
                    row.get(col.as_str()).and_then(|v| {
                        // Handle "answers" which may be an array
                        if let Some(arr) = v.as_array() {
                            arr.first().and_then(|a| a.as_str()).map(|s| s.to_string())
                        } else {
                            v.as_str().map(|s| s.to_string())
                        }
                    })
                });

                (prompt, completion)
            }
            ColumnFormat::Combined { cols, output_col } => {
                let parts: Vec<String> = cols
                    .iter()
                    .filter_map(|col| {
                        row.get(col.as_str()).and_then(|v| v.as_str()).map(|s| s.to_string())
                    })
                    .collect();

                if parts.is_empty() {
                    continue;
                }

                let prompt = parts.join("\n\n");

                let completion = output_col.as_ref().and_then(|col| {
                    row.get(col.as_str()).and_then(|v| v.as_str()).map(|s| s.to_string())
                });

                (prompt, completion)
            }
        };

        if prompt.is_empty() {
            continue;
        }

        // Tokenize prompt
        let prompt_ids = tokenizer.encode(&prompt, false)?;
        let prompt_len = prompt_ids.len();

        if prompt_len < 4 {
            continue;
        }

        // Determine output length
        let output_len = if let Some(fixed_len) = hf_output_len {
            fixed_len
        } else if let Some(ref comp) = completion {
            if !comp.is_empty() {
                let comp_ids = tokenizer.encode(comp, false)?;
                let len = comp_ids.len();
                if len == 0 { 128 } else { len }
            } else {
                if !warned_no_output {
                    eprintln!(
                        "WARNING: No output column detected and --hf-output-len not set. \
                         Using default output length of 128 tokens."
                    );
                    warned_no_output = true;
                }
                128
            }
        } else {
            if !warned_no_output {
                eprintln!(
                    "WARNING: No output column detected and --hf-output-len not set. \
                     Using default output length of 128 tokens."
                );
                warned_no_output = true;
            }
            128
        };

        samples.push(SampleRequest {
            prompt: Arc::from(prompt.as_str()),
            prompt_len,
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
                "Oversampled HF dataset from {original_len} to {} total samples.",
                samples.len()
            );
        }
    }

    if samples.is_empty() {
        return Err(BenchError::Config(
            "No valid samples after processing HF dataset. \
             Try a different --hf-text-column or check the dataset format."
                .into(),
        ));
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_chat_conversation() {
        let row = serde_json::json!({
            "conversation_id": "abc",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        });
        let format = detect_column_format(&row, None).unwrap();
        assert!(matches!(format, ColumnFormat::Chat(ref col) if col == "conversation"));
    }

    #[test]
    fn test_detect_chat_messages() {
        let row = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        });
        let format = detect_column_format(&row, None).unwrap();
        assert!(matches!(format, ColumnFormat::Chat(ref col) if col == "messages"));
    }

    #[test]
    fn test_detect_chat_sharegpt_format() {
        let row = serde_json::json!({
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"}
            ]
        });
        let format = detect_column_format(&row, None).unwrap();
        assert!(matches!(format, ColumnFormat::Chat(ref col) if col == "conversations"));
    }

    #[test]
    fn test_detect_plain_prompt() {
        let row = serde_json::json!({
            "prompt": "What is 2+2?",
            "completion": "4"
        });
        let format = detect_column_format(&row, None).unwrap();
        match format {
            ColumnFormat::Text {
                ref prompt_col,
                ref output_col,
            } => {
                assert_eq!(prompt_col, "prompt");
                assert_eq!(output_col.as_deref(), Some("completion"));
            }
            _ => panic!("Expected Text format"),
        }
    }

    #[test]
    fn test_detect_question_answer() {
        let row = serde_json::json!({
            "question": "What is AI?",
            "answer": "Artificial intelligence"
        });
        let format = detect_column_format(&row, None).unwrap();
        match format {
            ColumnFormat::Text {
                ref prompt_col,
                ref output_col,
            } => {
                assert_eq!(prompt_col, "question");
                assert_eq!(output_col.as_deref(), Some("answer"));
            }
            _ => panic!("Expected Text format"),
        }
    }

    #[test]
    fn test_detect_combined_context_input() {
        let row = serde_json::json!({
            "context": "The quick brown fox...",
            "input": "What animal was mentioned?",
            "answers": ["fox"]
        });
        let format = detect_column_format(&row, None).unwrap();
        match format {
            ColumnFormat::Combined {
                ref cols,
                ref output_col,
            } => {
                assert_eq!(cols, &["context", "input"]);
                assert_eq!(output_col.as_deref(), Some("answers"));
            }
            _ => panic!("Expected Combined format"),
        }
    }

    #[test]
    fn test_detect_turns_array() {
        let row = serde_json::json!({
            "turns": ["First turn prompt", "Second turn"],
            "answer": "The answer"
        });
        let format = detect_column_format(&row, None).unwrap();
        match format {
            ColumnFormat::Text {
                ref prompt_col,
                ref output_col,
            } => {
                assert_eq!(prompt_col, "turns");
                assert_eq!(output_col.as_deref(), Some("answer"));
            }
            _ => panic!("Expected Text format"),
        }
    }

    #[test]
    fn test_detect_user_override() {
        let row = serde_json::json!({
            "my_custom_col": "Hello world",
            "answer": "response"
        });
        let format = detect_column_format(&row, Some("my_custom_col")).unwrap();
        match format {
            ColumnFormat::Text {
                ref prompt_col,
                ref output_col,
            } => {
                assert_eq!(prompt_col, "my_custom_col");
                assert_eq!(output_col.as_deref(), Some("answer"));
            }
            _ => panic!("Expected Text format"),
        }
    }

    #[test]
    fn test_detect_user_override_chat_column() {
        let row = serde_json::json!({
            "my_chat": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
        });
        let format = detect_column_format(&row, Some("my_chat")).unwrap();
        assert!(matches!(format, ColumnFormat::Chat(ref col) if col == "my_chat"));
    }

    #[test]
    fn test_detect_user_override_missing_column() {
        let row = serde_json::json!({"text": "hello"});
        let result = detect_column_format(&row, Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_no_known_columns() {
        let row = serde_json::json!({"id": 1, "label": "positive"});
        let result = detect_column_format(&row, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_chat_prompt_role_content() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "What is AI?"}),
            serde_json::json!({"role": "assistant", "content": "AI is..."}),
        ];
        assert_eq!(
            extract_chat_prompt(&messages),
            Some("What is AI?".to_string())
        );
    }

    #[test]
    fn test_extract_chat_prompt_from_value() {
        let messages = vec![
            serde_json::json!({"from": "human", "value": "Hello"}),
            serde_json::json!({"from": "gpt", "value": "Hi!"}),
        ];
        assert_eq!(extract_chat_prompt(&messages), Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_chat_prompt_system_first() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "Tell me a joke"}),
            serde_json::json!({"role": "assistant", "content": "Why did..."}),
        ];
        assert_eq!(
            extract_chat_prompt(&messages),
            Some("Tell me a joke".to_string())
        );
    }

    #[test]
    fn test_extract_chat_prompt_no_user() {
        let messages = vec![serde_json::json!({"role": "system", "content": "You are helpful"})];
        assert_eq!(extract_chat_prompt(&messages), None);
    }

    #[test]
    fn test_extract_chat_completion_role_content() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hello!"}),
        ];
        assert_eq!(
            extract_chat_completion(&messages),
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_extract_chat_completion_gpt() {
        let messages = vec![
            serde_json::json!({"from": "human", "value": "Hi"}),
            serde_json::json!({"from": "gpt", "value": "Hello!"}),
        ];
        assert_eq!(
            extract_chat_completion(&messages),
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_is_chat_message_valid() {
        assert!(is_chat_message(
            &serde_json::json!({"role": "user", "content": "hi"})
        ));
        assert!(is_chat_message(
            &serde_json::json!({"from": "human", "value": "hi"})
        ));
    }

    #[test]
    fn test_is_chat_message_invalid() {
        assert!(!is_chat_message(&serde_json::json!({"text": "hi"})));
        assert!(!is_chat_message(&serde_json::json!("just a string")));
        assert!(!is_chat_message(&serde_json::json!(42)));
    }

    #[test]
    fn test_is_chat_array_valid() {
        let val = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]);
        assert!(is_chat_array(&val));
    }

    #[test]
    fn test_is_chat_array_invalid() {
        assert!(!is_chat_array(&serde_json::json!([])));
        assert!(!is_chat_array(&serde_json::json!(["a", "b"])));
        assert!(!is_chat_array(&serde_json::json!("not an array")));
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(
            sanitize_name("allenai/WildChat-4.8M"),
            "allenai_WildChat-4.8M"
        );
        assert_eq!(sanitize_name("simple-name"), "simple-name");
        assert_eq!(sanitize_name("org/sub/deep"), "org_sub_deep");
    }

    // --- extract_chat_prompt edge cases ---

    #[test]
    fn test_extract_chat_prompt_empty_messages() {
        let messages: Vec<serde_json::Value> = vec![];
        assert_eq!(extract_chat_prompt(&messages), None);
    }

    #[test]
    fn test_extract_chat_prompt_model_role_skipped() {
        // "model" role is not "user" or "human" — should be skipped
        let messages = vec![
            serde_json::json!({"role": "model", "content": "I am the model"}),
            serde_json::json!({"role": "assistant", "content": "Hello"}),
        ];
        assert_eq!(extract_chat_prompt(&messages), None);
    }

    #[test]
    fn test_extract_chat_prompt_model_role_before_user() {
        // "model" role before user: only "user" should be matched
        let messages = vec![
            serde_json::json!({"role": "model", "content": "intro"}),
            serde_json::json!({"role": "user", "content": "actual user message"}),
        ];
        assert_eq!(
            extract_chat_prompt(&messages),
            Some("actual user message".to_string())
        );
    }

    // --- extract_chat_completion edge cases ---

    #[test]
    fn test_extract_chat_completion_no_assistant() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "What is 2+2?"}),
            serde_json::json!({"role": "system", "content": "You are helpful"}),
        ];
        assert_eq!(extract_chat_completion(&messages), None);
    }

    #[test]
    fn test_extract_chat_completion_empty_messages() {
        let messages: Vec<serde_json::Value> = vec![];
        assert_eq!(extract_chat_completion(&messages), None);
    }

    #[test]
    fn test_extract_chat_completion_model_role_not_matched() {
        // "model" role is not "assistant" or "gpt" — should return None
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "model", "content": "I should not be returned"}),
        ];
        assert_eq!(extract_chat_completion(&messages), None);
    }

    // --- load_hf_dataset integration tests (require built-in tiktoken, no network) ---

    /// Build a gpt2 tokenizer using built-in tiktoken encoding (no network required).
    fn builtin_tokenizer() -> crate::tokenizer::TokenizerKind {
        crate::tokenizer::load_tokenizer("gpt2", false, None)
            .expect("gpt2 built-in tiktoken should always load without network")
    }

    /// Write JSON data to a unique temp file and return the path string.
    fn write_temp_json(name: &str, data: &serde_json::Value) -> String {
        let path = std::env::temp_dir().join(format!("vllm-bench-test-{name}.json"));
        std::fs::write(&path, serde_json::to_string(data).unwrap()).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_load_hf_dataset_chat_format() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {
                "conversation": [
                    {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
                    {"role": "assistant", "content": "The answer is 42, according to Douglas Adams."}
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "Tell me about quantum computing and its applications in modern science."},
                    {"role": "assistant", "content": "Quantum computing uses quantum bits to perform calculations."}
                ]
            }
        ]);
        let path = write_temp_json("chat-format", &data);

        let result =
            load_hf_dataset(&tok, &path, 2, Some(50), 42, "test-", None, false, false).unwrap();

        assert_eq!(result.len(), 2);
        assert!(
            result.iter().all(|r| r.expected_output_len == 50),
            "all samples should have fixed output len 50"
        );
    }

    #[test]
    fn test_load_hf_dataset_plain_text_format() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {"prompt": "What is the capital of France and why is it important in European history?", "completion": "Paris is the capital of France."},
            {"prompt": "Explain the difference between machine learning and deep learning in simple terms.", "completion": "Machine learning is a subset of AI."},
            {"prompt": "How does photosynthesis work in plants and what role does chlorophyll play?", "completion": "Photosynthesis converts light to energy."}
        ]);
        let path = write_temp_json("plain-text-format", &data);

        let result = load_hf_dataset(&tok, &path, 3, None, 0, "req-", None, false, false).unwrap();

        assert_eq!(result.len(), 3);
        // Without hf_output_len override, output len is derived from the completion tokens
        assert!(result.iter().all(|r| r.expected_output_len > 0));
    }

    #[test]
    fn test_load_hf_dataset_combined_format() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {
                "context": "The quick brown fox jumped over the lazy dog near the river bank.",
                "input": "What animal jumped over the dog in this sentence?",
                "answers": ["The quick brown fox"]
            },
            {
                "context": "Albert Einstein developed the theory of relativity in the early twentieth century.",
                "input": "Who developed the theory of relativity and when approximately?",
                "answers": ["Albert Einstein"]
            }
        ]);
        let path = write_temp_json("combined-format", &data);

        let result =
            load_hf_dataset(&tok, &path, 2, Some(64), 1, "comb-", None, false, false).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|r| r.expected_output_len == 64));
        // Combined format joins context + input with "\n\n"
        assert!(result.iter().all(|r| r.prompt.contains('\n')));
    }

    #[test]
    fn test_load_hf_dataset_empty_returns_error() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([]);
        let path = write_temp_json("empty-dataset", &data);

        let result = load_hf_dataset(&tok, &path, 5, None, 0, "test-", None, false, false);

        assert!(result.is_err(), "empty dataset should return an error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no rows") || err.contains("No rows") || err.contains("empty"),
            "error should mention empty/no rows: {err}"
        );
    }

    #[test]
    fn test_load_hf_dataset_all_rows_filtered_too_short() {
        let tok = builtin_tokenizer();
        // Very short prompts that will tokenize to fewer than 4 tokens and be filtered out
        let data = serde_json::json!([
            {"prompt": "Hi", "completion": "Ok"},
            {"prompt": "Yes", "completion": "No"},
            {"prompt": "Ok", "completion": "Fine"}
        ]);
        let path = write_temp_json("all-filtered", &data);

        let result = load_hf_dataset(&tok, &path, 3, None, 0, "test-", None, false, false);

        assert!(
            result.is_err(),
            "all rows being too short should return an error"
        );
    }

    #[test]
    fn test_load_hf_dataset_hf_output_len_override() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {"prompt": "Describe the history of the Roman Empire and its eventual decline over centuries.", "completion": "The Roman Empire fell for many complex reasons."},
            {"prompt": "What are the key differences between supervised and unsupervised machine learning?", "completion": "Supervised learning uses labeled data while unsupervised does not."},
            {"prompt": "Explain how neural networks are inspired by the human brain structure and function.", "completion": "Neural networks mimic brain neurons with layers of nodes."}
        ]);
        let path = write_temp_json("output-len-override", &data);

        let fixed_len = 77usize;
        let result = load_hf_dataset(
            &tok,
            &path,
            3,
            Some(fixed_len),
            42,
            "t-",
            None,
            false,
            false,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert!(
            result.iter().all(|r| r.expected_output_len == fixed_len),
            "all samples must use the fixed output length {fixed_len}"
        );
    }

    #[test]
    fn test_load_hf_dataset_no_oversample() {
        let tok = builtin_tokenizer();
        // 2 valid rows, but request 10 with no_oversample=true
        let data = serde_json::json!([
            {"prompt": "Explain how photosynthesis converts sunlight to energy in plant cells.", "completion": "Plants use chlorophyll to absorb sunlight."},
            {"prompt": "What is the difference between a virus and a bacterium in terms of biology?", "completion": "Viruses need host cells while bacteria are self-sufficient."}
        ]);
        let path = write_temp_json("no-oversample", &data);

        let result = load_hf_dataset(
            &tok,
            &path,
            10,
            Some(32),
            0,
            "t-",
            None,
            true, // no_oversample
            false,
        )
        .unwrap();

        // Should have at most 2 samples (the actual dataset size), not 10
        assert!(
            result.len() <= 2,
            "no_oversample should cap result at dataset size, got {}",
            result.len()
        );
    }

    #[test]
    fn test_load_hf_dataset_disable_shuffle_preserves_order() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {"prompt": "Alpha prompt about the first topic in our alphabetically ordered sequence.", "completion": "Alpha answer"},
            {"prompt": "Beta prompt about the second topic continuing from our ordered sequence here.", "completion": "Beta answer"},
            {"prompt": "Gamma prompt about the third item in our clearly ordered alphabetical sequence.", "completion": "Gamma answer"}
        ]);
        let path = write_temp_json("disable-shuffle", &data);

        let result = load_hf_dataset(
            &tok,
            &path,
            3,
            Some(20),
            99,
            "t-",
            None,
            false,
            true, // disable_shuffle
        )
        .unwrap();

        assert_eq!(result.len(), 3);
        // With disable_shuffle, rows come out in original order:
        // Alpha < Beta < Gamma (alphabetical), so first prompt contains "Alpha"
        assert!(
            result[0].prompt.to_lowercase().contains("alpha"),
            "first result should be Alpha prompt with shuffle disabled, got: {}",
            result[0].prompt
        );
        assert!(
            result[1].prompt.to_lowercase().contains("beta"),
            "second result should be Beta prompt, got: {}",
            result[1].prompt
        );
    }

    #[test]
    fn test_load_hf_dataset_request_id_prefix() {
        let tok = builtin_tokenizer();
        let data = serde_json::json!([
            {"prompt": "What is the boiling point of water at sea level under standard atmospheric pressure?", "completion": "Water boils at 100 degrees Celsius."},
            {"prompt": "Describe the structure of DNA and how genetic information is encoded within it.", "completion": "DNA is a double helix with base pairs."}
        ]);
        let path = write_temp_json("request-id-prefix", &data);

        let result =
            load_hf_dataset(&tok, &path, 2, Some(10), 0, "myprefix-", None, false, false).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].request_id.as_deref(), Some("myprefix-0"));
        assert_eq!(result[1].request_id.as_deref(), Some("myprefix-1"));
    }
}
