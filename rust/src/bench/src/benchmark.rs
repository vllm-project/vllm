// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

use crate::backends::{RequestFuncInput, RequestFuncOutput, get_backend};
use crate::cli::{BackendKind, DatasetName, LoraAssignment};
use crate::config::BenchConfig;
use crate::error::{BenchError, Result};
use crate::metrics::calculator::{calculate_embedding_metrics, calculate_metrics};
use crate::metrics::steady_state;
use crate::output::console::print_results;
use crate::output::json::{append_result, build_result_json, compute_result_filename, save_result};
use crate::rate_control::compute_schedule;
use crate::ready_checker::wait_for_endpoint;

/// Pre-resolve the hostname in `base_url` and pin all resolved IPs on the
/// client builder via [`reqwest::ClientBuilder::resolve_to_addrs`].  This
/// avoids repeated DNS lookups under high concurrency which can cause
/// transient "Temporary failure in name resolution" errors while preserving
/// happy-eyeballs and multi-A failover.
///
/// Skipped when: URL parse fails, host is already an IP, host is a loopback
/// name (resolved from `/etc/hosts`, no DNS pressure and dual-stack ambiguity
/// between `127.0.0.1` and `::1` breaks IPv4-only servers like vLLM), or
/// resolution fails.
pub fn pre_resolve_dns(
    base_url: &str,
    mut builder: reqwest::ClientBuilder,
) -> reqwest::ClientBuilder {
    let parsed = match url::Url::parse(base_url) {
        Ok(u) => u,
        Err(_) => return builder,
    };

    let host = match parsed.host_str() {
        Some(h) => h,
        None => return builder,
    };

    if host.parse::<std::net::IpAddr>().is_ok() {
        return builder;
    }

    let host_lower = host.to_ascii_lowercase();
    if host_lower == "localhost"
        || host_lower == "ip6-localhost"
        || host_lower.ends_with(".localhost")
    {
        return builder;
    }

    let port = parsed.port_or_known_default().unwrap_or(80);
    let addr_str = format!("{host}:{port}");

    match std::net::ToSocketAddrs::to_socket_addrs(&addr_str) {
        Ok(addrs) => {
            let mut v4 = Vec::new();
            let mut v6 = Vec::new();
            for addr in addrs {
                if addr.is_ipv4() {
                    v4.push(addr);
                } else {
                    v6.push(addr);
                }
            }
            v4.extend(v6);
            if !v4.is_empty() {
                let ips: Vec<_> = v4.iter().map(|a| a.ip()).collect();
                println!("Pre-resolved {host} -> {ips:?}");
                builder = builder.resolve_to_addrs(host, &v4);
            }
        }
        Err(e) => {
            eprintln!("Warning: DNS pre-resolution for '{host}' failed: {e}");
        }
    }

    builder
}

/// Raw speculative decoding metrics from the server's Prometheus endpoint.
#[derive(Debug, Clone)]
pub(crate) struct SpecDecodeMetrics {
    num_drafts: u64,
    num_draft_tokens: u64,
    num_accepted_tokens: u64,
    accepted_per_pos: HashMap<u64, u64>,
}

/// Computed speculative decoding statistics (delta between before/after benchmark).
#[derive(Debug, Clone)]
pub struct SpecDecodeStats {
    pub num_drafts: u64,
    pub draft_tokens: u64,
    pub accepted_tokens: u64,
    pub acceptance_rate: f64,
    pub acceptance_length: f64,
    pub per_position_acceptance_rates: Vec<f64>,
}

/// Fetch speculative decoding metrics from the server's Prometheus `/metrics` endpoint.
///
/// Returns None if speculative decoding is not enabled or metrics are not available.
pub(crate) async fn fetch_spec_decode_metrics(
    base_url: &str,
    client: &reqwest::Client,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) -> Option<SpecDecodeMetrics> {
    let metrics_url = format!("{base_url}/metrics");
    let mut request = client.get(&metrics_url);
    if let Some(headers) = extra_headers {
        for (k, v) in headers {
            request = request.header(k, v);
        }
    }
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        request = request.header("Authorization", format!("Bearer {api_key}"));
    }

    let response = match request.send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return None,
    };

    let text = match response.text().await {
        Ok(t) => t,
        Err(_) => return None,
    };

    let mut num_drafts: u64 = 0;
    let mut num_draft_tokens: u64 = 0;
    let mut num_accepted_tokens: u64 = 0;
    let mut accepted_per_pos: HashMap<u64, u64> = HashMap::new();
    let mut found_spec_decode = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with("vllm:spec_decode") {
            continue;
        }
        found_spec_decode = true;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let val = match parts.last().and_then(|s| s.parse::<f64>().ok()) {
            Some(v) => v as u64,
            None => continue,
        };

        if line.contains("num_drafts") {
            num_drafts += val;
        } else if line.contains("num_draft_tokens") {
            num_draft_tokens += val;
        } else if line.contains("num_accepted_tokens_per_pos") {
            // Parse position label: position="N"
            if let Some(start) = line.find("position=\"") {
                let start = start + "position=\"".len();
                if let Some(end) = line[start..].find('"')
                    && let Ok(pos) = line[start..start + end].parse::<u64>()
                {
                    *accepted_per_pos.entry(pos).or_insert(0) += val;
                }
            }
        } else if line.contains("num_accepted_tokens") {
            num_accepted_tokens += val;
        }
    }

    if !found_spec_decode {
        return None;
    }

    Some(SpecDecodeMetrics {
        num_drafts,
        num_draft_tokens,
        num_accepted_tokens,
        accepted_per_pos,
    })
}

/// Compute speculative decoding stats from before/after metrics snapshots.
pub(crate) fn compute_spec_decode_stats(
    before: &SpecDecodeMetrics,
    after: &SpecDecodeMetrics,
) -> Option<SpecDecodeStats> {
    let delta_drafts = after.num_drafts.saturating_sub(before.num_drafts);
    let delta_draft_tokens = after.num_draft_tokens.saturating_sub(before.num_draft_tokens);
    let delta_accepted = after.num_accepted_tokens.saturating_sub(before.num_accepted_tokens);

    if delta_draft_tokens == 0 {
        return None;
    }

    let mut per_pos_rates = Vec::new();
    if delta_drafts > 0 {
        let mut positions: Vec<u64> = before
            .accepted_per_pos
            .keys()
            .chain(after.accepted_per_pos.keys())
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        positions.sort();

        for pos in positions {
            let before_val = before.accepted_per_pos.get(&pos).copied().unwrap_or(0);
            let after_val = after.accepted_per_pos.get(&pos).copied().unwrap_or(before_val);
            let delta_pos = after_val.saturating_sub(before_val);
            per_pos_rates.push(delta_pos as f64 / delta_drafts as f64);
        }
    }

    let acceptance_rate = (delta_accepted as f64 / delta_draft_tokens as f64) * 100.0;
    let acceptance_length = if delta_drafts > 0 {
        1.0 + delta_accepted as f64 / delta_drafts as f64
    } else {
        0.0
    };

    Some(SpecDecodeStats {
        num_drafts: delta_drafts,
        draft_tokens: delta_draft_tokens,
        accepted_tokens: delta_accepted,
        acceptance_rate,
        acceptance_length,
        per_position_acceptance_rates: per_pos_rates,
    })
}

/// Pre-assign a LoRA adapter name per item based on the configured strategy.
///
/// Returns `None` when LoRA is not configured. With LoRA enabled, returns a
/// `Vec<Arc<str>>` of length `n` whose i-th entry is the adapter name to use
/// for the i-th item.
///
/// - `RoundRobin` cycles deterministically: `lora_modules[i % N]`.
/// - `Random` uses `StdRng::seed_from_u64(seed)` so the assignment is fully reproducible across
///   runs with the same seed.
///
/// "Item" is a request in single-shot mode and a conversation in multi-turn
/// mode (sticky across all turns of that conversation).
pub(crate) fn assign_lora_modules(
    lora_modules: &Option<Vec<Arc<str>>>,
    assignment: LoraAssignment,
    n: usize,
    seed: u64,
) -> Option<Vec<Arc<str>>> {
    let modules = lora_modules.as_ref()?;
    if modules.is_empty() {
        return None;
    }
    let m = modules.len();
    let mut out = Vec::with_capacity(n);
    match assignment {
        LoraAssignment::RoundRobin => {
            for i in 0..n {
                out.push(modules[i % m].clone());
            }
        }
        LoraAssignment::Random => {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..n {
                out.push(modules[rng.random_range(0..m)].clone());
            }
        }
    }
    Some(out)
}

/// Fetch the first model from the server's /v1/models endpoint.
async fn get_first_model_from_server(
    base_url: &str,
    client: &reqwest::Client,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) -> Result<(String, String)> {
    let url = format!("{base_url}/v1/models");
    let mut request = client.get(&url);
    if let Some(headers) = extra_headers {
        for (k, v) in headers {
            request = request.header(k, v);
        }
    }
    // Add API key from environment
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        request = request.header("Authorization", format!("Bearer {api_key}"));
    }

    let response = request.send().await?;
    let data: serde_json::Value = response.json().await?;

    if let Some(models) = data.get("data").and_then(|d| d.as_array())
        && let Some(first) = models.first()
    {
        let id = first.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let root = first.get("root").and_then(|v| v.as_str()).unwrap_or(&id).to_string();
        return Ok((id, root));
    }

    Err(BenchError::Config(format!(
        "No models found on the server at {base_url}"
    )))
}

/// Run the complete benchmark.
///
/// This is the core orchestrator matching Python's benchmark() + main_async().
pub async fn run_benchmark(config: &BenchConfig) -> Result<serde_json::Value> {
    // Validate backend name early
    let _ = get_backend(config.backend)?;

    // Build HTTP client with connection pool settings matching Python's aiohttp.TCPConnector
    // Force HTTP/1.1 to match aiohttp behavior (avoids HTTP/2 negotiation issues)
    let mut client_builder = reqwest::Client::builder()
        .pool_max_idle_per_host(config.max_concurrency.unwrap_or(2048).max(256))
        .timeout(std::time::Duration::from_secs(6 * 60 * 60))
        .connect_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .tcp_nodelay(true)
        .http1_only()
        .no_proxy();

    if config.insecure {
        client_builder = client_builder.danger_accept_invalid_certs(true);
    }

    client_builder = pre_resolve_dns(&config.base_url, client_builder);

    let client = client_builder
        .build()
        .map_err(|e| BenchError::Backend(format!("Failed to build HTTP client: {e}")))?;

    // Resolve model
    let (model_id, model_name) = if let Some(ref m) = config.model {
        (m.clone(), config.model_name.clone())
    } else {
        println!("Model not specified, fetching first model from server...");
        let (name, id) =
            get_first_model_from_server(&config.base_url, &client, &config.extra_headers).await?;
        println!("First model name: {name}, first model id: {id}");
        (id, Some(name))
    };

    // Load tokenizer (if needed)
    let tokenizer = if config.skip_tokenizer_init {
        None
    } else {
        let tid = config.tokenizer_id.as_deref().unwrap_or(&model_id);
        println!("Loading tokenizer: {tid}");
        let server_info = Some((config.base_url.as_str(), model_id.as_str()));
        let t = crate::tokenizer::load_tokenizer(tid, config.trust_remote_code, server_info)?;
        println!("Tokenizer loaded successfully.");
        Some(t)
    };
    let has_tokenizer = tokenizer.is_some();

    // Generate dataset
    let dataset_label = match config.dataset_name {
        DatasetName::Random => format!("{} random prompts", config.num_prompts),
        DatasetName::RandomMm => format!(
            "{} random multimodal prompts ({} items/req)",
            config.num_prompts, config.random_mm_base_items_per_request
        ),
        DatasetName::ShareGpt => format!(
            "{} prompts from ShareGPT ({})",
            config.num_prompts,
            config.dataset_path.as_deref().unwrap_or("auto-download")
        ),
        DatasetName::Sonnet => format!(
            "{} prompts from Sonnet ({}, isl={}, osl={}, prefix={})",
            config.num_prompts,
            config.dataset_path.as_deref().unwrap_or("built-in"),
            config.sonnet_input_len,
            config.sonnet_output_len,
            config.sonnet_prefix_len,
        ),
        DatasetName::SpeedBench => {
            let truncate_info = config
                .speed_bench_max_input_len
                .map(|n| format!(", truncated to {n} tokens"))
                .unwrap_or_default();
            format!(
                "{} prompts from SPEED-Bench ({}/{}{})",
                config.num_prompts,
                config.speed_bench_config,
                config.speed_bench_category.as_deref().unwrap_or("all"),
                truncate_info,
            )
        }
        DatasetName::Hf => format!(
            "{} prompts from HF dataset ({})",
            config.num_prompts,
            config.dataset_path.as_deref().unwrap_or("unknown"),
        ),
        DatasetName::Custom => format!(
            "{} prompts from custom JSONL ({})",
            config.num_prompts,
            config.dataset_path.as_deref().unwrap_or("unknown"),
        ),
        DatasetName::PrefixRepetition => format!(
            "{} prefix-repetition prompts ({} prefixes, prefix={}, suffix={})",
            config.num_prompts,
            config.prefix_repetition_num_prefixes,
            config.prefix_repetition_prefix_len,
            config.prefix_repetition_suffix_len,
        ),
        DatasetName::RandomRerank => format!(
            "{} random rerank requests (batch={}, reranker={})",
            config.num_prompts, config.random_batch_size, config.is_reranker,
        ),
    };
    println!("Generating {dataset_label}...");
    let gen_start = Instant::now();

    let mut input_requests = match config.dataset_name {
        DatasetName::Random => {
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| BenchError::Config("Random dataset requires a tokenizer".into()))?;
            crate::datasets::random::generate_random_dataset(
                tok,
                config.num_prompts,
                config.random_input_len,
                config.random_output_len,
                config.random_prefix_len,
                config.random_range_ratio,
                config.random_cache_hit_fraction,
                config.random_cache_ratio,
                config.seed,
                &config.request_id_prefix,
                config.prompt_token_ids,
                config.random_batch_size,
            )?
        }
        DatasetName::RandomMm => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("Random-MM dataset requires a tokenizer".into())
            })?;
            crate::datasets::random_mm::generate_random_mm_dataset(
                tok,
                config.num_prompts,
                config.random_input_len,
                config.random_output_len,
                config.random_prefix_len,
                config.random_range_ratio,
                config.seed,
                &config.request_id_prefix,
                config.random_mm_base_items_per_request,
                config.random_mm_num_mm_items_range_ratio,
                &config.random_mm_limit,
                &config.random_mm_buckets,
                config.enable_multimodal_chat,
            )?
        }
        DatasetName::ShareGpt => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("ShareGPT dataset requires a tokenizer".into())
            })?;
            let downloaded;
            let path = match config.dataset_path.as_deref() {
                Some(p) => p,
                None => {
                    downloaded = crate::datasets::sharegpt::download_sharegpt_dataset()?;
                    downloaded.as_str()
                }
            };
            crate::datasets::sharegpt::load_sharegpt_dataset(
                tok,
                path,
                config.num_prompts,
                config.sharegpt_output_len,
                config.seed,
                &config.request_id_prefix,
                config.no_oversample,
                config.disable_shuffle,
            )?
        }
        DatasetName::Sonnet => {
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| BenchError::Config("Sonnet dataset requires a tokenizer".into()))?;
            crate::datasets::sonnet::load_sonnet_dataset(
                tok,
                config.dataset_path.as_deref(),
                config.num_prompts,
                config.sonnet_input_len,
                config.sonnet_output_len,
                config.sonnet_prefix_len,
                config.seed,
                &config.request_id_prefix,
            )?
        }
        DatasetName::SpeedBench => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("SPEED-Bench dataset requires a tokenizer".into())
            })?;
            let downloaded;
            let path = match config.dataset_path.as_deref() {
                Some(p) => p,
                None => {
                    downloaded = crate::datasets::speed_bench::download_speed_bench(
                        config.speed_bench_config,
                    )?;
                    downloaded.as_str()
                }
            };
            let output_len = config.sharegpt_output_len.unwrap_or(config.random_output_len);
            crate::datasets::speed_bench::load_speed_bench_dataset(
                tok,
                path,
                config.num_prompts,
                output_len,
                config.seed,
                &config.request_id_prefix,
                config.speed_bench_category.as_deref(),
                config.no_oversample,
                config.disable_shuffle,
                config.speed_bench_max_input_len,
            )?
        }
        DatasetName::Hf => {
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| BenchError::Config("HF dataset requires a tokenizer".into()))?;
            let dataset_id = config.dataset_path.as_deref().ok_or_else(|| {
                BenchError::Config("--dataset-path is required for --dataset-name hf".into())
            })?;
            let (downloaded_path, _config, _split) =
                crate::datasets::hf_dataset::download_hf_dataset(
                    dataset_id,
                    config.hf_subset.as_deref(),
                    config.hf_split.as_deref(),
                    config.num_prompts,
                )?;
            crate::datasets::hf_dataset::load_hf_dataset(
                tok,
                &downloaded_path,
                config.num_prompts,
                config.hf_output_len,
                config.seed,
                &config.request_id_prefix,
                config.hf_text_column.as_deref(),
                config.no_oversample,
                config.disable_shuffle,
            )?
        }
        DatasetName::Custom => {
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| BenchError::Config("Custom dataset requires a tokenizer".into()))?;
            let path = config.dataset_path.as_deref().ok_or_else(|| {
                BenchError::Config("--dataset-path is required for --dataset-name custom".into())
            })?;
            crate::datasets::custom::load_custom_dataset(
                tok,
                path,
                config.num_prompts,
                config.custom_output_len,
                config.seed,
                &config.request_id_prefix,
                config.no_oversample,
                config.disable_shuffle,
            )?
        }
        DatasetName::PrefixRepetition => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("Prefix repetition dataset requires a tokenizer".into())
            })?;
            crate::datasets::prefix_repetition::generate_prefix_repetition_dataset(
                tok,
                config.num_prompts,
                config.prefix_repetition_prefix_len,
                config.prefix_repetition_suffix_len,
                config.prefix_repetition_num_prefixes,
                config.prefix_repetition_output_len,
                config.seed,
                &config.request_id_prefix,
                config.disable_shuffle,
            )?
        }
        DatasetName::RandomRerank => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("Random rerank dataset requires a tokenizer".into())
            })?;
            crate::datasets::random_rerank::generate_random_rerank_dataset(
                tok,
                config.num_prompts,
                config.random_input_len,
                config.random_range_ratio,
                config.seed,
                &config.request_id_prefix,
                config.random_batch_size,
                config.is_reranker,
            )?
        }
    };

    let gen_elapsed = gen_start.elapsed();
    println!(
        "Generated {} prompts in {:.2}s",
        input_requests.len(),
        gen_elapsed.as_secs_f64()
    );

    let filtered_count =
        filter_requests_by_max_model_len(&mut input_requests, config.max_model_len);
    if filtered_count > 0 {
        println!(
            "Filtered {filtered_count} prompt(s) above --max-model-len {}.",
            config.max_model_len.unwrap()
        );
    }
    if input_requests.is_empty() {
        return Err(BenchError::Config(
            "No requests remain after applying --max-model-len".into(),
        ));
    }

    // Dry run: just print stats and exit
    if config.dry_run {
        let total_input_tokens: usize = input_requests.iter().map(|r| r.prompt_len).sum();
        let total_output_tokens: usize = input_requests.iter().map(|r| r.expected_output_len).sum();
        println!("Dry run stats:");
        println!("  Total prompts: {}", input_requests.len());
        println!("  Total input tokens: {total_input_tokens}");
        println!("  Total expected output tokens: {total_output_tokens}");
        println!(
            "  Avg input tokens: {:.1}",
            total_input_tokens as f64 / input_requests.len() as f64
        );
        println!(
            "  Avg output tokens: {:.1}",
            total_output_tokens as f64 / input_requests.len() as f64
        );
        return Ok(serde_json::json!({"dry_run": true}));
    }

    // Build test input from first request
    let first = &input_requests[0];
    let test_input = RequestFuncInput {
        prompt: first.prompt.clone(),
        api_url: config.api_url.clone(),
        prompt_len: first.prompt_len,
        output_len: first.expected_output_len,
        model: model_id.clone(),
        model_name: model_name.clone(),
        logprobs: config.logprobs,
        extra_headers: config.extra_headers.clone(),
        extra_body: config.extra_body.clone(),
        ignore_eos: config.ignore_eos,
        request_id: first.request_id.clone(),
        messages: None,
        prompt_token_ids: first.prompt_token_ids.clone(),
        multi_modal_content: first.multi_modal_content.clone(),
        chat_messages_json: first.chat_messages_json.clone(),
        prompt_list: first.prompt_list.clone(),
    };

    // Ready check
    if config.ready_check_timeout_sec > 0 {
        println!("Starting initial single prompt test run...");
        let test_output = wait_for_endpoint(
            config.backend,
            &client,
            &test_input,
            config.ready_check_timeout_sec,
            5,
        )
        .await?;
        if !test_output.success {
            return Err(BenchError::Backend(format!(
                "Initial test run failed: {}",
                test_output.error
            )));
        }
        println!("Initial test run completed.");
    }

    // Verify and fix prompt token lengths against the server's /tokenize endpoint.
    // Runs after the ready check so a still-starting server isn't mistaken for a
    // tokenize failure, and after the dry-run exit so dry runs stay offline.
    // Uses a cache: if a previous run with the same model+server already verified OK,
    // skip entirely. Otherwise sample 10 prompts first — if all match, cache and skip.
    // If any mismatch, do full verify+fix for all prompts.
    // Skip verification when prompt_token_ids are set (token counts are exact by construction).
    let has_token_ids = input_requests.first().is_some_and(|r| r.prompt_token_ids.is_some());
    // Python aligns prompts to the server tokenizer for random AND prefix_repetition
    // (both are synthetic exact-length datasets).
    let verifiable_dataset = matches!(
        config.dataset_name,
        DatasetName::Random | DatasetName::PrefixRepetition
    );
    if verifiable_dataset && has_token_ids && !config.backend.is_pooling() {
        println!("Using prompt_token_ids, skipping server-side tokenizer verification.");
    }
    if verifiable_dataset && !has_token_ids && !config.backend.is_pooling() {
        let cache_key = tokenizer_verify_cache_key(&config.base_url, &model_id);
        if is_tokenizer_verified(&cache_key) {
            println!("Tokenizer verified in previous run (cached), skipping verification.");
        } else {
            let num_special =
                tokenizer.as_ref().map(|t| t.num_special_tokens_to_add()).unwrap_or(0);
            match sample_verify_prompts(
                &client,
                &config.base_url,
                &model_id,
                &input_requests,
                num_special,
                &config.extra_headers,
            )
            .await?
            {
                SampleVerifyOutcome::Passed => {
                    println!("Sample verification passed, skipping full verification.");
                    mark_tokenizer_verified(&cache_key);
                }
                SampleVerifyOutcome::Skipped(reason) => {
                    println!("Server /tokenize unavailable ({reason}), skipping verification.");
                }
                SampleVerifyOutcome::Mismatch => {
                    println!("Sample verification found mismatch, running full verify+fix...");
                    match verify_and_fix_prompt_lengths(
                        &client,
                        &config.base_url,
                        &model_id,
                        &mut input_requests,
                        num_special,
                        &config.extra_headers,
                    )
                    .await
                    {
                        Ok(()) => {
                            println!(
                                "All {} prompts verified: exact token length match.",
                                input_requests.len()
                            );
                            mark_tokenizer_verified(&cache_key);
                        }
                        Err(BenchError::TokenizeUnavailable(reason)) => {
                            println!(
                                "Server /tokenize became unavailable during verification \
                                 ({reason}); proceeding with client-side token counts."
                            );
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }

    // Warmup
    if config.num_warmups > 0 {
        println!("Warming up with {} requests...", config.num_warmups);
        run_warmup(
            config.backend,
            &client,
            &test_input,
            config.num_warmups,
            config.max_concurrency,
            config.request_rate,
            config.burstiness,
            config.seed,
            config.disable_tqdm,
        )
        .await;
        println!("Warmup run completed.");
    }

    // Start profiler if requested (immediate mode — no batch threshold)
    if config.profile && config.profile_batch_threshold.is_none() {
        start_profiler_immediate(&client, &config.base_url, &config.extra_headers).await;
    }

    // Threshold-based profiling: spawn background task that polls /metrics
    // and triggers start/stop profile when batch size is reached.
    // A oneshot channel lets us cancel the polling loop when the benchmark ends
    // (e.g. if the threshold is never reached).
    let profile_task = if let Some(threshold) = config.profile_batch_threshold {
        let poll_client = client.clone();
        let base_url = config.base_url.clone();
        let extra_headers = config.extra_headers.clone();
        let duration_secs = config.profile_duration;
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            profile_on_batch_threshold(
                &poll_client,
                &base_url,
                &extra_headers,
                threshold,
                duration_secs,
                cancel_rx,
            )
            .await;
        });
        Some((cancel_tx, handle))
    } else {
        None
    };

    // Fetch speculative decoding metrics before benchmark
    let spec_decode_before =
        fetch_spec_decode_metrics(&config.base_url, &client, &config.extra_headers).await;
    if spec_decode_before.is_some() {
        println!("Speculative decoding detected, will collect metrics.");
    }

    // Main benchmark
    println!("Starting main benchmark run...");
    let distribution = if config.burstiness == 1.0 {
        "Poisson process"
    } else {
        "Gamma distribution"
    };
    println!(
        "Traffic request rate: {}",
        if config.request_rate.is_infinite() {
            "inf".to_string()
        } else {
            format!("{}", config.request_rate)
        }
    );
    println!("Burstiness factor: {} ({distribution})", config.burstiness);
    println!(
        "Maximum request concurrency: {}",
        config.max_concurrency.unwrap_or(config.num_prompts)
    );

    // Pre-assign LoRA adapters to each request (None when --lora-modules not set).
    let lora_assignments = assign_lora_modules(
        &config.lora_modules,
        config.lora_assignment,
        input_requests.len(),
        config.seed,
    );
    if let (Some(modules), Some(_)) = (config.lora_modules.as_ref(), lora_assignments.as_ref()) {
        let names: Vec<&str> = modules.iter().map(|s| s.as_ref()).collect();
        println!(
            "LoRA adapters ({}): {:?} [assignment={:?}]",
            modules.len(),
            names,
            config.lora_assignment
        );
    }

    // Compute request schedule
    let schedule = compute_schedule(
        input_requests.len(),
        config.request_rate,
        config.burstiness,
        config.seed,
        config.ramp_up.as_ref(),
    );

    // Progress bar
    let pb = if config.disable_tqdm {
        None
    } else {
        let bar = ProgressBar::new(input_requests.len() as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        Some(bar)
    };

    // Semaphore for concurrency control
    let semaphore = config.max_concurrency.map(|mc| Arc::new(Semaphore::new(mc)));

    let benchmark_start = Instant::now();

    // Arc-wrap shared data to avoid cloning per request
    let shared_api_url = Arc::new(config.api_url.clone());
    let shared_model_id = Arc::new(model_id.clone());
    let shared_model_name = Arc::new(model_name.clone());
    let shared_extra_headers = Arc::new(config.extra_headers.clone());
    let shared_extra_body = Arc::new(config.extra_body.clone());
    let shared_backend = get_backend(config.backend)?;
    let shared_logprobs = config.logprobs;
    let shared_ignore_eos = config.ignore_eos;

    // Spawn all request tasks. Store prompt_len alongside handle so
    // we can preserve it in the panic recovery path.
    let mut handles: Vec<(usize, tokio::task::JoinHandle<RequestFuncOutput>)> =
        Vec::with_capacity(input_requests.len());

    for (i, (request, delay)) in input_requests.iter().zip(schedule.delays.iter()).enumerate() {
        let client = client.clone();
        let backend = shared_backend.clone();
        let sem = semaphore.clone();
        let pb = pb.clone();
        let api_url = shared_api_url.clone();
        let model = shared_model_id.clone();
        let model_name = shared_model_name.clone();
        let extra_headers = shared_extra_headers.clone();
        let extra_body = shared_extra_body.clone();

        // Per-request LoRA override: the adapter name replaces both `model` and
        // `model_name` in the outgoing payload (vLLM routes by name). Tokenizer,
        // /v1/models, /tokenize, etc. continue using the base model unchanged.
        let lora_name = lora_assignments.as_ref().map(|v| v[i].clone());

        let prompt = request.prompt.clone();
        let prompt_len = request.prompt_len;
        let output_len = request.expected_output_len;
        let request_id = request.request_id.clone();
        let prompt_token_ids = request.prompt_token_ids.clone();
        let multi_modal_content = request.multi_modal_content.clone();
        let chat_messages_json = request.chat_messages_json.clone();
        let prompt_list = request.prompt_list.clone();

        let delay_dur = std::time::Duration::from_secs_f64(*delay);
        let bench_start = benchmark_start;

        handles.push((
            prompt_len,
            tokio::spawn(async move {
                // Sleep until scheduled time
                let target = bench_start + delay_dur;
                let now = Instant::now();
                if target > now {
                    tokio::time::sleep(target - now).await;
                }

                // Acquire semaphore permit
                let _permit = if let Some(ref s) = sem {
                    Some(s.acquire().await.unwrap())
                } else {
                    None
                };

                let (req_model, req_model_name) = match lora_name.as_ref() {
                    Some(name) => (name.to_string(), Some(name.to_string())),
                    None => ((*model).clone(), (*model_name).clone()),
                };

                let input = RequestFuncInput {
                    prompt,
                    api_url: (*api_url).clone(),
                    prompt_len,
                    output_len,
                    model: req_model,
                    model_name: req_model_name,
                    logprobs: shared_logprobs,
                    extra_headers: (*extra_headers).clone(),
                    extra_body: (*extra_body).clone(),
                    ignore_eos: shared_ignore_eos,
                    request_id,
                    messages: None,
                    prompt_token_ids,
                    multi_modal_content,
                    chat_messages_json,
                    prompt_list,
                };

                // Send request, retry on connection errors
                let max_retries = 3;
                let mut output = None;

                for attempt in 0..=max_retries {
                    // Capture monotonic start time right before sending,
                    // relative to benchmark_start. Python uses perf_counter()
                    // for both start_time and ttft/itl, keeping them on the
                    // same clock. We do the same with Instant.
                    let request_instant = Instant::now();
                    let result = backend.send_request(&input, &client).await;

                    match result {
                        Ok(mut o) => {
                            // Override SystemTime-based start_time with monotonic offset
                            o.start_time =
                                request_instant.duration_since(bench_start).as_secs_f64();

                            // Retry on connection-level failures reported as !success
                            if !o.success && attempt < max_retries && is_connection_error(&o.error)
                            {
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    500 * (attempt as u64 + 1),
                                ))
                                .await;
                                continue;
                            }
                            output = Some(o);
                            break;
                        }
                        Err(e) => {
                            if attempt < max_retries {
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    500 * (attempt as u64 + 1),
                                ))
                                .await;
                                continue;
                            }
                            output = Some(RequestFuncOutput {
                                success: false,
                                error: e.to_string(),
                                prompt_len: input.prompt_len,
                                start_time: request_instant
                                    .duration_since(bench_start)
                                    .as_secs_f64(),
                                ..Default::default()
                            });
                            break;
                        }
                    }
                }

                if let Some(pb) = pb {
                    pb.inc(1);
                }

                output.unwrap()
            }),
        ));
    }

    // Collect all results, handling task panics gracefully
    let mut outputs = Vec::with_capacity(handles.len());
    for (prompt_len, handle) in handles {
        match handle.await {
            Ok(output) => outputs.push(output),
            Err(e) => {
                outputs.push(RequestFuncOutput {
                    success: false,
                    error: format!("Task panicked: {e}"),
                    prompt_len,
                    ..Default::default()
                });
            }
        }
    }

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    let benchmark_duration = benchmark_start.elapsed().as_secs_f64();

    // Fetch speculative decoding metrics after benchmark and compute stats
    let spec_decode_stats = if spec_decode_before.is_some() {
        let spec_decode_after =
            fetch_spec_decode_metrics(&config.base_url, &client, &config.extra_headers).await;
        match (spec_decode_before.as_ref(), spec_decode_after.as_ref()) {
            (Some(before), Some(after)) => compute_spec_decode_stats(before, after),
            _ => None,
        }
    } else {
        None
    };

    // Calculate metrics — pooling uses a dedicated path matching Python's
    // calculate_metrics_for_embeddings (uses server-reported prompt_len, e2el only).
    let (mut metrics, actual_output_lens) = if config.backend.is_pooling() {
        let m =
            calculate_embedding_metrics(&outputs, benchmark_duration, &config.selected_percentiles);
        (m, Vec::new())
    } else {
        calculate_metrics(
            &input_requests,
            &outputs,
            benchmark_duration,
            &config.selected_percentiles,
            has_tokenizer,
            &config.goodput,
        )
    };

    // Attach steady-state metrics when the closed-loop scope gate passes.
    let scope_ok = !config.no_steady_state
        && config.max_concurrency.is_some()
        && config.request_rate.is_infinite();
    if scope_ok {
        let target = config.max_concurrency;
        let min_window = config
            .steady_state_min_window
            .unwrap_or_else(|| (0.1 * benchmark_duration).max(10.0));
        if let Some(window) = steady_state::detect_window(
            &outputs,
            target,
            config.steady_state_threshold,
            min_window,
            benchmark_duration,
        ) {
            let ss = steady_state::compute(
                &outputs,
                &input_requests,
                &window,
                &config.selected_percentiles,
                config.backend.is_pooling(),
            );
            metrics.steady_state = Some(ss);
        }
    }

    // Print console output
    print_results(
        &metrics,
        benchmark_duration,
        config,
        has_tokenizer,
        spec_decode_stats.as_ref(),
    );

    // Stop profiler if requested (immediate mode — no batch threshold)
    if config.profile && config.profile_batch_threshold.is_none() {
        stop_profiler_immediate(&client, &config.base_url, &config.extra_headers).await;
    }

    // Signal the threshold-based profile task that the benchmark is done, then wait
    if let Some((cancel_tx, task)) = profile_task {
        let _ = cancel_tx.send(());
        if let Err(e) = task.await {
            eprintln!("WARNING: Profile background task failed: {e}");
        }
    }

    // Build result JSON
    let date_iso = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let dt_filename = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let result_json = build_result_json(
        config,
        &metrics,
        &actual_output_lens,
        &outputs,
        benchmark_duration,
        &date_iso,
        spec_decode_stats.as_ref(),
    );

    // Save if requested
    if config.save_result || config.append_result {
        let model_for_filename = config.model.as_deref().unwrap_or(&model_id);
        let file_name = compute_result_filename(config, model_for_filename, &dt_filename);

        // Create result directory if needed
        if let Some(ref dir) = config.result_dir {
            std::fs::create_dir_all(dir)?;
        }

        if config.append_result {
            append_result(&result_json, &file_name)?;
        } else {
            save_result(&result_json, &file_name)?;
        }
    }

    Ok(result_json)
}

fn filter_requests_by_max_model_len(
    requests: &mut Vec<crate::datasets::SampleRequest>,
    max_model_len: Option<usize>,
) -> usize {
    let Some(max_model_len) = max_model_len else {
        return 0;
    };

    let before = requests.len();
    requests.retain(|request| {
        request.prompt_len.saturating_add(request.expected_output_len) <= max_model_len
    });
    before - requests.len()
}

#[cfg(test)]
mod max_model_len_tests {
    use std::sync::Arc;

    use super::filter_requests_by_max_model_len;
    use crate::datasets::SampleRequest;

    fn sample(prompt_len: usize, expected_output_len: usize) -> SampleRequest {
        SampleRequest {
            prompt: Arc::from("prompt"),
            prompt_len,
            expected_output_len,
            request_id: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_requests_by_max_model_len_keeps_boundary() {
        let mut requests = vec![sample(80, 20), sample(81, 20), sample(30, 10)];

        let filtered = filter_requests_by_max_model_len(&mut requests, Some(100));

        assert_eq!(filtered, 1);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].prompt_len, 80);
        assert_eq!(requests[1].prompt_len, 30);
    }

    #[test]
    fn test_filter_requests_by_max_model_len_noop_without_limit() {
        let mut requests = vec![sample(80, 20), sample(81, 20)];

        let filtered = filter_requests_by_max_model_len(&mut requests, None);

        assert_eq!(filtered, 0);
        assert_eq!(requests.len(), 2);
    }
}

async fn run_warmup(
    backend: BackendKind,
    client: &reqwest::Client,
    test_input: &RequestFuncInput,
    num_warmups: usize,
    max_concurrency: Option<usize>,
    request_rate: f64,
    burstiness: f64,
    seed: u64,
    disable_tqdm: bool,
) {
    let pb = if disable_tqdm {
        None
    } else {
        let bar = ProgressBar::new(num_warmups as u64);
        bar.set_style(
            ProgressStyle::with_template("{spinner:.green} Warmup [{bar:30}] {pos}/{len}")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(bar)
    };

    let semaphore = max_concurrency.map(|mc| Arc::new(Semaphore::new(mc)));

    // Use the same rate-limited scheduling as the main benchmark run
    let schedule = compute_schedule(num_warmups, request_rate, burstiness, seed, None);
    let start = Instant::now();

    let mut handles = Vec::with_capacity(num_warmups);
    for i in 0..num_warmups {
        // Wait until scheduled time
        let target = std::time::Duration::from_secs_f64(schedule.delays[i]);
        let elapsed = start.elapsed();
        if target > elapsed {
            tokio::time::sleep(target - elapsed).await;
        }

        let client = client.clone();
        let input = test_input.clone();
        let sem = semaphore.clone();
        let pb = pb.clone();
        handles.push(tokio::spawn(async move {
            let _permit = if let Some(ref s) = sem {
                Some(s.acquire().await.unwrap())
            } else {
                None
            };
            if let Ok(b) = get_backend(backend) {
                let _ = b.send_request(&input, &client).await;
            }
            if let Some(pb) = pb {
                pb.inc(1);
            }
        }));
    }

    for handle in handles {
        let _ = handle.await;
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
}

/// Start the profiler (immediate mode — no batch threshold).
pub(crate) async fn start_profiler_immediate(
    client: &reqwest::Client,
    base_url: &str,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) {
    println!("Starting profiler...");
    let profile_url = format!("{base_url}/start_profile");
    match send_profile_request(client, &profile_url, extra_headers).await {
        Ok(true) => println!("Profiler started"),
        Ok(false) => eprintln!("WARNING: Profiler start request returned non-success"),
        Err(e) => eprintln!("WARNING: Failed to start profiler: {e}"),
    }
}

/// Stop the profiler (immediate mode — no batch threshold).
pub(crate) async fn stop_profiler_immediate(
    client: &reqwest::Client,
    base_url: &str,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) {
    println!("Stopping profiler...");
    let profile_url = format!("{base_url}/stop_profile");
    match send_profile_request(client, &profile_url, extra_headers).await {
        Ok(true) => println!("Profiler stopped"),
        Ok(false) => eprintln!("WARNING: Profiler stop request returned non-success"),
        Err(e) => eprintln!("WARNING: Failed to stop profiler: {e}"),
    }
}

/// Send a request to the vLLM profiler endpoint (start_profile / stop_profile).
/// Returns Ok(true) if the server responded with 200.
pub(crate) async fn send_profile_request(
    client: &reqwest::Client,
    url: &str,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) -> Result<bool> {
    let mut request = client.post(url);
    if let Some(headers) = extra_headers {
        for (k, v) in headers {
            request = request.header(k, v);
        }
    }
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        request = request.header("Authorization", format!("Bearer {api_key}"));
    }

    let resp = request
        .send()
        .await
        .map_err(|e| BenchError::Backend(format!("Profile request to {url} failed: {e}")))?;

    Ok(resp.status().is_success())
}

/// Parse `vllm:num_requests_running` from the server's `/metrics` Prometheus endpoint.
pub(crate) async fn fetch_num_requests_running(
    client: &reqwest::Client,
    base_url: &str,
) -> Option<usize> {
    let url = format!("{base_url}/metrics");
    let resp = client.get(&url).send().await.ok()?;
    let text = resp.text().await.ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("vllm:num_requests_running") {
            // Line format: `vllm:num_requests_running{model_name="..."} 42`
            // The value is after the last space.
            if let Some(val_str) = rest.rsplit(' ').next()
                && let Ok(v) = val_str.parse::<f64>()
            {
                return Some(v as usize);
            }
        }
    }
    None
}

/// Poll `/metrics` until `num_requests_running >= threshold`, then start the
/// profiler, wait `duration_secs`, and stop it.  If `cancel_rx` fires before
/// the threshold is reached (i.e. the benchmark finished), exits early.
pub(crate) async fn profile_on_batch_threshold(
    client: &reqwest::Client,
    base_url: &str,
    extra_headers: &Option<HashMap<String, String>>,
    threshold: usize,
    duration_secs: f64,
    mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
) {
    println!(
        "Waiting for batch size >= {threshold} before starting profiler \
         (will capture {duration_secs}s)..."
    );

    loop {
        if let Some(running) = fetch_num_requests_running(client, base_url).await
            && running >= threshold
        {
            println!("Batch size {running} >= {threshold}, starting profiler...");
            break;
        }
        // Wait 500ms or until the benchmark signals cancellation
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
            _ = &mut cancel_rx => {
                eprintln!(
                    "NOTE: Benchmark finished before batch threshold {threshold} was reached; \
                     profiling skipped."
                );
                return;
            }
        }
    }

    let start_url = format!("{base_url}/start_profile");
    match send_profile_request(client, &start_url, extra_headers).await {
        Ok(true) => println!("Profiler started"),
        Ok(false) => {
            eprintln!("WARNING: Profiler start request returned non-success");
            return;
        }
        Err(e) => {
            eprintln!("WARNING: Failed to start profiler: {e}");
            return;
        }
    }

    // Wait for the capture duration, but exit early if the benchmark finishes
    tokio::select! {
        _ = tokio::time::sleep(std::time::Duration::from_secs_f64(duration_secs)) => {}
        _ = &mut cancel_rx => {
            println!("Benchmark finished, stopping profiler early...");
        }
    }

    let stop_url = format!("{base_url}/stop_profile");
    match send_profile_request(client, &stop_url, extra_headers).await {
        Ok(true) => println!("Profiler stopped after capturing"),
        Ok(false) => eprintln!("WARNING: Profiler stop request returned non-success"),
        Err(e) => eprintln!("WARNING: Failed to stop profiler: {e}"),
    }
}

/// Verify and fix prompt token lengths against the server's /tokenize + /detokenize.
/// For each prompt, if the server's token count doesn't match that prompt's own
/// target (its generated length plus the special tokens the server adds), adjust
/// the token sequence (pad/truncate) and detokenize, repeating until exact match.
/// Runs up to 64 prompts concurrently.
async fn verify_and_fix_prompt_lengths(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    requests: &mut [crate::datasets::SampleRequest],
    num_special: usize,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) -> Result<()> {
    let tokenize_url = Arc::new(format!("{base_url}/tokenize"));
    let detokenize_url = Arc::new(format!("{base_url}/detokenize"));
    let api_key = Arc::new(std::env::var("OPENAI_API_KEY").ok());
    let extra_headers = Arc::new(extra_headers.clone());
    let concurrency = 64;
    let sem = Arc::new(Semaphore::new(concurrency));

    let pb = Arc::new(ProgressBar::new(requests.len() as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} Verifying [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let fixed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Spawn a task per prompt
    let mut handles: Vec<tokio::task::JoinHandle<Result<(usize, String)>>> =
        Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let client = client.clone();
        let tok_url = tokenize_url.clone();
        let detok_url = detokenize_url.clone();
        let model = model.to_string();
        let prompt = req.prompt.to_string(); // Convert Arc<str> to String for mutation
        let expected_input_len = req.prompt_len + num_special;
        let api_key = api_key.clone();
        let headers = extra_headers.clone();
        let sem = sem.clone();
        let pb = pb.clone();
        let fixed_count = fixed_count.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let mut prompt = prompt;
            let max_iterations = 20;
            let mut needed_fix = false;
            // Track systematic offset (e.g., server auto-adds BOS token).
            // If the same excess appears twice consecutively, compensate.
            let mut last_excess: Option<usize> = None;

            for _iter in 0..max_iterations {
                let tokens = server_tokenize(
                    &client, &tok_url, &model, &prompt, &api_key, &headers, i,
                ).await?;

                if tokens.len() == expected_input_len {
                    if needed_fix {
                        fixed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    pb.inc(1);
                    return Ok((i, prompt));
                }

                needed_fix = true;

                // Detect systematic offset: if the server consistently returns
                // expected + N tokens (e.g., auto-prepended BOS), reduce target
                // by N so the server's re-tokenization lands on the expected count.
                let excess = tokens.len().saturating_sub(expected_input_len);
                let compensate = if excess > 0 && last_excess == Some(excess) {
                    if _iter == 1 {
                        eprintln!(
                            "Prompt {i}: server consistently adds {excess} extra token(s) \
                             (likely BOS), compensating target to {}.",
                            expected_input_len.saturating_sub(excess),
                        );
                    }
                    excess
                } else {
                    0
                };
                last_excess = if excess > 0 { Some(excess) } else { None };

                let target = expected_input_len.saturating_sub(compensate);

                let mut adjusted = tokens;
                if adjusted.is_empty() {
                    return Err(BenchError::Tokenizer(format!(
                        "Prompt {i}: server returned no tokens, cannot fix prompt length"
                    )));
                }
                if adjusted.len() < target {
                    let pad_needed = target - adjusted.len();
                    let original_len = adjusted.len();
                    for j in 0..pad_needed {
                        adjusted.push(adjusted[j % original_len]);
                    }
                } else {
                    adjusted.truncate(target);
                }

                prompt = server_detokenize(
                    &client, &detok_url, &model, &adjusted, &api_key, &headers, i,
                ).await?;
            }

            Err(BenchError::Tokenizer(format!(
                "Prompt {i}: server verification failed to converge after {max_iterations} iterations"
            )))
        }));
    }

    // Collect results and write back
    for handle in handles {
        let result = match handle.await {
            Ok(r) => r,
            Err(e) => {
                return Err(BenchError::Tokenizer(format!(
                    "Verification task panicked: {e}"
                )));
            }
        };
        let (i, prompt) = result?;
        requests[i].prompt = Arc::from(prompt);
        // Server-side count the fix loop converged on for this prompt.
        requests[i].prompt_len += num_special;
    }

    pb.finish_and_clear();

    let fc = fixed_count.load(std::sync::atomic::Ordering::Relaxed);
    if fc > 0 {
        println!("Fixed {fc} prompt(s) via server tokenize/detokenize convergence.");
    }

    Ok(())
}

/// Call the server's /tokenize endpoint and return the token ID list.
/// Retries up to 3 times on transient errors (5xx, connection errors).
async fn server_tokenize(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    api_key: &Option<String>,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
    prompt_idx: usize,
) -> Result<Vec<u64>> {
    let max_retries = 3;

    for attempt in 0..=max_retries {
        let payload = serde_json::json!({
            "model": model,
            "prompt": prompt,
        });

        let mut request = client.post(url).json(&payload);
        if let Some(headers) = extra_headers {
            for (k, v) in headers {
                request = request.header(k, v);
            }
        }
        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                if attempt < max_retries {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        500 * (attempt as u64 + 1),
                    ))
                    .await;
                    continue;
                }
                return Err(BenchError::Tokenizer(format!(
                    "Server /tokenize failed for prompt {prompt_idx} after {max_retries} retries: {e}"
                )));
            }
        };

        if resp.status().is_server_error() && attempt < max_retries {
            tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt as u64 + 1))).await;
            continue;
        }

        if resp.status().is_client_error() {
            // Server doesn't expose /tokenize (404, e.g. Dynamo) or a gateway
            // rejects it with another 4xx (LLM-d/EPP returns 400) — caller
            // should skip verification. 5xx and connection errors stay fatal.
            return Err(BenchError::TokenizeUnavailable(format!(
                "HTTP {}",
                resp.status()
            )));
        }

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server /tokenize returned HTTP {} for prompt {prompt_idx}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| {
            BenchError::Tokenizer(format!(
                "Failed to parse /tokenize response for prompt {prompt_idx}: {e}"
            ))
        })?;

        let tokens = data.get("tokens").and_then(|t| t.as_array()).ok_or_else(|| {
            BenchError::Tokenizer(format!(
                "No 'tokens' array in /tokenize response for prompt {prompt_idx}"
            ))
        })?;

        return tokens
            .iter()
            .map(|v| {
                v.as_u64().ok_or_else(|| {
                    BenchError::Tokenizer("Invalid token ID in /tokenize response".to_string())
                })
            })
            .collect();
    }

    unreachable!()
}

/// Call the server's /detokenize endpoint and return the prompt text.
/// Retries up to 3 times on transient errors (5xx, connection errors).
async fn server_detokenize(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    tokens: &[u64],
    api_key: &Option<String>,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
    prompt_idx: usize,
) -> Result<String> {
    let max_retries = 3;

    for attempt in 0..=max_retries {
        let payload = serde_json::json!({
            "model": model,
            "tokens": tokens,
        });

        let mut request = client.post(url).json(&payload);
        if let Some(headers) = extra_headers {
            for (k, v) in headers {
                request = request.header(k, v);
            }
        }
        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                if attempt < max_retries {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        500 * (attempt as u64 + 1),
                    ))
                    .await;
                    continue;
                }
                return Err(BenchError::Tokenizer(format!(
                    "Server /detokenize failed for prompt {prompt_idx} after {max_retries} retries: {e}"
                )));
            }
        };

        if resp.status().is_server_error() && attempt < max_retries {
            tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt as u64 + 1))).await;
            continue;
        }

        if resp.status().is_client_error() {
            return Err(BenchError::TokenizeUnavailable(format!(
                "HTTP {}",
                resp.status()
            )));
        }

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server /detokenize returned HTTP {} for prompt {prompt_idx}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| {
            BenchError::Tokenizer(format!(
                "Failed to parse /detokenize response for prompt {prompt_idx}: {e}"
            ))
        })?;

        return data.get("prompt").and_then(|p| p.as_str()).map(|s| s.to_string()).ok_or_else(
            || {
                BenchError::Tokenizer(format!(
                    "No 'prompt' in /detokenize response for prompt {prompt_idx}"
                ))
            },
        );
    }

    unreachable!()
}

/// Check if an error message indicates a transient connection error worth retrying.
fn is_connection_error(error: &str) -> bool {
    let patterns = [
        "connection reset",
        "Connection reset",
        "connection error",
        "broken pipe",
        "connect error",
        "timed out",
        "connection refused",
        "Connection refused",
    ];
    patterns.iter().any(|p| error.contains(p))
}

// --- Tokenizer verification cache ---

/// Build a cache key from base_url + model_id.
fn tokenizer_verify_cache_key(base_url: &str, model_id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    base_url.hash(&mut h);
    model_id.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Cache directory for tokenizer verification markers.
fn verify_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("vllm-bench")
        .join("verified_tokenizers")
}

/// Check if tokenizer was previously verified for this model+server.
fn is_tokenizer_verified(cache_key: &str) -> bool {
    verify_cache_dir().join(cache_key).exists()
}

/// Mark tokenizer as verified for this model+server.
fn mark_tokenizer_verified(cache_key: &str) {
    let dir = verify_cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(cache_key), "");
}

/// Outcome of sample verification against the server's /tokenize.
enum SampleVerifyOutcome {
    /// All sampled prompts matched their expected token counts.
    Passed,
    /// At least one sampled prompt did not match; full verify+fix is needed.
    Mismatch,
    /// The server cannot tokenize (4xx from /tokenize); verification skipped.
    /// Unlike Passed, this must NOT be cached as "verified".
    Skipped(String),
}

/// Sample-verify a small number of prompts against the server's /tokenize.
/// Each prompt is checked against its own generated length (plus the special
/// tokens the server adds), so variable-length datasets
/// (--random-range-ratio < 1.0) and shared prefixes (--random-prefix-len)
/// verify correctly.
async fn sample_verify_prompts(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    requests: &[crate::datasets::SampleRequest],
    num_special: usize,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
) -> Result<SampleVerifyOutcome> {
    let sample_size = 10.min(requests.len());
    let tokenize_url = format!("{base_url}/tokenize");
    let api_key = std::env::var("OPENAI_API_KEY").ok();

    println!("Sampling {sample_size} prompts for verification...");

    for (i, request) in requests.iter().enumerate().take(sample_size) {
        let tokens = match server_tokenize(
            client,
            &tokenize_url,
            model,
            &request.prompt,
            &api_key,
            extra_headers,
            i,
        )
        .await
        {
            Ok(t) => t,
            Err(BenchError::TokenizeUnavailable(reason)) => {
                return Ok(SampleVerifyOutcome::Skipped(reason));
            }
            Err(e) => return Err(e),
        };

        let expected = request.prompt_len + num_special;
        if tokens.len() != expected {
            println!(
                "Prompt {i}: expected {expected} tokens, server returned {}",
                tokens.len()
            );
            return Ok(SampleVerifyOutcome::Mismatch);
        }
    }

    Ok(SampleVerifyOutcome::Passed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(
        drafts: u64,
        draft_tokens: u64,
        accepted_tokens: u64,
        per_pos: &[(u64, u64)],
    ) -> SpecDecodeMetrics {
        SpecDecodeMetrics {
            num_drafts: drafts,
            num_draft_tokens: draft_tokens,
            num_accepted_tokens: accepted_tokens,
            accepted_per_pos: per_pos.iter().copied().collect(),
        }
    }

    #[test]
    fn test_compute_spec_decode_stats_basic() {
        let before = make_metrics(100, 300, 200, &[(0, 90), (1, 70), (2, 50)]);
        let after = make_metrics(200, 600, 440, &[(0, 180), (1, 150), (2, 120)]);

        let stats = compute_spec_decode_stats(&before, &after).unwrap();

        assert_eq!(stats.num_drafts, 100);
        assert_eq!(stats.draft_tokens, 300);
        assert_eq!(stats.accepted_tokens, 240);
        assert!((stats.acceptance_rate - 80.0).abs() < 0.01);
        assert!((stats.acceptance_length - 3.4).abs() < 0.01);
        assert_eq!(stats.per_position_acceptance_rates.len(), 3);
        assert!((stats.per_position_acceptance_rates[0] - 0.9).abs() < 0.01);
        assert!((stats.per_position_acceptance_rates[1] - 0.8).abs() < 0.01);
        assert!((stats.per_position_acceptance_rates[2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_compute_spec_decode_stats_zero_draft_tokens_returns_none() {
        let before = make_metrics(0, 100, 80, &[]);
        let after = make_metrics(0, 100, 80, &[]);
        assert!(compute_spec_decode_stats(&before, &after).is_none());
    }

    #[test]
    fn test_compute_spec_decode_stats_zero_drafts() {
        let before = make_metrics(0, 0, 0, &[]);
        let after = make_metrics(0, 100, 80, &[]);

        let stats = compute_spec_decode_stats(&before, &after).unwrap();
        assert_eq!(stats.num_drafts, 0);
        assert_eq!(stats.draft_tokens, 100);
        assert_eq!(stats.accepted_tokens, 80);
        assert!((stats.acceptance_rate - 80.0).abs() < 0.01);
        assert!((stats.acceptance_length - 0.0).abs() < 0.01);
        assert!(stats.per_position_acceptance_rates.is_empty());
    }

    #[test]
    fn test_compute_spec_decode_stats_new_positions_in_after() {
        let before = make_metrics(50, 150, 100, &[(0, 40)]);
        let after = make_metrics(150, 450, 320, &[(0, 130), (1, 80)]);

        let stats = compute_spec_decode_stats(&before, &after).unwrap();
        assert_eq!(stats.num_drafts, 100);
        assert_eq!(stats.per_position_acceptance_rates.len(), 2);
        assert!((stats.per_position_acceptance_rates[0] - 0.9).abs() < 0.01);
        assert!((stats.per_position_acceptance_rates[1] - 0.8).abs() < 0.01);
    }

    fn lora_names(v: &[Arc<str>]) -> Vec<&str> {
        v.iter().map(|s| s.as_ref()).collect()
    }

    #[test]
    fn test_assign_lora_modules_none_when_unset() {
        assert!(assign_lora_modules(&None, LoraAssignment::Random, 5, 0).is_none());
    }

    #[test]
    fn test_assign_lora_modules_round_robin_cycles() {
        let modules = Some(vec![Arc::<str>::from("a"), Arc::from("b"), Arc::from("c")]);
        let out = assign_lora_modules(&modules, LoraAssignment::RoundRobin, 7, 42).unwrap();
        assert_eq!(lora_names(&out), vec!["a", "b", "c", "a", "b", "c", "a"]);
    }

    #[test]
    fn test_assign_lora_modules_random_is_seed_reproducible() {
        let modules = Some(vec![Arc::<str>::from("x"), Arc::from("y"), Arc::from("z")]);
        let a = assign_lora_modules(&modules, LoraAssignment::Random, 100, 7).unwrap();
        let b = assign_lora_modules(&modules, LoraAssignment::Random, 100, 7).unwrap();
        assert_eq!(lora_names(&a), lora_names(&b));

        // Different seed should (almost certainly) produce a different sequence.
        let c = assign_lora_modules(&modules, LoraAssignment::Random, 100, 8).unwrap();
        assert_ne!(lora_names(&a), lora_names(&c));
    }

    #[test]
    fn test_assign_lora_modules_random_covers_all_modules() {
        let modules = Some(vec![
            Arc::<str>::from("a"),
            Arc::from("b"),
            Arc::from("c"),
            Arc::from("d"),
        ]);
        let out = assign_lora_modules(&modules, LoraAssignment::Random, 1000, 0).unwrap();
        let unique: std::collections::HashSet<&str> = out.iter().map(|s| s.as_ref()).collect();
        assert_eq!(
            unique.len(),
            4,
            "all 4 adapters should be sampled in 1000 draws"
        );
    }

    #[test]
    fn test_assign_lora_modules_single_adapter() {
        let modules = Some(vec![Arc::<str>::from("only")]);
        for assignment in [LoraAssignment::Random, LoraAssignment::RoundRobin] {
            let out = assign_lora_modules(&modules, assignment, 5, 0).unwrap();
            assert!(out.iter().all(|s| s.as_ref() == "only"));
        }
    }
}
