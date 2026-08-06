// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use thiserror_ext::AsReport as _;
use tokio::sync::Semaphore;

use crate::backends::{Backend, RequestFuncInput, RequestFuncOutput, get_backend};
use crate::benchmark::{
    assign_lora_modules, compute_spec_decode_stats, fetch_spec_decode_metrics, pre_resolve_dns,
    profile_on_batch_threshold, start_profiler_immediate, stop_profiler_immediate,
};
use crate::cli::DatasetName;
use crate::config::BenchConfig;
use crate::datasets::MultiTurnConversation;
use crate::error::{BenchError, Result};
use crate::metrics::calculator::calculate_multi_turn_metrics;
use crate::output::console::print_multi_turn_results;
use crate::output::json::{
    append_result, build_multi_turn_result_json, compute_result_filename, save_result,
};

/// Output from a single turn within a conversation.
#[derive(Debug, Clone)]
pub struct TurnOutput {
    pub turn_index: usize,
    pub request_output: RequestFuncOutput,
    pub cumulative_input_tokens: usize,
}

/// Output from an entire conversation.
#[derive(Debug, Clone)]
pub struct ConversationOutput {
    pub conversation_id: String,
    pub turns: Vec<TurnOutput>,
    pub total_duration_ms: f64,
    pub all_success: bool,
}

/// Run the multi-turn conversation benchmark.
pub async fn run_multi_turn_benchmark(config: &BenchConfig) -> Result<serde_json::Value> {
    let _ = get_backend(config.backend)?;

    // Build HTTP client
    let concurrency = config
        .multi_turn_concurrency
        .unwrap_or(config.max_concurrency.unwrap_or(config.num_prompts));

    let mut client_builder = reqwest::Client::builder()
        .pool_max_idle_per_host(concurrency.max(256))
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
        tracing::info!(base_url = %config.base_url, "fetching first model from server");
        let (name, id) = get_first_model(&config.base_url, &client, &config.extra_headers).await?;
        tracing::info!(
            model_name = name,
            model_id = id,
            "selected first model from server"
        );
        (id, Some(name))
    };

    // Load tokenizer
    let tokenizer = if config.skip_tokenizer_init {
        None
    } else {
        let tid = config.tokenizer_id.as_deref().unwrap_or(&model_id);
        tracing::info!(tokenizer = tid, "loading tokenizer");
        let server_info = Some((config.base_url.as_str(), model_id.as_str()));
        let t =
            crate::tokenizer::load_tokenizer(tid, config.trust_remote_code, server_info).await?;
        Some(t)
    };

    // Generate/load conversations
    tracing::info!(
        dataset = ?config.dataset_name,
        conversations = config.num_prompts,
        "generating multi-turn conversations"
    );
    let gen_start = Instant::now();

    let mut conversations = match config.dataset_name {
        DatasetName::Random => {
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| BenchError::Config("Random dataset requires a tokenizer".into()))?;
            let prefix_sharing_config = if config.multi_turn_prefix_global_ratio > 0.0
                || config.multi_turn_prefix_conversation_ratio > 0.0
            {
                Some(crate::datasets::multi_turn::PrefixSharingConfig {
                    global_ratio: config.multi_turn_prefix_global_ratio,
                    conversation_ratio: config.multi_turn_prefix_conversation_ratio,
                })
            } else {
                None
            };
            let random_cfg = crate::datasets::multi_turn::MultiTurnRandomConfig {
                num_conversations: config.num_prompts,
                min_turns: config.multi_turn_min_turns,
                max_turns: config.multi_turn_max_turns,
                prefix_len: config.random_prefix_len,
                input_len: config.random_input_len,
                per_turn_input_len: config.per_turn_input_len,
                output_len: config.random_output_len,
                seed: config.seed,
                request_id_prefix: config.request_id_prefix.clone(),
                prefix_sharing_config,
            };
            crate::datasets::multi_turn::generate_multi_turn_random(tok, &random_cfg)?
        }
        DatasetName::ShareGpt => {
            let tok = tokenizer.as_ref().ok_or_else(|| {
                BenchError::Config("ShareGPT dataset requires a tokenizer".into())
            })?;
            let downloaded;
            let path = match config.dataset_path.as_deref() {
                Some(p) => p,
                None => {
                    downloaded = crate::datasets::sharegpt::download_sharegpt_dataset().await?;
                    downloaded.as_str()
                }
            };
            crate::datasets::multi_turn::load_sharegpt_multi_turn(
                tok,
                path,
                config.num_prompts,
                config.sharegpt_output_len,
                config.sharegpt_multi_turn_max_turns,
                config.seed,
                &config.request_id_prefix,
            )?
        }
        DatasetName::RandomMm => {
            return Err(BenchError::Config(
                "Random-MM multi-turn is not yet supported. Use 'random' or 'sharegpt' with --multi-turn.".into(),
            ));
        }
        DatasetName::Sonnet => {
            return Err(BenchError::Config(
                "Sonnet multi-turn is not yet supported. Use 'random' or 'sharegpt' with --multi-turn.".into(),
            ));
        }
        DatasetName::SpeedBench => {
            return Err(BenchError::Config(
                "SPEED-Bench multi-turn is not yet supported. Use 'random' or 'sharegpt' with --multi-turn.".into(),
            ));
        }
        DatasetName::Custom | DatasetName::PrefixRepetition | DatasetName::RandomRerank => {
            return Err(BenchError::Config(
                "This dataset does not support multi-turn. Use 'random' or 'sharegpt' with --multi-turn.".into(),
            ));
        }
        DatasetName::Hf => {
            return Err(BenchError::Config(
                "HF dataset multi-turn is not yet supported. Use 'random' or 'sharegpt' with --multi-turn.".into(),
            ));
        }
    };

    let no_history = config.multi_turn_prefix_global_ratio > 0.0
        || config.multi_turn_prefix_conversation_ratio > 0.0;

    if let Some(max_model_len) = config.max_model_len {
        let (filtered_conversations, filtered_turns) =
            filter_turns_by_max_model_len(&mut conversations, max_model_len, no_history);
        if filtered_turns > 0 || filtered_conversations > 0 {
            tracing::info!(
                filtered_turns,
                filtered_conversations,
                max_model_len,
                "filtered conversations above maximum model length"
            );
        }
        if conversations.is_empty() {
            return Err(BenchError::Config(
                "No conversations remain after applying --max-model-len".into(),
            ));
        }
    }

    let gen_elapsed = gen_start.elapsed();
    let total_turns: usize = conversations.iter().map(|c| c.turns.len()).sum();
    tracing::info!(
        conversations = conversations.len(),
        total_turns,
        elapsed_seconds = gen_elapsed.as_secs_f64(),
        "generated multi-turn conversations"
    );

    // Log prefix sharing info
    if no_history {
        let num_special = tokenizer.as_ref().map(|t| t.num_special_tokens_to_add()).unwrap_or(0);
        let real_input_len = config.random_input_len.saturating_sub(num_special);
        let global_tokens =
            (real_input_len as f64 * config.multi_turn_prefix_global_ratio).floor() as usize;
        let conv_tokens =
            (real_input_len as f64 * config.multi_turn_prefix_conversation_ratio).floor() as usize;
        let unique_tokens = real_input_len.saturating_sub(global_tokens + conv_tokens);
        tracing::info!(
            global_ratio = config.multi_turn_prefix_global_ratio,
            global_tokens,
            conversation_ratio = config.multi_turn_prefix_conversation_ratio,
            conversation_tokens = conv_tokens,
            unique_tokens,
            history_accumulation = false,
            "configured multi-turn prefix sharing"
        );
    }

    if config.dry_run {
        let total_user_tokens: usize = conversations
            .iter()
            .flat_map(|c| c.turns.iter())
            .map(|t| t.user_message_len)
            .sum();
        println!("Dry run stats:");
        println!("  Total conversations: {}", conversations.len());
        println!("  Total turns: {total_turns}");
        println!("  Total user message tokens: {total_user_tokens}");
        return Ok(serde_json::json!({"dry_run": true, "mode": "multi_turn"}));
    }

    // Ready check with a simple single request
    if config.ready_check_timeout_sec > 0 {
        let first_turn = &conversations[0].turns[0];
        let test_input = RequestFuncInput {
            prompt: first_turn.user_message.clone(),
            api_url: config.api_url.clone(),
            prompt_len: first_turn.user_message_len,
            output_len: first_turn.expected_output_len,
            model: model_id.clone(),
            model_name: model_name.clone(),
            logprobs: config.logprobs,
            extra_headers: config.extra_headers.clone(),
            extra_body: config.extra_body.clone(),
            ignore_eos: config.ignore_eos,
            request_id: None,
            ..Default::default()
        };

        tracing::info!("starting initial single-prompt test run");
        let test_output = crate::ready_checker::wait_for_endpoint(
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
        tracing::info!("initial single-prompt test run completed");
    }

    // For random datasets in multi-turn mode, auto-set min_tokens to enforce
    // output length without using ignore_eos (which causes unbounded context growth).
    // min_tokens + max_completion_tokens together control output length precisely.
    let extra_body = if config.dataset_name == DatasetName::Random && !config.ignore_eos {
        let mut body = config.extra_body.clone().unwrap_or_else(|| serde_json::json!({}));
        if let serde_json::Value::Object(ref mut map) = body
            && !map.contains_key("min_tokens")
        {
            map.insert(
                "min_tokens".to_string(),
                serde_json::json!(config.random_output_len),
            );
            tracing::info!(
                min_tokens = config.random_output_len,
                dataset = "random",
                "set minimum output tokens for multi-turn dataset"
            );
        }
        Some(body)
    } else {
        config.extra_body.clone()
    };

    // Fetch speculative decoding metrics before benchmark
    let spec_decode_before =
        fetch_spec_decode_metrics(&config.base_url, &client, &config.extra_headers).await;
    if spec_decode_before.is_some() {
        tracing::info!("detected speculative decoding; collecting metrics");
    }

    // Start profiler if requested (immediate mode — no batch threshold)
    if config.profile && config.profile_batch_threshold.is_none() {
        start_profiler_immediate(&client, &config.base_url, &config.extra_headers).await;
    }

    // Threshold-based profiling: spawn background task that polls /metrics
    // and triggers start/stop profile when batch size is reached.
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

    // Main benchmark
    tracing::info!(
        conversations = conversations.len(),
        total_turns,
        concurrency,
        inter_turn_delay_ms = config.multi_turn_delay_ms,
        "starting multi-turn benchmark"
    );

    let max_turn_count = conversations.iter().map(|c| c.turns.len()).max().unwrap_or(0);

    // Progress bar counts total turns
    let pb = if config.disable_tqdm {
        None
    } else {
        let bar = ProgressBar::new(total_turns as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} turns ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        Some(bar)
    };

    let benchmark_start = Instant::now();

    // Per-conversation LoRA assignment (sticky: all turns of a conversation
    // share the same adapter). None when --lora-modules not set.
    let lora_assignments = assign_lora_modules(
        &config.lora_modules,
        config.lora_assignment,
        conversations.len(),
        config.seed,
    );
    if let Some(modules) = config.lora_modules.as_ref() {
        let names: Vec<&str> = modules.iter().map(|s| s.as_ref()).collect();
        tracing::info!(
            adapters = modules.len(),
            names = ?names,
            assignment = ?config.lora_assignment,
            scope = "conversation",
            "assigned LoRA adapters"
        );
    }

    // Shared state
    let backend = get_backend(config.backend)?;
    let api_url = Arc::new(config.api_url.clone());
    let model = Arc::new(model_id.clone());
    let model_name_arc = Arc::new(model_name.clone());
    let extra_body = Arc::new(extra_body);
    let base_extra_headers = Arc::new(config.extra_headers.clone());
    let ignore_eos = config.ignore_eos;
    let logprobs = config.logprobs;
    let delay_ms = config.multi_turn_delay_ms;

    // Semaphore controls max in-flight requests (not conversations).
    // Each conversation acquires the permit only during the HTTP request,
    // releasing it before the inter-turn delay so other conversations can
    // immediately fill the slot.
    let semaphore = Arc::new(Semaphore::new(concurrency));

    // Spawn one task per conversation (all at once)
    let mut handles = Vec::with_capacity(conversations.len());
    for (i, conv) in conversations.into_iter().enumerate() {
        let client = client.clone();
        let backend = backend.clone();
        let api_url = api_url.clone();
        let model = model.clone();
        let model_name = model_name_arc.clone();
        let extra_body = extra_body.clone();
        let base_extra_headers = base_extra_headers.clone();
        let pb = pb.clone();
        let bench_start = benchmark_start;
        let semaphore = semaphore.clone();
        let lora_name = lora_assignments.as_ref().map(|v| v[i].clone());

        handles.push(tokio::spawn(async move {
            run_conversation(
                &conv,
                &backend,
                &client,
                &api_url,
                &model,
                &model_name,
                lora_name.as_deref(),
                &extra_body,
                &base_extra_headers,
                ignore_eos,
                logprobs,
                delay_ms,
                no_history,
                bench_start,
                pb.as_ref(),
                &semaphore,
            )
            .await
        }));
    }

    // Collect all conversation outputs
    let mut all_outputs: Vec<ConversationOutput> = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(output) => all_outputs.push(output),
            Err(e) => {
                tracing::error!(error = %e.as_report(), "conversation task panicked");
            }
        }
    }

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    let benchmark_duration = benchmark_start.elapsed().as_secs_f64();

    // Stop profiler if requested (immediate mode — no batch threshold)
    if config.profile && config.profile_batch_threshold.is_none() {
        stop_profiler_immediate(&client, &config.base_url, &config.extra_headers).await;
    }

    // Signal the threshold-based profile task that the benchmark is done, then wait
    if let Some((cancel_tx, task)) = profile_task {
        let _ = cancel_tx.send(());
        if let Err(e) = task.await {
            tracing::error!(error = %e.as_report(), "profiler background task failed");
        }
    }

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

    // Calculate metrics
    let mt_metrics = calculate_multi_turn_metrics(
        &all_outputs,
        benchmark_duration,
        &config.selected_percentiles,
        &config.goodput,
        max_turn_count,
    );

    // Print console output
    print_multi_turn_results(
        &mt_metrics,
        benchmark_duration,
        config,
        spec_decode_stats.as_ref(),
    );

    // Build result JSON
    let date_iso = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let dt_filename = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let result_json = build_multi_turn_result_json(
        config,
        &mt_metrics,
        &all_outputs,
        benchmark_duration,
        &date_iso,
        spec_decode_stats.as_ref(),
    );

    // Save if requested
    if config.save_result || config.append_result {
        let model_for_filename = config.model.as_deref().unwrap_or(&model_id);
        let file_name = compute_result_filename(config, model_for_filename, &dt_filename);

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

fn filter_turns_by_max_model_len(
    conversations: &mut Vec<MultiTurnConversation>,
    max_model_len: usize,
    no_history: bool,
) -> (usize, usize) {
    let before_conversations = conversations.len();
    let before_turns: usize = conversations.iter().map(|c| c.turns.len()).sum();

    for conversation in conversations.iter_mut() {
        if no_history {
            conversation.turns.retain(|turn| {
                turn.user_message_len.saturating_add(turn.expected_output_len) <= max_model_len
            });
            continue;
        }

        let keep_turns = valid_prefix_len_for_max_model_len(conversation, max_model_len);
        conversation.turns.truncate(keep_turns);
    }

    conversations.retain(|conversation| conversation.turns.len() >= 2);

    let after_conversations = conversations.len();
    let after_turns: usize = conversations.iter().map(|c| c.turns.len()).sum();

    (
        before_conversations - after_conversations,
        before_turns - after_turns,
    )
}

fn valid_prefix_len_for_max_model_len(
    conversation: &MultiTurnConversation,
    max_model_len: usize,
) -> usize {
    let mut cumulative_input_tokens = 0usize;
    let mut keep_turns = 0usize;

    for turn in &conversation.turns {
        cumulative_input_tokens = cumulative_input_tokens.saturating_add(turn.user_message_len);

        if cumulative_input_tokens.saturating_add(turn.expected_output_len) > max_model_len {
            break;
        }

        cumulative_input_tokens = cumulative_input_tokens.saturating_add(turn.expected_output_len);
        keep_turns += 1;
    }

    keep_turns
}

/// Run a single conversation: sequential turns, building up message history.
///
/// `lora_name`, when set, replaces both `model` and `model_name` in every
/// turn's request payload — sticky for the whole conversation so prefix-cache
/// reuse across turns isn't broken by mid-conversation adapter switches.
async fn run_conversation(
    conversation: &MultiTurnConversation,
    backend: &Backend,
    client: &reqwest::Client,
    api_url: &str,
    model: &str,
    model_name: &Option<String>,
    lora_name: Option<&str>,
    extra_body: &Option<serde_json::Value>,
    base_extra_headers: &Option<HashMap<String, String>>,
    ignore_eos: bool,
    logprobs: Option<usize>,
    delay_ms: u64,
    no_history: bool,
    bench_start: Instant,
    pb: Option<&ProgressBar>,
    semaphore: &Semaphore,
) -> ConversationOutput {
    let conv_start = Instant::now();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    let mut cumulative_tokens: usize = 0;
    let mut turn_outputs: Vec<TurnOutput> = Vec::new();
    let mut all_success = true;

    // Router affinity: all turns share same X-Session-ID
    let mut extra_headers = base_extra_headers.clone().unwrap_or_default();
    extra_headers.insert(
        "X-Session-ID".to_string(),
        conversation.conversation_id.clone(),
    );

    for (turn_idx, turn) in conversation.turns.iter().enumerate() {
        if no_history {
            // Prefix sharing mode: reset messages each turn, send only this turn's message
            messages.clear();
            messages.push(serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": &*turn.user_message}]
            }));
            cumulative_tokens = turn.user_message_len;
        } else {
            // Normal mode: accumulate history
            messages.push(serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": &*turn.user_message}]
            }));
            cumulative_tokens += turn.user_message_len;
        }

        // Set min_tokens per-request so each turn's output length matches
        // the dataset's expected length. Skip if ignore_eos is set (unbounded
        // generation) or the user already provided min_tokens via --extra-body.
        let turn_extra_body = if !ignore_eos {
            let mut body = extra_body.clone().unwrap_or_else(|| serde_json::json!({}));
            if let serde_json::Value::Object(ref mut map) = body
                && !map.contains_key("min_tokens")
            {
                map.insert(
                    "min_tokens".to_string(),
                    serde_json::json!(turn.expected_output_len),
                );
            }
            Some(body)
        } else {
            extra_body.clone()
        };

        let (req_model, req_model_name) = match lora_name {
            Some(name) => (name.to_string(), Some(name.to_string())),
            None => (model.to_string(), model_name.clone()),
        };

        let input = RequestFuncInput {
            prompt: turn.user_message.clone(),
            api_url: api_url.to_string(),
            prompt_len: cumulative_tokens,
            output_len: turn.expected_output_len,
            model: req_model,
            model_name: req_model_name,
            logprobs,
            extra_headers: Some(extra_headers.clone()),
            extra_body: turn_extra_body,
            ignore_eos,
            request_id: Some(format!("{}-turn{}", conversation.conversation_id, turn_idx)),
            messages: Some(serde_json::json!(messages)),
            ..Default::default()
        };

        // Acquire semaphore permit before sending the request.
        // Released after the response so the slot is free during inter-turn delay.
        let _permit = match semaphore.acquire().await {
            Ok(p) => p,
            Err(_) => {
                all_success = false;
                break;
            }
        };
        let request_instant = Instant::now();
        let result = backend.send_request(&input, client).await;
        drop(_permit);

        let output = match result {
            Ok(mut o) => {
                o.start_time = request_instant.duration_since(bench_start).as_secs_f64();
                o
            }
            Err(e) => RequestFuncOutput {
                success: false,
                error: e.to_string(),
                prompt_len: cumulative_tokens,
                start_time: request_instant.duration_since(bench_start).as_secs_f64(),
                ..Default::default()
            },
        };

        if let Some(pb) = pb {
            pb.inc(1);
        }

        if output.success {
            if !no_history {
                // Add assistant response to history for next turn
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": &output.generated_text
                }));
                cumulative_tokens += output.output_tokens;
            }

            turn_outputs.push(TurnOutput {
                turn_index: turn_idx,
                request_output: output,
                cumulative_input_tokens: cumulative_tokens,
            });
        } else {
            all_success = false;
            turn_outputs.push(TurnOutput {
                turn_index: turn_idx,
                request_output: output,
                cumulative_input_tokens: cumulative_tokens,
            });
            break; // Conversation stops on failure
        }

        // Inter-turn delay (semaphore NOT held — slot is free for others)
        if delay_ms > 0 && turn_idx + 1 < conversation.turns.len() {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
    }

    let total_duration_ms = conv_start.elapsed().as_secs_f64() * 1000.0;

    ConversationOutput {
        conversation_id: conversation.conversation_id.clone(),
        turns: turn_outputs,
        total_duration_ms,
        all_success,
    }
}

/// Fetch the first model from the server's /v1/models endpoint.
async fn get_first_model(
    base_url: &str,
    client: &reqwest::Client,
    extra_headers: &Option<HashMap<String, String>>,
) -> Result<(String, String)> {
    let url = format!("{base_url}/v1/models");
    let mut request = client.get(&url);
    if let Some(headers) = extra_headers {
        for (k, v) in headers {
            request = request.header(k, v);
        }
    }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{filter_turns_by_max_model_len, valid_prefix_len_for_max_model_len};
    use crate::datasets::{ConversationTurn, MultiTurnConversation};

    fn conversation(turns: &[(usize, usize)]) -> MultiTurnConversation {
        MultiTurnConversation {
            conversation_id: "conv-0".to_string(),
            turns: turns
                .iter()
                .map(|(user_message_len, expected_output_len)| ConversationTurn {
                    user_message: Arc::from("hello"),
                    user_message_len: *user_message_len,
                    expected_output_len: *expected_output_len,
                })
                .collect(),
        }
    }

    #[test]
    fn test_valid_prefix_len_for_max_model_len_with_history() {
        let conv = conversation(&[(40, 10), (45, 10)]);

        assert_eq!(valid_prefix_len_for_max_model_len(&conv, 105), 2);
        assert_eq!(valid_prefix_len_for_max_model_len(&conv, 104), 1);
    }

    #[test]
    fn test_filter_turns_by_max_model_len_truncates_history_conversations() {
        let mut conversations = vec![
            conversation(&[(40, 10), (45, 10), (1, 1)]),
            conversation(&[(100, 10), (1, 1)]),
        ];

        let (filtered_conversations, filtered_turns) =
            filter_turns_by_max_model_len(&mut conversations, 105, false);

        assert_eq!(filtered_conversations, 1);
        assert_eq!(filtered_turns, 3);
        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].turns.len(), 2);
    }

    #[test]
    fn test_filter_turns_by_max_model_len_retains_independent_no_history_turns() {
        let mut conversations = vec![conversation(&[(40, 10), (95, 10), (1, 1)])];

        let (filtered_conversations, filtered_turns) =
            filter_turns_by_max_model_len(&mut conversations, 104, true);

        assert_eq!(filtered_conversations, 0);
        assert_eq!(filtered_turns, 1);
        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].turns.len(), 2);
    }
}
