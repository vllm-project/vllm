// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Multimodal preprocessing benchmark against a live vLLM engine
//! (`vllm-bench mm-processor`).
//!
//! Mirrors `vllm bench mm-processor` (`vllm/benchmarks/mm_processor.py`):
//! spawn a managed headless Python engine with the model loaded, generate a
//! synthetic multimodal dataset with the random-mm generator, and submit every
//! request through the same chat pipeline used by the serving frontend
//! (render -> markers -> media fetch -> processor -> engine encode/decode).
//! Per-stage preprocessing latency is collected by the `TimingContext` /
//! `MultiModalTimingRegistry` hooks and drained via
//! `ChatRequestProcessor::mm_timing_stats`.
//!
//! Per-request end-to-end latency spans submission to final token, including
//! queueing and preprocessing (Python's fallback E2EL semantics).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Args;
use futures::future::join_all;
use serde::Serialize;
use tokio::sync::Semaphore;

use vllm_chat::{
    ChatContent, ChatContentPart, ChatLlm, ChatMessage, ChatRequest, LoadModelBackendsOptions,
    load_model_backends,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::Llm;
use vllm_managed_engine::cli::ManagedEngineArgs;
use vllm_managed_engine::{ManagedEngineConfig, ManagedEngineHandle};
use vllm_text::TextLlm;

use crate::config::RangeRatio;
use crate::datasets::SampleRequest;
use crate::datasets::random_mm::{
    MmBucketKey, MmLimitPerPrompt, generate_random_mm_dataset, parse_bucket_config,
    parse_limit_mm_per_prompt,
};
use crate::error::BenchError;
use crate::metrics::calculator::{mean, median_sorted, percentile_sorted, sort_clone, std_dev};
use crate::tokenizer::{TokenizerKind, load_tokenizer};

/// Scale factor to convert the seconds recorded by `TimingContext` into the
/// milliseconds reported by the benchmark (matching Python's `unit="ms"`).
const SEC_TO_MS: f64 = 1000.0;

/// Preferred column order for the printed table; any additional stages are
/// appended in alphabetical order.
const STAGE_ORDER: &[&str] = &[
    "preprocessor_total_ms",
    "media_fetch_ms",
    "preprocess_image_ms",
    "preprocess_video_ms",
    "preprocess_audio_ms",
    "prompt_expansion_ms",
];

/// Maximum time to wait for the managed engines to register with the
/// frontend transport (matches the Rust frontend default).
const ENGINE_READY_TIMEOUT: Duration = Duration::from_secs(600);

/// Maximum time to wait for the managed engine to drain on shutdown.
const ENGINE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Args, Debug, Clone)]
pub struct MmProcessorArgs {
    /// Model to serve and benchmark (Hugging Face id or local path).
    #[arg(long)]
    pub model: String,

    /// Number of prompts to process, excluding warmups.
    #[arg(long, default_value_t = 10)]
    pub num_prompts: usize,

    /// Number of warmup prompts processed and discarded before timing.
    #[arg(long, default_value_t = 1)]
    pub num_warmups: usize,

    /// Comma-separated percentiles to report, e.g. "50,90,99".
    #[arg(long, default_value = "99")]
    pub metric_percentiles: String,

    /// Path to write the aggregate stats as JSON.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Text input length per prompt (before range scaling).
    #[arg(long, default_value_t = 1024)]
    pub random_input_len: usize,

    /// Expected output length per prompt.
    #[arg(long, default_value_t = 128)]
    pub random_output_len: usize,

    /// Prefix token length per prompt.
    #[arg(long, default_value_t = 0)]
    pub random_prefix_len: usize,

    /// Input/output length range ratio: a float or `{"input": i, "output": o}`.
    #[arg(long, default_value = "0.0")]
    pub random_range_ratio: String,

    /// Base number of multimodal items per request.
    #[arg(long, default_value_t = 1)]
    pub random_mm_base_items_per_request: usize,

    /// Range ratio (in [0, 1]) for the number of mm items per request.
    #[arg(long, default_value_t = 0.0)]
    pub random_mm_num_mm_items_range_ratio: f64,

    /// Per-modality item caps, e.g. `{"image": 255, "video": 1}`.
    #[arg(long, default_value = "{\"image\": 255, \"video\": 1}")]
    pub random_mm_limit_mm_per_prompt: String,

    /// Bucket config, e.g. `{(256,256,1): 0.5, (720,1280,1): 0.5}`.
    #[arg(long, default_value = "{(256,256,1): 0.5, (720,1280,1): 0.5}")]
    pub random_mm_bucket_config: String,

    /// Seed for dataset generation.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Request-id prefix for per-request timing keys.
    #[arg(long, default_value = "mm-proc-")]
    pub request_id_prefix: String,

    /// Maximum number of requests submitted concurrently. Defaults to 1 to
    /// match Python's serial `LLMEngine` driver for a like-for-like
    /// preprocessing comparison.
    #[arg(long, default_value_t = 1)]
    pub max_concurrency: usize,

    /// Trust remote code for the tokenizer and the managed engine.
    #[arg(long, default_value_t = false)]
    pub trust_remote_code: bool,

    /// Optional chat-template override (inline template or path to a template
    /// file), mirroring `vllm serve --chat-template`. When set, it bypasses the
    /// model's `tokenizer_config.json` chat template.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// Managed Python headless-engine options. The benchmark spawns one
    /// engine deployment with the model loaded and drives it over the same
    /// transport as the Rust serving frontend.
    #[command(flatten)]
    pub engine: ManagedEngineArgs,
}

/// Entry point for `vllm-bench mm-processor`.
pub async fn run_mm_processor(args: MmProcessorArgs) -> Result<()> {
    let range_ratio =
        RangeRatio::parse(&args.random_range_ratio).context("invalid --random-range-ratio")?;
    let limit: MmLimitPerPrompt = parse_limit_mm_per_prompt(&args.random_mm_limit_mm_per_prompt)
        .context("invalid --random-mm-limit-mm-per-prompt")?;
    let buckets: Vec<(MmBucketKey, f64)> = parse_bucket_config(&args.random_mm_bucket_config)
        .context("invalid --random-mm-bucket-config")?;
    let percentiles = parse_percentiles(&args.metric_percentiles)?;
    let seed = args.seed.unwrap_or(0);

    if args.engine.data_parallel_size_local == Some(0) {
        anyhow::bail!(
            "`--data-parallel-size-local 0` is not supported; the benchmark requires \
             a local managed engine"
        );
    }

    // Offline tokenizer: local Hugging Face tokenizer or built-in tiktoken.
    let tokenizer = load_tokenizer(&args.model, args.trust_remote_code, None)
        .await
        .context("failed to load tokenizer")?;

    // Spawn the managed headless Python engine (real weights, real encoder).
    let handshake_port = args
        .engine
        .resolve_handshake_port()
        .context("failed to allocate handshake port")?;
    let mut python_args = args.engine.python_args.clone();
    if args.trust_remote_code {
        python_args.push("--trust-remote-code".to_string());
    }
    let engine_config = ManagedEngineConfig {
        python: args.engine.python.clone(),
        model: args.model.clone(),
        handshake_host: args.engine.handshake_host.clone(),
        handshake_port,
        data_parallel_size: args.engine.data_parallel_size,
        python_args,
    };
    eprintln!(
        "Spawning managed headless engine: python={} model={} data_parallel_size={}",
        engine_config.python, engine_config.model, engine_config.data_parallel_size
    );
    let engine = ManagedEngineHandle::spawn(engine_config)
        .await
        .context("failed to start managed Python headless engine")?;

    // Frontend backends: model configs + multimodal processor metadata.
    let backends = load_model_backends(
        &args.model,
        LoadModelBackendsOptions {
            chat_template: args.chat_template.clone(),
            ..LoadModelBackendsOptions::default()
        },
    )
    .await
    .context("failed to load model backends")?;

    if backends.chat_backend.multimodal_model_info().is_none() {
        anyhow::bail!(
            "model `{}` has no multimodal processor registered",
            args.model
        );
    }

    // Connect to the engine over the same transport the serving frontend uses.
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: format!("tcp://{}:{}", args.engine.handshake_host, handshake_port),
            advertised_host: args.engine.handshake_host.clone(),
            engine_count: args
                .engine
                .data_parallel_size_local
                .unwrap_or(args.engine.data_parallel_size),
            ready_timeout: ENGINE_READY_TIMEOUT,
            local_input_address: None,
            local_output_address: None,
        },
        coordinator_mode: None,
        model_name: args.model.clone(),
        client_index: 0,
    })
    .await
    .context("failed to connect to engine core")?;

    let chat = ChatLlm::new(
        TextLlm::new(Llm::new(client), backends.text_backend),
        backends.chat_backend,
    )
    .with_mm_processor_stats(true);

    let semaphore = Arc::new(Semaphore::new(args.max_concurrency));

    // Warmup: generate a separate dataset, process it, then drain and discard
    // the collected timings so cold caches do not skew the results.
    if args.num_warmups > 0 {
        let warmup = generate_dataset(
            &tokenizer,
            args.num_warmups,
            seed.wrapping_add(1),
            &args,
            range_ratio,
            &limit,
            &buckets,
        )?;
        eprintln!("Processing {} warmup requests...", warmup.len());
        run_batch(&chat, &warmup, &semaphore).await;
        let _ = chat.mm_timing_stats();
    }

    let samples = generate_dataset(
        &tokenizer,
        args.num_prompts,
        seed,
        &args,
        range_ratio,
        &limit,
        &buckets,
    )?;
    eprintln!("Processing {} requests...", samples.len());
    let start_time = Instant::now();
    let outcomes = run_batch(&chat, &samples, &semaphore).await;
    let total_time = start_time.elapsed();

    let completed = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
    let failed = outcomes.len() - completed;
    let e2el_times: Vec<f64> = outcomes
        .iter()
        .filter_map(|outcome| outcome.as_ref().ok().map(|elapsed| elapsed.as_secs_f64()))
        .collect();

    let per_request = chat.mm_timing_stats();
    let stats = aggregate_stats(per_request, &percentiles);

    eprintln!(
        "Processed {} requests (+{} warmup) for model `{}`",
        samples.len(),
        args.num_warmups,
        args.model
    );
    print_report(&stats, &percentiles);
    print_e2el_summary(completed, failed, total_time, &e2el_times, &percentiles);

    if let Some(path) = &args.output_json {
        write_result_json(path, completed, failed, &e2el_times, &percentiles, &stats)?;
        eprintln!("Wrote stats JSON to {path}");
    }

    chat.shutdown().await.context("failed to shut down chat facade")?;
    engine
        .shutdown(ENGINE_SHUTDOWN_TIMEOUT)
        .await
        .context("failed to shut down managed engine")?;
    Ok(())
}

/// Submit every sample through the full chat pipeline and drive each stream
/// to completion, returning per-request wall-clock latencies.
///
/// Concurrency is capped by `semaphore`; pass a `Semaphore::new(1)` to match
/// Python's serial `LLMEngine` driver.
async fn run_batch(
    chat: &ChatLlm,
    samples: &[SampleRequest],
    semaphore: &Arc<Semaphore>,
) -> Vec<Result<Duration>> {
    let submissions = samples.iter().map(|sample| {
        let request = build_chat_request(sample);
        let semaphore = semaphore.clone();
        async move {
            let _permit = semaphore.acquire().await.context("semaphore closed")?;
            let request = request?;
            let start = Instant::now();
            chat.chat(request)
                .await
                .context("failed to submit request")?
                .collect_message()
                .await
                .context("request failed")?;
            Ok(start.elapsed())
        }
    });
    join_all(submissions).await
}

fn generate_dataset(
    tokenizer: &TokenizerKind,
    num_requests: usize,
    seed: u64,
    args: &MmProcessorArgs,
    range_ratio: RangeRatio,
    limit: &MmLimitPerPrompt,
    buckets: &[(MmBucketKey, f64)],
) -> Result<Vec<SampleRequest>> {
    generate_random_mm_dataset(
        tokenizer,
        num_requests,
        args.random_input_len,
        args.random_output_len,
        args.random_prefix_len,
        range_ratio,
        seed,
        &args.request_id_prefix,
        args.random_mm_base_items_per_request,
        args.random_mm_num_mm_items_range_ratio,
        limit,
        buckets,
        false,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Build a chat request from a dataset sample: the decoded text prompt plus the
/// pre-serialized `image_url` fragments (base64 data URLs) as content parts.
///
/// Sampling matches the Python benchmark: greedy decode capped at the expected
/// output length.
fn build_chat_request(sample: &SampleRequest) -> Result<ChatRequest> {
    let mut request = ChatRequest::for_test();
    request.request_id = sample.request_id.clone().unwrap_or_else(|| "mm-proc".to_string());
    request.sampling_params.temperature = Some(0.0);
    request.sampling_params.max_tokens = Some(u32::try_from(sample.expected_output_len)?);

    let mut parts = vec![ChatContentPart::text(sample.prompt.as_ref())];
    if let Some(mm_items) = &sample.multi_modal_content {
        for fragment in mm_items.iter() {
            let url = extract_image_url(fragment)?;
            parts.push(ChatContentPart::image_url(url));
        }
    }
    request.messages = vec![ChatMessage::user(ChatContent::from(parts))];
    Ok(request)
}

/// Extract the `image_url.url` string from a pre-serialized content fragment
/// (`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`).
fn extract_image_url(fragment: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(fragment)
        .with_context(|| format!("invalid multimodal fragment: {fragment}"))?;
    let url = value
        .get("image_url")
        .and_then(|v| v.get("url"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| BenchError::Config("fragment missing image_url.url".to_string()))?;
    Ok(url.to_string())
}

fn parse_percentiles(raw: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: f64 = part.parse().with_context(|| format!("invalid percentile `{part}`"))?;
        if !(0.0..=100.0).contains(&value) {
            anyhow::bail!("percentile `{part}` out of range [0, 100]");
        }
        values.push(value);
    }
    Ok(values)
}

/// Aggregate per-request stage timings into `{statistic: {stage_ms: ms}}`.
fn aggregate_stats(
    per_request: HashMap<String, HashMap<String, f64>>,
    percentiles: &[f64],
) -> BTreeMap<String, BTreeMap<String, f64>> {
    let mut stage_values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (_request_id, stage_map) in per_request {
        for (stage, seconds) in stage_map {
            stage_values.entry(stage).or_default().push(seconds);
        }
    }

    let mut stats: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut mean_map = BTreeMap::new();
    let mut median_map = BTreeMap::new();
    let mut std_map = BTreeMap::new();
    let mut percentile_maps: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

    for (stage, values) in &stage_values {
        let sorted = sort_clone(values);
        let stage_ms = stage.replace("_secs", "_ms");
        mean_map.insert(stage_ms.clone(), mean(values) * SEC_TO_MS);
        median_map.insert(stage_ms.clone(), median_sorted(&sorted) * SEC_TO_MS);
        std_map.insert(stage_ms.clone(), std_dev(values) * SEC_TO_MS);
        for p in percentiles {
            percentile_maps
                .entry(format!("p{p}"))
                .or_default()
                .insert(stage_ms.clone(), percentile_sorted(&sorted, *p) * SEC_TO_MS);
        }
    }

    stats.insert("mean".to_string(), mean_map);
    stats.insert("median".to_string(), median_map);
    stats.insert("std".to_string(), std_map);
    stats.extend(percentile_maps);
    stats
}

/// Print a plain-text table with one row per statistic and one column per stage.
fn print_report(stats: &BTreeMap<String, BTreeMap<String, f64>>, percentiles: &[f64]) {
    let mut stat_order: Vec<String> = vec!["mean".into(), "median".into(), "std".into()];
    let mut pkeys: Vec<String> = percentiles.iter().map(|p| format!("p{p}")).collect();
    pkeys.sort_by(|a, b| {
        let av: f64 = a[1..].parse().unwrap_or(0.0);
        let bv: f64 = b[1..].parse().unwrap_or(0.0);
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });
    stat_order.extend(pkeys);
    for key in stats.keys() {
        if !stat_order.contains(key) {
            stat_order.push(key.clone());
        }
    }

    let mut stages: Vec<String> = {
        let mut set = BTreeSet::new();
        for map in stats.values() {
            set.extend(map.keys().cloned());
        }
        set.into_iter().collect()
    };
    stages.sort_by_key(|stage| {
        STAGE_ORDER
            .iter()
            .position(|s| *s == stage.as_str())
            .map(|i| (i as isize, stage.clone()))
            .unwrap_or((isize::MAX, stage.clone()))
    });

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut header = vec!["Statistic".to_string()];
    header.extend(stages.iter().cloned());
    rows.push(header);
    for stat in &stat_order {
        if let Some(map) = stats.get(stat) {
            let mut row = vec![stat.clone()];
            for stage in &stages {
                row.push(
                    map.get(stage).map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".to_string()),
                );
            }
            rows.push(row);
        }
    }

    let ncols = stages.len() + 1;
    let mut widths = vec![0usize; ncols];
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    for row in &rows {
        let line = row
            .iter()
            .enumerate()
            .map(|(i, cell)| format!("{cell:<width$}", width = widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
    }
}

/// Print the end-to-end latency summary (mirrors the Python report layout).
fn print_e2el_summary(
    completed: usize,
    failed: usize,
    total_time: Duration,
    e2el_times_secs: &[f64],
    percentiles: &[f64],
) {
    println!("Completed: {completed}, Failed: {failed}");
    if e2el_times_secs.is_empty() {
        println!("End-to-End Latency (ms): no completed requests");
        return;
    }

    let e2el_ms: Vec<f64> = e2el_times_secs.iter().map(|secs| secs * SEC_TO_MS).collect();
    let sorted = sort_clone(&e2el_ms);
    println!("End-to-End Latency (ms):");
    println!("  Mean:   {:.3}", mean(&e2el_ms));
    println!("  Median: {:.3}", median_sorted(&sorted));
    println!("  Std:    {:.3}", std_dev(&e2el_ms));
    for p in percentiles {
        println!("  P{p}:    {:.3}", percentile_sorted(&sorted, *p));
    }
    let throughput = completed as f64 / total_time.as_secs_f64();
    println!("Request throughput: {throughput:.3} req/s");
}

/// JSON result schema mirroring `vllm bench mm-processor`.
#[derive(Serialize)]
struct BenchmarkResultJson<'a> {
    completed: usize,
    failed: usize,
    mean_e2el_ms: f64,
    median_e2el_ms: f64,
    std_e2el_ms: f64,
    percentiles_e2el_ms: Vec<(f64, f64)>,
    mm_processor_stats: &'a BTreeMap<String, BTreeMap<String, f64>>,
}

fn write_result_json(
    path: &str,
    completed: usize,
    failed: usize,
    e2el_times_secs: &[f64],
    percentiles: &[f64],
    stats: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Result<()> {
    let e2el_ms: Vec<f64> = e2el_times_secs.iter().map(|secs| secs * SEC_TO_MS).collect();
    let sorted = sort_clone(&e2el_ms);
    let result = BenchmarkResultJson {
        completed,
        failed,
        mean_e2el_ms: mean(&e2el_ms),
        median_e2el_ms: median_sorted(&sorted),
        std_e2el_ms: std_dev(&e2el_ms),
        percentiles_e2el_ms: percentiles
            .iter()
            .map(|p| (*p, percentile_sorted(&sorted, *p)))
            .collect(),
        mm_processor_stats: stats,
    };
    let json = serde_json::to_string_pretty(&result).context("failed to serialize result")?;
    std::fs::write(path, format!("{json}\n")).with_context(|| format!("failed to write {path}"))?;
    Ok(())
}
