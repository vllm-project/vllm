// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use crate::cli::{
    BackendKind, BenchServeArgs, DatasetName, LoraAssignment, RampUpStrategy, SpeedBenchConfig,
};
use crate::datasets::random_mm::{MmBucketKey, MmLimitPerPrompt};
use crate::error::{BenchError, Result};

/// Parsed goodput SLO configuration.
#[derive(Debug, Clone, Default)]
pub struct GoodputConfig {
    /// TTFT SLO in milliseconds (None = not checked).
    pub ttft_ms: Option<f64>,
    /// TPOT SLO in milliseconds (None = not checked).
    pub tpot_ms: Option<f64>,
    /// E2EL SLO in milliseconds (None = not checked).
    pub e2el_ms: Option<f64>,
}

impl GoodputConfig {
    pub fn is_empty(&self) -> bool {
        self.ttft_ms.is_none() && self.tpot_ms.is_none() && self.e2el_ms.is_none()
    }
}

/// Ramp-up configuration.
#[derive(Debug, Clone)]
pub struct RampUpConfig {
    pub strategy: RampUpStrategy,
    pub start_rps: f64,
    pub end_rps: f64,
}

/// Range ratio for sampling input/output lengths, matching Python
/// `vllm bench serve`: lengths are drawn uniformly from [len*(1-r), len*(1+r)].
/// A single float applies to both; the JSON form '{"input": r1, "output": r2}'
/// controls them independently. Each ratio must be in [0, 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeRatio {
    pub input: f64,
    pub output: f64,
}

impl RangeRatio {
    /// Parse `--random-range-ratio`: a bare float or a JSON object with
    /// "input" and "output" keys.
    pub fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        let (input, output) = if let Ok(v) = trimmed.parse::<f64>() {
            (v, v)
        } else {
            let v: serde_json::Value = serde_json::from_str(trimmed).map_err(|_| {
                BenchError::Config(format!(
                    "Invalid --random-range-ratio '{raw}': expected a float or \
                     '{{\"input\": r1, \"output\": r2}}'"
                ))
            })?;
            let obj = v.as_object().ok_or_else(|| {
                BenchError::Config(
                    "--random-range-ratio JSON form must be an object with \
                     'input' and 'output' keys"
                        .into(),
                )
            })?;
            let get = |key: &str| -> Result<f64> {
                obj.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
                    BenchError::Config(format!(
                        "--random-range-ratio JSON form must contain a numeric '{key}' key"
                    ))
                })
            };
            (get("input")?, get("output")?)
        };

        for (name, r) in [("input", input), ("output", output)] {
            if !(0.0..1.0).contains(&r) {
                let hint = if r == 1.0 {
                    " NOTE: semantics now match Python vllm bench serve — lengths are \
                     sampled from [len*(1-r), len*(1+r)] and 0.0 means fixed length. \
                     The old Rust-only default 1.0 ([len*r, len]) is no longer valid."
                } else {
                    ""
                };
                return Err(BenchError::Config(format!(
                    "--random-range-ratio {name} ratio must be in [0, 1), got {r}.{hint}"
                )));
            }
        }
        Ok(Self { input, output })
    }

    /// Sampling interval for input lengths: [floor(len*(1-r)), ceil(len*(1+r))].
    pub fn input_bounds(&self, len: usize) -> (usize, usize) {
        let low = ((len as f64) * (1.0 - self.input)).floor() as usize;
        let high = ((len as f64) * (1.0 + self.input)).ceil() as usize;
        (low, high)
    }

    /// Sampling interval for output lengths, clamped to at least 1 token.
    pub fn output_bounds(&self, len: usize) -> (usize, usize) {
        let low = (((len as f64) * (1.0 - self.output)).floor() as usize).max(1);
        let high = (((len as f64) * (1.0 + self.output)).ceil() as usize).max(1);
        (low, high)
    }

    /// True when both ratios are 0 (every request uses the exact target lengths).
    pub fn is_fixed(&self) -> bool {
        self.input == 0.0 && self.output == 0.0
    }
}

/// Validated benchmark configuration derived from CLI args.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub backend: BackendKind,
    pub base_url: String,
    pub api_url: String,
    pub model: Option<String>,
    pub model_name: Option<String>,
    pub tokenizer_id: Option<String>,
    #[allow(dead_code)]
    pub tokenizer_mode: String,
    pub trust_remote_code: bool,
    pub skip_tokenizer_init: bool,
    pub dataset_name: DatasetName,
    pub dataset_path: Option<String>,
    pub max_model_len: Option<usize>,
    pub random_input_len: usize,
    pub random_output_len: usize,
    pub random_prefix_len: usize,
    pub random_range_ratio: RangeRatio,
    pub random_cache_hit_fraction: f64,
    pub random_cache_ratio: f64,
    /// Inputs per request for embeddings/pooling backends (1 = no batching).
    pub random_batch_size: usize,
    /// random-rerank: whether the served model is a reranker (default true).
    pub is_reranker: bool,
    pub custom_output_len: i64,
    pub prefix_repetition_prefix_len: usize,
    pub prefix_repetition_suffix_len: usize,
    pub prefix_repetition_num_prefixes: usize,
    pub prefix_repetition_output_len: usize,
    pub sharegpt_output_len: Option<usize>,
    pub sonnet_input_len: usize,
    pub sonnet_output_len: usize,
    pub sonnet_prefix_len: usize,
    pub no_oversample: bool,
    pub disable_shuffle: bool,
    pub num_prompts: usize,
    pub request_rate: f64,
    pub burstiness: f64,
    pub max_concurrency: Option<usize>,
    pub steady_state_threshold: f64,
    pub steady_state_min_window: Option<f64>,
    pub no_steady_state: bool,
    pub disable_tqdm: bool,
    pub num_warmups: usize,
    pub profile: bool,
    pub profile_batch_threshold: Option<usize>,
    pub profile_duration: f64,
    pub save_result: bool,
    pub save_detailed: bool,
    pub append_result: bool,
    pub result_dir: Option<String>,
    pub result_filename: Option<String>,
    pub seed: u64,
    pub ignore_eos: bool,
    pub insecure: bool,
    pub selected_percentile_metrics: Vec<String>,
    pub selected_percentiles: Vec<f64>,
    pub sweep_summary_percentiles: Vec<f64>,
    pub label: Option<String>,
    pub logprobs: Option<usize>,
    pub request_id_prefix: String,
    pub ready_check_timeout_sec: u64,
    pub extra_headers: Option<HashMap<String, String>>,
    pub extra_body: Option<serde_json::Value>,
    pub metadata: Option<Vec<(String, String)>>,
    pub dry_run: bool,
    pub goodput: GoodputConfig,
    pub ramp_up: Option<RampUpConfig>,
    pub multi_turn: bool,
    pub multi_turn_num_turns: usize,
    pub multi_turn_min_turns: usize,
    pub multi_turn_max_turns: usize,
    pub sharegpt_multi_turn_max_turns: Option<usize>,
    pub per_turn_input_len: usize,
    pub multi_turn_concurrency: Option<usize>,
    pub multi_turn_delay_ms: u64,
    pub multi_turn_prefix_global_ratio: f64,
    pub multi_turn_prefix_conversation_ratio: f64,
    pub speed_bench_config: SpeedBenchConfig,
    pub speed_bench_category: Option<String>,
    pub speed_bench_max_input_len: Option<usize>,
    pub hf_split: Option<String>,
    pub hf_subset: Option<String>,
    pub hf_output_len: Option<usize>,
    pub hf_text_column: Option<String>,
    pub reset_prefix_cache: bool,
    pub prompt_token_ids: bool,
    // --- Random multimodal dataset ---
    pub random_mm_base_items_per_request: usize,
    pub random_mm_num_mm_items_range_ratio: f64,
    pub random_mm_limit: MmLimitPerPrompt,
    pub random_mm_buckets: Vec<(MmBucketKey, f64)>,
    /// Datasets that support it pre-build the chat `messages` array
    /// (text + multimodal parts) instead of prompt + separate mm content.
    pub enable_multimodal_chat: bool,
    /// LoRA adapter names. None = no LoRA routing (use --model directly).
    /// Stored as Arc<str> so per-request override is a cheap clone.
    pub lora_modules: Option<Vec<Arc<str>>>,
    pub lora_assignment: LoraAssignment,
}

impl BenchConfig {
    pub fn from_args(args: &BenchServeArgs) -> Result<Self> {
        if args.burstiness <= 0.0 {
            return Err(BenchError::Config("Burstiness must be positive".into()));
        }

        if args.num_prompts == 0 {
            return Err(BenchError::Config(
                "--num-prompts must be at least 1".into(),
            ));
        }

        if args.request_rate <= 0.0 && !args.request_rate.is_infinite() {
            return Err(BenchError::Config(
                "--request-rate must be positive (or inf)".into(),
            ));
        }
        if args.max_model_len == Some(0) {
            return Err(BenchError::Config(
                "--max-model-len must be at least 1".into(),
            ));
        }

        let base_url = args.resolve_base_url();
        let api_url = args.resolve_api_url();

        let extra_headers = args.parse_headers()?;
        let mut extra_body = args.parse_extra_body()?;

        // Merge sampling parameters into extra_body (matches Python behavior).
        // Python collects non-None sampling params and merges them UNDER extra_body,
        // meaning extra_body keys take precedence over sampling params.
        {
            let mut sampling_params = serde_json::Map::new();
            if let Some(v) = args.top_p {
                sampling_params.insert("top_p".into(), serde_json::json!(v));
            }
            if let Some(v) = args.top_k {
                sampling_params.insert("top_k".into(), serde_json::json!(v));
            }
            if let Some(v) = args.min_p {
                sampling_params.insert("min_p".into(), serde_json::json!(v));
            }
            if let Some(v) = args.temperature {
                sampling_params.insert("temperature".into(), serde_json::json!(v));
            }
            if let Some(v) = args.frequency_penalty {
                sampling_params.insert("frequency_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = args.presence_penalty {
                sampling_params.insert("presence_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = args.repetition_penalty {
                sampling_params.insert("repetition_penalty".into(), serde_json::json!(v));
            }

            if !sampling_params.is_empty() {
                if !args.backend.is_openai_compatible() {
                    return Err(BenchError::Config(
                        "Sampling parameters are only supported by openai-compatible backends."
                            .into(),
                    ));
                }

                // Merge: sampling_params first, then extra_body on top (extra_body wins)
                let merged = match extra_body.take() {
                    Some(serde_json::Value::Object(existing)) => {
                        sampling_params.extend(existing);
                        sampling_params
                    }
                    Some(other) => {
                        // extra_body was not an object — just use sampling params
                        let value_type = match &other {
                            serde_json::Value::Null => "null",
                            serde_json::Value::Bool(_) => "boolean",
                            serde_json::Value::Number(_) => "number",
                            serde_json::Value::String(_) => "string",
                            serde_json::Value::Array(_) => "array",
                            serde_json::Value::Object(_) => unreachable!(),
                        };
                        tracing::warn!(
                            value_type,
                            "sampling parameters may be lost because --extra-body is not a JSON object"
                        );
                        sampling_params
                    }
                    None => sampling_params,
                };
                extra_body = Some(serde_json::Value::Object(merged));
            }
        }

        // Parse metadata
        let metadata = match &args.metadata {
            None => None,
            Some(items) => {
                let mut pairs = Vec::new();
                for item in items {
                    let (k, v) = item.split_once('=').ok_or_else(|| {
                        BenchError::Config("Invalid metadata format. Use KEY=VALUE".into())
                    })?;
                    pairs.push((k.trim().to_string(), v.trim().to_string()));
                }
                Some(pairs)
            }
        };

        // Parse goodput SLOs
        let goodput = parse_goodput(&args.goodput)?;

        // Parse ramp-up config
        let ramp_up = parse_ramp_up(args)?;

        // Default percentile metrics based on backend type
        let default_percentile_metrics = if args.backend.is_pooling() {
            "e2el"
        } else {
            "ttft,tpot,itl,e2el"
        };
        let percentile_metrics_str =
            args.percentile_metrics.as_deref().unwrap_or(default_percentile_metrics);
        let selected_percentile_metrics: Vec<String> =
            percentile_metrics_str.split(',').map(|s| s.trim().to_string()).collect();

        let metric_percentiles = parse_percentiles(&args.metric_percentiles, false)?;
        let sweep_summary_percentiles = args
            .sweep_summary_percentiles
            .as_deref()
            .map(|raw| parse_percentiles(raw, true))
            .transpose()?
            .unwrap_or_default();
        let mut selected_percentiles =
            merge_percentiles(&metric_percentiles, &sweep_summary_percentiles);
        // Always include p90 so it appears in console output and sweep summary
        if !selected_percentiles.contains(&90.0) {
            selected_percentiles.push(90.0);
        }

        let tokenizer_id = if args.skip_tokenizer_init {
            None
        } else {
            args.tokenizer.clone().or_else(|| args.model.clone())
        };

        // Resolve input/output lengths
        let random_input_len = args.resolved_random_input_len();
        let random_output_len = args.resolved_random_output_len();
        let per_turn_input_len = args.resolved_per_turn_input_len();

        // Normalized multi-turn turn counts (computed in validation block below, defaults
        // to num_turns if multi-turn mode is not active)
        let mut multi_turn_min_turns = args.multi_turn_num_turns;
        let mut multi_turn_max_turns = args.multi_turn_num_turns;

        // For random datasets with openai-compatible backends, default to ignore_eos.
        // Exception: multi-turn mode, where ignore_eos causes unbounded context growth
        // across turns. Multi-turn uses min_tokens instead for output length control.
        // Pooling backends don't generate tokens, so ignore_eos is irrelevant.
        let ignore_eos = if args.backend.is_pooling() {
            false
        } else {
            args.ignore_eos
                || ((args.dataset_name == DatasetName::Random
                    || args.dataset_name == DatasetName::RandomMm)
                    && args.backend.is_openai_compatible()
                    && !args.multi_turn)
        };

        // Pooling backends don't support multi-turn
        if args.backend.is_pooling() && args.multi_turn {
            return Err(BenchError::Config(
                "Pooling/embedding backends do not support --multi-turn".into(),
            ));
        }

        // LoRA validation. Adapter names must be non-empty after trim; pooling
        // backends are out of scope (vLLM LoRA routing is for generative paths).
        let lora_modules = match args.lora_modules.as_ref() {
            None => None,
            Some(names) => {
                if names.is_empty() {
                    return Err(BenchError::Config(
                        "--lora-modules requires at least one adapter name".into(),
                    ));
                }
                if args.backend.is_pooling() {
                    return Err(BenchError::Config(
                        "--lora-modules is not supported for pooling/embedding backends".into(),
                    ));
                }
                let mut out = Vec::with_capacity(names.len());
                for n in names {
                    let trimmed = n.trim();
                    if trimmed.is_empty() {
                        return Err(BenchError::Config(
                            "--lora-modules contains an empty adapter name".into(),
                        ));
                    }
                    out.push(Arc::<str>::from(trimmed));
                }
                Some(out)
            }
        };

        // Random-MM validation and config parsing
        let (random_mm_limit, random_mm_buckets) = if args.dataset_name == DatasetName::RandomMm {
            if args.backend != BackendKind::OpenaiChat {
                return Err(BenchError::Config(
                    "Multi-modal content (images) is only supported on 'openai-chat' backend."
                        .into(),
                ));
            }
            let limit = crate::datasets::random_mm::parse_limit_mm_per_prompt(
                &args.random_mm_limit_mm_per_prompt,
            )?;
            let buckets =
                crate::datasets::random_mm::parse_bucket_config(&args.random_mm_bucket_config)?;
            (limit, buckets)
        } else {
            (MmLimitPerPrompt::default(), Vec::new())
        };

        // Note: --dataset-path is optional for sharegpt (auto-downloads) and
        // sonnet (uses built-in Shakespeare's sonnets).

        // Range ratio (Python semantics: [len*(1-r), len*(1+r)], each r in [0,1))
        let random_range_ratio = RangeRatio::parse(&args.random_range_ratio)?;

        // Batched inputs only make sense for pooling backends (the generation
        // backends send one prompt per request).
        if args.random_batch_size == 0 {
            return Err(BenchError::Config(
                "--random-batch-size must be at least 1".into(),
            ));
        }
        if args.random_batch_size > 1
            && !args.backend.is_pooling()
            && args.dataset_name != DatasetName::RandomRerank
        {
            return Err(BenchError::Config(
                "--random-batch-size > 1 is only supported with embeddings/pooling backends".into(),
            ));
        }

        // random-rerank validation (mirrors Python RandomDatasetForReranking)
        let is_reranker = !args.no_reranker;
        if args.dataset_name == DatasetName::RandomRerank {
            if !args.backend.is_pooling() {
                return Err(BenchError::Config(
                    "--dataset-name random-rerank requires an embeddings/pooling backend \
                     (e.g. --backend vllm-rerank)"
                        .into(),
                ));
            }
            if !is_reranker && (args.num_prompts < 2 || args.random_batch_size < 2) {
                return Err(BenchError::Config(
                    "--no-reranker requires --num-prompts > 1 and --random-batch-size > 1 \
                     (the query is folded into the first batch slot)"
                        .into(),
                ));
            }
        }

        // Custom dataset validation
        if args.dataset_name == DatasetName::Custom {
            match args.dataset_path.as_deref() {
                None => {
                    return Err(BenchError::Config(
                        "--dataset-path is required for --dataset-name custom \
                         (a JSONL file with {\"prompt\": ..., \"output_tokens\": ...} lines)"
                            .into(),
                    ));
                }
                Some(p) if !p.ends_with(".jsonl") => {
                    return Err(BenchError::Config(
                        "Only JSONL format is supported for the custom dataset".into(),
                    ));
                }
                _ => {}
            }
            if !args.skip_chat_template {
                tracing::warn!(
                    dataset = "custom",
                    "client-side chat template rendering is unsupported; sending prompts raw"
                );
            }
        }

        // Prefix repetition validation
        if args.dataset_name == DatasetName::PrefixRepetition {
            if args.prefix_repetition_num_prefixes == 0 {
                return Err(BenchError::Config(
                    "--prefix-repetition-num-prefixes must be at least 1".into(),
                ));
            }
            if args.num_prompts < args.prefix_repetition_num_prefixes {
                return Err(BenchError::Config(format!(
                    "--num-prompts ({}) must be >= --prefix-repetition-num-prefixes ({})",
                    args.num_prompts, args.prefix_repetition_num_prefixes
                )));
            }
        }

        // HF dataset validation
        if args.dataset_name == DatasetName::Hf && args.dataset_path.is_none() {
            return Err(BenchError::Config(
                "--dataset-path is required for --dataset-name hf \
                 (set to a HuggingFace dataset ID, e.g. 'allenai/WildChat-4.8M')"
                    .into(),
            ));
        }
        if let Some(len) = args.hf_output_len
            && len == 0
        {
            return Err(BenchError::Config(
                "--hf-output-len must be at least 1".into(),
            ));
        }

        // Multi-turn validation
        if args.multi_turn {
            if args.backend != BackendKind::OpenaiChat {
                return Err(BenchError::Config(
                    "--multi-turn requires --backend openai-chat".into(),
                ));
            }
            if args.multi_turn_num_turns == 0 {
                return Err(BenchError::Config(
                    "--multi-turn-num-turns must be at least 1".into(),
                ));
            }

            // Normalize and validate min/max turns. ShareGPT only consumes max_turns
            // (the loader walks all available turns up to the cap), so the
            // min/num/max coupling used for synthetic generation does not apply.
            if args.dataset_name == DatasetName::ShareGpt {
                if args.multi_turn_max_turns == 1 {
                    return Err(BenchError::Config(
                        "--multi-turn-max-turns must be at least 2 for ShareGPT multi-turn".into(),
                    ));
                }
            } else {
                (multi_turn_min_turns, multi_turn_max_turns) =
                    match (args.multi_turn_min_turns, args.multi_turn_max_turns) {
                        (0, 0) => (args.multi_turn_num_turns, args.multi_turn_num_turns),
                        (m, 0) => (m, args.multi_turn_num_turns),
                        (0, x) => (args.multi_turn_num_turns, x),
                        (m, x) => (m, x),
                    };
                if multi_turn_min_turns < 1 {
                    return Err(BenchError::Config(
                        "--multi-turn-min-turns must be at least 1".into(),
                    ));
                }
                if multi_turn_min_turns > multi_turn_max_turns {
                    return Err(BenchError::Config(
                        "--multi-turn-min-turns must be <= --multi-turn-max-turns".into(),
                    ));
                }
            }

            if ignore_eos {
                tracing::warn!(
                    ignore_eos,
                    multi_turn = true,
                    "output length limits may be ignored, causing unbounded context growth"
                );
            }

            // Validate prefix sharing ratios
            let pg = args.multi_turn_prefix_global_ratio;
            let pc = args.multi_turn_prefix_conversation_ratio;
            if !(0.0..=1.0).contains(&pg) {
                return Err(BenchError::Config(
                    "--multi-turn-prefix-global-ratio must be in [0.0, 1.0]".into(),
                ));
            }
            if !(0.0..=1.0).contains(&pc) {
                return Err(BenchError::Config(
                    "--multi-turn-prefix-conversation-ratio must be in [0.0, 1.0]".into(),
                ));
            }
            if pg + pc >= 1.0 {
                return Err(BenchError::Config(
                    "--multi-turn-prefix-global-ratio + --multi-turn-prefix-conversation-ratio must be < 1.0 (unique suffix required)".into(),
                ));
            }
            if (pg > 0.0 || pc > 0.0) && args.dataset_name != DatasetName::Random {
                return Err(BenchError::Config(
                    "Prefix sharing (--multi-turn-prefix-global-ratio / --multi-turn-prefix-conversation-ratio) only works with --dataset-name random".into(),
                ));
            }
        }

        if !(args.steady_state_threshold > 0.0 && args.steady_state_threshold <= 1.0) {
            return Err(BenchError::Config(format!(
                "--steady-state-threshold must be in (0.0, 1.0], got {}",
                args.steady_state_threshold
            )));
        }
        if let Some(mw) = args.steady_state_min_window
            && mw < 0.0
        {
            return Err(BenchError::Config(format!(
                "--steady-state-min-window must be >= 0, got {mw}"
            )));
        }

        if args.profile_batch_threshold.is_some() && !args.profile {
            return Err(BenchError::Config(
                "--profile-batch-threshold requires --profile".into(),
            ));
        }
        if args.profile_duration <= 0.0 {
            return Err(BenchError::Config(
                "--profile-duration must be positive".into(),
            ));
        }
        if args.profile_batch_threshold.is_none() && args.profile_duration != 5.0 {
            return Err(BenchError::Config(
                "--profile-duration requires --profile-batch-threshold".into(),
            ));
        }

        Ok(BenchConfig {
            backend: args.backend,
            base_url,
            api_url,
            model: args.model.clone(),
            model_name: args.served_model_name.clone(),
            tokenizer_id,
            tokenizer_mode: args.tokenizer_mode.clone(),
            trust_remote_code: args.trust_remote_code,
            skip_tokenizer_init: args.skip_tokenizer_init,
            dataset_name: args.dataset_name,
            dataset_path: args.dataset_path.clone(),
            max_model_len: args.max_model_len,
            random_input_len,
            random_output_len,
            random_prefix_len: args.random_prefix_len,
            random_range_ratio,
            random_batch_size: args.random_batch_size,
            is_reranker,
            custom_output_len: args.output_len.map(|v| v as i64).unwrap_or(args.custom_output_len),
            prefix_repetition_prefix_len: args.prefix_repetition_prefix_len,
            prefix_repetition_suffix_len: args.prefix_repetition_suffix_len,
            prefix_repetition_num_prefixes: args.prefix_repetition_num_prefixes,
            prefix_repetition_output_len: args
                .output_len
                .unwrap_or(args.prefix_repetition_output_len),
            random_cache_hit_fraction: args.random_cache_hit_fraction,
            random_cache_ratio: args.random_cache_ratio,
            sharegpt_output_len: args.sharegpt_output_len,
            sonnet_input_len: args.sonnet_input_len,
            sonnet_output_len: args.sonnet_output_len,
            sonnet_prefix_len: args.sonnet_prefix_len,
            no_oversample: args.no_oversample,
            disable_shuffle: args.disable_shuffle,
            num_prompts: args.num_prompts,
            request_rate: args.request_rate,
            burstiness: args.burstiness,
            max_concurrency: args.max_concurrency,
            steady_state_threshold: args.steady_state_threshold,
            steady_state_min_window: args.steady_state_min_window,
            no_steady_state: args.no_steady_state,
            disable_tqdm: args.disable_tqdm,
            num_warmups: args.num_warmups,
            profile: args.profile,
            profile_batch_threshold: args.profile_batch_threshold,
            profile_duration: args.profile_duration,
            save_result: args.save_result,
            save_detailed: args.save_detailed,
            append_result: args.append_result,
            result_dir: args.result_dir.clone(),
            result_filename: args.result_filename.clone(),
            seed: args.seed,
            ignore_eos,
            insecure: args.insecure,
            selected_percentile_metrics,
            selected_percentiles,
            sweep_summary_percentiles,
            label: args.label.clone(),
            logprobs: args.logprobs,
            request_id_prefix: args.get_request_id_prefix(),
            ready_check_timeout_sec: args.ready_check_timeout_sec,
            extra_headers,
            extra_body,
            metadata,
            dry_run: args.dry_run,
            goodput,
            ramp_up,
            multi_turn: args.multi_turn,
            multi_turn_num_turns: args.multi_turn_num_turns,
            multi_turn_min_turns,
            multi_turn_max_turns,
            sharegpt_multi_turn_max_turns: if args.multi_turn
                && args.dataset_name == DatasetName::ShareGpt
                && args.multi_turn_max_turns != 0
            {
                Some(args.multi_turn_max_turns)
            } else {
                None
            },
            per_turn_input_len,
            multi_turn_concurrency: args.multi_turn_concurrency,
            multi_turn_delay_ms: args.multi_turn_delay_ms,
            multi_turn_prefix_global_ratio: args.multi_turn_prefix_global_ratio,
            multi_turn_prefix_conversation_ratio: args.multi_turn_prefix_conversation_ratio,
            speed_bench_config: args.speed_bench_config,
            speed_bench_category: args.speed_bench_category.clone(),
            speed_bench_max_input_len: args.speed_bench_max_input_len,
            hf_split: args.hf_split.clone(),
            hf_subset: args.hf_subset.clone(),
            hf_output_len: args.hf_output_len,
            hf_text_column: args.hf_text_column.clone(),
            reset_prefix_cache: args.reset_prefix_cache,
            prompt_token_ids: args.prompt_token_ids,
            random_mm_base_items_per_request: args.random_mm_base_items_per_request,
            random_mm_num_mm_items_range_ratio: args.random_mm_num_mm_items_range_ratio,
            random_mm_limit,
            random_mm_buckets,
            enable_multimodal_chat: args.enable_multimodal_chat,
            lora_modules,
            lora_assignment: args.lora_assignment,
        })
    }
}

fn parse_percentiles(raw: &str, dedupe: bool) -> Result<Vec<f64>> {
    let mut percentiles = Vec::new();
    for s in raw.split(',') {
        let p = s
            .trim()
            .parse::<f64>()
            .map_err(|_| BenchError::Config(format!("Invalid percentile: {s}")))?;
        if !(0.0..=100.0).contains(&p) {
            return Err(BenchError::Config(format!(
                "Percentile must be in [0, 100], got: {p}"
            )));
        }
        if !dedupe || !percentiles.contains(&p) {
            percentiles.push(p);
        }
    }
    Ok(percentiles)
}

fn merge_percentiles(metric_percentiles: &[f64], summary_percentiles: &[f64]) -> Vec<f64> {
    let mut merged = Vec::with_capacity(metric_percentiles.len() + summary_percentiles.len());
    for &p in metric_percentiles {
        if !merged.contains(&p) {
            merged.push(p);
        }
    }
    for &p in summary_percentiles {
        if !merged.contains(&p) {
            merged.push(p);
        }
    }
    merged
}

fn parse_goodput(goodput_args: &Option<Vec<String>>) -> Result<GoodputConfig> {
    let items = match goodput_args {
        None => return Ok(GoodputConfig::default()),
        Some(items) => items,
    };

    let valid_names = ["ttft", "tpot", "e2el"];
    let mut config = GoodputConfig::default();

    for item in items {
        let (name, val_str) = item.split_once(':').ok_or_else(|| {
            BenchError::Config(format!(
                "Invalid goodput format: '{item}'. Use KEY:VALUE (e.g. ttft:100)"
            ))
        })?;

        let val: f64 = val_str
            .trim()
            .parse()
            .map_err(|_| BenchError::Config(format!("Invalid goodput value: '{val_str}'")))?;

        if val < 0.0 {
            return Err(BenchError::Config(format!(
                "Goodput SLO value must be non-negative, got: {name}:{val}"
            )));
        }

        if !valid_names.contains(&name) {
            return Err(BenchError::Config(format!(
                "Invalid goodput metric: '{name}'. Valid: {valid_names:?}"
            )));
        }

        match name {
            "ttft" => config.ttft_ms = Some(val),
            "tpot" => config.tpot_ms = Some(val),
            "e2el" => config.e2el_ms = Some(val),
            _ => unreachable!(),
        }
    }

    Ok(config)
}

fn parse_ramp_up(args: &BenchServeArgs) -> Result<Option<RampUpConfig>> {
    let strategy = match args.ramp_up_strategy {
        None => return Ok(None),
        Some(s) => s,
    };

    let start_rps = args.ramp_up_start_rps.ok_or_else(|| {
        BenchError::Config("--ramp-up-start-rps is required when --ramp-up-strategy is set".into())
    })?;

    let end_rps = args.ramp_up_end_rps.ok_or_else(|| {
        BenchError::Config("--ramp-up-end-rps is required when --ramp-up-strategy is set".into())
    })?;

    if start_rps <= 0.0 || end_rps <= 0.0 {
        return Err(BenchError::Config(
            "Ramp-up RPS values must be positive".into(),
        ));
    }

    Ok(Some(RampUpConfig {
        strategy,
        start_rps,
        end_rps,
    }))
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::cli::BenchServeArgs;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: BenchServeArgs,
    }

    fn parse_args<I, T>(args: I) -> BenchServeArgs
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        TestCli::parse_from(args).args
    }

    fn base_multi_turn_args() -> Vec<&'static str> {
        vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
        ]
    }

    #[test]
    fn test_prefix_sharing_defaults_to_zero() {
        let args = base_multi_turn_args();
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();
        assert_eq!(config.multi_turn_prefix_global_ratio, 0.0);
        assert_eq!(config.multi_turn_prefix_conversation_ratio, 0.0);
    }

    #[test]
    fn test_prefix_sharing_valid() {
        let mut args = base_multi_turn_args();
        args.extend([
            "--multi-turn-prefix-global-ratio",
            "0.1",
            "--multi-turn-prefix-conversation-ratio",
            "0.8",
        ]);
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();
        assert!((config.multi_turn_prefix_global_ratio - 0.1).abs() < 1e-10);
        assert!((config.multi_turn_prefix_conversation_ratio - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_prefix_sharing_sum_exceeds_one_fails() {
        let mut args = base_multi_turn_args();
        args.extend([
            "--multi-turn-prefix-global-ratio",
            "0.6",
            "--multi-turn-prefix-conversation-ratio",
            "0.6",
        ]);
        let args = parse_args(args);
        assert!(BenchConfig::from_args(&args).is_err());
    }

    #[test]
    fn test_prefix_sharing_sum_equals_one_fails() {
        let mut args = base_multi_turn_args();
        args.extend([
            "--multi-turn-prefix-global-ratio",
            "0.5",
            "--multi-turn-prefix-conversation-ratio",
            "0.5",
        ]);
        let args = parse_args(args);
        assert!(BenchConfig::from_args(&args).is_err());
    }

    #[test]
    fn test_prefix_sharing_out_of_range_fails() {
        let mut args = base_multi_turn_args();
        args.extend(["--multi-turn-prefix-global-ratio", "1.5"]);
        let args = parse_args(args);
        assert!(BenchConfig::from_args(&args).is_err());
    }

    #[test]
    fn test_prefix_sharing_requires_random_dataset() {
        let args = vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
            "--dataset-name",
            "sharegpt",
            "--multi-turn-prefix-global-ratio",
            "0.1",
        ];
        let args = parse_args(args);
        assert!(BenchConfig::from_args(&args).is_err());
    }

    #[test]
    fn test_sharegpt_multi_turn_max_turns_default_uncapped() {
        let args = vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
            "--dataset-name",
            "sharegpt",
        ];
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();

        assert_eq!(config.multi_turn_max_turns, 3);
        assert_eq!(config.sharegpt_multi_turn_max_turns, None);
    }

    #[test]
    fn test_sharegpt_multi_turn_max_turns_2_succeeds() {
        // Regression: previously the (0, x) normalization arm produced
        // (min=multi_turn_num_turns=3, max=2), tripping the min>max check
        // with a misleading error about a flag the user never set.
        let args = vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
            "--dataset-name",
            "sharegpt",
            "--multi-turn-max-turns",
            "2",
        ];
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();
        assert_eq!(config.sharegpt_multi_turn_max_turns, Some(2));
    }

    #[test]
    fn test_sharegpt_multi_turn_max_turns_1_rejected() {
        let args = vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
            "--dataset-name",
            "sharegpt",
            "--multi-turn-max-turns",
            "1",
        ];
        let args = parse_args(args);
        let err = BenchConfig::from_args(&args).unwrap_err().to_string();
        assert!(
            err.contains("at least 2 for ShareGPT"),
            "expected ShareGPT-specific error, got: {err}"
        );
    }

    #[test]
    fn test_sharegpt_multi_turn_max_turns_explicit_cap() {
        let args = vec![
            "vllm-bench",
            "--backend",
            "openai-chat",
            "--multi-turn",
            "--model",
            "test-model",
            "--dataset-name",
            "sharegpt",
            "--multi-turn-max-turns",
            "20",
        ];
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();

        assert_eq!(config.sharegpt_multi_turn_max_turns, Some(20));
    }

    #[test]
    fn test_sweep_summary_percentiles_default_empty() {
        let args = base_multi_turn_args();
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();

        assert!(config.sweep_summary_percentiles.is_empty());
        assert_eq!(config.selected_percentiles, vec![99.0, 90.0]);
    }

    #[test]
    fn test_sweep_summary_percentiles_are_deduped_and_merged() {
        let mut args = base_multi_turn_args();
        args.extend([
            "--metric-percentiles",
            "99,95",
            "--sweep-summary-percentiles",
            "90,95,90",
        ]);
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();

        assert_eq!(config.sweep_summary_percentiles, vec![90.0, 95.0]);
        assert_eq!(config.selected_percentiles, vec![99.0, 95.0, 90.0]);
    }

    #[test]
    fn test_invalid_sweep_summary_percentile_fails() {
        let mut args = base_multi_turn_args();
        args.extend(["--sweep-summary-percentiles", "101"]);
        let args = parse_args(args);
        assert!(BenchConfig::from_args(&args).is_err());
    }

    #[test]
    fn test_max_model_len_is_configured() {
        let args = vec![
            "vllm-bench",
            "--model",
            "test-model",
            "--max-model-len",
            "4096",
        ];
        let args = parse_args(args);
        let config = BenchConfig::from_args(&args).unwrap();

        assert_eq!(config.max_model_len, Some(4096));
    }

    #[test]
    fn test_tokenizer_id_deferred_when_model_is_unspecified() {
        let args = parse_args(["vllm-bench"]);
        let config = BenchConfig::from_args(&args).unwrap();

        assert_eq!(config.tokenizer_id, None);
    }

    #[test]
    fn test_zero_max_model_len_fails() {
        let args = vec![
            "vllm-bench",
            "--model",
            "test-model",
            "--max-model-len",
            "0",
        ];
        let args = parse_args(args);

        assert!(BenchConfig::from_args(&args).is_err());
    }
    #[test]
    fn test_range_ratio_parse_float() {
        let rr = RangeRatio::parse("0.2").unwrap();
        assert_eq!(rr.input, 0.2);
        assert_eq!(rr.output, 0.2);
        assert!(!rr.is_fixed());
        assert!(RangeRatio::parse("0.0").unwrap().is_fixed());
    }

    #[test]
    fn test_range_ratio_parse_dict() {
        let rr = RangeRatio::parse(r#"{"input": 0.1, "output": 0.5}"#).unwrap();
        assert_eq!(rr.input, 0.1);
        assert_eq!(rr.output, 0.5);
    }

    #[test]
    fn test_range_ratio_rejects_old_default_with_hint() {
        let err = RangeRatio::parse("1.0").unwrap_err().to_string();
        assert!(err.contains("semantics now match Python"), "got: {err}");
        assert!(RangeRatio::parse("-0.1").is_err());
        assert!(RangeRatio::parse("{\"input\": 0.1}").is_err());
        assert!(RangeRatio::parse("abc").is_err());
    }

    #[test]
    fn test_range_ratio_bounds_python_semantics() {
        // Python: [floor(len*(1-r)), ceil(len*(1+r))], output low clamped to 1
        let rr = RangeRatio::parse("0.2").unwrap();
        assert_eq!(rr.input_bounds(100), (80, 120));
        assert_eq!(rr.output_bounds(100), (80, 120));
        let rr = RangeRatio::parse("0.99").unwrap();
        assert_eq!(rr.output_bounds(1).0, 1); // clamped
        let fixed = RangeRatio::parse("0.0").unwrap();
        assert_eq!(fixed.input_bounds(8192), (8192, 8192));
    }
}
