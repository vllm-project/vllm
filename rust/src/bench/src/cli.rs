// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;

use clap::Parser;

/// Backend type for the benchmark endpoint.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "openai")]
    Openai,
    #[value(name = "openai-chat")]
    OpenaiChat,
    #[value(name = "openai-embeddings")]
    OpenaiEmbeddings,
    #[value(name = "openai-embeddings-chat")]
    OpenaiEmbeddingsChat,
    #[value(name = "vllm-pooling")]
    VllmPooling,
    #[value(name = "vllm-rerank")]
    VllmRerank,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Openai => "openai",
            Self::OpenaiChat => "openai-chat",
            Self::OpenaiEmbeddings => "openai-embeddings",
            Self::OpenaiEmbeddingsChat => "openai-embeddings-chat",
            Self::VllmPooling => "vllm-pooling",
            Self::VllmRerank => "vllm-rerank",
        }
    }

    /// Return true if the backend is compatible with OpenAI-style API and sampling parameters.
    pub fn is_openai_compatible(self) -> bool {
        match self {
            Self::Vllm | Self::Openai | Self::OpenaiChat => true,
            Self::OpenaiEmbeddings
            | Self::OpenaiEmbeddingsChat
            | Self::VllmPooling
            | Self::VllmRerank => false,
        }
    }

    /// Return true if the backend is a pooling/embedding backend (non-generative).
    pub fn is_pooling(self) -> bool {
        matches!(
            self,
            Self::OpenaiEmbeddings
                | Self::OpenaiEmbeddingsChat
                | Self::VllmPooling
                | Self::VllmRerank
        )
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dataset to benchmark with.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetName {
    #[value(name = "random")]
    Random,
    #[value(name = "random-mm")]
    RandomMm,
    #[value(name = "sharegpt")]
    ShareGpt,
    #[value(name = "sonnet")]
    Sonnet,
    #[value(name = "speed-bench")]
    SpeedBench,
    #[value(name = "hf")]
    Hf,
    #[value(name = "custom")]
    Custom,
    #[value(name = "prefix_repetition", alias = "prefix-repetition")]
    PrefixRepetition,
    #[value(name = "random-rerank")]
    RandomRerank,
}

/// Ramp-up strategy for request rate.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RampUpStrategy {
    #[value(name = "linear")]
    Linear,
    #[value(name = "exponential")]
    Exponential,
}

/// Strategy for assigning LoRA modules to requests.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraAssignment {
    #[value(name = "random")]
    Random,
    #[value(name = "round-robin")]
    RoundRobin,
}

/// SPEED-Bench dataset split/config.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedBenchConfig {
    #[value(name = "qualitative")]
    Qualitative,
    #[value(name = "throughput_1k")]
    Throughput1k,
    #[value(name = "throughput_2k")]
    Throughput2k,
    #[value(name = "throughput_8k")]
    Throughput8k,
    #[value(name = "throughput_16k")]
    Throughput16k,
    #[value(name = "throughput_32k")]
    Throughput32k,
}

impl SpeedBenchConfig {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qualitative => "qualitative",
            Self::Throughput1k => "throughput_1k",
            Self::Throughput2k => "throughput_2k",
            Self::Throughput8k => "throughput_8k",
            Self::Throughput16k => "throughput_16k",
            Self::Throughput32k => "throughput_32k",
        }
    }
}

impl fmt::Display for SpeedBenchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// High-performance benchmark client for vLLM serving endpoints.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "vllm-bench",
    about = "Benchmark online serving throughput",
    version
)]
pub struct Cli {
    /// The type of backend or endpoint to use for the benchmark.
    #[arg(long, default_value = "openai")]
    pub backend: BackendKind,

    /// Server or API base url if not using http host and port.
    #[arg(long)]
    pub base_url: Option<String>,

    /// Server host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Server port.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// API endpoint. Auto-selected based on --backend if not specified.
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Name of the model. If not specified, will fetch from server.
    #[arg(long)]
    pub model: Option<String>,

    /// The model name used in the API (for --served-model-name).
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Name or path of the tokenizer.
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Tokenizer mode (auto, hf, slow, mistral).
    #[arg(long, default_value = "auto")]
    pub tokenizer_mode: String,

    /// Skip initialization of tokenizer.
    #[arg(long, default_value_t = false)]
    pub skip_tokenizer_init: bool,

    /// Trust remote code for tokenizer.
    #[arg(long, default_value_t = false)]
    pub trust_remote_code: bool,

    /// Dataset name.
    #[arg(long, default_value = "random")]
    pub dataset_name: DatasetName,

    /// General input length for datasets.
    #[arg(long)]
    pub input_len: Option<usize>,

    /// General output length for datasets.
    #[arg(long)]
    pub output_len: Option<usize>,

    /// Maximum model context length. Requests with prompt_len + output_len above this are filtered
    /// out.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Random dataset input length.
    #[arg(long, default_value_t = 1024)]
    pub random_input_len: usize,

    /// Random dataset output length.
    #[arg(long, default_value_t = 128)]
    pub random_output_len: usize,

    /// Random dataset prefix length.
    #[arg(long, default_value_t = 0)]
    pub random_prefix_len: usize,

    /// Per-turn input length for turns 1+ in multi-turn mode.
    /// 0 = fallback to --random-input-len for all turns.
    /// Mirrors sglang bench_multiturn.py --sub-question-input-length.
    #[arg(long, default_value_t = 0)]
    pub per_turn_input_len: usize,

    /// Range ratio for sampling input/output lengths, matching Python
    /// `vllm bench serve`: lengths are drawn uniformly from
    /// [len*(1-r), len*(1+r)]. 0.0 (the default) = exact target lengths.
    /// Accepts a single float in [0, 1) or a JSON object
    /// '{"input": r1, "output": r2}' for independent control.
    /// NOTE: semantics changed — the old Rust-only form sampled [len*r, len]
    /// with default 1.0; old values like 1.0 are now rejected.
    #[arg(long, default_value = "0.0")]
    pub random_range_ratio: String,

    /// Batch multiple generated inputs into one request (embeddings/pooling
    /// backends only). E.g. 8 sends "input": [t1..t8] per request. Mirrors
    /// Python --random-batch-size. Default 1 = no batching.
    #[arg(long, default_value_t = 1)]
    pub random_batch_size: usize,

    /// random-rerank: the served model is NOT a reranker (embedding-based
    /// scoring). Changes query/document length accounting to mirror Python
    /// --no-reranker.
    #[arg(long, default_value_t = false)]
    pub no_reranker: bool,

    /// Bimodal prefix-cache (random dataset): fraction of prompts that are "warm"
    /// and reuse a shared cached prefix. 0.0 = off (default). E.g. 0.8 = 80% warm
    /// (prefix-cache hit), 20% cold (full prefill). Requires --random-cache-ratio > 0
    /// and --prompt-token-ids. In this mode --random-input-len is the TOTAL length.
    #[arg(long, default_value_t = 0.0)]
    pub random_cache_hit_fraction: f64,

    /// Bimodal prefix-cache (random dataset): fraction of each WARM prompt's length
    /// that is the shared cached prefix. 0.0 = off (default). E.g. 0.95 = 95% cached,
    /// 5% unique suffix. Used with --random-cache-hit-fraction.
    #[arg(long, default_value_t = 0.0)]
    pub random_cache_ratio: f64,

    /// Send prompt as token ID arrays instead of text strings.
    /// By default, prompts are decoded to text for maximum
    /// compatibility. Enable this for pure vLLM deployments to skip server-side
    /// tokenization (faster, exact token counts).
    #[arg(long, default_value_t = false)]
    pub prompt_token_ids: bool,

    // --- Random multimodal dataset ---
    /// Base number of multimodal items (images/videos) per request.
    #[arg(long, default_value_t = 1)]
    pub random_mm_base_items_per_request: usize,

    /// Range ratio for varying the number of multimodal items per request.
    /// Items sampled from [floor(n*(1-r)), ceil(n*(1+r))].
    #[arg(long, default_value_t = 0.0)]
    pub random_mm_num_mm_items_range_ratio: f64,

    /// Per-modality hard caps as JSON, e.g. '{"image": 3, "video": 0}'.
    #[arg(long, default_value = "{\"image\": 255, \"video\": 1}")]
    pub random_mm_limit_mm_per_prompt: String,

    /// Bucket config mapping (height,width,num_frames) to probability.
    /// Uses Python-style syntax: '{(256,256,1): 0.5, (720,1280,1): 0.5}'.
    /// num_frames=1 means image, num_frames>1 means video.
    #[arg(long, default_value = "{(256,256,1): 0.5, (720,1280,1): 0.5}")]
    pub random_mm_bucket_config: String,

    /// Enable multimodal chat transformation for datasets that support it.
    /// The dataset pre-builds the OpenAI chat `messages` array (text part +
    /// multimodal items) at generation time, and the request sends it verbatim.
    /// Mirrors Python's --enable-multimodal-chat. Currently applies to random-mm.
    #[arg(long, default_value_t = false)]
    pub enable_multimodal_chat: bool,

    // --- Custom dataset (JSONL) ---
    /// Output tokens per request for the custom dataset. Set to -1 to use the
    /// per-line "output_tokens" field from the JSONL file instead.
    #[arg(long, default_value_t = 256, allow_negative_numbers = true)]
    pub custom_output_len: i64,

    /// Skip applying a chat template to custom dataset prompts.
    /// NOTE: the Rust client never renders chat templates client-side, so this
    /// is always effectively on; passing it silences the informational notice.
    #[arg(long, default_value_t = false)]
    pub skip_chat_template: bool,

    // --- Prefix repetition dataset ---
    /// Shared-prefix token length for the prefix_repetition dataset.
    #[arg(long, default_value_t = 256)]
    pub prefix_repetition_prefix_len: usize,

    /// Per-request random suffix token length for the prefix_repetition dataset.
    #[arg(long, default_value_t = 256)]
    pub prefix_repetition_suffix_len: usize,

    /// Number of distinct shared prefixes for the prefix_repetition dataset.
    /// Requests are split evenly across prefixes (num-prompts / num-prefixes each).
    #[arg(long, default_value_t = 10)]
    pub prefix_repetition_num_prefixes: usize,

    /// Output tokens per request for the prefix_repetition dataset.
    #[arg(long, default_value_t = 128)]
    pub prefix_repetition_output_len: usize,

    /// Number of prompts to generate.
    #[arg(long, default_value_t = 1000)]
    pub num_prompts: usize,

    /// Number of requests per second. Use "inf" for all at once.
    #[arg(long, default_value_t = f64::INFINITY)]
    pub request_rate: f64,

    /// Burstiness factor of request generation.
    #[arg(long, default_value_t = 1.0)]
    pub burstiness: f64,

    /// Maximum number of concurrent requests.
    #[arg(long)]
    pub max_concurrency: Option<usize>,

    /// Fraction of --max-concurrency at which the steady-state window opens.
    /// Range: (0.0, 1.0]. Used only when --max-concurrency is set and
    /// --request-rate is inf.
    #[arg(long, default_value_t = 0.95)]
    pub steady_state_threshold: f64,

    /// Minimum steady-state window duration in seconds. Below this, a warning
    /// is attached. If unset, computed as max(10.0, 0.1 * run_duration).
    #[arg(long)]
    pub steady_state_min_window: Option<f64>,

    /// Disable steady-state metrics computation entirely.
    #[arg(long, default_value_t = false)]
    pub no_steady_state: bool,

    /// Disable tqdm progress bar.
    #[arg(long, default_value_t = false)]
    pub disable_tqdm: bool,

    /// Number of warmup requests.
    #[arg(long, default_value_t = 0)]
    pub num_warmups: usize,

    /// Use vLLM profiling. --profiler-config must be provided on the server.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Minimum server batch size (num_requests_running) before starting the
    /// profiler. When set, profiling is deferred until the /metrics endpoint
    /// reports at least this many running requests, then captures for
    /// --profile-duration seconds. Requires --profile.
    #[arg(long)]
    pub profile_batch_threshold: Option<usize>,

    /// How many seconds to capture once the batch threshold is reached.
    /// Defaults to 5. Requires --profile and --profile-batch-threshold.
    #[arg(long, default_value_t = 5.0)]
    pub profile_duration: f64,

    /// Save benchmark results to a JSON file.
    #[arg(long, default_value_t = false)]
    pub save_result: bool,

    /// Save detailed per-request results.
    #[arg(long, default_value_t = false)]
    pub save_detailed: bool,

    /// Directory to save benchmark JSON results.
    #[arg(long)]
    pub result_dir: Option<String>,

    /// Filename to save benchmark JSON results.
    #[arg(long)]
    pub result_filename: Option<String>,

    /// Random seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Set ignore_eos flag when sending the benchmark request.
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,

    /// Comma-separated list of metrics to report percentiles for.
    #[arg(long)]
    pub percentile_metrics: Option<String>,

    /// Comma-separated list of percentiles for selected metrics.
    #[arg(long, default_value = "99")]
    pub metric_percentiles: String,

    /// Comma-separated list of extra percentiles to show in sweep summaries.
    #[arg(long)]
    pub sweep_summary_percentiles: Option<String>,

    /// The label (prefix) of the benchmark results.
    #[arg(long)]
    pub label: Option<String>,

    /// Number of logprobs-per-token to compute.
    #[arg(long)]
    pub logprobs: Option<usize>,

    /// Prefix for request IDs.
    #[arg(long)]
    pub request_id_prefix: Option<String>,

    /// Maximum time to wait for endpoint readiness in seconds.
    #[arg(long, default_value_t = 0)]
    pub ready_check_timeout_sec: u64,

    /// Key-value pairs for extra headers (KEY=VALUE).
    #[arg(long = "header", num_args = 1..)]
    pub headers: Option<Vec<String>>,

    /// JSON string for extra body parameters.
    #[arg(long)]
    pub extra_body: Option<String>,

    /// Key-value pairs for metadata (KEY=VALUE).
    #[arg(long = "metadata", num_args = 1..)]
    pub metadata: Option<Vec<String>>,

    /// Dry run: only generate dataset and print stats, don't benchmark.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    // --- Sampling parameters ---
    /// Top-p sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub top_k: Option<i64>,

    /// Min-p sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub min_p: Option<f64>,

    /// Temperature sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Frequency penalty sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub frequency_penalty: Option<f64>,

    /// Presence penalty sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub presence_penalty: Option<f64>,

    /// Repetition penalty sampling parameter. Only affects openai-compatible backends.
    #[arg(long)]
    pub repetition_penalty: Option<f64>,

    // --- SSL ---
    /// Disable SSL certificate verification.
    #[arg(long, default_value_t = false)]
    pub insecure: bool,

    // --- Ramp-up ---
    /// Ramp-up strategy for request rate (linear or exponential).
    #[arg(long)]
    pub ramp_up_strategy: Option<RampUpStrategy>,

    /// Starting request rate for ramp-up (RPS).
    #[arg(long)]
    pub ramp_up_start_rps: Option<f64>,

    /// Ending request rate for ramp-up (RPS).
    #[arg(long)]
    pub ramp_up_end_rps: Option<f64>,

    // --- Goodput ---
    /// Service level objectives for goodput as "KEY:VALUE" pairs (e.g. ttft:100 tpot:50 e2el:500).
    /// Values are in milliseconds.
    #[arg(long = "goodput", num_args = 1..)]
    pub goodput: Option<Vec<String>>,

    // --- Result ---
    /// Append the benchmark result to the existing JSON file.
    #[arg(long, default_value_t = false)]
    pub append_result: bool,

    // --- ShareGPT dataset ---
    /// Path to dataset file (required for sharegpt dataset).
    #[arg(long)]
    pub dataset_path: Option<String>,

    /// Override output length for ShareGPT dataset.
    #[arg(long)]
    pub sharegpt_output_len: Option<usize>,

    /// Do not oversample if dataset is smaller than num_prompts.
    #[arg(long, default_value_t = false)]
    pub no_oversample: bool,

    /// Do not shuffle the dataset.
    #[arg(long, default_value_t = false)]
    pub disable_shuffle: bool,

    // --- Sonnet dataset ---
    /// Number of input tokens per request (sonnet dataset).
    #[arg(long, default_value_t = crate::datasets::sonnet::DEFAULT_INPUT_LEN)]
    pub sonnet_input_len: usize,

    /// Number of output tokens per request (sonnet dataset).
    #[arg(long, default_value_t = crate::datasets::sonnet::DEFAULT_OUTPUT_LEN)]
    pub sonnet_output_len: usize,

    /// Number of prefix tokens shared across requests (sonnet dataset).
    #[arg(long, default_value_t = crate::datasets::sonnet::DEFAULT_PREFIX_LEN)]
    pub sonnet_prefix_len: usize,

    /// SPEED-Bench config/split (qualitative, throughput_1k, throughput_2k, throughput_8k,
    /// throughput_16k, throughput_32k).
    #[arg(long, default_value = "qualitative")]
    pub speed_bench_config: SpeedBenchConfig,

    /// Filter SPEED-Bench by category (e.g. low_entropy, high_entropy, coding, math).
    #[arg(long)]
    pub speed_bench_category: Option<String>,

    /// Truncate SPEED-Bench prompts to at most this many tokens.
    /// Useful for creating custom input lengths from larger splits (e.g. --speed-bench-config
    /// throughput_16k --speed-bench-max-input-len 10240).
    #[arg(long)]
    pub speed_bench_max_input_len: Option<usize>,

    // --- HuggingFace dataset ---
    /// HuggingFace dataset split (e.g. train, test, validation).
    #[arg(long)]
    pub hf_split: Option<String>,

    /// HuggingFace dataset subset/config name.
    #[arg(long)]
    pub hf_subset: Option<String>,

    /// Fixed output length for HF dataset requests (overrides dataset-derived length).
    #[arg(long)]
    pub hf_output_len: Option<usize>,

    /// Column name containing the prompt text. Auto-detected if not specified.
    #[arg(long)]
    pub hf_text_column: Option<String>,

    // --- Compare mode ---
    /// Compare two benchmark result JSON files (e.g. --compare a.json b.json).
    /// Prints side-by-side metrics with delta and % change. Skips benchmarking.
    #[arg(long = "compare", num_args = 2, value_names = ["FILE_A", "FILE_B"])]
    pub compare: Option<Vec<String>>,

    // --- Sweep mode ---
    /// Sweep over max-concurrency values (comma-separated, e.g. --sweep-max-concurrency
    /// 1,10,50,100,500).
    #[arg(long)]
    pub sweep_max_concurrency: Option<String>,

    /// When sweeping concurrency, set num_prompts = concurrency * this factor for each sweep
    /// point.
    #[arg(long)]
    pub sweep_num_prompts_factor: Option<usize>,

    /// Sweep over request-rate values (comma-separated, supports "inf", e.g. --sweep-request-rate
    /// 1,10,100,inf).
    #[arg(long)]
    pub sweep_request_rate: Option<String>,

    /// Reset the server's prefix cache before each sweep iteration.
    /// Requires VLLM_SERVER_DEV_MODE=1 on the vLLM server.
    #[arg(long, default_value_t = false)]
    pub reset_prefix_cache: bool,

    // --- Multi-run ---
    /// Number of benchmark runs for statistical aggregation.
    #[arg(long, default_value_t = 1)]
    pub num_runs: usize,

    // --- Multi-turn conversation benchmark ---
    /// Enable multi-turn conversation benchmark mode.
    #[arg(long, default_value_t = false)]
    pub multi_turn: bool,

    /// Number of turns per conversation in synthetic multi-turn mode.
    #[arg(long, default_value_t = 3)]
    pub multi_turn_num_turns: usize,

    /// Minimum turns per conversation. 0 = use --multi-turn-num-turns.
    #[arg(long, default_value_t = 0)]
    pub multi_turn_min_turns: usize,

    /// Maximum turns per conversation.
    /// For synthetic multi-turn, 0 = use --multi-turn-num-turns.
    /// For ShareGPT multi-turn, 0 = uncapped.
    #[arg(long, default_value_t = 0)]
    pub multi_turn_max_turns: usize,

    /// Number of concurrent conversations (defaults to max-concurrency or num-prompts).
    #[arg(long)]
    pub multi_turn_concurrency: Option<usize>,

    /// Delay between turns in milliseconds (simulates user think time).
    #[arg(long, default_value_t = 0)]
    pub multi_turn_delay_ms: u64,

    /// Fraction of per-turn input tokens shared across ALL conversations (0.0–1.0).
    /// When > 0, enables prefix sharing mode: each turn sends a fixed-length message
    /// (no history accumulation). Only works with --dataset-name random.
    #[arg(long, default_value_t = 0.0)]
    pub multi_turn_prefix_global_ratio: f64,

    /// Fraction of per-turn input tokens shared within each conversation (0.0–1.0).
    /// When > 0, enables prefix sharing mode: each turn sends a fixed-length message
    /// (no history accumulation). Only works with --dataset-name random.
    #[arg(long, default_value_t = 0.0)]
    pub multi_turn_prefix_conversation_ratio: f64,

    // --- LoRA ---
    /// LoRA adapter names registered on the server (server-side
    /// `--lora-modules name=path`). Each request's `model` field is rewritten
    /// to one of these names; tokenizer and other endpoints keep using --model.
    /// In multi-turn mode, one adapter is assigned per conversation (sticky
    /// across turns).
    #[arg(long = "lora-modules", num_args = 1..)]
    pub lora_modules: Option<Vec<String>>,

    /// Strategy for assigning LoRA adapters to requests.
    /// 'random' (default) picks uniformly at random; 'round-robin' cycles
    /// through `--lora-modules` deterministically (i % N).
    #[arg(long = "lora-assignment", default_value = "random")]
    pub lora_assignment: LoraAssignment,
}

impl Cli {
    /// Resolve the base URL from explicit --base-url or from --host/--port.
    pub fn resolve_base_url(&self) -> String {
        if let Some(ref base) = self.base_url {
            base.clone()
        } else {
            format!("http://{}:{}", self.host, self.port)
        }
    }

    /// Resolve the API endpoint, auto-selecting based on backend if not explicit.
    pub fn resolve_endpoint(&self) -> String {
        if let Some(ref ep) = self.endpoint {
            return ep.clone();
        }
        match self.backend {
            BackendKind::OpenaiChat => "/v1/chat/completions".to_string(),
            BackendKind::Vllm | BackendKind::Openai => "/v1/completions".to_string(),
            BackendKind::OpenaiEmbeddings | BackendKind::OpenaiEmbeddingsChat => {
                "/v1/embeddings".to_string()
            }
            BackendKind::VllmPooling => "/v1/pooling".to_string(),
            BackendKind::VllmRerank => "/v1/rerank".to_string(),
        }
    }

    /// Resolve the full API URL.
    pub fn resolve_api_url(&self) -> String {
        format!("{}{}", self.resolve_base_url(), self.resolve_endpoint())
    }

    /// Parse extra headers from KEY=VALUE pairs.
    pub fn parse_headers(
        &self,
    ) -> crate::error::Result<Option<std::collections::HashMap<String, String>>> {
        match &self.headers {
            None => Ok(None),
            Some(items) => {
                let mut map = std::collections::HashMap::new();
                for item in items {
                    let (k, v) = item.split_once('=').ok_or_else(|| {
                        crate::error::BenchError::Config(
                            "Invalid header format. Use KEY=VALUE".into(),
                        )
                    })?;
                    map.insert(k.trim().to_string(), v.trim().to_string());
                }
                Ok(Some(map))
            }
        }
    }

    /// Parse extra body JSON.
    pub fn parse_extra_body(&self) -> crate::error::Result<Option<serde_json::Value>> {
        match &self.extra_body {
            None => Ok(None),
            Some(s) => {
                let v: serde_json::Value = serde_json::from_str(s).map_err(|e| {
                    crate::error::BenchError::Config(format!("Invalid --extra-body JSON: {e}"))
                })?;
                Ok(Some(v))
            }
        }
    }

    /// Generate the request ID prefix (auto-generate if not provided).
    pub fn get_request_id_prefix(&self) -> String {
        self.request_id_prefix
            .clone()
            .unwrap_or_else(|| format!("bench-{}-", &uuid::Uuid::new_v4().to_string()[..8]))
    }

    /// Resolve input/output lengths, applying --input-len/--output-len overrides.
    pub fn resolved_random_input_len(&self) -> usize {
        self.input_len.unwrap_or(self.random_input_len)
    }

    pub fn resolved_random_output_len(&self) -> usize {
        self.output_len.unwrap_or(self.random_output_len)
    }

    pub fn resolved_per_turn_input_len(&self) -> usize {
        if self.per_turn_input_len > 0 {
            self.per_turn_input_len
        } else {
            self.resolved_random_input_len()
        }
    }
}
