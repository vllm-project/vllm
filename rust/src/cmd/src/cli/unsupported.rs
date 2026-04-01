#![allow(clippy::doc_lazy_continuation)]

use std::str::FromStr;

use clap::Args;

/// Marker type for frontend-owned `serve` arguments that `vllm-rs` recognizes but does not
/// support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {}

impl FromStr for Unsupported {
    type Err = String;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err("argument is not implemented in Rust frontend yet".to_string())
    }
}

/// Frontend-owned Python `serve` arguments that `vllm-rs` recognizes but does not support yet.
#[derive(Debug, Clone, PartialEq, Eq, Default, Args)]
#[command(next_help_heading = "Options not implemented in Rust frontend yet")]
pub struct UnsupportedArgs {
    #[command(flatten)]
    top_level: TopLevelUnsupportedArgs,
    #[command(flatten)]
    engine: EngineUnsupportedArgs,
    #[command(flatten)]
    server: ServerUnsupportedArgs,
}

/// Frontend-owned Python `vllm serve` top-level arguments that `vllm-rs` recognizes but does not
/// support yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.entrypoints.openai.cli_args.make_arg_parser(...)`
/// - `vllm.entrypoints.cli.serve.ServeSubcommand.subparser_init(...)`
///
/// These are not part of `EngineArgs`, `AsyncEngineArgs`, `BaseFrontendArgs`, or `FrontendArgs`.
/// They live on the `serve` command itself and control managed-engine / multi-process orchestration
/// rather than the shared frontend runtime config.
#[derive(Debug, Clone, PartialEq, Eq, Default, Args)]
pub struct TopLevelUnsupportedArgs {
    /// How many API server processes to run. Defaults to data_parallel_size if not specified.
    #[arg(long)]
    pub api_server_count: Option<Unsupported>,

    /// Read CLI options from a config file. Must be a YAML with the following options:
    /// https://docs.vllm.ai/en/latest/configuration/serve_args.html
    #[arg(long)]
    pub config: Option<Unsupported>,

    /// Launch a gRPC server instead of the HTTP OpenAI-compatible server. Requires:
    /// pip install vllm[grpc].
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub grpc: Option<Unsupported>,
}

/// Frontend-owned Python engine arguments that `vllm-rs` recognizes but does not support yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.engine.arg_utils.EngineArgs.add_cli_args(...)`
/// - `vllm.engine.arg_utils.AsyncEngineArgs.add_cli_args(...)`
///
/// These arguments are declared through the Python engine-args surface, but they are still
/// frontend-owned: the API server / AsyncLLM layer reads them for tokenizer setup, request
/// validation, routing, logging, and other frontend behavior, so Rust must recognize them rather
/// than treating them as pure engine passthrough.
#[derive(Debug, Clone, PartialEq, Eq, Default, Args)]
pub struct EngineUnsupportedArgs {
    /// Name or path of the Hugging Face tokenizer to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    pub tokenizer: Option<Unsupported>,

    /// Tokenizer mode:
    ///
    /// - "auto" will use the tokenizer from `mistral_common` for Mistral models if available,
    ///   otherwise it will use the "hf" tokenizer.
    ///
    /// - "hf" will use the fast tokenizer if available.
    ///
    /// - "slow" will always use the slow tokenizer.
    ///
    /// - "mistral" will always use the tokenizer from `mistral_common`.
    ///
    /// - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.
    ///
    /// - "qwen_vl" will always use the tokenizer from `qwen_vl`.
    ///
    /// - Other custom values can be supported via plugins.
    #[arg(long)]
    pub tokenizer_mode: Option<Unsupported>,

    /// Trust remote code (e.g., from HuggingFace) when downloading the model
    /// and tokenizer.
    #[arg(
        long,
        visible_alias = "no-trust-remote-code",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub trust_remote_code: Option<Unsupported>,

    /// Random seed for reproducibility.
    ///
    /// We must set the global seed because otherwise,
    /// different tensor parallel workers would sample different tokens,
    /// leading to inconsistent results.
    #[arg(long)]
    pub seed: Option<Unsupported>,

    /// Name or path of the Hugging Face config to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    pub hf_config_path: Option<Unsupported>,

    /// Allowing API requests to read local images or videos from directories
    /// specified by the server file system. This is a security risk. Should only
    /// be enabled in trusted environments.
    #[arg(long)]
    pub allowed_local_media_path: Option<Unsupported>,

    /// If set, only media URLs that belong to this domain can be used for
    /// multi-modal inputs.
    #[arg(long)]
    pub allowed_media_domains: Option<Unsupported>,

    /// The specific revision to use for the tokenizer on the Hugging Face Hub.
    /// It can be a branch name, a tag name, or a commit id. If unspecified, will
    /// use the default version.
    #[arg(long)]
    pub tokenizer_revision: Option<Unsupported>,

    /// Maximum number of log probabilities to return when `logprobs` is
    /// specified in `SamplingParams`. The default value comes the default for the
    /// OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    /// vocab_size) logprobs are allowed to be returned and it may cause OOM.
    #[arg(long)]
    pub max_logprobs: Option<Unsupported>,

    /// Indicates the content returned in the logprobs and prompt_logprobs.
    /// Supported mode:
    /// 1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    /// Raw means the values before applying any logit processors, like bad words.
    /// Processed means the values after applying all processors, including
    /// temperature and top_k/top_p.
    #[arg(long)]
    pub logprobs_mode: Option<Unsupported>,

    /// Skip initialization of tokenizer and detokenizer. Expects valid
    /// `prompt_token_ids` and `None` for prompt from the input. The generated
    /// output will contain token ids.
    #[arg(
        long,
        visible_alias = "no-skip-tokenizer-init",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub skip_tokenizer_init: Option<Unsupported>,

    /// If `True`, enables passing text embeddings as inputs via the
    /// `prompt_embeds` key.
    ///
    /// WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    /// Only enable this flag for trusted users!
    #[arg(
        long,
        visible_alias = "no-enable-prompt-embeds",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_prompt_embeds: Option<Unsupported>,

    /// The model name(s) used in the API. If multiple names are provided, the
    /// server will respond to any of the provided names. The model name in the
    /// model field of a response will be the first name in this list. If not
    /// specified, the model name will be the same as the `--model` argument. Noted
    /// that this name(s) will also be used in `model_name` tag content of
    /// prometheus metrics, if multiple names provided, metrics tag will take the
    /// first one.
    #[arg(long)]
    pub served_model_name: Option<Unsupported>,

    /// The token to use as HTTP bearer authorization for remote files. If
    /// `True`, will use the token generated when running `hf auth login`
    /// (stored in `~/.cache/huggingface/token`).
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub hf_token: Option<Unsupported>,

    /// If a dictionary, contains arguments to be forwarded to the Hugging Face
    /// config. If a callable, it is called to update the HuggingFace config.
    #[arg(long)]
    pub hf_overrides: Option<Unsupported>,

    /// The folder path to the generation config. Defaults to `"auto"`, the
    /// generation config will be loaded from model path. If set to `"vllm"`, no
    /// generation config is loaded, vLLM defaults will be used. If set to a folder
    /// path, the generation config will be loaded from the specified folder path.
    /// If `max_new_tokens` is specified in generation config, then it sets a
    /// server-wide limit on the number of output tokens for all requests.
    #[arg(long)]
    pub generation_config: Option<Unsupported>,

    /// IOProcessor plugin name to load at model startup
    #[arg(long)]
    pub io_processor_plugin: Option<Unsupported>,

    /// Path to a dynamically reasoning parser plugin that can be dynamically
    /// loaded and registered.
    #[arg(long)]
    pub reasoning_parser_plugin: Option<Unsupported>,

    /// Rank of the data parallel group.
    #[arg(long, env = "VLLM_DP_RANK")]
    pub data_parallel_rank: Option<Unsupported>,

    /// Whether to use "hybrid" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. Enables running an AsyncLLM
    /// and API server on a "per-node" basis where vLLM load balances
    /// between local data parallel ranks, but an external LB balances
    /// between vLLM nodes/replicas. Set explicitly in conjunction with
    /// --data-parallel-start-rank.
    #[arg(
        long,
        visible_alias = "no-data-parallel-hybrid-lb",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub data_parallel_hybrid_lb: Option<Unsupported>,

    /// Whether to use "external" DP LB mode. Applies only to online serving
    /// and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    /// wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    /// is provided explicitly to vllm serve.
    #[arg(
        long,
        visible_alias = "no-data-parallel-external-lb",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub data_parallel_external_lb: Option<Unsupported>,

    /// This feature is work in progress and no prefill optimization takes place
    /// with this flag enabled currently.
    #[arg(
        long,
        visible_alias = "no-kv-sharing-fast-prefill",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub kv_sharing_fast_prefill: Option<Unsupported>,

    /// The maximum number of input items and options allowed per
    /// prompt for each modality.
    #[arg(long)]
    pub limit_mm_per_prompt: Option<Unsupported>,

    /// Additional args passed to process media inputs, keyed by modalities.
    #[arg(long)]
    pub media_io_kwargs: Option<Unsupported>,

    /// Arguments to be forwarded to the model's processor for multi-modal data,
    /// e.g., image processor.
    #[arg(long)]
    pub mm_processor_kwargs: Option<Unsupported>,

    /// The size (in GiB) of the multi-modal processor cache.
    #[arg(long)]
    pub mm_processor_cache_gb: Option<Unsupported>,

    /// Type of cache to use for the multi-modal preprocessor/mapper.
    #[arg(long)]
    pub mm_processor_cache_type: Option<Unsupported>,

    /// If True, enable handling of LoRA adapters.
    #[arg(
        long,
        visible_alias = "no-enable-lora",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_lora: Option<Unsupported>,

    /// Dictionary mapping specific modalities to LoRA model paths.
    #[arg(long)]
    pub default_mm_loras: Option<Unsupported>,

    /// Target URL to which OpenTelemetry traces will be sent.
    #[arg(long)]
    pub otlp_traces_endpoint: Option<Unsupported>,

    /// It makes sense to set this only if `--otlp-traces-endpoint` is set.
    #[arg(long)]
    pub collect_detailed_traces: Option<Unsupported>,

    /// Maximum number of sequences to be processed in a single iteration.
    #[arg(long)]
    pub max_num_seqs: Option<Unsupported>,

    /// The interval (or buffer size) for streaming in terms of token length.
    #[arg(long)]
    pub stream_interval: Option<Unsupported>,

    /// Structured outputs configuration.
    #[arg(long)]
    pub structured_outputs_config: Option<Unsupported>,

    /// Profiling configuration.
    #[arg(long)]
    pub profiler_config: Option<Unsupported>,

    /// Disable logging statistics.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub disable_log_stats: Option<Unsupported>,

    /// Log aggregate rather than per-engine statistics when using data parallelism.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub aggregate_engine_logging: Option<Unsupported>,

    /// Log requests.
    #[arg(
        long,
        visible_alias = "no-enable-log-requests",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_log_requests: Option<Unsupported>,
}

/// Frontend-owned Python OpenAI server arguments that `vllm-rs` recognizes but does not support
/// yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.entrypoints.openai.cli_args.BaseFrontendArgs`
/// - `vllm.entrypoints.openai.cli_args.FrontendArgs`
///
/// These are not engine args. They belong to the Python OpenAI-compatible frontend / API-server
/// layer itself, for example chat-template configuration, tool/frontend behavior, UDS / TLS /
/// CORS / HTTP server settings, and other northbound server knobs.
#[derive(Debug, Clone, PartialEq, Eq, Default, Args)]
pub struct ServerUnsupportedArgs {
    /// LoRA modules configurations in either 'name=path' format or JSON format
    /// or JSON list format. Example (old format): `'name=path'` Example (new
    /// format): `{"name": "name", "path": "lora_path",
    /// "base_model_name": "id"}`
    #[arg(long)]
    pub lora_modules: Option<Unsupported>,

    /// The file path to the chat template, or the template in single-line form
    /// for the specified model.
    #[arg(long)]
    pub chat_template: Option<Unsupported>,

    /// The format to render message content within a chat template.
    ///
    /// * "string" will render the content as a string. Example: `"Hello World"`
    /// * "openai" will render the content as a list of dictionaries, similar to OpenAI schema.
    ///   Example: `[{"type": "text", "text": "Hello world!"}]`
    #[arg(long)]
    pub chat_template_content_format: Option<Unsupported>,

    /// Whether to trust the chat template provided in the request. If False,
    /// the server will always use the chat template specified by `--chat-template`
    /// or the ones from tokenizer.
    #[arg(
        long,
        visible_alias = "no-trust-request-chat-template",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub trust_request_chat_template: Option<Unsupported>,

    /// Default keyword arguments to pass to the chat template renderer.
    /// These will be merged with request-level chat_template_kwargs,
    /// with request values taking precedence. Useful for setting default
    /// behavior for reasoning models. Example: '{"enable_thinking": false}'
    /// to disable thinking mode by default for Qwen3/DeepSeek models.
    #[arg(long)]
    pub default_chat_template_kwargs: Option<Unsupported>,

    /// The role name to return if `request.add_generation_prompt=true`.
    #[arg(long)]
    pub response_role: Option<Unsupported>,

    /// When `--max-logprobs` is specified, represents single tokens as
    /// strings of the form 'token_id:{token_id}' so that tokens that are not
    /// JSON-encodable can be identified.
    #[arg(
        long,
        visible_alias = "no-return-tokens-as-token-ids",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub return_tokens_as_token_ids: Option<Unsupported>,

    /// If specified, will run the OpenAI frontend server in the same process as
    /// the model serving engine.
    #[arg(
        long,
        visible_alias = "no-disable-frontend-multiprocessing",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub disable_frontend_multiprocessing: Option<Unsupported>,

    /// Enable auto tool choice for supported models. Use `--tool-call-parser`
    /// to specify which parser to use.
    #[arg(
        long,
        visible_alias = "no-enable-auto-tool-choice",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_auto_tool_choice: Option<Unsupported>,

    /// If specified, exclude tool definitions in prompts when
    /// tool_choice='none'.
    #[arg(
        long,
        visible_alias = "no-exclude-tools-when-tool-choice-none",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub exclude_tools_when_tool_choice_none: Option<Unsupported>,

    /// Special the tool parser plugin write to parse the model-generated tool
    /// into OpenAI API format, the name register in this plugin can be used in
    /// `--tool-call-parser`.
    #[arg(long)]
    pub tool_parser_plugin: Option<Unsupported>,

    /// Comma-separated list of host:port pairs (IPv4, IPv6, or hostname).
    /// Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for
    /// built-in demo tools (browser and Python code interpreter). WARNING:
    /// The `demo` Python tool executes model-generated code in Docker without
    /// network isolation by default. See the security guide for more
    /// information.
    #[arg(long)]
    pub tool_server: Option<Unsupported>,

    /// Path to logging config JSON file for both vllm and uvicorn
    #[arg(long, /* env = "VLLM_LOGGING_CONFIG_PATH" */)]
    pub log_config_file: Option<Unsupported>,

    /// Max number of prompt characters or prompt ID numbers being printed in
    /// log. The default of None means unlimited.
    #[arg(long)]
    pub max_log_len: Option<Unsupported>,

    /// If set to True, enable prompt_tokens_details in usage.
    #[arg(
        long,
        visible_alias = "no-enable-prompt-tokens-details",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_prompt_tokens_details: Option<Unsupported>,

    /// If set to True, enable tracking server_load_metrics in the app state.
    #[arg(
        long,
        visible_alias = "no-enable-server-load-tracking",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_server_load_tracking: Option<Unsupported>,

    /// If set to True, including usage on every request.
    #[arg(
        long,
        visible_alias = "no-enable-force-include-usage",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_force_include_usage: Option<Unsupported>,

    /// Enable the `/tokenizer_info` endpoint. May expose chat
    /// templates and other tokenizer configuration.
    #[arg(
        long,
        visible_alias = "no-enable-tokenizer-info-endpoint",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_tokenizer_info_endpoint: Option<Unsupported>,

    /// If set to True, log model outputs (generations).
    /// Requires `--enable-log-requests`. As with `--enable-log-requests`,
    /// information is only logged at INFO level at maximum.
    #[arg(
        long,
        visible_alias = "no-enable-log-outputs",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_log_outputs: Option<Unsupported>,

    /// If set to False, output deltas will not be logged. Relevant only if
    /// --enable-log-outputs is set.
    #[arg(
        long,
        visible_alias = "no-enable-log-deltas",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_log_deltas: Option<Unsupported>,

    /// If set to True, log the stack trace of error responses
    #[arg(
        long,
        // env = "VLLM_SERVER_DEV_MODE",
        visible_alias = "no-log-error-stack",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub log_error_stack: Option<Unsupported>,

    /// If set to True, only enable the Tokens In<>Out endpoint.
    /// This is intended for use in a Disaggregated Everything setup.
    #[arg(
        long,
        visible_alias = "no-tokens-only",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub tokens_only: Option<Unsupported>,

    /// Unix domain socket path. If set, host and port arguments are ignored.
    #[arg(long)]
    pub uds: Option<Unsupported>,

    /// Log level for uvicorn.
    #[arg(long)]
    pub uvicorn_log_level: Option<Unsupported>,

    /// Disable uvicorn access log.
    #[arg(
        long,
        visible_alias = "no-disable-uvicorn-access-log",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub disable_uvicorn_access_log: Option<Unsupported>,

    /// Comma-separated list of endpoint paths to exclude from uvicorn access
    /// logs. This is useful to reduce log noise from high-frequency endpoints
    /// like health checks. Example: "/health,/metrics,/ping".
    /// When set, access logs for requests to these paths will be suppressed
    /// while keeping logs for other endpoints.
    #[arg(long)]
    pub disable_access_log_for_endpoints: Option<Unsupported>,

    /// Allow credentials.
    #[arg(
        long,
        visible_alias = "no-allow-credentials",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub allow_credentials: Option<Unsupported>,

    /// Allowed origins.
    #[arg(long)]
    pub allowed_origins: Option<Unsupported>,

    /// Allowed methods.
    #[arg(long)]
    pub allowed_methods: Option<Unsupported>,

    /// Allowed headers.
    #[arg(long)]
    pub allowed_headers: Option<Unsupported>,

    /// If provided, the server will require one of these keys to be presented in
    /// the header.
    #[arg(long)]
    pub api_key: Option<Unsupported>,

    /// The file path to the SSL key file.
    #[arg(long)]
    pub ssl_keyfile: Option<Unsupported>,

    /// The file path to the SSL cert file.
    #[arg(long)]
    pub ssl_certfile: Option<Unsupported>,

    /// The CA certificates file.
    #[arg(long)]
    pub ssl_ca_certs: Option<Unsupported>,

    /// Refresh SSL Context when SSL certificate files change
    #[arg(
        long,
        visible_alias = "no-enable-ssl-refresh",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_ssl_refresh: Option<Unsupported>,

    /// Whether client certificate is required (see stdlib ssl module's).
    #[arg(long)]
    pub ssl_cert_reqs: Option<Unsupported>,

    /// SSL cipher suites for HTTPS (TLS 1.2 and below only).
    /// Example: 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305'
    #[arg(long)]
    pub ssl_ciphers: Option<Unsupported>,

    /// FastAPI root_path when app is behind a path based routing proxy.
    #[arg(long)]
    pub root_path: Option<Unsupported>,

    /// Additional ASGI middleware to apply to the app. We accept multiple
    /// --middleware arguments. The value should be an import path. If a function
    /// is provided, vLLM will add it to the server using
    /// `@app.middleware('http')`. If a class is provided, vLLM will
    /// add it to the server using `app.add_middleware()`.
    #[arg(long)]
    pub middleware: Option<Unsupported>,

    /// If specified, API server will add X-Request-Id header to responses.
    #[arg(
        long,
        visible_alias = "no-enable-request-id-headers",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_request_id_headers: Option<Unsupported>,

    /// Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint.
    #[arg(
        long,
        visible_alias = "no-disable-fastapi-docs",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub disable_fastapi_docs: Option<Unsupported>,

    /// Maximum size (bytes) of an incomplete HTTP event (header or body) for
    /// h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB).
    #[arg(long)]
    pub h11_max_incomplete_event_size: Option<Unsupported>,

    /// Maximum number of HTTP headers allowed in a request for h11 parser.
    /// Helps mitigate header abuse. Default: 256.
    #[arg(long)]
    pub h11_max_header_count: Option<Unsupported>,

    /// Enable offline FastAPI documentation for air-gapped environments.
    /// Uses vendored static assets bundled with vLLM.
    #[arg(
        long,
        visible_alias = "no-enable-offline-docs",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_offline_docs: Option<Unsupported>,
}
