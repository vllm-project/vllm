// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#![allow(clippy::doc_lazy_continuation)]

use std::fmt::Display;
use std::str::FromStr;

use clap::Args;
use clap::builder::{TypedValueParser, ValueParserFactory};
use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize};

/// Marker type for frontend-owned `serve` arguments that `vllm-rs` recognizes
/// but does not support yet.
///
/// When passed as JSON args, it can be deserialized from any value, and
/// serializes back to the original value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Unsupported(pub serde_json::Value);

impl FromStr for Unsupported {
    type Err = String;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err("argument is not implemented in Rust frontend yet

Remove this argument, or unset the environment variable bound to it, to continue."
            .to_string())
    }
}

/// Marker type for no-op arguments that are accepted by the Rust frontend but
/// have no effect.
///
/// When passed as JSON args, it can be deserialized from any value, but always
/// serializes back to `null`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Noop;

impl<'de> Deserialize<'de> for Noop {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Noop)
    }
}

impl ValueParserFactory for Noop {
    type Parser = NoopValueParser;

    fn value_parser() -> Self::Parser {
        NoopValueParser
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NoopValueParser;

#[track_caller]
fn noop_warn(arg: impl Display) {
    tracing::warn!("argument '{arg}' currently has no effect in Rust frontend, ignoring");
}

impl TypedValueParser for NoopValueParser {
    type Value = Noop;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        _value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        if let Some(arg) = arg {
            noop_warn(arg);
        }
        Ok(Noop)
    }
}

/// Frontend-owned Python `serve` arguments that `vllm-rs` recognizes but does
/// not support yet.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Default, Args, Serialize, Deserialize)]
#[command(next_help_heading = "Options not implemented in Rust frontend yet")]
pub struct UnsupportedArgs {
    #[command(flatten)]
    #[serde(default, flatten)]
    top_level: TopLevelUnsupportedArgs,
    #[command(flatten)]
    #[serde(default, flatten)]
    engine: EngineUnsupportedArgs,
    #[command(flatten)]
    #[serde(default, flatten)]
    server: ServerUnsupportedArgs,
}

impl UnsupportedArgs {
    /// Check whether any unsupported arguments are set, and if so, return an
    /// error listing them. Also warn about any no-op arguments that are set
    /// but will be ignored.
    pub(crate) fn check(&self) -> Result<(), String> {
        let value = serde_json::to_value(self).unwrap();
        let map = value.as_object().unwrap();
        let cmd = Self::augment_args(clap::Command::new("unsupported"));
        let mut unsupported = Vec::new();
        let mut env_bound = false;

        for (key, value) in map {
            if value.is_null() {
                noop_warn(key);
                continue;
            }
            // Python resolves an environment variable into the argument bound
            // to it, so name the variable too.
            let env = (cmd.get_arguments())
                .find(|arg| arg.get_id() == key)
                .and_then(|arg| arg.get_env());
            env_bound |= env.is_some();
            unsupported.push(match env {
                Some(env) => format!("{key} ({})", env.to_string_lossy()),
                None => key.clone(),
            });
        }

        if !unsupported.is_empty() {
            unsupported.sort_unstable();
            let bullets = unsupported.into_iter().map(|key| format!("- {key}")).join("\n");
            let remove = if env_bound {
                "Remove these arguments, or unset the environment variables bound to them, to continue."
            } else {
                "Remove these arguments to continue."
            };
            return Err(format!(
                "The following arguments are not implemented in Rust frontend yet:
{bullets}

{remove}"
            ));
        }

        Ok(())
    }
}

/// Frontend-owned Python `vllm serve` top-level arguments that `vllm-rs`
/// recognizes but does not support yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.entrypoints.common.cli_args.make_arg_parser(...)`
/// - `vllm.entrypoints.cli.serve.ServeSubcommand.subparser_init(...)`
///
/// These are not part of `EngineArgs`, `AsyncEngineArgs`, `BaseFrontendArgs`,
/// or `FrontendArgs`. They live on the `serve` command itself and control
/// managed-engine / multi-process orchestration rather than the shared frontend
/// runtime config.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Default, Args, Serialize, Deserialize)]
pub struct TopLevelUnsupportedArgs {
    /// How many API server processes to run. Defaults to data_parallel_size if
    /// not specified.
    #[arg(long, hide = true)]
    pub api_server_count: Option<Noop>,

    /// Read CLI options from a config file. Must be a YAML with the following
    /// options: https://docs.vllm.ai/en/latest/configuration/serve_args.html
    #[arg(long)]
    pub config: Option<Unsupported>,

    /// Launch a gRPC server instead of the HTTP OpenAI-compatible server.
    /// Requires: pip install vllm[grpc].
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    pub grpc: Option<Unsupported>,
}

/// Frontend-owned Python engine arguments that `vllm-rs` recognizes but does
/// not support yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.engine.arg_utils.EngineArgs.add_cli_args(...)`
/// - `vllm.engine.arg_utils.AsyncEngineArgs.add_cli_args(...)`
///
/// These arguments are declared through the Python engine-args surface, but
/// they are still frontend-owned: the API server / AsyncLLM layer reads them
/// for tokenizer setup, request validation, routing, logging, and other
/// frontend behavior, so Rust must recognize them rather than treating them as
/// pure engine passthrough.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Default, Args, Serialize, Deserialize)]
pub struct EngineUnsupportedArgs {
    /// Allowing API requests to read local images or videos from directories
    /// specified by the server file system. This is a security risk. Should
    /// only be enabled in trusted environments.
    #[arg(long)]
    pub allowed_local_media_path: Option<Unsupported>,

    /// If set, only media URLs that belong to this domain can be used for
    /// multi-modal inputs.
    #[arg(long)]
    pub allowed_media_domains: Option<Unsupported>,

    /// If `True`, enables passing text embeddings as inputs via the
    /// `prompt_embeds` key.
    ///
    /// WARNING: The vLLM engine may crash if incorrect shape of embeddings is
    /// passed. Only enable this flag for trusted users!
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_prompt_embeds: Option<Unsupported>,

    /// IOProcessor plugin name to load at model startup
    #[arg(long)]
    pub io_processor_plugin: Option<Unsupported>,

    /// Number of worker threads in the renderer thread pool. The pool is
    /// consumed by the async renderer path to parallelize tokenization, chat
    /// template rendering, and multimodal preprocessing across concurrent
    /// requests.
    #[arg(long)]
    pub renderer_num_workers: Option<Noop>,

    /// Path to a dynamically reasoning parser plugin that can be dynamically
    /// loaded and registered.
    #[arg(long)]
    pub reasoning_parser_plugin: Option<Unsupported>,

    /// Enable fault tolerance for detailed error recovery, such as scaling
    /// down fault DPEngineCore.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_fault_tolerance: Option<Unsupported>,

    /// Enable elastic expert parallelism with stateless NCCL groups for DP/EP.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_elastic_ep: Option<Unsupported>,

    /// If `True`, LoRA support for the tower (vision encoder) and connector
    /// of multimodal models will be enabled. This is an experimental feature
    /// and currently only supports some MM models such as the Qwen VL series.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_tower_connector_lora: Option<Unsupported>,

    /// Additional args passed to process media inputs, keyed by modalities.
    #[arg(long)]
    pub media_io_kwargs: Option<Unsupported>,

    /// The size (in GiB) of the multi-modal processor cache.
    #[arg(long)]
    pub mm_processor_cache_gb: Option<Unsupported>,

    /// Type of cache to use for the multi-modal preprocessor/mapper.
    #[arg(long)]
    pub mm_processor_cache_type: Option<Unsupported>,

    /// Hash algorithm to use for multi-modal input caching. Use `"sha256"` or
    /// `"sha512"` for FIPS-compliant deployments.
    #[arg(long)]
    pub mm_hasher_algorithm: Option<Unsupported>,

    /// IPC (inter-process communication) method for multimodal tensors.
    /// - "direct_rpc": Use msgspec serialization via RPC
    /// - "torch_shm": Use torch.multiprocessing shared memory for zero-copy IPC
    #[arg(long)]
    pub mm_tensor_ipc: Option<Unsupported>,

    /// Device the HF multi-modal processor runs the image/video transform on.
    /// Convenience for `--mm-processor-kwargs '{"device": ...}'`.
    #[arg(long)]
    pub mm_processor_device: Option<Unsupported>,

    /// Amount of GPU memory (in GiB) sequestered on the engine's device for
    /// GPU-side multimodal work in the API-server (frontend) process, such as
    /// hardware video decoding.
    #[arg(long)]
    pub mm_ipc_gpu_memory_gb: Option<Unsupported>,

    /// Target URL to which OpenTelemetry traces will be sent.
    #[arg(long)]
    pub otlp_traces_endpoint: Option<Unsupported>,

    /// It makes sense to set this only if `--otlp-traces-endpoint` is set.
    #[arg(long)]
    pub collect_detailed_traces: Option<Unsupported>,

    /// Log aggregate rather than per-engine statistics when using data
    /// parallelism.
    ///
    /// Accepted as a no-op: the Rust frontend's periodic stats line always
    /// aggregates across engines.
    #[arg(long, default_missing_value = "true", num_args = 0..=1, hide = true)]
    pub aggregate_engine_logging: Option<Noop>,
}

/// Frontend-owned Python OpenAI server arguments that `vllm-rs` recognizes but
/// does not support yet.
///
/// Source of truth in Python vLLM:
/// - `vllm.entrypoints.common.cli_args.BaseFrontendArgs`
/// - `vllm.entrypoints.common.cli_args.FrontendArgs`
/// - `vllm.envs`, for switches that Python exposes only as environment
///   variables, which `vllm-rs` binds to hidden flags
///
/// These are not engine args. They belong to the Python OpenAI-compatible
/// frontend / API-server layer itself, for example chat-template configuration,
/// tool/frontend behavior, TLS / CORS / HTTP server settings, and other
/// northbound server knobs.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Default, Args, Serialize, Deserialize)]
pub struct ServerUnsupportedArgs {
    /// Whether to trust per-request multimodal kwargs (`mm_processor_kwargs`
    /// and `media_io_kwargs`). If False, the server rejects non-empty values
    /// because they can change multimodal preprocessing resource usage.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub trust_request_mm_kwargs: Option<Unsupported>,

    /// The role name to return if `request.add_generation_prompt=true`.
    #[arg(long)]
    pub response_role: Option<Unsupported>,

    /// Enable auto tool choice for supported models. Use `--tool-call-parser`
    /// to specify which parser to use.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1,
        hide = true
    )]
    pub enable_auto_tool_choice: Option<Noop>,

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

    /// If set to True, enable tracking server_load_metrics in the app state.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_server_load_tracking: Option<Noop>,

    /// Log level for uvicorn.
    ///
    /// Accepted as a no-op: the Rust frontend does not run uvicorn.
    #[arg(long, hide = true)]
    pub uvicorn_log_level: Option<Noop>,

    /// Refresh SSL Context when SSL certificate files change
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_ssl_refresh: Option<Unsupported>,

    /// Additional ASGI middleware to apply to the app. We accept multiple
    /// --middleware arguments. The value should be an import path. If a
    /// function is provided, vLLM will add it to the server using
    /// `@app.middleware('http')`. If a class is provided, vLLM will
    /// add it to the server using `app.add_middleware()`.
    #[arg(long)]
    pub middleware: Option<Unsupported>,

    /// If set to True, log the full body of every API server response,
    /// including streamed SSE payloads, for debugging. CAUTION: logged
    /// responses can include sensitive information and should be avoided
    /// in production.
    #[arg(
        long,
        visible_alias = "no-debug-log-api-server-response",
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub debug_log_api_server_response: Option<Unsupported>,

    /// Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint.
    ///
    /// Accepted as a no-op: the Rust frontend does not serve these docs
    /// endpoints at all.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1,
        hide = true
    )]
    pub disable_fastapi_docs: Option<Noop>,

    /// Maximum size (bytes) of an incomplete HTTP event (header or body) for
    /// h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB).
    #[arg(long)]
    pub h11_max_incomplete_event_size: Option<Unsupported>,

    /// Maximum number of HTTP headers allowed in a request for h11 parser.
    /// Helps mitigate header abuse. Default: 256.
    #[arg(long)]
    pub h11_max_header_count: Option<Unsupported>,

    /// Maximum number of tokens allowed in /tokenize, /detokenize, and
    /// /v1/messages/count_tokens requests. Like Python, it can also be set
    /// through `VLLM_MAX_TOKENIZE_TOKENS`.
    #[arg(long, env = "VLLM_MAX_TOKENIZE_TOKENS")]
    pub max_tokenize_tokens: Option<Unsupported>,

    /// Enable `features.kwargs_refs` in `/inference/v1/generate`, which loads
    /// processed multimodal tensors that clients upload to an S3-compatible
    /// store.
    #[arg(long)]
    pub mm_tensor_refs_config: Option<Unsupported>,

    /// Enable offline FastAPI documentation for air-gapped environments.
    /// Uses vendored static assets bundled with vLLM.
    ///
    /// Accepted as a no-op: the Rust frontend does not serve these docs
    /// endpoints at all.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1,
        hide = true
    )]
    pub enable_offline_docs: Option<Noop>,

    /// If set, run pooling score MaxSim on GPU in the API server process.
    /// Can significantly improve late-interaction scoring performance.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_flash_late_interaction: Option<Noop>,

    /// Enable the `/tokenizer_info` endpoint. May expose chat
    /// templates and other tokenizer configuration.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    pub enable_tokenizer_info_endpoint: Option<Unsupported>,

    /// Expose the Cohere Chat v2 API at `/cohere/v2/chat`. Like Python, it is
    /// set only through `VLLM_ENABLE_COHERE_API`.
    #[arg(long, hide = true, env = "VLLM_ENABLE_COHERE_API")]
    pub enable_cohere_api: Option<Unsupported>,

    /// Keep Responses API responses in memory for retrieval, `background`, and
    /// `previous_response_id`. Like Python, it is set only through
    /// `VLLM_ENABLE_RESPONSES_API_STORE`.
    #[arg(long, hide = true, env = "VLLM_ENABLE_RESPONSES_API_STORE")]
    pub enable_responses_api_store: Option<Unsupported>,
}
