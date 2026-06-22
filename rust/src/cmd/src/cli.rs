//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

mod unsupported;

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use educe::Educe;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use uuid::Uuid;
use vllm_engine_core_client::TransportMode;
use vllm_managed_engine::ManagedEngineConfig;
use vllm_managed_engine::cli::{ManagedEngineArgs, repartition_managed_engine_args};
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};

use crate::cli::unsupported::UnsupportedArgs;

/// Top-level parser for the `vllm-rs` binary.
#[derive(Debug, Parser)]
#[command(
    name = "vllm-rs",
    about = "Rust frontend and managed-engine CLI for vLLM."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn parse() -> Self {
        Self::try_parse_from(std::env::args_os()).unwrap_or_else(|error| error.exit())
    }

    pub fn try_parse_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let args: Vec<OsString> = itr.into_iter().map(Into::into).collect();
        let repartitioned_args = repartition_managed_engine_args::<Self>(&args, Some("serve"))?;
        <Self as Parser>::try_parse_from(&repartitioned_args).inspect(|cli| {
            if let Command::Serve(serve) = &cli.command
                && serve.debug_cli
            {
                println!(
                    "Original CLI args: {}\n",
                    args.join(OsStr::new(" ")).display()
                );
                println!(
                    "Repartitioned CLI args: {}\n",
                    repartitioned_args.join(OsStr::new(" ")).display()
                );
                println!(
                    "Passthrough Python args: {}",
                    serve.managed_engine.python_args.join(" ")
                );
                std::process::exit(0);
            }
        })
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Command {
    /// Run the Rust OpenAI frontend as a Python-supervised worker.
    Frontend(FrontendArgs),
    /// Launch a managed Python headless engine, then run the Rust OpenAI
    /// frontend.
    Serve(ServeArgs),
}

/// Runtime arguments shared by the external-engine and managed-engine paths.
#[derive(Educe, Clone, Args, PartialEq, Eq, Deserialize)]
#[educe(Debug)]
pub struct SharedRuntimeArgs {
    #[serde(rename = "model_tag")]
    /// Model identifier or local model directory used for backend loading and
    /// public model ID.
    pub model: String,

    /// Maximum time to wait for the expected engines to register on the
    /// frontend transport.
    #[arg(
        long = "engine-ready-timeout-secs",
        env = "VLLM_ENGINE_READY_TIMEOUT_S",
        default_value_t = default_engine_ready_timeout_secs()
    )]
    #[serde(default = "default_engine_ready_timeout_secs")]
    pub engine_ready_timeout_secs: u64,

    /// Select the tool call parser depending on the model that you're using.
    /// Use `auto` to infer from the model or `none` to disable parsing.
    #[arg(long, default_value_t)]
    #[serde(default)]
    pub tool_call_parser: ParserSelection,
    /// Select the reasoning parser depending on the model that you're using.
    /// Use `auto` to infer from the model or `none` to disable parsing.
    #[arg(long, default_value_t)]
    #[serde(default)]
    pub reasoning_parser: ParserSelection,
    /// Select the chat renderer implementation.
    #[arg(long = "tokenizer-mode", default_value_t)]
    #[serde(default, rename = "tokenizer_mode")]
    pub renderer: RendererSelection,
    /// Disable multimodal inputs and treat the model as language-only.
    #[arg(long)]
    #[serde(default)]
    pub language_model_only: bool,
    /// Override the maximum model context length. When set, the frontend uses
    /// this value instead of the model's `max_position_embeddings` from
    /// `config.json`.
    #[arg(long)]
    pub max_model_len: Option<u32>,
    /// TCP port for the gRPC Generate service. When not set, no gRPC server is
    /// started.
    #[arg(long)]
    #[serde(default)]
    pub grpc_port: Option<u16>,
    /// Maximum time to wait for active requests to drain during shutdown.
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub shutdown_timeout: u64,

    /// The file path to the chat template, or the template in single-line form
    /// for the specified model.
    #[arg(long)]
    #[serde(default)]
    pub chat_template: Option<String>,

    /// Default keyword arguments to pass to the chat template renderer.
    ///
    /// These will be merged with request-level chat_template_kwargs, with
    /// request values taking precedence. Useful for setting default
    /// behavior for reasoning models.
    ///
    /// Example: `{"enable_thinking": false}` to disable thinking mode by
    /// default for Qwen3/DeepSeek models.
    #[arg(long, value_parser = parse_json::<HashMap<String, Value>>, value_name = "JSON")]
    #[serde(default)]
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,

    /// The format to render message content within a chat template.
    ///
    /// * "auto" detects the format from the template
    /// * "string" renders content as a string. Example: `"Hello World"`
    /// * "openai" renders content as a list of dictionaries, similar to OpenAI schema. Example:
    ///   `[{"type": "text", "text": "Hello world!"}]`
    #[arg(long, default_value_t)]
    #[serde(default)]
    pub chat_template_content_format: ChatTemplateContentFormatOption,

    /// Log a summary line for each completed request, including prompt/output
    /// token counts and finish reason.
    #[arg(long)]
    #[serde(default)]
    pub enable_log_requests: bool,

    /// If specified, API server will add X-Request-Id header to responses.
    #[arg(
        long,
        default_missing_value = "true",
        num_args = 0..=1
    )]
    #[serde(default)]
    pub enable_request_id_headers: bool,

    /// Disable periodic logging of engine statistics (throughput, queue depth,
    /// cache usage).
    #[arg(long)]
    #[serde(default)]
    pub disable_log_stats: bool,

    /// The model name(s) used in the API. If multiple names are provided, the
    /// server will respond to any of the provided names. The model name in the
    /// model field of a response will be the first name in this list. If not
    /// specified, the model name will be the same as the `--model` argument.
    /// Noted that this name(s) will also be used in `model_name` tag
    /// content of prometheus metrics, if multiple names provided, metrics
    /// tag will take the first one.
    #[arg(long, num_args = 0..)]
    #[serde(default)]
    pub served_model_name: Vec<String>,

    /// HTTP bind host for the OpenAI-compatible server (passed to frontend in bootstrap mode).
    #[arg(long, default_value = "127.0.0.1")]
    #[serde(default = "default_frontend_host", alias = "host")]
    pub frontend_host: String,

    /// HTTP bind port for the OpenAI-compatible server (passed to frontend in bootstrap mode).
    #[arg(long, default_value_t = 8000)]
    #[serde(default = "default_frontend_port", alias = "port")]
    pub frontend_port: u16,

    /// Path to TLS certificate file (PEM format). If set, enables TLS on the HTTP server.
    #[arg(long)]
    #[serde(default, alias = "ssl_certfile")]
    pub ssl_cert_file: Option<String>,

    /// Path to TLS private key file (PEM format). Required if ssl_cert_file is set.
    #[arg(long)]
    #[serde(default, alias = "ssl_keyfile")]
    pub ssl_key_file: Option<String>,

    /// Path to CA certificates file (PEM format) for client certificate verification.
    #[arg(long)]
    #[serde(default, alias = "ssl_ca_certs")]
    pub ssl_ca_certs: Option<String>,

    /// Client certificate requirement level: 0=CERT_NONE (no client auth), 2=CERT_REQUIRED (mandatory).
    /// 
    /// Note: CERT_OPTIONAL (value 1) is NOT SUPPORTED due to rustls v0.23 limitations.
    /// If ssl_cert_reqs=1 is specified, the server will log a warning and fall back to CERT_NONE.
    /// 
    /// - 0 (CERT_NONE): Client certificates are not required. Server does not request or verify client certs.
    /// - 2 (CERT_REQUIRED): Client certificates are mandatory. Clients must provide valid certificates
    ///   signed by the CA specified in ssl_ca_certs. Connections without valid certs will be rejected.
    /// - 1 (CERT_OPTIONAL): NOT SUPPORTED - will fallback to CERT_NONE with warning.
    ///   Rustls v0.23 has no TLS-level support for optional client authentication.
    #[arg(long, default_value = "0")]
    #[serde(default)]
    pub ssl_cert_reqs: u32,

    /// SSL cipher suites (colon-separated). Rustls only supports modern secure cipher suites
    /// and does not allow weak or legacy ciphers (by design).
    /// 
    /// IMPORTANT: This parameter is ACCEPTED FOR VALIDATION AND LOGGING ONLY.
    /// It does NOT restrict which ciphers the server actually accepts - rustls always uses its
    /// hardcoded secure defaults and provides no API to configure or restrict cipher suites.
    /// The server will always offer all rustls default ciphers regardless of this setting.
    /// 
    /// This parameter exists purely for CLI compatibility with Python/Uvicorn.
    /// Supported formats:
    /// - OpenSSL names: 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256'
    /// - Rustls internal names: 'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384'
    #[arg(long)]
    #[serde(default, alias = "ssl_ciphers")]
    pub ssl_ciphers: Option<String>,

    /// Unsupported Python vLLM frontend arguments recognized but not yet
    /// implemented in Rust.
    #[educe(Debug(ignore))]
    #[command(flatten)]
    #[serde(default, flatten)]
    pub unsupported: UnsupportedArgs,
}

impl SharedRuntimeArgs {
    /// Maximum time to wait for the expected engines to register on the
    /// frontend transport.
    pub fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.engine_ready_timeout_secs)
    }

    /// Maximum time to wait for active requests to drain during shutdown.
    pub fn shutdown_timeout(&self) -> Duration {
        Duration::from_secs(self.shutdown_timeout)
    }

    /// Build the OpenAI-server config for the Python-bootstrap worker contract.
    ///
    /// The resulting config binds the Python-supplied transport addresses and
    /// inherits an already open HTTP listener from the supervisor process,
    /// unless TLS is enabled in which case it binds its own socket.
    fn into_bootstrapped_config(
        self,
        listen_fd: i32,
        input_address: String,
        output_address: String,
        coordinator_address: Option<String>,
        engine_count: usize,
    ) -> Config {
        let ready_timeout = self.ready_timeout();
        let shutdown_timeout = self.shutdown_timeout();

        // Determine listener mode: use TLS if cert/key are provided, otherwise use inherited fd
        let listener_mode = match (&self.ssl_cert_file, &self.ssl_key_file) {
            (Some(cert), Some(key)) => HttpListenerMode::BindTcpTls {
                host: self.frontend_host.clone(),
                port: self.frontend_port,
                cert_path: cert.clone(),
                key_path: key.clone(),
                ca_certs_path: self.ssl_ca_certs.clone(),
                ssl_cert_reqs: self.ssl_cert_reqs,
                ssl_ciphers: self.ssl_ciphers.clone(),
            },
            (Some(_), None) => panic!("ssl_cert_file specified but ssl_key_file is missing"),
            (None, Some(_)) => panic!("ssl_key_file specified but ssl_cert_file is missing"),
            (None, None) => HttpListenerMode::InheritedFd { fd: listen_fd },
        };

        Config {
            transport_mode: TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_count,
                ready_timeout,
            },
            coordinator_mode: match coordinator_address {
                Some(address) => CoordinatorMode::External { address },
                None => CoordinatorMode::None,
            },
            model: self.model,
            served_model_name: self.served_model_name,
            listener_mode,
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            renderer: self.renderer,
            language_model_only: self.language_model_only,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
            enable_request_id_headers: self.enable_request_id_headers,
            disable_log_stats: self.disable_log_stats,
            grpc_port: self.grpc_port,
            shutdown_timeout,
        }
    }

    /// Build the OpenAI-server config for the managed `serve` path that still
    /// owns the startup handshake and binds its own HTTP listener.
    fn into_managed_config(
        self,
        listener_mode: HttpListenerMode,
        handshake_address: String,
        advertised_host: String,
        engine_count: usize,
        local_input_address: Option<String>,
        local_output_address: Option<String>,
    ) -> Config {
        let ready_timeout = self.ready_timeout();
        let shutdown_timeout = self.shutdown_timeout();

        Config {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count,
                ready_timeout,
                local_input_address,
                local_output_address,
            },
            coordinator_mode: CoordinatorMode::MaybeInProc,
            model: self.model,
            served_model_name: self.served_model_name,
            listener_mode,
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            renderer: self.renderer,
            language_model_only: self.language_model_only,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
            enable_request_id_headers: self.enable_request_id_headers,
            disable_log_stats: self.disable_log_stats,
            grpc_port: self.grpc_port,
            shutdown_timeout,
        }
    }
}

fn default_engine_ready_timeout_secs() -> u64 {
    600
}

fn default_frontend_host() -> String {
    "127.0.0.1".to_string()
}

fn default_frontend_port() -> u16 {
    8000
}

fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_str(value).map_err(|e| format!("invalid JSON object: {}", e.as_report()))
}

fn parse_runtime_args_json(value: &str) -> Result<SharedRuntimeArgs, String> {
    let args: SharedRuntimeArgs = serde_json::from_str(value)
        .map_err(|e| format!("invalid JSON arguments: {}", e.as_report()))?;
    args.unsupported.check()?;
    Ok(args)
}

/// Arguments for running the Rust frontend as a Python-bootstrapped worker.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct FrontendArgs {
    /// Inherited listening socket file descriptor passed by the Python
    /// supervisor.
    #[arg(long)]
    pub listen_fd: i32,
    /// Frontend input ROUTER socket address that the Python engines will
    /// connect to.
    #[arg(long)]
    pub input_address: String,
    /// Frontend output PULL socket address that the Python engines will push
    /// responses to.
    #[arg(long)]
    pub output_address: String,
    /// Optional Python-owned frontend-side DP coordinator socket address for
    /// external coordinator mode in the bootstrapped frontend path, i.e.,
    /// `stats_update_address`.
    #[arg(long)]
    pub coordinator_address: Option<String>,
    /// Total number of data-parallel engines expected for this frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,

    /// Shared frontend arguments as one JSON object.
    #[arg(long = "args-json", value_parser = parse_runtime_args_json, value_name = "JSON")]
    pub runtime: SharedRuntimeArgs,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime.into_bootstrapped_config(
            self.listen_fd,
            self.input_address,
            self.output_address,
            self.coordinator_address,
            self.engine_count,
        )
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the
/// user.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
#[command(override_usage = "vllm-rs serve <MODEL> [OPTIONS] [-- <PYTHON_ARGS>...]")]
pub struct ServeArgs {
    /// Only launch the managed Python headless engine and do not start the Rust
    /// frontend.
    #[arg(long)]
    pub headless: bool,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Unix domain socket path. If set, host and port arguments are ignored.
    #[arg(long)]
    pub uds: Option<String>,

    /// Flag to print debug information about CLI argument parsing and exit.
    #[educe(Debug(ignore))]
    #[arg(long, hide = true, env = "VLLM_RS_DEBUG_CLI")]
    pub debug_cli: bool,

    /// Shared frontend arguments (includes TLS arguments: ssl_cert_file, ssl_key_file, ssl_ca_certs, ssl_cert_reqs, ssl_ciphers).
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,

    /// Managed Python headless-engine arguments.
    #[command(flatten)]
    pub managed_engine: ManagedEngineArgs,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config used after the managed Python
    /// engine starts.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        // Prefer IPC sockets for local engine input/output.
        let (local_input_address, local_output_address) =
            self.managed_engine.frontend_local_only().then(frontend_ipc_addresses).unzip();
        let listener_mode = match &self.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => {
                // Check if TLS is requested
                match (&self.runtime.ssl_cert_file, &self.runtime.ssl_key_file) {
                    (Some(cert), Some(key)) => HttpListenerMode::BindTcpTls {
                        host: self.host.clone(),
                        port: self.port,
                        cert_path: cert.clone(),
                        key_path: key.clone(),
                        ca_certs_path: self.runtime.ssl_ca_certs.clone(),
                        ssl_cert_reqs: self.runtime.ssl_cert_reqs,
                        ssl_ciphers: self.runtime.ssl_ciphers.clone(),
                    },
                    (Some(_), None) => panic!("ssl_cert_file specified but ssl_key_file is missing"),
                    (None, Some(_)) => panic!("ssl_key_file specified but ssl_cert_file is missing"),
                    (None, None) => HttpListenerMode::BindTcp {
                        host: self.host.clone(),
                        port: self.port,
                    },
                }
            }
        };

        self.runtime.clone().into_managed_config(
            listener_mode,
            handshake_address,
            self.managed_engine.handshake_host.clone(),
            self.managed_engine.data_parallel_size,
            local_input_address,
            local_output_address,
        )
    }

    /// Build the managed Python-engine spawn configuration with the given
    /// handshake port.
    pub fn to_managed_engine_config(&self, handshake_port: u16) -> ManagedEngineConfig {
        self.managed_engine.clone().into_config(
            self.runtime.model.clone(),
            self.runtime.max_model_len,
            self.runtime.language_model_only,
            handshake_port,
        )
    }
}

/// Allocate fresh IPC endpoints for one managed frontend instance.
fn frontend_ipc_addresses() -> (String, String) {
    let preferred_base_path = std::env::var_os("VLLM_RPC_BASE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let input_name = format!("vllm-rs-i-{}", Uuid::new_v4().simple());
    let output_name = format!("vllm-rs-o-{}", Uuid::new_v4().simple());

    let input = preferred_base_path.join(input_name);
    let output = preferred_base_path.join(output_name);

    (
        format!("ipc://{}", input.to_string_lossy()),
        format!("ipc://{}", output.to_string_lossy()),
    )
}

#[cfg(test)]
mod tests;
