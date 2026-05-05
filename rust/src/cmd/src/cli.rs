//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

mod serve_validate;
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
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};

use crate::cli::unsupported::UnsupportedArgs;
use crate::managed_engine::ManagedEngineConfig;

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
        let repartitioned_args = serve_validate::repartition_serve_args(&args)?;
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
                println!("Passthrough Python args: {}", serve.python_args.join(" "));
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
    /// * "openai" renders content as a list of dictionaries, similar to OpenAI
    ///   schema. Example: `[{"type": "text", "text": "Hello world!"}]`
    #[arg(long, default_value_t)]
    #[serde(default)]
    pub chat_template_content_format: ChatTemplateContentFormatOption,

    /// Log a summary line for each completed request, including prompt/output
    /// token counts and finish reason.
    #[arg(long)]
    #[serde(default)]
    pub enable_log_requests: bool,

    /// Disable periodic logging of engine statistics (throughput, queue depth,
    /// cache usage).
    #[arg(long)]
    #[serde(default)]
    pub disable_log_stats: bool,

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
    /// inherits an already open HTTP listener from the supervisor process.
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
            listener_mode: HttpListenerMode::InheritedFd { fd: listen_fd },
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            renderer: self.renderer,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
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
            listener_mode,
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            renderer: self.renderer,
            chat_template: self.chat_template,
            default_chat_template_kwargs: self.default_chat_template_kwargs,
            chat_template_content_format: self.chat_template_content_format,
            enable_log_requests: self.enable_log_requests,
            disable_log_stats: self.disable_log_stats,
            grpc_port: self.grpc_port,
            shutdown_timeout,
        }
    }
}

fn default_engine_ready_timeout_secs() -> u64 {
    600
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
    /// Python executable used to launch the managed headless vLLM engine.
    #[arg(long, env = "VLLM_RS_PYTHON", default_value = "python3")]
    pub python: String,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Unix domain socket path. If set, host and port arguments are ignored.
    #[arg(long)]
    pub uds: Option<String>,
    /// Host/IP used both for the managed-engine handshake endpoint and the
    /// frontend-advertised input/output ZMQ socket addresses.
    #[arg(
        long = "data-parallel-address",
        visible_alias = "handshake-host",
        default_value = "127.0.0.1"
    )]
    pub handshake_host: String,
    /// Optional TCP port for the managed-engine handshake / data-parallel RPC
    /// endpoint.
    ///
    /// When omitted, the CLI allocates an ephemeral port automatically.
    #[arg(
        long = "data-parallel-rpc-port",
        visible_alias = "handshake-port",
        value_parser = clap::value_parser!(u16).range(1..)
    )]
    pub handshake_port: Option<u16>,
    /// Number of data parallel replicas across the whole deployment.
    #[arg(long, default_value_t = 1)]
    pub data_parallel_size: usize,
    /// Number of data parallel replicas to run on this node.
    #[arg(long)]
    pub data_parallel_size_local: Option<usize>,

    /// Flag to print debug information about CLI argument parsing and exit.
    #[educe(Debug(ignore))]
    #[arg(long, hide = true, env = "VLLM_RS_DEBUG_CLI")]
    pub debug_cli: bool,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,

    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main
    /// serve ...`.
    ///
    /// Arguments after an explicit `--` are forwarded verbatim. Before `--`,
    /// `vllm-rs serve` automatically keeps recognized frontend options on
    /// the Rust side and forwards everything else to Python.
    #[arg(
        last = true,
        allow_hyphen_values = true,
        help_heading = "Passthrough arguments"
    )]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the handshake address shared by the Rust frontend and managed
    /// Python engine.
    pub fn handshake_address(&self, handshake_port: u16) -> String {
        format!("tcp://{}:{}", self.handshake_host, handshake_port)
    }

    /// Build the OpenAI-server runtime config used after the managed Python
    /// engine starts.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        // Prefer IPC sockets for local engine input/output.
        let (local_input_address, local_output_address) =
            self.frontend_local_only().then(frontend_ipc_addresses).unzip();
        let listener_mode = match &self.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => HttpListenerMode::BindTcp {
                host: self.host.clone(),
                port: self.port,
            },
        };

        self.runtime.clone().into_managed_config(
            listener_mode,
            handshake_address,
            self.handshake_host.clone(),
            self.data_parallel_size,
            local_input_address,
            local_output_address,
        )
    }

    /// Build the managed Python-engine spawn configuration for one resolved
    /// handshake port.
    pub fn into_managed_engine_config(self, handshake_port: u16) -> ManagedEngineConfig {
        let mut python_args = self.python_args;
        // Manually forward some args to the Python engine.
        if let Some(max_model_len) = self.runtime.max_model_len {
            python_args.push("--max-model-len".to_string());
            python_args.push(max_model_len.to_string());
        }
        if let Some(data_parallel_size_local) = self.data_parallel_size_local {
            python_args.push("--data-parallel-size-local".to_string());
            python_args.push(data_parallel_size_local.to_string());
        }

        ManagedEngineConfig {
            python: self.python,
            model: self.runtime.model,
            handshake_host: self.handshake_host,
            handshake_port,
            data_parallel_size: self.data_parallel_size,
            python_args,
        }
    }

    fn local_engine_count(&self) -> usize {
        self.data_parallel_size_local.unwrap_or(self.data_parallel_size)
    }

    /// Return whether the managed Rust frontend only needs to communicate with
    /// colocated engines.
    fn frontend_local_only(&self) -> bool {
        self.data_parallel_size_local != Some(0)
            && self.local_engine_count() == self.data_parallel_size
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
