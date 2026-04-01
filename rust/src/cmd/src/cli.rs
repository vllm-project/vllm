//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

mod serve_validate;
mod unsupported;

use std::ffi::OsString;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use educe::Educe;
use vllm_openai_server::Config;

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
        let normalized_args = serve_validate::normalize_python_arg_aliases(&args);
        <Self as Parser>::try_parse_from(normalized_args.clone())
            .map_err(|error| serve_validate::rewrite_unknown_arg_error(&normalized_args, error))
            .and_then(serve_validate::validate_passthrough_args)
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Command {
    /// Run the Rust OpenAI frontend against an already running headless Python engine.
    Frontend(FrontendArgs),
    /// Launch a managed Python headless engine, then run the Rust OpenAI frontend.
    Serve(ServeArgs),
}

/// Runtime arguments shared by the external-engine and managed-engine paths.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct SharedRuntimeArgs {
    /// Hugging Face model identifier used both for backend loading and public model ID.
    pub model: String,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Total number of data-parallel engines expected to join the shared handshake socket.
    #[arg(long, visible_alias = "data-parallel-size", default_value_t = 1)]
    pub engine_count: usize,
    /// Maximum time to wait for the engine handshake to complete.
    #[arg(long, env = "VLLM_ENGINE_READY_TIMEOUT_S", default_value_t = 300)]
    pub ready_timeout_secs: u64,
    /// Select the tool call parser depending on the model that you're using.
    /// When not specified, the parser is auto-detected from the model.
    #[arg(long)]
    pub tool_call_parser: Option<String>,
    /// Select the reasoning parser depending on the model that you're using.
    /// When not specified, the parser is auto-detected from the model.
    #[arg(long)]
    pub reasoning_parser: Option<String>,
    /// Override the maximum model context length. When set, the frontend uses this value
    /// instead of the model's `max_position_embeddings` from `config.json`.
    #[arg(long)]
    pub max_model_len: Option<u32>,
    /// Unsupported Python vLLM frontend arguments recognized but not yet implemented in Rust.
    #[educe(Debug(ignore))]
    #[command(flatten)]
    pub unsupported: UnsupportedArgs,
}

impl SharedRuntimeArgs {
    /// Build one OpenAI-server runtime config for the resolved handshake address.
    fn into_config(self, handshake_address: String, advertised_host: String) -> Config {
        Config {
            handshake_address,
            engine_count: self.engine_count,
            model: self.model,
            host: self.host,
            port: self.port,
            advertised_host,
            ready_timeout: Duration::from_secs(self.ready_timeout_secs),
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            max_model_len: self.max_model_len,
        }
    }
}

/// Arguments for connecting the Rust frontend to an already running headless engine.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct FrontendArgs {
    /// Host/IP advertised by the frontend to headless engines for shared input/output ZMQ sockets.
    #[arg(long, env = "VLLM_HOST_IP", default_value = "127.0.0.1")]
    pub advertised_host: String,
    /// Headless vLLM engine handshake endpoint, for example `tcp://127.0.0.1:62100`.
    #[arg(long)]
    pub handshake_address: String,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime
            .into_config(self.handshake_address, self.advertised_host)
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the user.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct ServeArgs {
    /// Only launch the managed Python headless engine and do not start the Rust frontend.
    #[arg(long)]
    pub headless: bool,
    /// Python executable used to launch the managed headless vLLM engine.
    #[arg(long, env = "VLLM_RS_PYTHON", default_value = "python3")]
    pub python: String,
    /// Host/IP used both for the managed-engine handshake endpoint and the frontend-advertised
    /// input/output ZMQ socket addresses.
    #[arg(
        long = "data-parallel-address",
        visible_alias = "handshake-host",
        default_value = "127.0.0.1"
    )]
    pub handshake_host: String,
    /// Optional TCP port for the managed-engine handshake / data-parallel RPC endpoint.
    ///
    /// When omitted, the CLI allocates an ephemeral port automatically.
    #[arg(
        long = "data-parallel-rpc-port",
        visible_alias = "handshake-port",
        value_parser = clap::value_parser!(u16).range(1..)
    )]
    pub handshake_port: Option<u16>,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,

    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main serve ...`.
    ///
    /// These arguments must be placed after `--` so the Rust frontend can parse its own options
    /// first and then pass the remaining argv through to Python unchanged.
    #[arg(
        last = true,
        allow_hyphen_values = true,
        help_heading = "Passthrough arguments"
    )]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config that should connect to the managed engine.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        self.runtime
            .clone()
            .into_config(handshake_address, self.handshake_host.clone())
    }

    /// Build the managed Python-engine spawn configuration for one resolved handshake port.
    pub fn into_managed_engine_config(self, handshake_port: u16) -> ManagedEngineConfig {
        let mut python_args = self.python_args;
        // Forward `--max-model-len` to the Python engine so both sides agree on the limit.
        if let Some(max_model_len) = self.runtime.max_model_len {
            python_args.push("--max-model-len".to_string());
            python_args.push(max_model_len.to_string());
        }
        ManagedEngineConfig {
            python: self.python,
            model: self.runtime.model,
            handshake_host: self.handshake_host,
            handshake_port,
            engine_count: self.runtime.engine_count,
            python_args,
        }
    }
}

#[cfg(test)]
mod tests;
