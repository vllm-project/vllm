//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

mod serve_validate;
mod unsupported;

use std::ffi::{OsStr, OsString};
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use educe::Educe;
use vllm_engine_core_client::TransportMode;
use vllm_server::{Config, CoordinatorMode, HttpListenerMode};

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
    /// Launch a managed Python headless engine, then run the Rust OpenAI frontend.
    Serve(ServeArgs),
}

/// Runtime arguments shared by the external-engine and managed-engine paths.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct SharedRuntimeArgs {
    /// Hugging Face model identifier used both for backend loading and public model ID.
    pub model: String,
    /// Total number of data-parallel engines expected for this frontend.
    #[arg(long, visible_alias = "data-parallel-size", default_value_t = 1)]
    pub engine_count: usize,
    /// Maximum time to wait for the expected engines to register on the frontend transport.
    #[arg(
        long = "engine-ready-timeout-secs",
        env = "VLLM_ENGINE_READY_TIMEOUT_S",
        default_value_t = 300
    )]
    pub engine_ready_timeout_secs: u64,
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
    /// Maximum time to wait for the expected engines to register on the frontend transport.
    fn ready_timeout(&self) -> Duration {
        Duration::from_secs(self.engine_ready_timeout_secs)
    }

    /// Build the OpenAI-server config for the Python-bootstrap worker contract.
    ///
    /// The resulting config binds the Python-supplied transport addresses and inherits an already
    /// open HTTP listener from the supervisor process.
    fn into_bootstrapped_config(
        self,
        listen_fd: i32,
        input_address: String,
        output_address: String,
    ) -> Config {
        Config {
            transport_mode: TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_count: self.engine_count,
                ready_timeout: self.ready_timeout(),
            },
            // TODO: this might be an external Python process once we support it.
            coordinator_mode: CoordinatorMode::None,
            model: self.model,
            listener_mode: HttpListenerMode::InheritedFd { fd: listen_fd },
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            max_model_len: self.max_model_len,
        }
    }

    /// Build the OpenAI-server config for the managed `serve` path that still owns the startup
    /// handshake and binds its own HTTP listener.
    fn into_managed_config(
        self,
        host: String,
        port: u16,
        handshake_address: String,
        advertised_host: String,
    ) -> Config {
        Config {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count: self.engine_count,
                ready_timeout: self.ready_timeout(),
                local_input_address: None,
                local_output_address: None,
            },
            coordinator_mode: CoordinatorMode::MaybeInProc,
            model: self.model,
            listener_mode: HttpListenerMode::Bind { host, port },
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            max_model_len: self.max_model_len,
        }
    }
}

/// Arguments for running the Rust frontend as a Python-bootstrapped worker.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct FrontendArgs {
    /// Inherited listening socket file descriptor passed by the Python supervisor.
    #[arg(long)]
    pub listen_fd: i32,
    /// Frontend input ROUTER socket address that the Python engines will connect to.
    #[arg(long)]
    pub input_address: String,
    /// Frontend output PULL socket address that the Python engines will push responses to.
    #[arg(long)]
    pub output_address: String,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime.into_bootstrapped_config(
            self.listen_fd,
            self.input_address,
            self.output_address,
        )
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the user.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
#[command(override_usage = "vllm-rs serve <MODEL> [OPTIONS] [-- <PYTHON_ARGS>...]")]
pub struct ServeArgs {
    /// Only launch the managed Python headless engine and do not start the Rust frontend.
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

    /// Flag to print debug information about CLI argument parsing and exit.
    #[educe(Debug(ignore))]
    #[arg(long, hide = true, env = "VLLM_RS_DEBUG_CLI")]
    pub debug_cli: bool,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,

    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main serve ...`.
    ///
    /// Arguments after an explicit `--` are forwarded verbatim. Before `--`, `vllm-rs serve`
    /// automatically keeps recognized frontend options on the Rust side and forwards everything
    /// else to Python.
    #[arg(
        last = true,
        allow_hyphen_values = true,
        help_heading = "Passthrough arguments"
    )]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config used after the managed Python engine starts.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        self.runtime.clone().into_managed_config(
            self.host.clone(),
            self.port,
            handshake_address,
            self.handshake_host.clone(),
        )
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
