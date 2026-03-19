//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use vllm_openai_server::Config;

use crate::managed_engine::{MANAGED_ENGINE_HANDSHAKE_HOST, ManagedEngineConfig};

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

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Command {
    /// Run the Rust OpenAI frontend against an already running headless Python engine.
    Frontend(FrontendArgs),
    /// Launch a managed Python headless engine, then run the Rust OpenAI frontend.
    Serve(ServeArgs),
}

/// Runtime arguments shared by the external-engine and managed-engine paths.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct FrontendRuntimeArgs {
    /// Hugging Face model identifier used both for backend loading and public model ID.
    pub model: String,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Local host/IP announced to the headless engine for ZMQ sockets.
    #[arg(long, env = "VLLM_HOST_IP", default_value = "127.0.0.1")]
    pub engine_local_host: String,
    /// Maximum time to wait for the engine handshake to complete.
    #[arg(long, env = "VLLM_ENGINE_READY_TIMEOUT_S", default_value_t = 30)]
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
}

impl FrontendRuntimeArgs {
    /// Build one OpenAI-server runtime config for the resolved handshake address.
    fn into_config(self, handshake_address: String) -> Config {
        Config {
            handshake_address,
            model: self.model,
            host: self.host,
            port: self.port,
            engine_local_host: self.engine_local_host,
            ready_timeout: Duration::from_secs(self.ready_timeout_secs),
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            max_model_len: self.max_model_len,
        }
    }
}

/// Arguments for connecting the Rust frontend to an already running headless engine.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct FrontendArgs {
    #[command(flatten)]
    pub runtime: FrontendRuntimeArgs,
    /// Headless vLLM engine handshake endpoint, for example `tcp://127.0.0.1:62100`.
    #[arg(long)]
    pub handshake_address: String,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime.into_config(self.handshake_address)
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the user.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct ServeArgs {
    #[command(flatten)]
    pub runtime: FrontendRuntimeArgs,
    /// Python executable used to launch the managed headless vLLM engine.
    #[arg(long, env = "VLLM_RS_PYTHON", default_value = "python3")]
    pub python: String,
    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main serve ...`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config that should connect to the managed engine.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        self.runtime.clone().into_config(handshake_address)
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
            handshake_host: MANAGED_ENGINE_HANDSHAKE_HOST.to_string(),
            handshake_port,
            python_args,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;
    use expect_test::expect;

    use super::Cli;

    #[test]
    fn serve_args_forward_python_flags_without_separator() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "../vllm/.venv/bin/python",
            "--max-model-len",
            "512",
            "--dtype",
            "float16",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Serve(
                    ServeArgs {
                        runtime: FrontendRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            engine_local_host: "127.0.0.1",
                            ready_timeout_secs: 30,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: Some(
                                512,
                            ),
                        },
                        python: "../vllm/.venv/bin/python",
                        python_args: [
                            "--dtype",
                            "float16",
                        ],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }
}
