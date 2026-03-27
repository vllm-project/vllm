//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use vllm_openai_server::Config;

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
}

impl FrontendRuntimeArgs {
    /// Build one OpenAI-server runtime config for the resolved handshake address.
    fn into_config(
        self,
        handshake_address: String,
        engine_count: usize,
        advertised_host: String,
    ) -> Config {
        Config {
            handshake_address,
            engine_count,
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
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct FrontendArgs {
    #[command(flatten)]
    pub runtime: FrontendRuntimeArgs,
    /// Host/IP advertised by the frontend to headless engines for shared input/output ZMQ sockets.
    #[arg(long, env = "VLLM_HOST_IP", default_value = "127.0.0.1")]
    pub advertised_host: String,
    /// Headless vLLM engine handshake endpoint, for example `tcp://127.0.0.1:62100`.
    #[arg(long)]
    pub handshake_address: String,
    /// Number of engines expected to connect on the shared handshake socket.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        self.runtime.into_config(
            self.handshake_address,
            self.engine_count,
            self.advertised_host,
        )
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the user.
#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct ServeArgs {
    #[command(flatten)]
    pub runtime: FrontendRuntimeArgs,
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
    /// Total number of data-parallel engines expected to join the shared handshake socket.
    #[arg(
        long = "data-parallel-size",
        visible_alias = "engine-count",
        default_value_t = 1
    )]
    pub engine_count: usize,
    /// Additional arguments forwarded to `python -m vllm.entrypoints.cli.main serve ...`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub python_args: Vec<String>,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config that should connect to the managed engine.
    pub fn to_frontend_config(&self, handshake_address: String) -> Config {
        self.runtime.clone().into_config(
            handshake_address,
            self.engine_count,
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
            engine_count: self.engine_count,
            python_args,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;
    use expect_test::expect;

    use super::{Cli, Command};

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
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: Some(
                                512,
                            ),
                        },
                        headless: false,
                        python: "../vllm/.venv/bin/python",
                        handshake_host: "127.0.0.1",
                        handshake_port: None,
                        engine_count: 1,
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

    #[test]
    fn frontend_args_accept_engine_count() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "frontend",
            "Qwen/Qwen3-0.6B",
            "--handshake-address",
            "tcp://127.0.0.1:62100",
            "--engine-count",
            "2",
        ])
        .unwrap();

        expect![[r#"
            Cli {
                command: Frontend(
                    FrontendArgs {
                        runtime: FrontendRuntimeArgs {
                            model: "Qwen/Qwen3-0.6B",
                            host: "127.0.0.1",
                            port: 8000,
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                        },
                        advertised_host: "127.0.0.1",
                        handshake_address: "tcp://127.0.0.1:62100",
                        engine_count: 2,
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_accept_handshake_aliases() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--python",
            "python3",
            "--handshake-host",
            "10.99.48.128",
            "--handshake-port",
            "13345",
            "--engine-count",
            "4",
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
                            ready_timeout_secs: 300,
                            tool_call_parser: None,
                            reasoning_parser: None,
                            max_model_len: None,
                        },
                        headless: false,
                        python: "python3",
                        handshake_host: "10.99.48.128",
                        handshake_port: Some(
                            13345,
                        ),
                        engine_count: 4,
                        python_args: [],
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&cli);
    }

    #[test]
    fn serve_args_accept_data_parallel_primary_flags() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--data-parallel-address",
            "10.99.48.128",
            "--data-parallel-rpc-port",
            "13345",
            "--data-parallel-size",
            "4",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        assert!(!args.headless);
        assert_eq!(args.handshake_host, "10.99.48.128");
        assert_eq!(args.handshake_port, Some(13345));
        assert_eq!(args.engine_count, 4);
    }

    #[test]
    fn serve_args_accept_headless_mode() {
        let cli =
            Cli::try_parse_from(["vllm-rs", "serve", "Qwen/Qwen3-0.6B", "--headless"]).unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        assert!(args.headless);
    }

    #[test]
    fn serve_frontend_config_uses_dp_address_for_both_handshake_and_transport_host() {
        let cli = Cli::try_parse_from([
            "vllm-rs",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--handshake-host",
            "10.99.48.128",
            "--engine-count",
            "4",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve args");
        };
        let config = args.to_frontend_config("tcp://10.99.48.128:29550".to_string());

        expect![[r#"
            Config {
                handshake_address: "tcp://10.99.48.128:29550",
                engine_count: 4,
                model: "Qwen/Qwen3-0.6B",
                host: "127.0.0.1",
                port: 8000,
                advertised_host: "10.99.48.128",
                ready_timeout: 300s,
                tool_call_parser: None,
                reasoning_parser: None,
                max_model_len: None,
            }
        "#]]
        .assert_debug_eq(&config);
    }
}
