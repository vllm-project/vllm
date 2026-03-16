use std::time::Duration;

use clap::Parser;

/// CLI arguments for the minimal OpenAI-compatible server binary.
#[derive(Debug, Parser)]
#[command(about = "Minimal OpenAI-compatible server for vLLM-frontend-rs.")]
pub struct Args {
    /// Headless vLLM engine handshake endpoint, for example `tcp://127.0.0.1:62100`.
    #[arg(long)]
    pub handshake_address: String,
    /// Hugging Face model identifier used both for backend loading and public model ID.
    #[arg(long)]
    pub model: String,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub bind_host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Local host/IP announced to the headless engine for ZMQ sockets.
    #[arg(long, default_value = "127.0.0.1")]
    pub engine_local_host: String,
    /// Maximum time to wait for the engine handshake to complete.
    #[arg(long, default_value_t = 30)]
    pub ready_timeout_secs: u64,
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Headless vLLM engine handshake endpoint.
    pub handshake_address: String,
    /// Backend model identifier and exposed OpenAI model ID.
    pub model: String,
    /// HTTP bind host.
    pub bind_host: String,
    /// HTTP bind port.
    pub port: u16,
    /// Local host/IP used when connecting to the engine.
    pub engine_local_host: String,
    /// Maximum time to wait for the engine to become ready.
    pub ready_timeout: Duration,
}

impl Config {
    /// Parse one runtime config from CLI arguments.
    pub fn parse() -> Self {
        Self::from(Args::parse())
    }

    /// Render the HTTP bind address as `host:port`.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.bind_host, self.port)
    }
}

impl From<Args> for Config {
    fn from(value: Args) -> Self {
        Self {
            handshake_address: value.handshake_address,
            model: value.model,
            bind_host: value.bind_host,
            port: value.port,
            engine_local_host: value.engine_local_host,
            ready_timeout: Duration::from_secs(value.ready_timeout_secs),
        }
    }
}
