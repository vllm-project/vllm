use std::time::Duration;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Minimal OpenAI-compatible server for vLLM-frontend-rs.")]
pub struct Args {
    #[arg(long)]
    pub handshake_address: String,
    #[arg(long)]
    pub model: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub bind_host: String,
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    pub engine_local_host: String,
    #[arg(long, default_value_t = 30)]
    pub ready_timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub handshake_address: String,
    pub model: String,
    pub bind_host: String,
    pub port: u16,
    pub engine_local_host: String,
    pub ready_timeout: Duration,
}

impl Config {
    pub fn parse() -> Self {
        Self::from(Args::parse())
    }

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
