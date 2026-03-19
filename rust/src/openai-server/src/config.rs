use std::time::Duration;

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Headless vLLM engine handshake endpoint.
    pub handshake_address: String,
    /// Backend model identifier and exposed OpenAI model ID.
    pub model: String,
    /// HTTP bind host.
    pub host: String,
    /// HTTP bind port.
    pub port: u16,
    /// Local host/IP used when connecting to the engine.
    pub engine_local_host: String,
    /// Maximum time to wait for the engine to become ready.
    pub ready_timeout: Duration,
    /// Explicit tool call parser name, or `None` for model-based auto-detection.
    pub tool_call_parser: Option<String>,
    /// Explicit reasoning parser name, or `None` for model-based auto-detection.
    pub reasoning_parser: Option<String>,
    /// Override for the maximum model context length. Takes priority over the model's
    /// `max_position_embeddings` from `config.json`.
    pub max_model_len: Option<u32>,
}

impl Config {
    /// Render the HTTP bind address as `host:port`.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
