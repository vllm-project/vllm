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
}

impl Config {
    /// Render the HTTP bind address as `host:port`.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
