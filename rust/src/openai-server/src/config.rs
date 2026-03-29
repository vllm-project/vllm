use std::time::Duration;

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Shared handshake endpoint used by one or more headless vLLM engines.
    pub handshake_address: String,
    /// Number of engines expected to connect on the shared handshake socket.
    pub engine_count: usize,
    /// Backend model identifier and exposed OpenAI model ID.
    pub model: String,
    /// HTTP bind host.
    pub host: String,
    /// HTTP bind port.
    pub port: u16,
    /// Host/IP advertised by the frontend to engines for shared input/output sockets.
    pub advertised_host: String,
    /// Enable the in-process wave coordinator for single-frontend deployments.
    pub enable_inproc_coordinator: bool,
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
