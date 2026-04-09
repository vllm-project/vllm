use anyhow::Result;
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    Bind { host: String, port: u16 },
    /// Adopt an already-open listening socket inherited from a supervisor process.
    InheritedFd { fd: i32 },
}

/// Which coordinator implementation should be active when one is present for a frontend client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if there are mutliple
    /// engines and the model is MoE.
    MaybeInProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Requested frontend-side coordinator behavior.
    pub coordinator_mode: CoordinatorMode,
    /// Backend model identifier and exposed OpenAI model ID.
    pub model: String,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Explicit tool call parser name, or `None` for model-based auto-detection.
    pub tool_call_parser: Option<String>,
    /// Explicit reasoning parser name, or `None` for model-based auto-detection.
    pub reasoning_parser: Option<String>,
    /// Override for the maximum model context length. Takes priority over the model's
    /// `max_position_embeddings` from `config.json`.
    pub max_model_len: Option<u32>,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, suppress periodic stats logging (throughput, queue depth, cache usage).
    pub disable_log_stats: bool,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(
            self.tool_call_parser.as_deref(),
            self.reasoning_parser.as_deref(),
        )?;

        Ok(())
    }

    /// Return the number of engines implied by the configured transport mode.
    pub fn engine_count(&self) -> usize {
        match &self.transport_mode {
            TransportMode::HandshakeOwner { engine_count, .. }
            | TransportMode::Bootstrapped { engine_count, .. } => *engine_count,
        }
    }

    /// Resolve the effective coordinator mode.
    pub fn effective_coordinator_mode(
        &self,
        model_is_moe: bool,
    ) -> Option<EngineCoreCoordinatorMode> {
        match &self.coordinator_mode {
            CoordinatorMode::None => None,
            CoordinatorMode::MaybeInProc => {
                if model_is_moe && self.engine_count() > 1 {
                    Some(EngineCoreCoordinatorMode::InProc)
                } else {
                    None
                }
            }
            CoordinatorMode::External { address } => Some(EngineCoreCoordinatorMode::External {
                address: address.clone(),
            }),
        }
    }
}
