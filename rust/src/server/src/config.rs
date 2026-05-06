use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp { host: String, port: u16 },
    /// Bind a fresh Unix domain listener on the given filesystem path.
    BindUnix { path: String },
    /// Adopt an already-open listening socket inherited from a supervisor
    /// process.
    InheritedFd { fd: i32 },
}

/// Which coordinator implementation should be active when one is present for a
/// frontend client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if
    /// there are mutliple engines and the model is MoE.
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
    /// Backend model identifier used for engine-core loading.
    pub model: String,
    /// Model name(s) exposed to clients via the OpenAI API. When non-empty,
    /// the first entry is used as the primary ID in responses and all entries
    /// are accepted in requests. When empty, falls back to `model`.
    pub served_model_name: Vec<String>,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Tool-call parser selection.
    pub tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    pub reasoning_parser: ParserSelection,
    /// Chat renderer selection.
    pub renderer: RendererSelection,
    /// Server-default chat template override, as a file path or inline
    /// template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, suppress periodic stats logging (throughput, queue depth,
    /// cache usage).
    pub disable_log_stats: bool,
    /// TCP port for the gRPC Generate service. When `None`, no gRPC server is
    /// started.
    pub grpc_port: Option<u16>,
    /// Maximum time to wait for active HTTP/gRPC requests to drain on shutdown.
    pub shutdown_timeout: Duration,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine
    /// startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;

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
