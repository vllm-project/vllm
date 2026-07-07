use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use anyhow::{Result, bail};
use axum::http::{HeaderName, HeaderValue, Method};
use educe::Educe;
use serde::Serialize;
use serde_json::Value;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// Default keep-alive idle timeout (seconds); also the head-read bound
/// when keep-alive is disabled (`0`).
pub const DEFAULT_KEEP_ALIVE_TIMEOUT: Duration = Duration::from_secs(5);

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if
    /// there are multiple engines and the model is MoE.
    MaybeInProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// HTTP/API-server behavior switches that affect route-layer responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub struct ApiServerOptions {
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, include prompt token cache details in response usage.
    pub enable_prompt_tokens_details: bool,
    /// When `true`, set `X-Request-Id` on every HTTP response.
    pub enable_request_id_headers: bool,
}

/// CORS settings mirroring Python's `CORSMiddleware`; the default is permissive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorsConfig {
    /// Allowed origins. `["*"]` allows any origin.
    pub allow_origins: Vec<String>,
    /// Allowed methods. `["*"]` allows the standard method set.
    pub allow_methods: Vec<String>,
    /// Allowed request headers. `["*"]` mirrors the requested headers.
    pub allow_headers: Vec<String>,
    /// Whether to allow credentials (cookies, authorization headers).
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allow_origins: vec!["*".to_string()],
            allow_methods: vec!["*".to_string()],
            allow_headers: vec!["*".to_string()],
            allow_credentials: false,
        }
    }
}

impl CorsConfig {
    /// Validate that non-wildcard values parse into HTTP types, so the CORS
    /// layer can be built infallibly after startup validation has run.
    pub fn validate(&self) -> Result<()> {
        for origin in &self.allow_origins {
            if origin != "*" {
                origin.parse::<HeaderValue>().map_err(|e| {
                    anyhow::anyhow!("invalid --allowed-origins value {origin:?}: {e}")
                })?;
            }
        }
        for method in &self.allow_methods {
            if method != "*" {
                method.parse::<Method>().map_err(|e| {
                    anyhow::anyhow!("invalid --allowed-methods value {method:?}: {e}")
                })?;
            }
        }
        for header in &self.allow_headers {
            if header != "*" {
                header.parse::<HeaderName>().map_err(|e| {
                    anyhow::anyhow!("invalid --allowed-headers value {header:?}: {e}")
                })?;
            }
        }
        Ok(())
    }
}

/// TLS settings mirroring Python's uvicorn `ssl_*` arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TlsConfig {
    /// PEM certificate chain file. Required when TLS is configured; may also
    /// hold the private key (combined PEM) when `key_file` is unset.
    pub cert_file: Option<String>,
    /// PEM private key file. When `None`, the key is read from `cert_file`
    /// (combined PEM).
    pub key_file: Option<String>,
    /// PEM CA bundle used to verify client certificates (mTLS). Required when
    /// `cert_reqs` is non-zero.
    pub ca_certs: Option<String>,
    /// Client-certificate requirement, mirroring Python's `ssl.CERT_*`:
    /// 0 = none, 1 = optional, 2 = required.
    pub cert_reqs: i32,
    /// OpenSSL cipher string for TLS 1.2 and below, mirroring Python's
    /// `ssl.set_ciphers`. `None` keeps the forward-secret AEAD default.
    pub ciphers: Option<String>,
}

impl TlsConfig {
    /// Structurally validate the TLS arguments; the cert/key material is parsed
    /// later, when the OpenSSL context is built.
    pub fn validate(&self) -> Result<()> {
        if self.cert_file.is_none() {
            bail!(
                "--ssl-certfile is required to enable TLS; \
                 --ssl-keyfile/--ssl-ca-certs/--ssl-cert-reqs/--ssl-ciphers \
                 cannot be used without it"
            );
        }
        if !matches!(self.cert_reqs, 0..=2) {
            bail!(
                "--ssl-cert-reqs must be 0 (none), 1 (optional), or 2 (required), got {}",
                self.cert_reqs
            );
        }
        if self.cert_reqs != 0 && self.ca_certs.is_none() {
            bail!(
                "--ssl-ca-certs is required when --ssl-cert-reqs is {} \
                 (client certificate verification)",
                self.cert_reqs
            );
        }
        Ok(())
    }
}

/// OpenTelemetry / observability settings, mirroring Python's
/// `ObservabilityConfig`. A `None` endpoint disables tracing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct ObservabilityConfig {
    /// Target endpoint for the OTLP trace exporter. When `None`, no trace
    /// provider is initialized.
    pub otlp_traces_endpoint: Option<String>,
    /// Opaque `--collect-detailed-traces` value, forwarded verbatim to the
    /// engine (the frontend does not interpret it). Requires an endpoint.
    pub collect_detailed_traces: Option<String>,
}

impl ObservabilityConfig {
    /// Validate the tracing configuration, failing closed.
    pub fn validate(&self) -> Result<()> {
        if self.collect_detailed_traces.is_some() && self.otlp_traces_endpoint.is_none() {
            bail!("--collect-detailed-traces requires --otlp-traces-endpoint to be set");
        }
        if self.otlp_traces_endpoint.is_some() {
            crate::otel::validate_protocol_env()?;
        }
        Ok(())
    }
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Educe, Clone, PartialEq, Eq, Serialize)]
#[educe(Debug)]
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
    /// Disable frontend-side multimodal preprocessing and render the model as
    /// language-only.
    pub language_model_only: bool,
    /// Server-default chat template override, as a file path or inline
    /// template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Optional maximum number of top log probabilities accepted by the
    /// frontend. `None` delegates to the text layer default.
    pub max_logprobs: Option<i32>,
    /// HTTP/API-server behavior switches.
    pub api_server_options: ApiServerOptions,
    /// CORS settings applied to every HTTP response.
    pub cors: CorsConfig,
    /// TLS settings. `None` serves plaintext HTTP; `Some` terminates TLS at the
    /// listener.
    pub tls: Option<TlsConfig>,
    /// API keys accepted as bearer tokens for guarded routes.
    #[serde(skip_serializing)]
    #[educe(Debug(method(fmt_redacted_api_keys)))]
    pub api_keys: Vec<String>,
    /// When `true`, suppress periodic stats logging (throughput, queue depth,
    /// cache usage).
    pub disable_log_stats: bool,
    /// TCP port for the gRPC Generate service. When `None`, no gRPC server is
    /// started.
    pub grpc_port: Option<u16>,
    /// Maximum time to wait for active HTTP/gRPC requests to drain on shutdown.
    pub shutdown_timeout: Duration,
    /// Maximum idle time on a keep-alive HTTP connection before the server
    /// closes it (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, default 5s).
    pub keep_alive_timeout: Duration,
    /// Profiler mode that registers `/start_profile` and `/stop_profile`
    /// routes when present.
    pub profiler: Option<String>,
    /// OpenTelemetry / observability settings (OTLP trace export).
    pub observability: ObservabilityConfig,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine
    /// startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;
        self.cors.validate()?;
        if let Some(tls) = &self.tls {
            tls.validate()?;
        }
        self.observability.validate()?;
        if let Some(max_logprobs) = self.max_logprobs
            && max_logprobs < -1
        {
            bail!(
                "max_logprobs must be non-negative or -1, got {}",
                max_logprobs
            );
        }

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

struct RedactedApiKeys<'a>(&'a [String]);

impl fmt::Debug for RedactedApiKeys<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.debug_list().finish()
        } else {
            write!(f, "[<redacted>; {}]", self.0.len())
        }
    }
}

fn fmt_redacted_api_keys(api_keys: &[String], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(&RedactedApiKeys(api_keys), f)
}
