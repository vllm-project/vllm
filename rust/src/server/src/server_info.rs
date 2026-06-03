use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use serde_json::{Value, json};
use vllm_engine_core_client::TransportMode;

use crate::config::{Config, CoordinatorMode, HttpListenerMode};

const SENSITIVE_VLLM_ENV_PATTERNS: &[&str] =
    &["KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServerInfoConfigFormat {
    Text,
    Json,
}

/// Snapshot returned by `/server_info`.
#[derive(Debug, Clone)]
pub(crate) struct ServerInfoSnapshot {
    vllm_config_text: String,
    vllm_config_json: Value,
    vllm_env: BTreeMap<String, String>,
    system_env: BTreeMap<String, String>,
}

impl ServerInfoSnapshot {
    pub(crate) fn from_served_model_names(served_model_names: &[String]) -> Self {
        let vllm_config_json = serialize_value(ServedModelNamesConfig { served_model_names });

        Self {
            vllm_config_text: render_config_text(&vllm_config_json),
            vllm_config_json,
            vllm_env: collect_vllm_env(),
            system_env: collect_system_env(),
        }
    }

    /// Capture the runtime configuration fields available to the Rust frontend.
    pub(crate) fn from_config(config: &Config) -> Self {
        let vllm_config_json = serialize_value(RuntimeServerInfoConfig::from(config));

        Self {
            vllm_config_text: render_config_text(&vllm_config_json),
            vllm_config_json,
            vllm_env: collect_vllm_env(),
            system_env: collect_system_env(),
        }
    }

    pub(crate) fn response(&self, config_format: ServerInfoConfigFormat) -> Value {
        let vllm_config = match config_format {
            ServerInfoConfigFormat::Text => Value::String(self.vllm_config_text.clone()),
            ServerInfoConfigFormat::Json => self.vllm_config_json.clone(),
        };

        json!({
            "vllm_config": vllm_config,
            "vllm_env": self.vllm_env.clone(),
            "system_env": self.system_env.clone(),
        })
    }
}

#[derive(Debug, Serialize)]
struct ServedModelNamesConfig<'a> {
    #[serde(rename = "served_model_name")]
    served_model_names: &'a [String],
}

#[derive(Debug, Serialize)]
struct RuntimeServerInfoConfig {
    transport_mode: TransportModeInfo,
    coordinator_mode: CoordinatorModeInfo,
    model: String,
    served_model_name: Vec<String>,
    listener_mode: HttpListenerModeInfo,
    tool_call_parser: String,
    reasoning_parser: String,
    renderer: String,
    chat_template: Option<String>,
    default_chat_template_kwargs: Option<HashMap<String, Value>>,
    chat_template_content_format: String,
    enable_log_requests: bool,
    disable_log_stats: bool,
    grpc_port: Option<u16>,
    shutdown_timeout_secs: f64,
    engine_count: usize,
}

impl From<&Config> for RuntimeServerInfoConfig {
    fn from(config: &Config) -> Self {
        Self {
            transport_mode: TransportModeInfo::from(&config.transport_mode),
            coordinator_mode: CoordinatorModeInfo::from(&config.coordinator_mode),
            model: config.model.clone(),
            served_model_name: config.served_model_name.clone(),
            listener_mode: HttpListenerModeInfo::from(&config.listener_mode),
            tool_call_parser: config.tool_call_parser.to_string(),
            reasoning_parser: config.reasoning_parser.to_string(),
            renderer: config.renderer.to_string(),
            chat_template: config.chat_template.clone(),
            default_chat_template_kwargs: config.default_chat_template_kwargs.clone(),
            chat_template_content_format: config.chat_template_content_format.to_string(),
            enable_log_requests: config.enable_log_requests,
            disable_log_stats: config.disable_log_stats,
            grpc_port: config.grpc_port,
            shutdown_timeout_secs: config.shutdown_timeout.as_secs_f64(),
            engine_count: config.engine_count(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum TransportModeInfo {
    HandshakeOwner {
        handshake_address: String,
        advertised_host: String,
        engine_count: usize,
        ready_timeout_secs: f64,
        local_input_address: Option<String>,
        local_output_address: Option<String>,
    },
    Bootstrapped {
        input_address: String,
        output_address: String,
        engine_count: usize,
        ready_timeout_secs: f64,
    },
}

impl From<&TransportMode> for TransportModeInfo {
    fn from(transport_mode: &TransportMode) -> Self {
        match transport_mode {
            TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count,
                ready_timeout,
                local_input_address,
                local_output_address,
            } => Self::HandshakeOwner {
                handshake_address: handshake_address.clone(),
                advertised_host: advertised_host.clone(),
                engine_count: *engine_count,
                ready_timeout_secs: ready_timeout.as_secs_f64(),
                local_input_address: local_input_address.clone(),
                local_output_address: local_output_address.clone(),
            },
            TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_count,
                ready_timeout,
            } => Self::Bootstrapped {
                input_address: input_address.clone(),
                output_address: output_address.clone(),
                engine_count: *engine_count,
                ready_timeout_secs: ready_timeout.as_secs_f64(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum CoordinatorModeInfo {
    Named(CoordinatorModeName),
    External {
        mode: CoordinatorModeExternalTag,
        address: String,
    },
}

impl From<&CoordinatorMode> for CoordinatorModeInfo {
    fn from(coordinator_mode: &CoordinatorMode) -> Self {
        match coordinator_mode {
            CoordinatorMode::None => Self::Named(CoordinatorModeName::None),
            CoordinatorMode::MaybeInProc => Self::Named(CoordinatorModeName::MaybeInProc),
            CoordinatorMode::External { address } => Self::External {
                mode: CoordinatorModeExternalTag::External,
                address: address.clone(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CoordinatorModeName {
    None,
    MaybeInProc,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CoordinatorModeExternalTag {
    External,
}

#[derive(Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum HttpListenerModeInfo {
    BindTcp { host: String, port: u16 },
    BindUnix { path: String },
    InheritedFd { fd: i32 },
}

impl From<&HttpListenerMode> for HttpListenerModeInfo {
    fn from(listener_mode: &HttpListenerMode) -> Self {
        match listener_mode {
            HttpListenerMode::BindTcp { host, port } => Self::BindTcp {
                host: host.clone(),
                port: *port,
            },
            HttpListenerMode::BindUnix { path } => Self::BindUnix { path: path.clone() },
            HttpListenerMode::InheritedFd { fd } => Self::InheritedFd { fd: *fd },
        }
    }
}

fn serialize_value(value: impl Serialize) -> Value {
    serde_json::to_value(value).expect("server info value must serialize")
}

fn render_config_text(config: &Value) -> String {
    match config {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| format!("{key}={}", render_config_text_value(value)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => render_config_text_value(config),
    }
}

fn render_config_text_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn collect_vllm_env() -> BTreeMap<String, String> {
    std::env::vars().filter(|(key, _)| is_public_vllm_env_key(key)).collect()
}

fn is_public_vllm_env_key(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    key.starts_with("VLLM_")
        && !SENSITIVE_VLLM_ENV_PATTERNS.iter().any(|pattern| key.contains(pattern))
}

fn collect_system_env() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("arch".to_string(), std::env::consts::ARCH.to_string()),
        ("family".to_string(), std::env::consts::FAMILY.to_string()),
        ("os".to_string(), std::env::consts::OS.to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
    use vllm_engine_core_client::TransportMode;

    use super::{ServerInfoConfigFormat, ServerInfoSnapshot, is_public_vllm_env_key};
    use crate::config::{Config, CoordinatorMode, HttpListenerMode};

    #[test]
    fn server_info_env_filter_excludes_sensitive_vllm_keys() {
        for key in [
            "VLLM_API_KEY",
            "VLLM_AUTH_TOKEN",
            "VLLM_SECRET",
            "VLLM_PASSWORD",
            "VLLM_CREDENTIAL_FILE",
            "vllm_token",
        ] {
            assert!(!is_public_vllm_env_key(key), "{key}");
        }
    }

    #[test]
    fn server_info_env_filter_includes_public_vllm_keys() {
        assert!(is_public_vllm_env_key("VLLM_LOGGING_LEVEL"));
        assert!(is_public_vllm_env_key("VLLM_USE_MODELSCOPE"));
        assert!(!is_public_vllm_env_key("OTHER_ENV"));
    }

    #[test]
    fn server_info_config_json_uses_stable_values() {
        let config = Config {
            transport_mode: TransportMode::Bootstrapped {
                input_address: "tcp://127.0.0.1:0".to_string(),
                output_address: "tcp://127.0.0.1:1".to_string(),
                engine_count: 2,
                ready_timeout: Duration::from_secs(5),
            },
            coordinator_mode: CoordinatorMode::MaybeInProc,
            model: "test-model".to_string(),
            served_model_name: vec!["served-model".to_string()],
            listener_mode: HttpListenerMode::BindTcp {
                host: "127.0.0.1".to_string(),
                port: 8000,
            },
            tool_call_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::None,
            renderer: RendererSelection::Hf,
            chat_template: None,
            default_chat_template_kwargs: None,
            chat_template_content_format: ChatTemplateContentFormatOption::OpenAi,
            enable_log_requests: true,
            enable_request_id_headers: false,
            disable_log_stats: false,
            grpc_port: Some(9000),
            shutdown_timeout: Duration::from_secs(10),
        };

        let snapshot = ServerInfoSnapshot::from_config(&config);
        let response = snapshot.response(ServerInfoConfigFormat::Json);
        let vllm_config = &response["vllm_config"];

        assert_eq!(vllm_config["transport_mode"]["mode"], "bootstrapped");
        assert_eq!(vllm_config["transport_mode"]["engine_count"], 2);
        assert_eq!(vllm_config["coordinator_mode"], "maybe_in_proc");
        assert_eq!(vllm_config["listener_mode"]["mode"], "bind_tcp");
        assert_eq!(vllm_config["tool_call_parser"], "auto");
        assert_eq!(vllm_config["reasoning_parser"], "none");
        assert_eq!(vllm_config["renderer"], "hf");
        assert_eq!(vllm_config["chat_template_content_format"], "openai");
    }
}
