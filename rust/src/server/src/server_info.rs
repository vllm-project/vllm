use std::collections::BTreeMap;

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
        let vllm_config_json = json!({
            "served_model_name": served_model_names,
        });

        Self {
            vllm_config_text: render_config_text(&vllm_config_json),
            vllm_config_json,
            vllm_env: collect_vllm_env(),
            system_env: collect_system_env(),
        }
    }

    /// Capture the runtime configuration fields available to the Rust frontend.
    pub(crate) fn from_config(config: &Config) -> Self {
        let vllm_config_json = json!({
            "transport_mode": transport_mode_config(&config.transport_mode),
            "coordinator_mode": coordinator_mode_config(&config.coordinator_mode),
            "model": config.model.clone(),
            "served_model_name": config.served_model_name.clone(),
            "listener_mode": listener_mode_config(&config.listener_mode),
            "tool_call_parser": config.tool_call_parser.to_string(),
            "reasoning_parser": config.reasoning_parser.to_string(),
            "renderer": config.renderer.to_string(),
            "chat_template": config.chat_template.clone(),
            "default_chat_template_kwargs": config.default_chat_template_kwargs.clone(),
            "chat_template_content_format": config.chat_template_content_format.to_string(),
            "enable_log_requests": config.enable_log_requests,
            "disable_log_stats": config.disable_log_stats,
            "grpc_port": config.grpc_port,
            "shutdown_timeout_secs": config.shutdown_timeout.as_secs_f64(),
            "engine_count": config.engine_count(),
        });

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

fn transport_mode_config(transport_mode: &TransportMode) -> Value {
    match transport_mode {
        TransportMode::HandshakeOwner {
            handshake_address,
            advertised_host,
            engine_count,
            ready_timeout,
            local_input_address,
            local_output_address,
        } => json!({
            "mode": "handshake_owner",
            "handshake_address": handshake_address,
            "advertised_host": advertised_host,
            "engine_count": engine_count,
            "ready_timeout_secs": ready_timeout.as_secs_f64(),
            "local_input_address": local_input_address,
            "local_output_address": local_output_address,
        }),
        TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_count,
            ready_timeout,
        } => json!({
            "mode": "bootstrapped",
            "input_address": input_address,
            "output_address": output_address,
            "engine_count": engine_count,
            "ready_timeout_secs": ready_timeout.as_secs_f64(),
        }),
    }
}

fn coordinator_mode_config(coordinator_mode: &CoordinatorMode) -> Value {
    match coordinator_mode {
        CoordinatorMode::None => json!("none"),
        CoordinatorMode::MaybeInProc => json!("maybe_in_proc"),
        CoordinatorMode::External { address } => json!({
            "mode": "external",
            "address": address,
        }),
    }
}

fn listener_mode_config(listener_mode: &HttpListenerMode) -> Value {
    match listener_mode {
        HttpListenerMode::BindTcp { host, port } => json!({
            "mode": "bind_tcp",
            "host": host,
            "port": port,
        }),
        HttpListenerMode::BindUnix { path } => json!({
            "mode": "bind_unix",
            "path": path,
        }),
        HttpListenerMode::InheritedFd { fd } => json!({
            "mode": "inherited_fd",
            "fd": fd,
        }),
    }
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
