use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::config::Config;

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
    /// Capture the runtime configuration fields available to the Rust frontend.
    pub(crate) fn from_config(config: &Config) -> Self {
        let vllm_config_json =
            serde_json::to_value(config).expect("server info value must serialize");

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
    use std::collections::BTreeSet;

    use serde_json::{Value, json};

    use super::{is_public_vllm_env_key, render_config_text};

    #[test]
    fn render_config_text_formats_config_snapshot() {
        let rendered = render_config_text(&json!({
            "model": "test-model",
            "served_model_name": ["served-model"],
            "chat_template": null,
            "enable_log_requests": true,
        }));
        let lines = rendered.lines().collect::<BTreeSet<_>>();

        assert_eq!(
            lines,
            BTreeSet::from([
                "chat_template=None",
                "enable_log_requests=true",
                "model=test-model",
                "served_model_name=[\"served-model\"]",
            ])
        );
        assert_eq!(
            render_config_text(&Value::String("inline".to_string())),
            "inline"
        );
        assert_eq!(render_config_text(&Value::Null), "None");
    }

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
}
