// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Startup processor configuration.

use std::collections::HashMap;

use llm_multimodal::{Modality, PreProcessorConfig};
use serde_json::Value;

use crate::error::{Result, bail_multimodal, multimodal};

const SCOPES: [&str; 4] = [
    "images_kwargs",
    "videos_kwargs",
    "audio_kwargs",
    "text_kwargs",
];

/// Resolve shared options and modality-specific overrides at model load.
pub(super) fn resolve(kwargs: &HashMap<String, Value>) -> Result<HashMap<Modality, Value>> {
    if kwargs.get("text_kwargs").is_some_and(|value| {
        !value.is_null() && !value.as_object().is_some_and(|values| values.is_empty())
    }) {
        bail_multimodal!("mm_processor_kwargs.text_kwargs is not supported");
    }
    let flat = kwargs
        .iter()
        .filter(|(key, _)| !SCOPES.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    let mut resolved = HashMap::new();
    for (modality, scope) in [
        (Modality::Image, "images_kwargs"),
        (Modality::Video, "videos_kwargs"),
        (Modality::Audio, "audio_kwargs"),
    ] {
        let mut values = Value::Object(flat.clone());
        if let Some(scoped) = kwargs.get(scope).filter(|value| !value.is_null()) {
            if !scoped.is_object() {
                bail_multimodal!("mm_processor_kwargs.{scope} must be an object");
            }
            merge(&mut values, scoped);
        }
        if values.as_object().is_some_and(|values| !values.is_empty()) {
            resolved.insert(modality, values);
        }
    }
    Ok(resolved)
}

/// Merge startup options before the model processor interprets its config.
pub(super) fn apply(mut config: Value, overrides: Option<&Value>) -> Result<PreProcessorConfig> {
    if let Some(overrides) = overrides {
        merge(&mut config, overrides);
    }
    PreProcessorConfig::from_value(config)
        .map_err(|error| multimodal!("failed to parse processor config: {error}"))
}

/// Recursively merge mappings; other values replace the checkpoint value.
fn merge(config: &mut Value, overrides: &Value) {
    if let (Value::Object(config), Value::Object(overrides)) = (&mut *config, overrides) {
        for (key, value) in overrides {
            merge(config.entry(key.clone()).or_insert(Value::Null), value);
        }
    } else {
        *config = overrides.clone();
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;

    use super::*;

    #[test]
    fn model_specific_values_survive_scoped_config_merging() {
        let kwargs = resolve(
            &serde_json::from_value(json!({
                "custom_options": {"width": 512, "height": 256},
                "images_kwargs": {"custom_options": {"width": 1024}},
                "audio_kwargs": {"max_samples": 16000},
            }))
            .unwrap(),
        )
        .unwrap();
        let mut configs = serde_json::Map::new();
        for modality in [Modality::Image, Modality::Video, Modality::Audio] {
            let config = apply(
                json!({"custom_options": {"checkpoint": true}}),
                kwargs.get(&modality),
            )
            .unwrap();
            configs.insert(
                modality.to_string(),
                serde_json::to_value(config.extra).unwrap(),
            );
        }
        let mut configs = Value::Object(configs);
        configs.sort_all_objects();
        expect![[r#"
            {
              "audio": {
                "custom_options": {
                  "checkpoint": true,
                  "height": 256,
                  "width": 512
                },
                "max_samples": 16000
              },
              "image": {
                "custom_options": {
                  "checkpoint": true,
                  "height": 256,
                  "width": 1024
                }
              },
              "video": {
                "custom_options": {
                  "checkpoint": true,
                  "height": 256,
                  "width": 512
                }
              }
            }"#]]
        .assert_eq(&serde_json::to_string_pretty(&configs).unwrap());
    }
}
