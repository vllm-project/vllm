// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// An object-shaped JSON Merge Patch (RFC 7396) for the model's HF config.
/// Objects merge recursively, arrays/scalars replace, and null removes a key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct HfOverrides(pub Map<String, Value>);

impl HfOverrides {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Apply the patch to an in-memory config, preserving the source file.
    pub fn apply(&self, config: &mut Value) {
        merge_object(config, &self.0);
    }
}

/// Apply RFC 7396 object-member updates at every nesting level.
fn merge_object(target: &mut Value, patch: &Map<String, Value>) {
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    let target = target.as_object_mut().unwrap();
    for (key, value) in patch {
        match value {
            Value::Null => {
                target.remove(key);
            }
            Value::Object(object) => {
                merge_object(target.entry(key.clone()).or_insert(Value::Null), object)
            }
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::HfOverrides;

    #[test]
    fn merge_patch_preserves_siblings_replaces_arrays_and_deletes_null_members() {
        let mut config = json!({
            "rope_parameters": {"rope_type": "yarn", "factor": 2},
            "text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 2}},
            "architectures": ["Old", "Other"],
            "sliding_window": 4096
        });
        let patch: HfOverrides = serde_json::from_value(json!({
            "rope_parameters": {"factor": 4},
            "text_config": {"rope_parameters": {"factor": 4}},
            "architectures": ["New"],
            "sliding_window": null,
            "missing": null
        }))
        .unwrap();
        patch.apply(&mut config);
        assert_eq!(
            config,
            json!({
                "rope_parameters": {"rope_type": "yarn", "factor": 4},
                "text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4}},
                "architectures": ["New"]
            })
        );
    }

    #[test]
    fn merge_patch_creates_objects_and_preserves_nulls_inside_replacement_arrays() {
        let mut config = json!({"a": 1, "b": [1, 2]});
        let patch: HfOverrides = serde_json::from_value(json!({
            "a": {"nested": {"removed": null}},
            "b": {},
            "c": [null, {"literal": null}]
        }))
        .unwrap();
        patch.apply(&mut config);
        assert_eq!(
            config,
            json!({"a": {"nested": {}}, "b": {}, "c": [null, {"literal": null}]})
        );
    }
}
