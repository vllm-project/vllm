// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One measured adapter transition on the worker. Mirrors `LoRALoadTiming`
/// in vllm/v1/notifications.py.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraLoadTiming {
    pub adapter_name: String,
    /// `load`: read from disk into the CPU cache. `activate`: moved from the
    /// CPU cache into a GPU slot.
    pub transition: String,
    pub seconds: f64,
}

/// The set of loaded LoRA adapters changed, or an adapter transition
/// completed.
///
/// A full snapshot of the worker's adapter caches, so consumers replace their
/// state rather than merge. Python encodes it with `omit_defaults=True`, so
/// every field needs `#[serde(default)]`. Mirrors `LoRALoadEvent` in
/// vllm/v1/notifications.py.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoraLoadEvent {
    /// Adapters activated into GPU slots (sorted).
    #[serde(default)]
    pub gpu_adapters: Vec<String>,
    /// Adapters resident in the CPU cache (sorted, superset of `gpu_adapters`).
    #[serde(default)]
    pub cpu_adapters: Vec<String>,
    /// Adapters pinned in the caches (sorted).
    #[serde(default)]
    pub pinned_adapters: Vec<String>,
    /// Adapter transitions completed since the previous event, in order.
    #[serde(default)]
    pub loads: Vec<LoraLoadTiming>,
}

/// Open escape hatch for out-of-tree producers.
///
/// Plugins namespace under `key`; frontends ignore keys they don't know.
/// Mirrors `CustomNotification` in vllm/v1/notifications.py.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CustomNotification {
    pub key: String,
    #[serde(default)]
    pub payload: BTreeMap<String, rmpv::Value>,
}

/// Engine-level events carried on `EngineCoreOutputs::engine_notifications`.
///
/// Map-encoded with a `"type"` discriminator. Version-lockstep with the
/// engine: an unknown tag is a deployment error and fails the decode.
/// Mirrors vllm/v1/notifications.py.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineNotification {
    LoraLoadEvent(LoraLoadEvent),
    Custom(CustomNotification),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::decode_msgpack;

    fn hex_bytes(hex: &str) -> Vec<u8> {
        hex::decode(hex).unwrap()
    }

    /// Python: `encode(LoRALoadEvent(gpu_adapters=["alpha"],
    /// cpu_adapters=["alpha", "beta"], pinned_adapters=["alpha"]))`
    const PYTHON_LORA_LOAD_EVENT: &str = "84a474797065af6c6f72615f6c6f61645f6576656e74ac6770755f616461707465727391a5616c706861ac6370755f616461707465727392a5616c706861a462657461af70696e6e65645f616461707465727391a5616c706861";

    /// Python: `encode(LoRALoadEvent())`, where `omit_defaults=True` strips
    /// everything but the tag.
    const PYTHON_LORA_LOAD_EVENT_EMPTY: &str = "81a474797065af6c6f72615f6c6f61645f6576656e74";

    #[test]
    fn engine_event_decodes_python_lora_load_event() {
        let event: EngineNotification = decode_msgpack(&hex_bytes(PYTHON_LORA_LOAD_EVENT)).unwrap();
        expect_test::expect![[r#"
            LoraLoadEvent(
                LoraLoadEvent {
                    gpu_adapters: [
                        "alpha",
                    ],
                    cpu_adapters: [
                        "alpha",
                        "beta",
                    ],
                    pinned_adapters: [
                        "alpha",
                    ],
                    loads: [],
                },
            )
        "#]]
        .assert_debug_eq(&event);
    }

    /// Python: `encode(LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"],
    /// loads=[LoRALoadTiming("alpha", "load", 0.25), LoRALoadTiming("alpha", "activate", 0.5)]))`
    const PYTHON_LORA_LOAD_EVENT_WITH_LOADS: &str = "84a474797065af6c6f72615f6c6f61645f6576656e74ac6770755f616461707465727391a5616c706861ac6370755f616461707465727391a5616c706861a56c6f6164739283ac616461707465725f6e616d65a5616c706861aa7472616e736974696f6ea46c6f6164a77365636f6e6473cb3fd000000000000083ac616461707465725f6e616d65a5616c706861aa7472616e736974696f6ea86163746976617465a77365636f6e6473cb3fe0000000000000";

    #[test]
    fn engine_event_decodes_python_lora_load_event_with_timings() {
        let event: EngineNotification =
            decode_msgpack(&hex_bytes(PYTHON_LORA_LOAD_EVENT_WITH_LOADS)).unwrap();
        assert_eq!(
            event,
            EngineNotification::LoraLoadEvent(LoraLoadEvent {
                gpu_adapters: vec!["alpha".to_string()],
                cpu_adapters: vec!["alpha".to_string()],
                pinned_adapters: vec![],
                loads: vec![
                    LoraLoadTiming {
                        adapter_name: "alpha".to_string(),
                        transition: "load".to_string(),
                        seconds: 0.25,
                    },
                    LoraLoadTiming {
                        adapter_name: "alpha".to_string(),
                        transition: "activate".to_string(),
                        seconds: 0.5,
                    },
                ],
            })
        );
    }

    #[test]
    fn engine_event_decodes_lora_load_event_omitted_defaults() {
        let event: EngineNotification =
            decode_msgpack(&hex_bytes(PYTHON_LORA_LOAD_EVENT_EMPTY)).unwrap();
        assert_eq!(
            event,
            EngineNotification::LoraLoadEvent(LoraLoadEvent::default())
        );
    }

    /// Python: `encode(CustomNotification(key="my_plugin",
    /// payload={"count": 5, "name": "foo"}))`
    const PYTHON_CUSTOM: &str = "83a474797065a6637573746f6da36b6579a96d795f706c7567696ea77061796c6f616482a5636f756e7405a46e616d65a3666f6f";

    /// Python: `encode(CustomNotification(key="my_plugin"))`, where
    /// `omit_defaults=True` strips the empty payload.
    const PYTHON_CUSTOM_EMPTY: &str = "82a474797065a6637573746f6da36b6579a96d795f706c7567696e";

    #[test]
    fn engine_event_decodes_python_custom_notification() {
        let event: EngineNotification = decode_msgpack(&hex_bytes(PYTHON_CUSTOM)).unwrap();
        expect_test::expect![[r#"
            Custom(
                CustomNotification {
                    key: "my_plugin",
                    payload: {
                        "count": Integer(
                            PosInt(
                                5,
                            ),
                        ),
                        "name": String(
                            Utf8String {
                                s: Ok(
                                    "foo",
                                ),
                            },
                        ),
                    },
                },
            )
        "#]]
        .assert_debug_eq(&event);
    }

    #[test]
    fn engine_event_decodes_custom_omitted_payload() {
        let event: EngineNotification = decode_msgpack(&hex_bytes(PYTHON_CUSTOM_EMPTY)).unwrap();
        assert_eq!(
            event,
            EngineNotification::Custom(CustomNotification {
                key: "my_plugin".to_string(),
                payload: BTreeMap::new(),
            })
        );
    }

    #[test]
    fn engine_event_unknown_tag_fails_fast() {
        // An unknown event type must fail the decode, not be skipped.
        let value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("type"),
                rmpv::Value::from("graceful_shutdown_started"),
            ),
            (rmpv::Value::from("deadline_seconds"), rmpv::Value::from(30)),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();

        let result: Result<EngineNotification, _> = decode_msgpack(&bytes);
        assert!(result.is_err());
    }
}
