// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Open escape hatch for out-of-tree producers (plugins).
///
/// The union fails fast on unknown tags, so plugins can't add their own struct
/// type. They emit this instead: namespace under `key`, arbitrary `payload`.
/// Frontends that don't know the `key` ignore it. `payload` is Python's
/// `dict[str, Any]` with `omit_defaults=True`, hence `#[serde(default)]`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/v1/notifications.py>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CustomNotification {
    /// Producer-chosen namespace, e.g. the plugin name.
    pub key: String,
    /// Arbitrary msgpack data, opaque to this frontend.
    #[serde(default)]
    pub payload: BTreeMap<String, rmpv::Value>,
}

/// Rare engine-level event notifications carried on
/// `EngineCoreOutputs::engine_notifications`.
///
/// These are engine-scoped state transitions (as opposed to the per-request
/// lifecycle events in `EngineCoreOutput::events` and the per-step sampling
/// in `SchedulerStats`). The union is map-encoded with a `"type"`
/// discriminator field; msgspec dispatches on each struct's tag. Like the
/// rest of `EngineCoreOutputs`, the union is version-lockstep with the
/// engine: an unknown tag is a deployment error and fails the decode.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/v1/notifications.py>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineNotification {
    Custom(CustomNotification),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::decode_msgpack;

    fn hex_bytes(hex: &str) -> Vec<u8> {
        hex::decode(hex).unwrap()
    }

    /// Captured from Python:
    /// `msgspec.msgpack.encode(CustomNotification(key="my_plugin",
    /// payload={"count": 5, "name": "foo"}))`
    const PYTHON_CUSTOM: &str = "83a474797065a6637573746f6da36b6579a96d795f706c7567696ea77061796c6f616482a5636f756e7405a46e616d65a3666f6f";

    /// Captured from Python:
    /// `msgspec.msgpack.encode(CustomNotification(key="my_plugin"))`,
    /// where `omit_defaults=True` strips the empty payload.
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
        // The union is version-lockstep with the engine; an event type this
        // frontend does not know about must fail the decode, not be skipped.
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
