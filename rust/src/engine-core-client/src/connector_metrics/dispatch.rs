// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use rmpv::Value;

use super::adapter::{DescriptorDrivenAdapter, connector_id_from_payload_descriptor};
use super::descriptor::METRICS_DESCRIPTOR_KEY;

/// Builtin connector class names claimed by typed adapters (not generic).
const BUILTIN_CONNECTOR_IDS: &[&str] = &[
    "NixlConnector",
    "NixlPullConnector",
    "NixlPushConnector",
    "MooncakeStoreConnector",
];

/// Observe opaque third-party connector stats (`Multi.other` / `Other`).
///
/// Untagged decoding cannot tell a flat connector payload from a MultiConnector
/// child map, so this does not use class-name spelling. A key is a child when
/// its value is a map containing `_metrics_descriptor`, or when that key equals
/// an already-registered `connector_id` (data-only ticks). A descriptor
/// `connector_id` that differs from the child key is still registered under
/// that key, with one warning per pair. Map-valued Nixl / Mooncake builtin ids
/// are skipped. If any child is identified, remaining non-builtin keys are
/// observed as children too (no descriptor: warn once and drop). Otherwise the
/// map is one flat payload.
pub(crate) fn observe_opaque_connector_stats(
    generic: &DescriptorDrivenAdapter,
    model_name: &str,
    engine: u32,
    map: &BTreeMap<String, Value>,
) {
    if map.is_empty() {
        return;
    }

    let registered = generic.registered_ids();
    if map_has_child_connector(map, &registered) {
        for (key, value) in map {
            if is_builtin_connector_id(key) {
                continue;
            }
            generic.observe_multi_child(key, model_name, engine, value);
        }
        return;
    }

    // Flat single-connector payload. Prefer ``_metrics_descriptor.connector_id``,
    // else the sole already-registered descriptor (data-only ticks after one-shot).
    let payload = Value::Map(
        map.iter().map(|(k, v)| (Value::String(k.as_str().into()), v.clone())).collect(),
    );
    match resolve_flat_connector_id(generic, map) {
        Some(connector_id) => generic.observe(&connector_id, model_name, engine, &payload),
        None => generic.observe("<unknown>", model_name, engine, &payload),
    }
}

/// True when `map` is a MultiConnector child map rather than one flat payload.
fn map_has_child_connector(map: &BTreeMap<String, Value>, registered: &[String]) -> bool {
    map.iter().any(|(key, value)| is_child_connector_entry(key, value, registered))
}

/// Child payload: map containing `_metrics_descriptor`, an already-registered
/// connector id, or a builtin Nixl/Mooncake id (those are skipped by the caller).
fn is_child_connector_entry(key: &str, value: &Value, registered: &[String]) -> bool {
    if !value_is_map(value) {
        return false;
    }
    if is_builtin_connector_id(key) {
        return true;
    }
    map_contains_string_key(value, METRICS_DESCRIPTOR_KEY) || is_registered(registered, key)
}

fn is_registered(registered: &[String], key: &str) -> bool {
    registered.iter().any(|id| id == key)
}

fn value_is_map(value: &Value) -> bool {
    matches!(value, Value::Map(_))
}

fn map_contains_string_key(value: &Value, key: &str) -> bool {
    let Value::Map(entries) = value else {
        return false;
    };
    entries.iter().any(|(entry_key, _)| match entry_key {
        Value::String(text) => text.as_str().is_some_and(|text| text == key),
        _ => false,
    })
}

fn resolve_flat_connector_id(
    generic: &DescriptorDrivenAdapter,
    map: &BTreeMap<String, Value>,
) -> Option<String> {
    // 1) Payload carries descriptor → use its connector_id.
    if map.contains_key(METRICS_DESCRIPTOR_KEY) {
        return connector_id_from_payload_descriptor(map);
    }
    // 2) Data-only after one-shot: exactly one descriptor already registered.
    let ids = generic.registered_ids();
    if ids.len() == 1 {
        return ids.into_iter().next();
    }
    None
}

fn is_builtin_connector_id(id: &str) -> bool {
    BUILTIN_CONNECTOR_IDS.contains(&id)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rmpv::Value;
    use vllm_metrics::Metrics;

    use super::observe_opaque_connector_stats;
    use crate::connector_metrics::DescriptorDrivenAdapter;
    use crate::connector_metrics::descriptor::METRICS_DESCRIPTOR_KEY;

    fn test_adapter() -> (&'static Metrics, DescriptorDrivenAdapter) {
        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        (metrics, DescriptorDrivenAdapter::empty(metrics))
    }

    fn counter_descriptor(connector_id: &str, name: &str, samples_path: &str) -> Value {
        let document = serde_json::json!({
            "descriptor_version": 1,
            "connector_id": connector_id,
            "metrics": [{
                "name": name,
                "type": "counter",
                "documentation": "test counter",
                "samples_path": samples_path,
                "sample_kind": "inc_by_u64"
            }]
        });
        rmpv::ext::to_value(&document).expect("descriptor")
    }

    fn map_value(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(entries.into_iter().map(|(k, v)| (Value::String(k.into()), v)).collect())
    }

    fn insert_child(map: &mut BTreeMap<String, Value>, key: &str, value: Value) {
        map.insert(key.to_string(), value);
    }

    /// Class name has neither "Connector" nor a "Store" suffix.
    #[test]
    fn multi_child_without_connector_in_name_registers_then_data_only() {
        let (metrics, generic) = test_adapter();

        let mut first = BTreeMap::new();
        insert_child(
            &mut first,
            "BlobCache",
            map_value(vec![
                (
                    METRICS_DESCRIPTOR_KEY,
                    counter_descriptor("BlobCache", "blob_hits", "hits"),
                ),
                ("hits", Value::from(2u64)),
            ]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &first);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("blob_hits_total{engine=\"0\",model_name=\"model\"} 2"),
            "descriptor tick should register BlobCache:\n{rendered}"
        );
        assert_eq!(generic.registered_ids(), vec!["BlobCache".to_string()]);
        assert_eq!(generic.id_mismatch_warning_count(), 0);

        let mut second = BTreeMap::new();
        insert_child(
            &mut second,
            "BlobCache",
            map_value(vec![("hits", Value::from(5u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &second);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("blob_hits_total{engine=\"0\",model_name=\"model\"} 7"),
            "data-only tick should keep observing BlobCache:\n{rendered}"
        );
    }

    /// Adopted child registers; sibling without a descriptor is warned and dropped.
    #[test]
    fn multi_mixes_adopted_child_and_non_adopted_child() {
        let (metrics, generic) = test_adapter();

        let mut payload = BTreeMap::new();
        insert_child(
            &mut payload,
            "BlobCache",
            map_value(vec![
                (
                    METRICS_DESCRIPTOR_KEY,
                    counter_descriptor("BlobCache", "blob_hits", "hits"),
                ),
                ("hits", Value::from(3u64)),
            ]),
        );
        insert_child(
            &mut payload,
            "LegacyBackend",
            map_value(vec![("hits", Value::from(9u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &payload);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("blob_hits_total{engine=\"0\",model_name=\"model\"} 3"),
            "adopted child should record:\n{rendered}"
        );
        assert!(
            !rendered.contains("legacy"),
            "non-adopted child must not create series:\n{rendered}"
        );
        assert_eq!(generic.registered_ids(), vec!["BlobCache".to_string()]);
        assert!(generic.has_warned_missing("LegacyBackend"));
        assert!(!generic.has_warned_missing("BlobCache"));
    }

    /// A flat field whose name looks like a class is still one connector payload.
    #[test]
    fn flat_payload_with_class_like_field_name_stays_flat() {
        let (metrics, generic) = test_adapter();

        let mut first = BTreeMap::new();
        first.insert(
            METRICS_DESCRIPTOR_KEY.to_string(),
            counter_descriptor("acme_cache", "acme_puts", "CustomConnector"),
        );
        first.insert("CustomConnector".to_string(), Value::from(4u64));
        first.insert(
            "PayloadStore".to_string(),
            map_value(vec![("inner", Value::from(1u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &first);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("acme_puts_total{engine=\"0\",model_name=\"model\"} 4"),
            "class-like field names must stay flat fields:\n{rendered}"
        );
        assert_eq!(generic.registered_ids(), vec!["acme_cache".to_string()]);
        assert_eq!(generic.id_mismatch_warning_count(), 0);
        assert!(!generic.has_warned_missing("CustomConnector"));
        assert!(!generic.has_warned_missing("PayloadStore"));

        let mut second = BTreeMap::new();
        second.insert("CustomConnector".to_string(), Value::from(1u64));
        second.insert(
            "PayloadStore".to_string(),
            map_value(vec![("inner", Value::from(1u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &second);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("acme_puts_total{engine=\"0\",model_name=\"model\"} 5"),
            "data-only flat tick should follow the sole descriptor:\n{rendered}"
        );
        assert_eq!(generic.registered_ids(), vec!["acme_cache".to_string()]);
    }

    /// Multi child registration also pre-creates zero series for every engine.
    #[test]
    fn multi_child_registration_precreates_zero_series_for_all_engines() {
        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let generic = DescriptorDrivenAdapter::new(metrics, "model", &[0, 1]);

        let mut first = BTreeMap::new();
        insert_child(
            &mut first,
            "BlobCache",
            map_value(vec![
                (
                    METRICS_DESCRIPTOR_KEY,
                    counter_descriptor("BlobCache", "blob_hits", "hits"),
                ),
                // No hits yet — descriptor-only bootstrap style.
            ]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &first);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("blob_hits_total{engine=\"0\",model_name=\"model\"} 0"),
            "multi child engine 0 must be pre-created:\n{rendered}"
        );
        assert!(
            rendered.contains("blob_hits_total{engine=\"1\",model_name=\"model\"} 0"),
            "multi child engine 1 must be pre-created before traffic:\n{rendered}"
        );

        // Data-only tick updates engine 0 only.
        let mut second = BTreeMap::new();
        insert_child(
            &mut second,
            "BlobCache",
            map_value(vec![("hits", Value::from(4u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &second);
        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("blob_hits_total{engine=\"0\",model_name=\"model\"} 4"),
            "data-only multi tick should update engine 0:\n{rendered}"
        );
        assert!(
            rendered.contains("blob_hits_total{engine=\"1\",model_name=\"model\"} 0"),
            "engine 1 must stay at 0:\n{rendered}"
        );
    }

    /// Builtin ids stay on the typed Nixl/Mooncake path even if a descriptor is attached.
    #[test]
    fn builtin_connector_ids_in_opaque_map_are_skipped() {
        let (metrics, generic) = test_adapter();

        let mut payload = BTreeMap::new();
        for id in ["NixlConnector", "MooncakeStoreConnector"] {
            let metric_name = format!("builtin_should_not_{id}");
            insert_child(
                &mut payload,
                id,
                map_value(vec![
                    (
                        METRICS_DESCRIPTOR_KEY,
                        counter_descriptor(id, &metric_name, "hits"),
                    ),
                    ("hits", Value::from(5u64)),
                ]),
            );
        }
        observe_opaque_connector_stats(&generic, "model", 0, &payload);

        let rendered = metrics.render().unwrap();
        assert!(
            !rendered.contains("builtin_should_not"),
            "builtin ids must not register generic series:\n{rendered}"
        );
        assert!(generic.registered_ids().is_empty());
        assert!(!generic.has_warned_missing("NixlConnector"));
        assert!(!generic.has_warned_missing("MooncakeStoreConnector"));
    }

    /// Descriptor id differs from the Multi child class name: warn once, keep the series.
    #[test]
    fn multi_child_descriptor_id_mismatch_warns_and_follows_class_name() {
        let (metrics, generic) = test_adapter();

        let mut first = BTreeMap::new();
        insert_child(
            &mut first,
            "ExampleConnector",
            map_value(vec![
                (
                    METRICS_DESCRIPTOR_KEY,
                    counter_descriptor("example", "example_puts", "puts"),
                ),
                ("puts", Value::from(2u64)),
            ]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &first);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("example_puts_total{engine=\"0\",model_name=\"model\"} 2"),
            "mismatch should still record under the class-name binding:\n{rendered}"
        );
        assert_eq!(
            generic.registered_ids(),
            vec!["ExampleConnector".to_string()]
        );
        assert!(generic.has_warned_id_mismatch("ExampleConnector", "example"));
        assert_eq!(generic.id_mismatch_warning_count(), 1);

        // Same pair again must not add another warning, and must not register `example`.
        observe_opaque_connector_stats(&generic, "model", 0, &first);
        assert_eq!(generic.id_mismatch_warning_count(), 1);
        assert_eq!(
            generic.registered_ids(),
            vec!["ExampleConnector".to_string()]
        );

        let mut second = BTreeMap::new();
        insert_child(
            &mut second,
            "ExampleConnector",
            map_value(vec![("puts", Value::from(3u64))]),
        );
        observe_opaque_connector_stats(&generic, "model", 0, &second);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("example_puts_total{engine=\"0\",model_name=\"model\"} 7"),
            "data-only tick should keep updating the class-name series:\n{rendered}"
        );
        assert!(!generic.registered_ids().iter().any(|id| id == "example"));
    }
}
