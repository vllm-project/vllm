// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use parking_lot::RwLock;
use rmpv::Value;
use tracing::warn;
use vllm_metrics::{
    ConnectorMetricLabels, F64Counter, F64Gauge, Family, Histogram, MetricConstructor, Metrics,
    U64Counter, U64Gauge, connector_metric_labels,
};

use super::descriptor::{
    DESCRIPTOR_VERSION_V1, METRICS_DESCRIPTOR_KEY, MetricDef, MetricType, MetricsDescriptorV1,
    SampleKind,
};

#[derive(Clone)]
struct HistogramBuckets(Arc<Vec<f64>>);

impl MetricConstructor<Histogram> for HistogramBuckets {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.0.iter().copied())
    }
}

enum DynamicFamily {
    Counter(Family<ConnectorMetricLabels, U64Counter>),
    CounterF64(Family<ConnectorMetricLabels, F64Counter>),
    GaugeU64(Family<ConnectorMetricLabels, U64Gauge>),
    GaugeF64(Family<ConnectorMetricLabels, F64Gauge>),
    Histogram(Family<ConnectorMetricLabels, Histogram, HistogramBuckets>),
}

impl DynamicFamily {
    fn clone_family(&self) -> Result<Self, String> {
        Ok(match self {
            Self::Counter(f) => Self::Counter(f.clone()),
            Self::CounterF64(f) => Self::CounterF64(f.clone()),
            Self::GaugeU64(f) => Self::GaugeU64(f.clone()),
            Self::GaugeF64(f) => Self::GaugeF64(f.clone()),
            Self::Histogram(f) => Self::Histogram(f.clone()),
        })
    }
}

struct BoundMetric {
    def: MetricDef,
    family: DynamicFamily,
}

struct DescriptorInstance {
    metrics: Vec<BoundMetric>,
}

/// Registers and observes descriptor-driven connector metrics.
///
/// Descriptors arrive via the reserved stats key ``_metrics_descriptor`` (Python
/// connectors emit it once when the Rust frontend is enabled). There is no
/// env-path or ``include_str!`` builtin loader.
pub(crate) struct DescriptorDrivenAdapter {
    metrics: &'static Metrics,
    /// Model name applied to every pre-created connector series.
    model_name: String,
    /// Engine indices known at frontend startup (DP size); used to pre-create
    /// zero-valued series when a descriptor registers, matching Python Prom.
    engine_indices: Vec<u32>,
    instances: RwLock<BTreeMap<String, DescriptorInstance>>,
    warned_missing: Mutex<BTreeSet<String>>,
    warned_bad_descriptor: Mutex<BTreeSet<String>>,
    /// `(class_name, descriptor connector_id)` pairs already warned.
    warned_id_mismatch: Mutex<BTreeSet<(String, String)>>,
}

impl DescriptorDrivenAdapter {
    /// Empty adapter; descriptors register lazily from payload ``_metrics_descriptor``.
    ///
    /// When ``engine_indices`` is non-empty, registration also pre-creates a
    /// zero series per engine (and per ``const_labels``) so ``/metrics`` matches
    /// the Python frontend at idle / for DP engines that never saw traffic.
    pub(crate) fn new(metrics: &'static Metrics, model_name: &str, engine_indices: &[u32]) -> Self {
        Self {
            metrics,
            model_name: model_name.to_string(),
            engine_indices: engine_indices.to_vec(),
            instances: RwLock::new(BTreeMap::new()),
            warned_missing: Mutex::new(BTreeSet::new()),
            warned_bad_descriptor: Mutex::new(BTreeSet::new()),
            warned_id_mismatch: Mutex::new(BTreeSet::new()),
        }
    }

    /// Construct an empty adapter without engine pre-creation (tests).
    #[cfg(test)]
    pub(crate) fn empty(metrics: &'static Metrics) -> Self {
        Self::new(metrics, "", &[])
    }

    pub(crate) fn registered_ids(&self) -> Vec<String> {
        self.instances.read().keys().cloned().collect()
    }

    /// Validate and register one descriptor document (idempotent per connector_id).
    pub(crate) fn ensure_registered(&self, descriptor: MetricsDescriptorV1) -> Result<(), String> {
        descriptor.validate()?;
        {
            let instances = self.instances.read();
            if instances.contains_key(&descriptor.connector_id) {
                return Ok(());
            }
        }
        let instance = self.build_instance(&descriptor)?;
        let mut instances = self.instances.write();
        instances.entry(descriptor.connector_id).or_insert(instance);
        Ok(())
    }

    /// Observe one flat connector stats object for ``connector_id``.
    ///
    /// When the payload carries ``_metrics_descriptor``, that document's
    /// ``connector_id`` is used for registration and for this observe binding.
    /// Flat payloads have no outer class-name key, so a mismatch is not checked.
    pub(crate) fn observe(
        &self,
        connector_id: &str,
        model_name: &str,
        engine: u32,
        payload: &Value,
    ) {
        self.observe_inner(connector_id, model_name, engine, payload, false);
    }

    /// Observe one MultiConnector child. ``class_name`` is the map key.
    ///
    /// Registration uses ``class_name`` so later data-only ticks, which only
    /// carry that key, keep updating the same series. A descriptor
    /// ``connector_id`` that differs is warned once per pair.
    pub(crate) fn observe_multi_child(
        &self,
        class_name: &str,
        model_name: &str,
        engine: u32,
        payload: &Value,
    ) {
        self.observe_inner(class_name, model_name, engine, payload, true);
    }

    fn observe_inner(
        &self,
        connector_id: &str,
        model_name: &str,
        engine: u32,
        payload: &Value,
        bind_to_class_name: bool,
    ) {
        let mut payload = payload.clone();
        let mut bound_id = connector_id.to_string();
        if let Some(descriptor_value) = map_remove(&mut payload, METRICS_DESCRIPTOR_KEY) {
            match rmpv_to_descriptor(&descriptor_value) {
                Ok(mut descriptor) => {
                    if bind_to_class_name && descriptor.connector_id != connector_id {
                        self.warn_id_mismatch(connector_id, &descriptor.connector_id);
                        descriptor.connector_id = connector_id.to_string();
                    }
                    bound_id = descriptor.connector_id.clone();
                    if let Err(err) = self.ensure_registered(descriptor) {
                        self.warn_bad_descriptor(&bound_id, &err);
                    }
                }
                Err(err) => self.warn_bad_descriptor(connector_id, &err),
            }
        }
        let _ = map_remove(&mut payload, "_n_steps");

        let instances = self.instances.read();
        let Some(instance) = instances.get(&bound_id) else {
            drop(instances);
            self.warn_missing(&bound_id);
            return;
        };

        for bound in &instance.metrics {
            observe_metric(bound, model_name, engine, &payload);
        }
    }

    fn build_instance(
        &self,
        descriptor: &MetricsDescriptorV1,
    ) -> Result<DescriptorInstance, String> {
        // Multiple MetricDefs may share one Prom name (e.g. path/outcome
        // const_labels). Register each unique name once and reuse the Family.
        let mut families: BTreeMap<String, DynamicFamily> = BTreeMap::new();
        let mut metrics = Vec::with_capacity(descriptor.metrics.len());
        for def in &descriptor.metrics {
            let reg_name = strip_counter_total_suffix(&def.name, &def.metric_type).to_string();
            if !families.contains_key(&reg_name) {
                let family = self.register_family(def, &reg_name)?;
                families.insert(reg_name.clone(), family);
            }
            let family = families.get(&reg_name).expect("family just inserted").clone_family()?;
            metrics.push(BoundMetric {
                def: def.clone(),
                family,
            });
        }
        let instance = DescriptorInstance { metrics };
        self.precreate_zero_series(&instance);
        Ok(instance)
    }

    /// Pre-create a zero-valued series for every known engine so idle / unused
    /// DP engines still appear on ``/metrics`` (Python ``create_metric_per_engine``).
    ///
    /// Only labels fully known at registration time are created: ``model_name``,
    /// ``engine``, and descriptor ``const_labels``. Data-dependent label values
    /// (e.g. Offloading transfer_type nested under sample maps) cannot be
    /// pre-created here; those series still appear on first observation.
    fn precreate_zero_series(&self, instance: &DescriptorInstance) {
        if self.engine_indices.is_empty() {
            return;
        }
        for bound in &instance.metrics {
            let const_labels: Vec<(String, String)> =
                bound.def.const_labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            for &engine in &self.engine_indices {
                let labels = connector_metric_labels(&self.model_name, engine, &const_labels);
                match &bound.family {
                    DynamicFamily::Counter(family) => {
                        let _ = family.get_or_create(&labels);
                    }
                    DynamicFamily::CounterF64(family) => {
                        let _ = family.get_or_create(&labels);
                    }
                    DynamicFamily::GaugeU64(family) => {
                        let _ = family.get_or_create(&labels);
                    }
                    DynamicFamily::GaugeF64(family) => {
                        let _ = family.get_or_create(&labels);
                    }
                    DynamicFamily::Histogram(family) => {
                        let _ = family.get_or_create(&labels);
                    }
                }
            }
        }
    }

    fn register_family(&self, def: &MetricDef, name: &str) -> Result<DynamicFamily, String> {
        // prometheus-client appends '.' to help on register; strip one trailing
        // period from descriptor docs so HELP matches Python (single '.').
        let help = if def.documentation.is_empty() {
            format!("KV connector metric {}", def.name)
        } else {
            strip_trailing_help_period(&def.documentation).to_string()
        };

        Ok(match def.metric_type {
            MetricType::Counter => match def.sample_kind {
                SampleKind::IncByF64 => {
                    let family: Family<ConnectorMetricLabels, F64Counter> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::CounterF64(family)
                }
                SampleKind::IncByU64 | SampleKind::IncBySumU64 => {
                    let family: Family<ConnectorMetricLabels, U64Counter> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::Counter(family)
                }
                other => {
                    return Err(format!(
                        "counter metric '{}' has unsupported sample_kind {other:?}",
                        def.name
                    ));
                }
            },
            MetricType::Gauge => match def.sample_kind {
                SampleKind::SetU64 => {
                    let family: Family<ConnectorMetricLabels, U64Gauge> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::GaugeU64(family)
                }
                SampleKind::SetF64 => {
                    let family: Family<ConnectorMetricLabels, F64Gauge> = Family::default();
                    self.metrics.with_dynamic_registry(|registry| {
                        registry.register(name, help, family.clone());
                    });
                    DynamicFamily::GaugeF64(family)
                }
                other => {
                    return Err(format!(
                        "gauge metric '{}' has unsupported sample_kind {other:?}",
                        def.name
                    ));
                }
            },
            MetricType::Histogram => {
                let buckets = def
                    .buckets
                    .clone()
                    .ok_or_else(|| format!("histogram '{}' missing buckets", def.name))?;
                let family =
                    Family::<ConnectorMetricLabels, Histogram, HistogramBuckets>::new_with_constructor(
                        HistogramBuckets(Arc::new(buckets)),
                    );
                self.metrics.with_dynamic_registry(|registry| {
                    registry.register(name, help, family.clone());
                });
                DynamicFamily::Histogram(family)
            }
        })
    }

    /// Whether a missing-descriptor warning was already emitted for `connector_id`.
    #[cfg(test)]
    pub(crate) fn has_warned_missing(&self, connector_id: &str) -> bool {
        self.warned_missing.lock().expect("warn set").contains(connector_id)
    }

    /// Whether an invalid-descriptor warning was already emitted for `connector_id`.
    #[cfg(test)]
    pub(crate) fn has_warned_bad_descriptor(&self, connector_id: &str) -> bool {
        self.warned_bad_descriptor.lock().expect("warn set").contains(connector_id)
    }

    /// Number of distinct invalid-descriptor warnings.
    #[cfg(test)]
    pub(crate) fn bad_descriptor_warning_count(&self) -> usize {
        self.warned_bad_descriptor.lock().expect("warn set").len()
    }

    /// Whether this class-name / descriptor-id pair was already warned.
    #[cfg(test)]
    pub(crate) fn has_warned_id_mismatch(&self, class_name: &str, connector_id: &str) -> bool {
        self.warned_id_mismatch
            .lock()
            .expect("warn set")
            .iter()
            .any(|(class, id)| class == class_name && id == connector_id)
    }

    /// Number of distinct class-name / descriptor-id mismatch warnings.
    #[cfg(test)]
    pub(crate) fn id_mismatch_warning_count(&self) -> usize {
        self.warned_id_mismatch.lock().expect("warn set").len()
    }

    fn warn_missing(&self, connector_id: &str) {
        let mut warned = self.warned_missing.lock().expect("warn set");
        if warned.insert(connector_id.to_string()) {
            warn!(
                connector_id,
                "KV connector stats collected but no metrics descriptor is registered; \
                 connectors must emit _metrics_descriptor once in their stats payload"
            );
        }
    }

    fn warn_id_mismatch(&self, class_name: &str, connector_id: &str) {
        let mut warned = self.warned_id_mismatch.lock().expect("warn set");
        if warned.insert((class_name.to_string(), connector_id.to_string())) {
            warn!(
                connector_class = class_name,
                connector_id,
                "KV connector metrics descriptor connector_id `{}` must equal the connector \
                 class name `{}` used as the MultiConnector key",
                connector_id,
                class_name,
            );
        }
    }

    fn warn_bad_descriptor(&self, connector_id: &str, err: &str) {
        let mut warned = self.warned_bad_descriptor.lock().expect("warn set");
        if warned.insert(connector_id.to_string()) {
            warn!(
                connector_id,
                error = err,
                "ignoring invalid KV connector metrics descriptor"
            );
        }
    }
}

fn strip_counter_total_suffix<'a>(name: &'a str, metric_type: &MetricType) -> &'a str {
    if *metric_type == MetricType::Counter {
        name.strip_suffix("_total").unwrap_or(name)
    } else {
        name
    }
}

/// Drop one trailing '.' so prometheus-client's mandatory append yields a single period.
fn strip_trailing_help_period(help: &str) -> &str {
    help.strip_suffix('.').unwrap_or(help)
}

fn rmpv_to_descriptor(value: &Value) -> Result<MetricsDescriptorV1, String> {
    // Round-trip via JSON so descriptor parsing stays serde_json-based.
    let json = serde_json::to_value(value).map_err(|err| err.to_string())?;
    let descriptor: MetricsDescriptorV1 =
        serde_json::from_value(json).map_err(|err| err.to_string())?;
    if descriptor.descriptor_version != DESCRIPTOR_VERSION_V1 {
        return Err(format!(
            "unsupported descriptor_version {}",
            descriptor.descriptor_version
        ));
    }
    Ok(descriptor)
}

/// Parse ``_metrics_descriptor`` from a flat stats map, if present.
pub(crate) fn connector_id_from_payload_descriptor(
    map: &BTreeMap<String, Value>,
) -> Option<String> {
    let value = map.get(METRICS_DESCRIPTOR_KEY)?;
    match rmpv_to_descriptor(value) {
        Ok(descriptor) => Some(descriptor.connector_id),
        Err(_) => None,
    }
}

fn map_remove(payload: &mut Value, key: &str) -> Option<Value> {
    let Value::Map(entries) = payload else {
        return None;
    };
    let idx = entries.iter().position(|(k, _)| match k {
        Value::String(s) => s.as_str().is_some_and(|s| s == key),
        _ => false,
    })?;
    Some(entries.remove(idx).1)
}

fn observe_metric(bound: &BoundMetric, model_name: &str, engine: u32, payload: &Value) {
    let const_labels: Vec<(String, String)> =
        bound.def.const_labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let labels = connector_metric_labels(model_name, engine, &const_labels);
    let Some(sample) = value_at_path(payload, &bound.def.samples_path) else {
        return;
    };
    // Offloading (and similar) nest values under label-tuple map keys;
    // msgspec encodes `()` as an empty MessagePack array.
    let sample = unwrap_label_tuple_map(sample);
    let scale = bound.def.scale;

    match (&bound.family, &bound.def.sample_kind) {
        (DynamicFamily::Counter(family), SampleKind::IncByU64) => {
            if let Some(v) = as_u64(sample)
                && v != 0
            {
                family.get_or_create(&labels).inc_by(v);
            }
        }
        (DynamicFamily::Counter(family), SampleKind::IncBySumU64) => {
            let sum = sum_u64(sample);
            if sum != 0 {
                family.get_or_create(&labels).inc_by(sum);
            }
        }
        (DynamicFamily::CounterF64(family), SampleKind::IncByF64) => {
            if let Some(v) = as_f64(sample)
                && v != 0.0
            {
                family.get_or_create(&labels).inc_by(v);
            }
        }
        (DynamicFamily::GaugeU64(family), SampleKind::SetU64) => {
            if let Some(v) = as_u64(sample) {
                family.get_or_create(&labels).set(v);
            }
        }
        (DynamicFamily::GaugeF64(family), SampleKind::SetF64) => {
            if let Some(v) = as_f64(sample) {
                family.get_or_create(&labels).set(v);
            }
        }
        (DynamicFamily::Histogram(family), SampleKind::ObserveEachF64) => {
            for v in as_f64_list(sample) {
                family.get_or_create(&labels).observe(v * scale);
            }
        }
        (DynamicFamily::Histogram(family), SampleKind::ObserveEachU64AsF64) => {
            for v in as_u64_list(sample) {
                family.get_or_create(&labels).observe(v as f64 * scale);
            }
        }
        _ => {}
    }
}

/// Unwrap Offloading-style `{label_tuple: value}` maps to the unlabeled value.
///
/// Prefer the empty-tuple key (`Array([])`). If absent, use the sole entry or
/// leave the map unchanged (callers then no-op on type mismatch).
fn unwrap_label_tuple_map(sample: &Value) -> &Value {
    let Value::Map(entries) = sample else {
        return sample;
    };
    if entries.is_empty() {
        return sample;
    }
    let all_array_keys = entries.iter().all(|(k, _)| matches!(k, Value::Array(_)));
    if !all_array_keys {
        return sample;
    }
    if let Some((_, v)) = entries.iter().find(|(k, _)| matches!(k, Value::Array(a) if a.is_empty()))
    {
        return v;
    }
    if entries.len() == 1 {
        return &entries[0].1;
    }
    sample
}

fn value_at_path<'a>(payload: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let mut cur = payload;
    for part in path.split('.') {
        cur = map_get(cur, part)?;
    }
    Some(cur)
}

fn map_get<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
    match payload {
        Value::Map(entries) => entries.iter().find_map(|(k, v)| match k {
            Value::String(s) if s.as_str().is_some_and(|s| s == key) => Some(v),
            _ => None,
        }),
        _ => None,
    }
}

fn as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Integer(i) => i.as_u64().or_else(|| i.as_i64().and_then(|v| u64::try_from(v).ok())),
        Value::F64(v) => Some(*v as u64),
        Value::F32(v) => Some(*v as u64),
        _ => None,
    }
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::F64(v) => Some(*v),
        Value::F32(v) => Some(f64::from(*v)),
        Value::Integer(i) => i.as_i64().map(|v| v as f64).or_else(|| i.as_u64().map(|v| v as f64)),
        _ => None,
    }
}

fn sum_u64(value: &Value) -> u64 {
    match value {
        Value::Array(items) => items.iter().filter_map(as_u64).sum(),
        other => as_u64(other).unwrap_or(0),
    }
}

fn as_f64_list(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(items) => items.iter().filter_map(as_f64).collect(),
        other => as_f64(other).into_iter().collect(),
    }
}

fn as_u64_list(value: &Value) -> Vec<u64> {
    match value {
        Value::Array(items) => items.iter().filter_map(as_u64).collect(),
        other => as_u64(other).into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;
    use vllm_metrics::Metrics;

    use super::DescriptorDrivenAdapter;
    use crate::connector_metrics::descriptor::METRICS_DESCRIPTOR_KEY;

    #[test]
    fn registration_precreates_zero_series_for_all_engines() {
        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let adapter = DescriptorDrivenAdapter::new(metrics, "model", &[0, 1]);
        let descriptor = serde_json::json!({
            "descriptor_version": 1,
            "connector_id": "FakeConnector",
            "metrics": [
                {
                    "name": "fake_puts",
                    "type": "counter",
                    "documentation": "puts",
                    "samples_path": "puts",
                    "sample_kind": "inc_by_u64"
                },
                {
                    "name": "fake_latency_seconds",
                    "type": "histogram",
                    "documentation": "latency",
                    "samples_path": "latency",
                    "sample_kind": "observe_each_f64",
                    "buckets": [0.01, 0.1, 1.0]
                },
                {
                    "name": "fake_gauge",
                    "type": "gauge",
                    "documentation": "gauge",
                    "samples_path": "gauge",
                    "sample_kind": "set_f64",
                    "const_labels": {"path": "local"}
                }
            ]
        });
        // Descriptor-only payload (no samples): still registers + pre-creates.
        let payload = Value::Map(vec![(
            Value::String(METRICS_DESCRIPTOR_KEY.into()),
            rmpv::ext::to_value(&descriptor).expect("descriptor"),
        )]);
        adapter.observe("FakeConnector", "model", 0, &payload);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("fake_puts_total{engine=\"0\",model_name=\"model\"} 0"),
            "engine 0 counter missing at 0:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_puts_total{engine=\"1\",model_name=\"model\"} 0"),
            "engine 1 counter must exist before any traffic:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_latency_seconds_count{engine=\"1\",model_name=\"model\"} 0"),
            "engine 1 histogram count must be 0:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_gauge{engine=\"1\",model_name=\"model\",path=\"local\"}"),
            "const_labels series must be pre-created:\n{rendered}"
        );
        let gauge_zero = rendered
            .contains("fake_gauge{engine=\"1\",model_name=\"model\",path=\"local\"} 0")
            || rendered
                .contains("fake_gauge{engine=\"1\",model_name=\"model\",path=\"local\"} 0.0");
        assert!(gauge_zero, "pre-created gauge must be zero:\n{rendered}");

        // Data-only tick on engine 0 still updates that engine only.
        let data_only = Value::Map(vec![(Value::String("puts".into()), Value::from(3u64))]);
        adapter.observe("FakeConnector", "model", 0, &data_only);
        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("fake_puts_total{engine=\"0\",model_name=\"model\"} 3"),
            "data-only tick should update engine 0:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_puts_total{engine=\"1\",model_name=\"model\"} 0"),
            "engine 1 must stay at 0:\n{rendered}"
        );
    }

    #[test]
    fn invalid_descriptor_warns_once_and_does_not_register() {
        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let adapter = DescriptorDrivenAdapter::empty(metrics);
        let descriptor = serde_json::json!({
            "descriptor_version": 1,
            "connector_id": "ExampleConnector",
            "metrics": [{
                "name": "example_latency",
                "type": "histogram",
                "documentation": "latency",
                "samples_path": "latency",
                "sample_kind": "observe_each_f64"
            }]
        });
        let payload = Value::Map(vec![
            (
                Value::String(METRICS_DESCRIPTOR_KEY.into()),
                rmpv::ext::to_value(&descriptor).expect("descriptor"),
            ),
            (Value::String("latency".into()), Value::F64(1.0)),
        ]);

        adapter.observe("ignored-for-valid-id", "model", 0, &payload);
        adapter.observe("ignored-for-valid-id", "model", 0, &payload);

        assert!(adapter.has_warned_bad_descriptor("ExampleConnector"));
        assert_eq!(adapter.bad_descriptor_warning_count(), 1);
        assert!(adapter.registered_ids().is_empty());
        assert!(
            !metrics.render().unwrap().contains("example_latency"),
            "invalid histogram must not register a series"
        );
    }

    #[test]
    fn descriptor_help_trailing_period_stripped_before_register() {
        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let adapter = DescriptorDrivenAdapter::empty(metrics);
        let descriptor = serde_json::json!({
            "descriptor_version": 1,
            "connector_id": "HelpPeriodConnector",
            "metrics": [
                {
                    "name": "help_with_period",
                    "type": "counter",
                    "documentation": "Foo.",
                    "samples_path": "n",
                    "sample_kind": "inc_by_u64"
                },
                {
                    "name": "help_without_period",
                    "type": "counter",
                    "documentation": "Bar",
                    "samples_path": "m",
                    "sample_kind": "inc_by_u64"
                }
            ]
        });
        let payload = Value::Map(vec![(
            Value::String(METRICS_DESCRIPTOR_KEY.into()),
            rmpv::ext::to_value(&descriptor).expect("descriptor"),
        )]);
        adapter.observe("HelpPeriodConnector", "model", 0, &payload);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("# HELP help_with_period Foo."),
            "help ending in '.' must expose a single period:\n{rendered}"
        );
        assert!(
            !rendered.contains("# HELP help_with_period Foo.."),
            "must not double the trailing period:\n{rendered}"
        );
        assert!(
            rendered.contains("# HELP help_without_period Bar."),
            "help without period must still get prometheus-client's single '.':\n{rendered}"
        );
        assert!(
            !rendered.contains("# HELP help_without_period Bar.."),
            "bare help must not become double period:\n{rendered}"
        );
    }

    #[test]
    fn strip_trailing_help_period_helper() {
        assert_eq!(super::strip_trailing_help_period("Foo."), "Foo");
        assert_eq!(super::strip_trailing_help_period("Foo"), "Foo");
        assert_eq!(super::strip_trailing_help_period("Foo.."), "Foo.");
        assert_eq!(super::strip_trailing_help_period(""), "");
    }
}
