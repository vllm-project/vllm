// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The JSON log formatter for `--logging-config.formatter json`.
//!
//! Each event is one JSON object per line. Its first keys are Python's, in
//! Python's order: what `pythonjsonlogger` writes for
//! `vllm.logging_utils.formatter.JSON_FORMAT`, plus the `vllm_process_name`
//! record attribute, except `processName`, the multiprocessing name. That is
//! the contract log pipelines key on, so nothing else can replace them.
//! Tracing's structure, which Python has no counterpart for, follows under two
//! keys of its own:
//!
//! - `spans`: the names of the enclosing spans, from the root;
//! - `fields`: the fields of those spans and of the event, in one object, so a
//!   field such as `request_id` has the same path wherever it was recorded. A
//!   field replaces one of the same name from an enclosing span, as an inner
//!   variable shadows an outer one.

use std::{fmt, process};

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{Number, Value};
use serde_json_fmt::JsonFormat;
use time::macros::format_description;
use tracing::field::{Field, Visit};
use tracing::span::Record;
use tracing::{Event, Subscriber};
use tracing_log::NormalizeEvent as _;
use tracing_subscriber::field::RecordFields;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::registry::LookupSpan;

use crate::{TimeFormat, VllmLocalTimer, level_name};

/// Python `logging`'s default `asctime`, such as `2026-11-08 19:12:01,839`.
const ASCTIME_FORMAT: TimeFormat =
    format_description!("[year]-[month]-[day] [hour]:[minute]:[second],[subsecond digits:3]");

/// Field values by name, in recording order.
type FieldMap<'a> = IndexMap<&'a str, Value>;

/// Formats each event as one JSON line; see the module docs for its shape.
pub(crate) struct VllmJsonEventFormatter {
    process_label: String,
    pid: u32,
    timer: VllmLocalTimer,
    /// Python `json.dumps`'s default separators, `", "` and `": "`.
    json: JsonFormat,
}

impl VllmJsonEventFormatter {
    pub(crate) fn new(process_label: &str) -> Self {
        let json = JsonFormat::new()
            .comma(", ")
            .expect("literal comma separator is valid JSON")
            .colon(": ")
            .expect("literal colon separator is valid JSON");
        Self {
            process_label: process_label.to_owned(),
            pid: process::id(),
            timer: VllmLocalTimer::new(ASCTIME_FORMAT),
            json,
        }
    }
}

impl<S> FormatEvent<S, JsonFields> for VllmJsonEventFormatter
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, JsonFields>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // An event bridged from the `log` crate (dependencies such as `hf-hub`)
        // has the `log` target; its real target is in `log.*` fields.
        let normalized_meta = event.normalized_metadata();
        let meta = normalized_meta.as_ref().unwrap_or_else(|| event.metadata());
        let mut asctime = String::new();
        if self.timer.format_time(&mut Writer::new(&mut asctime)).is_err() {
            asctime = "<unknown time>".to_owned();
        }

        let spans: Vec<_> =
            ctx.event_scope().into_iter().flat_map(|scope| scope.from_root()).collect();
        let span_names: Vec<&str> = spans.iter().map(|span| span.metadata().name()).collect();
        // Hold the extensions so that `fields` can borrow the names in them.
        let extensions: Vec<_> = spans.iter().map(|span| span.extensions()).collect();
        let mut fields = FieldMap::new();
        for extension in &extensions {
            if let Some(span_fields) = extension.get::<FormattedFields<JsonFields>>() {
                fields.extend(parse_fields(&span_fields.fields));
            }
        }
        let mut event_fields = JsonVisitor::default();
        event.record(&mut event_fields);
        let message = event_fields.0.shift_remove("message").unwrap_or_else(|| Value::from(""));
        if normalized_meta.is_some() {
            event_fields.0.retain(|name, _| !name.starts_with("log."));
        }
        fields.extend(event_fields.0);

        let line = JsonLine {
            asctime: &asctime,
            levelname: level_name(meta.level()),
            name: meta.target(),
            process: self.pid,
            message: &message,
            vllm_process_name: &self.process_label,
            spans: &span_names,
            fields: &fields,
        };
        let line = self.json.format_to_string(&line).map_err(|_| fmt::Error)?;
        writer.write_str(&line)?;
        writer.write_char('\n')
    }
}

/// One output line; its keys are written in field order.
#[derive(Serialize)]
struct JsonLine<'a> {
    asctime: &'a str,
    levelname: &'static str,
    name: &'a str,
    process: u32,
    message: &'a Value,
    /// Python's role name for the process, as in the text prefix.
    vllm_process_name: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    spans: &'a [&'a str],
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    fields: &'a FieldMap<'a>,
}

/// Records span fields as a JSON object, which [`VllmJsonEventFormatter`]
/// merges into each event inside the span.
#[derive(Debug, Default)]
pub(crate) struct JsonFields;

impl<'writer> FormatFields<'writer> for JsonFields {
    fn format_fields<R: RecordFields>(
        &self,
        mut writer: Writer<'writer>,
        fields: R,
    ) -> fmt::Result {
        let mut visitor = JsonVisitor::default();
        fields.record(&mut visitor);
        let object = serde_json::to_string(&visitor.0).map_err(|_| fmt::Error)?;
        writer.write_str(&object)
    }

    /// Merge fields recorded after the span was created, such as an
    /// `engine_request_id` declared `Empty`, into its object.
    fn add_fields(
        &self,
        current: &'writer mut FormattedFields<Self>,
        fields: &Record<'_>,
    ) -> fmt::Result {
        let mut visitor = JsonVisitor(parse_fields(&current.fields));
        fields.record(&mut visitor);
        let object = serde_json::to_string(&visitor.0).map_err(|_| fmt::Error)?;
        current.fields = object;
        Ok(())
    }
}

/// Parse an object written by [`JsonFields`]. Its keys are field names, which
/// never need escaping, so they borrow from `object`.
fn parse_fields(object: &str) -> FieldMap<'_> {
    serde_json::from_str(object).unwrap_or_default()
}

/// Collects fields as JSON values: numbers and booleans as such, everything
/// else (`%display`, `?debug`, errors, non-finite floats) as its formatted
/// string.
#[derive(Debug, Default)]
struct JsonVisitor<'a>(FieldMap<'a>);

impl JsonVisitor<'_> {
    fn insert(&mut self, field: &Field, value: impl Into<Value>) {
        // `r#type = ...` names its field `r#type`.
        let name = field.name();
        let name = name.strip_prefix("r#").unwrap_or(name);
        self.0.insert(name, value.into());
    }
}

impl Visit for JsonVisitor<'_> {
    fn record_f64(&mut self, field: &Field, value: f64) {
        match Number::from_f64(value) {
            Some(number) => self.insert(field, number),
            None => self.insert(field, value.to_string()),
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, value);
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert(field, format!("{value:?}"));
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use expect_test::expect;
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    use super::*;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Buffer {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `emit` under the JSON formatter and return its lines, with the
    /// time and pid replaced by placeholders.
    fn json_lines(emit: impl FnOnce()) -> Vec<String> {
        let buffer = Buffer::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_writer(buffer.clone())
            .with_ansi(false)
            .fmt_fields(JsonFields)
            .event_format(VllmJsonEventFormatter::new("RustFrontend"));
        let guard = tracing_subscriber::registry().with(layer).set_default();
        emit();
        drop(guard);

        let output = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        let pid = format!(r#""process": {}, "#, process::id());
        output
            .lines()
            .map(|line| {
                let asctime: Value =
                    serde_json::from_str::<Value>(line).unwrap()["asctime"].clone();
                assert!(line.contains(&pid), "{line}");
                line.replacen(&asctime.to_string(), r#""<asctime>""#, 1).replacen(
                    &pid,
                    r#""process": "<pid>", "#,
                    1,
                )
            })
            .collect()
    }

    /// An event from the `log` crate keeps its own target as `name`, without
    /// the bridge's `log.*` fields.
    #[test]
    fn log_crate_events_keep_their_target() {
        use tracing_log::log;

        let lines = json_lines(|| {
            tracing_log::format_trace(
                &log::Record::builder()
                    .target("hf_hub")
                    .level(log::Level::Warn)
                    .module_path_static(Some("hf_hub::api"))
                    .file_static(Some("src/api.rs"))
                    .line(Some(7))
                    .args(format_args!("probe"))
                    .build(),
            )
            .unwrap();
        });

        expect![[r#"
            [
                "{\"asctime\": \"<asctime>\", \"levelname\": \"WARNING\", \"name\": \"hf_hub\", \"process\": \"<pid>\", \"message\": \"probe\", \"vllm_process_name\": \"RustFrontend\"}",
            ]
        "#]]
        .assert_debug_eq(&lines);
    }

    #[test]
    fn asctime_matches_python_logging_default() {
        let asctime = time::macros::datetime!(2026-11-08 19:12:01.839).format(ASCTIME_FORMAT);

        assert_eq!(asctime.unwrap(), "2026-11-08 19:12:01,839");
    }

    /// Python's keys come first with `json.dumps` spacing; span names and the
    /// span and event fields, including ones recorded later, follow.
    #[test]
    fn json_lines_use_python_keys_then_spans_and_fields() {
        let lines = json_lines(|| {
            tracing::warn!(target: "vllm_server::routes", "Started server on {}", "0.0.0.0:8000");
            let span = tracing::info_span!(
                "chat_completions",
                request_id = %"chatcmpl-1",
                engine_request_id = tracing::field::Empty,
            );
            span.in_scope(|| {
                tracing::info!(target: "vllm_server::routes", prompt_tokens = 12, "accepted");
                tracing::Span::current().record("engine_request_id", "chatcmpl-1-0");
                tracing::error!(
                    target: "vllm_server::routes",
                    cancelled = true,
                    ratio = 0.5,
                    "failed: {}",
                    "中文"
                );
            });
        });

        expect![[r#"
            [
                "{\"asctime\": \"<asctime>\", \"levelname\": \"WARNING\", \"name\": \"vllm_server::routes\", \"process\": \"<pid>\", \"message\": \"Started server on 0.0.0.0:8000\", \"vllm_process_name\": \"RustFrontend\"}",
                "{\"asctime\": \"<asctime>\", \"levelname\": \"INFO\", \"name\": \"vllm_server::routes\", \"process\": \"<pid>\", \"message\": \"accepted\", \"vllm_process_name\": \"RustFrontend\", \"spans\": [\"chat_completions\"], \"fields\": {\"request_id\": \"chatcmpl-1\", \"prompt_tokens\": 12}}",
                "{\"asctime\": \"<asctime>\", \"levelname\": \"ERROR\", \"name\": \"vllm_server::routes\", \"process\": \"<pid>\", \"message\": \"failed: 中文\", \"vllm_process_name\": \"RustFrontend\", \"spans\": [\"chat_completions\"], \"fields\": {\"request_id\": \"chatcmpl-1\", \"engine_request_id\": \"chatcmpl-1-0\", \"cancelled\": true, \"ratio\": 0.5}}",
            ]
        "#]]
        .assert_debug_eq(&lines);
    }

    /// Fields named like a top-level key stay under `fields`; an inner span's
    /// or the event's field replaces an outer one of the same name.
    #[test]
    fn fields_never_replace_top_level_keys() {
        let lines = json_lines(|| {
            let outer =
                tracing::info_span!("outer", request_id = "outer-1", message = "span message");
            outer.in_scope(|| {
                let inner =
                    tracing::info_span!("inner", request_id = "inner-1", levelname = "DEBUG");
                inner.in_scope(|| {
                    tracing::info!(target: "vllm_server", name = "field name", r#type = "raw", "event message");
                });
            });
        });

        expect![[r#"
            [
                "{\"asctime\": \"<asctime>\", \"levelname\": \"INFO\", \"name\": \"vllm_server\", \"process\": \"<pid>\", \"message\": \"event message\", \"vllm_process_name\": \"RustFrontend\", \"spans\": [\"outer\", \"inner\"], \"fields\": {\"request_id\": \"inner-1\", \"message\": \"span message\", \"levelname\": \"DEBUG\", \"name\": \"field name\", \"type\": \"raw\"}}",
            ]
        "#]]
        .assert_debug_eq(&lines);
    }
}
