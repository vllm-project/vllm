// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The JSON log formatter, matching Python vLLM's `--logging-config.formatter
//! json` (`pythonjsonlogger` with `vllm.logging_utils.formatter.JSON_FORMAT`).

use std::{fmt, process};

use serde_json::{Map, Number, Value};
use time::macros::format_description;
use tracing::field::{Field, Visit};
use tracing::span::Record;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::field::RecordFields;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::registry::LookupSpan;

use crate::VllmLocalTimer;

/// Python `logging`'s default `asctime`, such as `2026-11-08 19:12:01,839`.
const ASCTIME_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[year]-[month]-[day] [hour]:[minute]:[second],[subsecond digits:3]");

/// Formats each event as one JSON object per line, with Python's keys in
/// Python's order: `asctime`, `levelname`, `name` (the tracing target),
/// `processName`, `process`, `message`, then the extras: `vllm_process_name`,
/// the fields of the enclosing spans from the root, and the event's other
/// fields, like `pythonjsonlogger`'s `extra`. A later field replaces an
/// earlier one with the same name.
#[derive(Debug, Clone)]
pub(crate) struct VllmJsonEventFormatter {
    process_label: String,
    pid: u32,
    timer: VllmLocalTimer,
}

impl VllmJsonEventFormatter {
    pub(crate) fn new(process_label: &str) -> Self {
        Self {
            process_label: process_label.to_owned(),
            pid: process::id(),
            timer: VllmLocalTimer::default(),
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
        let meta = event.metadata();
        let mut fields = JsonVisitor::default();
        event.record(&mut fields);
        let message = fields.0.remove("message").unwrap_or_else(|| Value::from(""));

        let mut object = Map::new();
        let asctime = self.timer.now().format(ASCTIME_FORMAT).map_err(|_| fmt::Error)?;
        object.insert("asctime".into(), asctime.into());
        object.insert("levelname".into(), level_name(meta.level()).into());
        object.insert("name".into(), meta.target().into());
        // Python's `processName` is the multiprocessing name; the Rust frontend
        // has only its label, which Python also sets as `vllm_process_name`.
        object.insert("processName".into(), self.process_label.as_str().into());
        object.insert("process".into(), self.pid.into());
        object.insert("message".into(), message);
        object.insert(
            "vllm_process_name".into(),
            self.process_label.as_str().into(),
        );
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                let ext = span.extensions();
                if let Some(span_fields) = ext.get::<FormattedFields<JsonFields>>() {
                    object.extend(parse_object(&span_fields.fields));
                }
            }
        }
        object.extend(fields.0);

        let line = serde_json::to_string(&object).map_err(|_| fmt::Error)?;
        writeln!(writer, "{line}")
    }
}

/// Python's level names, as the text formatter writes them.
fn level_name(level: &Level) -> &'static str {
    match *level {
        Level::TRACE => "TRACE",
        Level::DEBUG => "DEBUG",
        Level::INFO => "INFO",
        Level::WARN => "WARNING",
        Level::ERROR => "ERROR",
    }
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
        write!(writer, "{}", Value::Object(visitor.0))
    }

    /// Merge fields recorded after the span was created, such as an
    /// `engine_request_id` declared `Empty`, into its object.
    fn add_fields(
        &self,
        current: &'writer mut FormattedFields<Self>,
        fields: &Record<'_>,
    ) -> fmt::Result {
        let mut visitor = JsonVisitor(parse_object(&current.fields));
        fields.record(&mut visitor);
        current.fields = Value::Object(visitor.0).to_string();
        Ok(())
    }
}

/// Parse an object written by [`JsonFields`].
fn parse_object(fields: &str) -> Map<String, Value> {
    serde_json::from_str(fields).unwrap_or_default()
}

/// Collects fields as JSON values: numbers and booleans as such, everything
/// else (`%display`, `?debug`, errors) as its formatted string.
#[derive(Debug, Default)]
struct JsonVisitor(Map<String, Value>);

impl JsonVisitor {
    fn insert(&mut self, field: &Field, value: impl Into<Value>) {
        self.0.insert(field.name().to_owned(), value.into());
    }
}

impl Visit for JsonVisitor {
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

    #[test]
    fn asctime_matches_python_logging_default() {
        let asctime = time::macros::datetime!(2026-11-08 19:12:01.839).format(ASCTIME_FORMAT);

        assert_eq!(asctime.unwrap(), "2026-11-08 19:12:01,839");
    }

    /// Each event is one line with Python's keys first, then the extras:
    /// `vllm_process_name`, span fields (including ones recorded later), and
    /// event fields.
    #[test]
    fn json_lines_use_python_keys_and_extras() {
        let buffer = Buffer::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_writer(buffer.clone())
            .with_ansi(false)
            .fmt_fields(JsonFields)
            .event_format(VllmJsonEventFormatter::new("RustFrontend"));
        let _guard = tracing_subscriber::registry().with(layer).set_default();

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
                request_id = "chatcmpl-1-override",
                cancelled = true,
                ratio = 0.5,
                "failed"
            );
        });

        let output = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        let lines: Vec<String> = output
            .lines()
            .map(|line| {
                let mut object: Map<String, Value> = serde_json::from_str(line).unwrap();
                assert!(object["asctime"].is_string());
                assert_eq!(object["process"], Value::from(process::id()));
                object.insert("asctime".into(), "<asctime>".into());
                object.insert("process".into(), "<pid>".into());
                serde_json::to_string(&object).unwrap()
            })
            .collect();
        expect![[r#"
            [
                "{\"asctime\":\"<asctime>\",\"levelname\":\"WARNING\",\"name\":\"vllm_server::routes\",\"processName\":\"RustFrontend\",\"process\":\"<pid>\",\"message\":\"Started server on 0.0.0.0:8000\",\"vllm_process_name\":\"RustFrontend\"}",
                "{\"asctime\":\"<asctime>\",\"levelname\":\"INFO\",\"name\":\"vllm_server::routes\",\"processName\":\"RustFrontend\",\"process\":\"<pid>\",\"message\":\"accepted\",\"vllm_process_name\":\"RustFrontend\",\"request_id\":\"chatcmpl-1\",\"prompt_tokens\":12}",
                "{\"asctime\":\"<asctime>\",\"levelname\":\"ERROR\",\"name\":\"vllm_server::routes\",\"processName\":\"RustFrontend\",\"process\":\"<pid>\",\"message\":\"failed\",\"vllm_process_name\":\"RustFrontend\",\"request_id\":\"chatcmpl-1-override\",\"engine_request_id\":\"chatcmpl-1-0\",\"ratio\":0.5,\"cancelled\":true}",
            ]
        "#]]
        .assert_debug_eq(&lines);
    }
}
