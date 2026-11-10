// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared tracing subscriber and log formatting for vLLM Rust binaries.

pub mod timing;

use std::str::FromStr;
use std::{env, fmt, process};

use serde_with::{DeserializeFromStr, SerializeDisplay};
use time::UtcOffset;
use time::macros::format_description;
use tracing::level_filters::LevelFilter;
use tracing::subscriber::DefaultGuard;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{Layer, Registry};

const CYAN: &str = "\x1b[0;36m";
const GREY: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const WHITE: &str = "\x1b[37m";
const RESET: &str = "\x1b[0m";
const VLLM_TIME_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[month]-[day] [hour]:[minute]:[second]");

/// Python's logging level names, as `--log-level`, `--logging-config` and
/// `VLLM_LOGGING_LEVEL` spell them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SerializeDisplay, DeserializeFromStr)]
pub enum LogLevel {
    Critical,
    Error,
    Warning,
    Info,
    Debug,
    Notset,
}

impl LogLevel {
    /// The level from `VLLM_LOGGING_LEVEL`, `INFO` when it is unset or invalid.
    pub fn from_env() -> Self {
        Self::from_env_value(env::var("VLLM_LOGGING_LEVEL").ok().as_deref())
    }

    fn from_env_value(value: Option<&str>) -> Self {
        value.and_then(|value| value.parse().ok()).unwrap_or(Self::Info)
    }

    /// The Python level name, which Python's `--log-level` choices accept.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::Error => "ERROR",
            Self::Warning => "WARNING",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Notset => "NOTSET",
        }
    }
}

/// Case-insensitive, with Python logging's `WARN` and `FATAL` aliases.
impl FromStr for LogLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_uppercase().as_str() {
            "CRITICAL" | "FATAL" => Ok(Self::Critical),
            "ERROR" => Ok(Self::Error),
            "WARNING" | "WARN" => Ok(Self::Warning),
            "INFO" => Ok(Self::Info),
            "DEBUG" => Ok(Self::Debug),
            "NOTSET" => Ok(Self::Notset),
            _ => Err(format!(
                "`{value}` is not a log level; expected one of CRITICAL, ERROR, WARNING, INFO, \
                 DEBUG, NOTSET"
            )),
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Critical | LogLevel::Error => LevelFilter::ERROR,
            LogLevel::Warning => LevelFilter::WARN,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Debug => LevelFilter::DEBUG,
            // Python's `vllm` logger at NOTSET defers to the root logger,
            // which is at WARNING unless something else configures it.
            LogLevel::Notset => LevelFilter::WARN,
        }
    }
}

/// Install the process-wide vLLM-style tracing subscriber at `log_level`.
pub fn init_tracing(process_label: &str, log_level: LogLevel) {
    let _ = tracing_subscriber::registry()
        .with(vllm_fmt_layer(process_label, log_level))
        .try_init();
}

/// Install the vLLM-style tracing subscriber at `log_level` on the current
/// thread until the guard drops, for events emitted before the process-wide
/// level is known.
pub fn set_default_tracing(process_label: &str, log_level: LogLevel) -> DefaultGuard {
    tracing_subscriber::registry()
        .with(vllm_fmt_layer(process_label, log_level))
        .set_default()
}

/// Install the process-wide vLLM-style tracing subscriber, with an
/// additional layer (e.g. the multimodal preprocessing timing collector).
pub fn init_tracing_with<L>(process_label: &str, log_level: LogLevel, layer: L)
where
    L: Layer<Registry> + Send + Sync + 'static,
{
    let _ = tracing_subscriber::registry()
        .with(layer)
        .with(vllm_fmt_layer(process_label, log_level))
        .try_init();
}

/// The vLLM-style fmt layer filtered by `log_level` / `RUST_LOG`.
fn vllm_fmt_layer<S>(
    process_label: &str,
    log_level: LogLevel,
) -> impl Layer<S> + Send + Sync + 'static + use<S>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    let filter = build_targets_filter(log_level, env::var("RUST_LOG").ok().as_deref());
    let formatter = VllmEventFormatter::new(process_label);

    tracing_subscriber::fmt::layer()
        .event_format(formatter)
        .with_writer(std::io::stderr)
        .with_filter(filter)
}

/// Build the CLI log filter by merging the vLLM-style default level with
/// Rust-style target overrides.
///
/// Precedence:
/// - Start from `log_level` (by default `VLLM_LOGGING_LEVEL`) as the default level for all
///   targets.
/// - If `RUST_LOG` contains a global default level such as `warn`, it overrides `log_level`.
/// - Any explicit target directives in `RUST_LOG`, such as `hyper=info`, override whichever default
///   level is active for those targets only.
fn build_targets_filter(log_level: LogLevel, rust_log: Option<&str>) -> Targets {
    let mut filter = Targets::new().with_default(LevelFilter::from(log_level));

    if let Some(rust_log) = rust_log
        && !rust_log.is_empty()
    {
        let rust_log_targets: Targets = rust_log.parse().expect("failed to parse `RUST_LOG`");
        if let Some(default_level) = rust_log_targets.default_level() {
            filter = filter.with_default(default_level);
        }
        filter = filter.with_targets(rust_log_targets);
    }

    filter
}

#[derive(Debug, Clone, Copy)]
struct VllmLocalTimer {
    local_offset: UtcOffset,
}

impl Default for VllmLocalTimer {
    fn default() -> Self {
        let local_offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
        Self { local_offset }
    }
}

impl FormatTime for VllmLocalTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let now = time::OffsetDateTime::now_utc().to_offset(self.local_offset);
        let formatted = now.format(VLLM_TIME_FORMAT).map_err(|_| fmt::Error)?;
        w.write_str(&formatted)
    }
}

#[derive(Debug, Clone)]
struct VllmEventFormatter {
    prefix: String,
    timer: VllmLocalTimer,
}

impl VllmEventFormatter {
    fn new(process_label: &str) -> Self {
        Self {
            prefix: format!("({process_label} pid={})", process::id()),
            timer: VllmLocalTimer::default(),
        }
    }

    fn write_process_prefix(&self, writer: &mut Writer<'_>, ansi: bool) -> fmt::Result {
        write_colored(writer, ansi, Some(CYAN), &self.prefix)?;
        writer.write_char(' ')
    }

    fn write_level(&self, writer: &mut Writer<'_>, level: &Level, ansi: bool) -> fmt::Result {
        let (text, color) = match *level {
            Level::TRACE => ("TRACE", WHITE),
            Level::DEBUG => ("DEBUG", WHITE),
            Level::INFO => ("INFO", GREEN),
            Level::WARN => ("WARNING", YELLOW),
            Level::ERROR => ("ERROR", RED),
        };
        write_colored(writer, ansi, Some(color), text)
    }

    fn write_timestamp(&self, writer: &mut Writer<'_>, ansi: bool) -> fmt::Result {
        if ansi {
            writer.write_str(GREY)?;
        }
        if self.timer.format_time(writer).is_err() {
            writer.write_str("<unknown time>")?;
        }
        if ansi {
            writer.write_str(RESET)?;
        }
        Ok(())
    }

    fn write_location(
        &self,
        writer: &mut Writer<'_>,
        file: Option<&str>,
        line: Option<u32>,
        full_path: bool,
        ansi: bool,
    ) -> fmt::Result {
        let Some(file) = file else {
            return Ok(());
        };
        let file = if full_path {
            file
        } else {
            shorten_file_path(file)
        };
        if ansi {
            writer.write_str(GREY)?;
        }
        match line {
            Some(line) => write!(writer, "[{file}:{line}]")?,
            None => write!(writer, "[{file}]")?,
        }
        if ansi {
            writer.write_str(RESET)?;
        }
        Ok(())
    }

    fn write_scope<S, N>(&self, ctx: &FmtContext<'_, S, N>, writer: &mut Writer<'_>) -> fmt::Result
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
        N: for<'writer> FormatFields<'writer> + 'static,
    {
        let Some(scope) = ctx.event_scope() else {
            return Ok(());
        };

        let mut seen = false;
        for span in scope.from_root() {
            if seen {
                writer.write_str(":")?;
            }
            seen = true;
            writer.write_str(span.metadata().name())?;

            let ext = span.extensions();
            if let Some(fields) = ext.get::<FormattedFields<N>>()
                && !fields.is_empty()
            {
                write!(writer, "{{{fields}}}")?;
            }
        }

        if seen {
            writer.write_str(": ")?;
        }

        Ok(())
    }
}

impl<S, N> FormatEvent<S, N> for VllmEventFormatter
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();
        let ansi = writer.has_ansi_escapes();

        self.write_process_prefix(&mut writer, ansi)?;
        self.write_level(&mut writer, meta.level(), ansi)?;
        writer.write_char(' ')?;
        self.write_timestamp(&mut writer, ansi)?;
        writer.write_char(' ')?;
        // Use the full file path only when DEBUG (or more verbose) is enabled anywhere,
        // independent of the level of this particular event. Filenames alone are often
        // ambiguous, but full paths are too noisy for normal INFO-level operation.
        let full_path = LevelFilter::current() >= LevelFilter::DEBUG;
        self.write_location(&mut writer, meta.file(), meta.line(), full_path, ansi)?;
        writer.write_char(' ')?;
        self.write_scope(ctx, &mut writer)?;
        ctx.format_fields(writer.by_ref(), event)?;
        writer.write_char('\n')
    }
}

/// Shorten a source file path for log output while preserving enough context
/// for common Rust entrypoint and module filenames.
///
/// - For `mod.rs`, keep the parent directory as `parent/mod.rs`.
/// - For `src/lib.rs` and `src/main.rs`, keep one additional component as `crate/src/lib.rs` or
///   `crate/src/main.rs` when available.
/// - Other files are displayed as just the basename.
fn shorten_file_path(file: &str) -> &str {
    let mut parts = file.rsplit('/');
    let name = parts.next().unwrap_or(file);
    let parent = parts.next();
    let grandparent = parts.next();

    let Some(parent) = parent else {
        return file;
    };

    if name == "mod.rs" {
        return &file[file.len() - parent.len() - 1 - name.len()..];
    }

    if !matches!(name, "lib.rs" | "main.rs") || parent != "src" {
        return name;
    }
    let Some(grandparent) = grandparent else {
        return file;
    };

    &file[file.len() - grandparent.len() - 1 - parent.len() - 1 - name.len()..]
}

fn write_colored(
    writer: &mut Writer<'_>,
    ansi: bool,
    color: Option<&str>,
    text: &str,
) -> fmt::Result {
    if ansi {
        if let Some(color) = color {
            writer.write_str(color)?;
        }
        writer.write_str(text)?;
        if color.is_some() {
            writer.write_str(RESET)?;
        }
        return Ok(());
    }

    writer.write_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatter_prefix_uses_process_label() {
        let formatter = VllmEventFormatter::new("Bench");

        assert_eq!(formatter.prefix, format!("(Bench pid={})", process::id()));
    }

    #[test]
    fn rust_log_target_overrides_are_merged_with_vllm_default_level() {
        let filter = build_targets_filter(LogLevel::Debug, Some("hyper=warn,tower=error"));

        assert_eq!(filter.to_string(), "tower=error,hyper=warn,debug");
    }

    #[test]
    fn rust_log_default_level_overrides_vllm_default_level() {
        let filter = build_targets_filter(LogLevel::Debug, Some("warn,hyper=info"));

        assert_eq!(filter.to_string(), "hyper=info,warn");
    }

    #[test]
    fn notset_level_matches_python_root_logger_default() {
        let filter = build_targets_filter(LogLevel::Notset, None);

        assert_eq!(filter.to_string(), "warn");
    }

    #[test]
    fn invalid_vllm_level_falls_back_to_info() {
        assert_eq!(LogLevel::from_env_value(Some("bogus")), LogLevel::Info);
        assert_eq!(LogLevel::from_env_value(None), LogLevel::Info);
        assert_eq!(LogLevel::from_env_value(Some("warn")), LogLevel::Warning);
    }

    #[test]
    fn location_path_uses_filename_for_non_ambiguous_files() {
        assert_eq!(shorten_file_path("src/cmd/src/logging.rs"), "logging.rs");
        assert_eq!(shorten_file_path("src/chat/lib.rs"), "lib.rs");
        assert_eq!(shorten_file_path("src/chat/main.rs"), "main.rs");
        assert_eq!(shorten_file_path("src/chat/src/xmod.rs"), "xmod.rs");
    }

    #[test]
    fn location_path_keeps_more_context_for_common_entrypoint_filenames() {
        assert_eq!(shorten_file_path("src/lib.rs"), "src/lib.rs");
        assert_eq!(shorten_file_path("src/chat/src/lib.rs"), "chat/src/lib.rs");
        assert_eq!(shorten_file_path("src/cmd/src/main.rs"), "cmd/src/main.rs");
        assert_eq!(shorten_file_path("mod.rs"), "mod.rs");
        assert_eq!(shorten_file_path("src/chat/src/tool/mod.rs"), "tool/mod.rs");
    }
}
