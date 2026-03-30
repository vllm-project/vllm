use std::{fmt, process};

use time::UtcOffset;
use time::macros::format_description;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{FmtContext, FormattedFields};
use tracing_subscriber::registry::LookupSpan;

const CYAN: &str = "\x1b[0;36m";
const GREY: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const WHITE: &str = "\x1b[37m";
const RESET: &str = "\x1b[0m";
const VLLM_TIME_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[month]-[day] [hour]:[minute]:[second]");

const PROCESS_LABEL: &str = "RustFrontend";

/// Install the process-wide vLLM-style tracing subscriber for the CLI binary.
pub(crate) fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));
    let formatter = VllmEventFormatter::new();

    let _ = tracing_subscriber::fmt()
        .event_format(formatter)
        .with_env_filter(filter)
        .try_init();
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
    fn new() -> Self {
        Self {
            prefix: format!("({} pid={})", PROCESS_LABEL, process::id()),
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
        ansi: bool,
    ) -> fmt::Result {
        let Some(file) = file else {
            return Ok(());
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
        self.write_location(&mut writer, meta.file(), meta.line(), ansi)?;
        writer.write_char(' ')?;
        self.write_scope(ctx, &mut writer)?;
        ctx.format_fields(writer.by_ref(), event)?;
        writer.write_char('\n')
    }
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
