// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Generated-output logging for `--enable-log-outputs`, mirroring Python's
//! `RequestLogger.log_outputs`.

use tracing::{debug, info};

/// Which part of a response one output log line covers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LoggedOutput {
    /// The complete output of a non-streaming response.
    Response,
    /// One streamed delta.
    StreamDelta,
    /// The accumulated output of a finished stream.
    StreamComplete,
}

/// Log one generated output like Python's `RequestLogger.log_outputs`: the
/// output text and finish reason at INFO, the output token IDs at DEBUG. The
/// request ID comes from the enclosing request span.
pub(crate) fn log_output(
    kind: LoggedOutput,
    output: &str,
    token_ids: Option<&[u32]>,
    finish_reason: Option<&str>,
) {
    let stream = match kind {
        LoggedOutput::Response => None,
        LoggedOutput::StreamDelta => Some("delta"),
        LoggedOutput::StreamComplete => Some("complete"),
    };
    if let Some(token_ids) = token_ids {
        debug!(stream, ?token_ids, "generated response details");
    }
    info!(stream, ?output, finish_reason, "generated response");
}

/// Describe a chat output for the log the way Python does: the content, or
/// the tool calls as `[tool_calls: name(arguments), ...]` when there is no
/// content.
pub(crate) fn chat_output_text<'a>(
    content: &str,
    tool_calls: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    if !content.is_empty() {
        return content.to_string();
    }
    let tool_calls = tool_calls
        .into_iter()
        .map(|(name, arguments)| format!("{name}({arguments})"))
        .collect::<Vec<_>>();
    if tool_calls.is_empty() {
        String::new()
    } else {
        format!("[tool_calls: {}]", tool_calls.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::chat_output_text;

    #[test]
    fn chat_output_text_prefers_content_then_describes_tool_calls() {
        let tool_calls = [("get_weather", r#"{"city": "Paris"}"#), ("get_time", "{}")];
        assert_eq!(
            chat_output_text("Let me check.", tool_calls),
            "Let me check."
        );
        assert_eq!(
            chat_output_text("", tool_calls),
            r#"[tool_calls: get_weather({"city": "Paris"}), get_time({})]"#
        );
        assert_eq!(chat_output_text("", []), "");
    }
}
