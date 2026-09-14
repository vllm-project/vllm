// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt::Write as _;

use tracing::{Level, enabled, info};
use vllm_chat::{AssistantContentBlock, AssistantMessage};

use super::types::ChatMessageDelta;

pub(super) fn log_message(
    message: &AssistantMessage,
    include_reasoning: bool,
    finish_reason: &str,
    streaming: bool,
) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut output = String::new();
    for block in &message.content {
        match block {
            AssistantContentBlock::Text { text } => output.push_str(text),
            AssistantContentBlock::Reasoning { text } if include_reasoning => {
                let _ = write!(output, "[reasoning: {text}]");
            }
            AssistantContentBlock::ToolCall(call) => {
                let _ = write!(output, "[tool_calls: {}({})]", call.name, call.arguments);
            }
            AssistantContentBlock::Reasoning { .. } => {}
        }
    }
    let stream_info = if streaming {
        " (streaming complete)"
    } else {
        ""
    };
    info!("Generated response{stream_info}: output: {output:?}, finish_reason: {finish_reason}");
}

pub(super) fn log_delta(delta: &ChatMessageDelta) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut output = delta.content.clone().unwrap_or_default();
    if let Some(reasoning) = &delta.reasoning
        && !reasoning.is_empty()
    {
        let _ = write!(output, "[reasoning: {reasoning}]");
    }
    if let Some(calls) = &delta.tool_calls {
        for call in calls {
            if let Some(function) = &call.function
                && let Some(arguments) = &function.arguments
                && !arguments.is_empty()
            {
                let _ = write!(output, "[tool_calls: {arguments}]");
            }
        }
    }
    if !output.is_empty() {
        info!("Generated response (streaming delta): output: {output:?}, finish_reason: None");
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use expect_test::expect;
    use vllm_chat::AssistantToolCall;

    use super::*;
    use crate::routes::openai::utils::types::{FunctionCallDelta, ToolCallDelta};

    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn capture_logs(level: Level, f: impl FnOnce()) -> String {
        let logs = LogBuffer::default();
        let buffer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_target(false)
            .with_level(false)
            .with_max_level(level)
            .with_writer(move || buffer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        String::from_utf8(logs.0.lock().unwrap().clone()).unwrap()
    }

    fn message() -> AssistantMessage {
        AssistantMessage {
            content: vec![
                AssistantContentBlock::Reasoning {
                    text: "I should use the weather tool.".into(),
                },
                AssistantContentBlock::Text {
                    text: "Let me check the weather.".into(),
                },
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: "call-1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"San Francisco"}"#.into(),
                }),
            ],
        }
    }

    #[test]
    fn message_output_includes_reasoning_text_and_tool_calls() {
        let logs = capture_logs(Level::INFO, || {
            log_message(&message(), true, "tool_calls", false);
        });
        expect![["Generated response: output: \"\
             [reasoning: I should use the weather tool.]\
             Let me check the weather.\
             [tool_calls: get_weather({\\\"city\\\":\\\"San Francisco\\\"})]\", \
             finish_reason: tool_calls\n"]]
        .assert_eq(&logs);
    }

    #[test]
    fn streaming_complete_output_uses_python_style_format() {
        let logs = capture_logs(Level::INFO, || {
            log_message(&message(), false, "streaming_complete", true);
        });
        expect![["Generated response (streaming complete): output: \"\
             Let me check the weather.\
             [tool_calls: get_weather({\\\"city\\\":\\\"San Francisco\\\"})]\", \
             finish_reason: streaming_complete\n"]]
        .assert_eq(&logs);
    }

    #[test]
    fn message_output_is_silent_when_info_is_disabled() {
        let logs = capture_logs(Level::WARN, || {
            log_message(&message(), true, "tool_calls", false);
        });
        assert!(logs.is_empty());
    }

    #[test]
    fn delta_output_includes_content_reasoning_and_tool_arguments() {
        let delta = ChatMessageDelta {
            content: Some("Let me check the weather.".into()),
            reasoning: Some("I should use the weather tool.".into()),
            tool_calls: Some(vec![ToolCallDelta {
                index: 0,
                id: Some("call-1".into()),
                tool_type: Some("function".into()),
                function: Some(FunctionCallDelta {
                    name: Some("get_weather".into()),
                    arguments: Some(r#"{"city":"San Francisco"}"#.into()),
                }),
            }]),
            ..Default::default()
        };
        let logs = capture_logs(Level::INFO, || {
            log_delta(&delta);
        });
        expect![["Generated response (streaming delta): output: \"\
             Let me check the weather.\
             [reasoning: I should use the weather tool.]\
             [tool_calls: {\\\"city\\\":\\\"San Francisco\\\"}]\", \
             finish_reason: None\n"]]
        .assert_eq(&logs);
    }
}
