// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::borrow::Cow;
use std::time::Instant;

use futures::StreamExt;

use super::streaming::StreamedResponseHandler;
use super::{RequestFuncInput, RequestFuncOutput, build_headers};
use crate::error::Result;

/// Responses API streaming event (minimal fields).
#[derive(serde::Deserialize)]
struct ResponsesEvent {
    #[serde(rename = "type")]
    event_type: Option<String>,
    delta: Option<String>,
    response: Option<ResponsesEventResponse>,
    error: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct ResponsesEventResponse {
    usage: Option<ResponsesUsage>,
}

#[derive(serde::Deserialize)]
struct ResponsesUsage {
    output_tokens: Option<u64>,
}

/// Backend for the OpenAI Responses API (/v1/responses).
#[derive(Clone)]
pub struct OpenAIResponsesBackend;

impl OpenAIResponsesBackend {
    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        let headers_map = build_headers(None, &input.extra_headers, &input.request_id);
        let mut request = client.post(&input.api_url).json(&build_payload(input));
        for (k, v) in &headers_map {
            request = request.header(k, v);
        }

        let st = Instant::now();
        let failed = |error: String| RequestFuncOutput {
            prompt_len: input.prompt_len,
            error,
            ..Default::default()
        };

        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => return Ok(failed(format!("{e:#}"))),
        };
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Ok(failed(if body.is_empty() {
                format!("HTTP {status}")
            } else {
                format!("HTTP {status}: {body}")
            }));
        }

        let mut stream = ResponsesStream::new(input, st);
        let mut handler = StreamedResponseHandler::new();
        let mut body = response.bytes_stream();
        while let Some(chunk_result) = body.next().await {
            let chunk_bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => return Ok(failed(format!("Stream error: {e}"))),
            };
            // Unlike the other backends, chunks are not trimmed: a chunk can end
            // between an event's `event:` and `data:` lines.
            for message in handler.add_chunk(&chunk_bytes) {
                stream.handle_message(message, Instant::now());
            }
        }
        Ok(stream.finish())
    }
}

fn build_payload(input: &RequestFuncInput) -> serde_json::Value {
    let model = input.model_name.as_deref().unwrap_or(&input.model);
    let mut payload = serde_json::json!({
        "model": model,
        "input": input.prompt,
        "max_output_tokens": input.output_len,
        "stream": true,
        // `store` defaults to true. A server started with
        // VLLM_ENABLE_RESPONSES_API_STORE=1 would keep every response for the
        // lifetime of the process.
        "store": false,
    });
    if input.ignore_eos {
        payload["ignore_eos"] = serde_json::json!(true);
    }
    if let Some(serde_json::Value::Object(map)) = input.extra_body.as_ref() {
        for (k, v) in map {
            payload[k] = v.clone();
        }
    }
    payload
}

/// Timing and usage accumulated from one Responses API event stream.
struct ResponsesStream {
    output: RequestFuncOutput,
    st: Instant,
    most_recent_timestamp: Instant,
    first_token_received: bool,
    saw_terminal_event: bool,
    stream_error: Option<String>,
}

impl ResponsesStream {
    fn new(input: &RequestFuncInput, st: Instant) -> Self {
        Self {
            output: RequestFuncOutput {
                prompt_len: input.prompt_len,
                itl: Vec::with_capacity(input.output_len.max(1)),
                ..Default::default()
            },
            st,
            most_recent_timestamp: st,
            first_token_received: false,
            saw_terminal_event: false,
            stream_error: None,
        }
    }

    fn handle_message(&mut self, message: &str, timestamp: Instant) {
        let mut data_lines = message
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|data| data.strip_prefix(' ').unwrap_or(data));
        // SSE comments (keep-alives) have no data line.
        let Some(first) = data_lines.next() else {
            return;
        };
        // Per the SSE spec, a payload spread over several `data:` lines is
        // joined with newlines.
        let data: Cow<'_, str> = match data_lines.next() {
            None => Cow::Borrowed(first),
            Some(second) => Cow::Owned(
                [first, second].into_iter().chain(data_lines).collect::<Vec<_>>().join("\n"),
            ),
        };
        let Ok(event) = serde_json::from_str::<ResponsesEvent>(&data) else {
            return;
        };

        match event.event_type.as_deref() {
            // Reasoning deltas count toward TTFT and ITL because the server is
            // already decoding. Only output text is collected.
            Some(kind @ ("response.output_text.delta" | "response.reasoning_text.delta")) => {
                if self.first_token_received {
                    self.output
                        .itl
                        .push(timestamp.duration_since(self.most_recent_timestamp).as_secs_f64());
                } else {
                    self.first_token_received = true;
                    self.output.ttft = timestamp.duration_since(self.st).as_secs_f64();
                }
                if kind == "response.output_text.delta"
                    && let Some(delta) = &event.delta
                {
                    self.output.generated_text.push_str(delta);
                }
                // Only token events move the end of the request, so
                // `latency - ttft` equals `sum(itl)`.
                self.most_recent_timestamp = timestamp;
            }
            // vLLM reports truncation at the output limit as `response.completed`
            // with `status: "incomplete"`; the spec also has `response.incomplete`.
            Some("response.completed" | "response.incomplete") => {
                self.saw_terminal_event = true;
                if let Some(n) = event.response.and_then(|r| r.usage?.output_tokens) {
                    self.output.output_tokens = n as usize;
                }
            }
            kind if matches!(kind, Some("error" | "response.failed")) || event.error.is_some() => {
                self.stream_error =
                    Some(event.error.map_or_else(|| data.to_string(), |e| e.to_string()));
            }
            _ => {}
        }
    }

    fn finish(mut self) -> RequestFuncOutput {
        self.output.latency = self.most_recent_timestamp.duration_since(self.st).as_secs_f64();
        if let Some(error) = self.stream_error {
            self.output.error = error;
        } else if !self.first_token_received {
            self.output.error = "Never received a token delta to calculate TTFT. \
                                 This response will be marked as failed!"
                .to_string();
        } else if !self.saw_terminal_event {
            // The terminal event carries the usage block, so without it the
            // token counts are missing as well as the tail of the response.
            self.output.error = "Stream ended without a terminal event. \
                                 This response will be marked as failed!"
                .to_string();
        } else {
            self.output.success = true;
        }
        self.output
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    fn event(kind: &str, fields: &str) -> String {
        format!("event: {kind}\ndata: {{\"type\":\"{kind}\"{fields}}}\n\n")
    }

    fn text_delta(delta: &str) -> String {
        event(
            "response.output_text.delta",
            &format!(",\"delta\":\"{delta}\""),
        )
    }

    fn completed(output_tokens: u64) -> String {
        event(
            "response.completed",
            &format!(",\"response\":{{\"usage\":{{\"output_tokens\":{output_tokens}}}}}"),
        )
    }

    /// Feeds raw stream chunks through the same path as `send_request`,
    /// with chunk `i` arriving at `10 * (i + 1)` ms.
    fn run(chunks: &[String]) -> RequestFuncOutput {
        let st = Instant::now();
        let mut stream = ResponsesStream::new(&RequestFuncInput::default(), st);
        let mut handler = StreamedResponseHandler::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let timestamp = st + Duration::from_millis(10 * (i as u64 + 1));
            for message in handler.add_chunk(chunk.as_bytes()) {
                stream.handle_message(message, timestamp);
            }
        }
        stream.finish()
    }

    #[test]
    fn test_reasoning_starts_ttft_and_latency_ends_at_last_token() {
        let output = run(&[
            event("response.reasoning_text.delta", ",\"delta\":\"hmm\""),
            text_delta("Hi"),
            text_delta(" there"),
            completed(3),
        ]);

        assert!(output.success, "{}", output.error);
        assert_eq!(output.ttft, 0.01);
        assert_eq!(output.itl, vec![0.01, 0.01]);
        // The terminal event arrives at 40 ms but does not extend the request.
        assert_eq!(output.latency, 0.03);
        assert_eq!(output.generated_text, "Hi there");
        assert_eq!(output.output_tokens, 3);
    }

    #[test]
    fn test_event_line_split_from_data_line() {
        let delta = text_delta("Hi");
        let (event_line, data_line) = delta.split_at(delta.find("data:").unwrap());
        let output = run(&[
            event_line.to_string(),
            data_line.to_string(),
            event("response.incomplete", ""),
        ]);

        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "Hi");
    }

    #[test]
    fn test_multi_line_data_payload() {
        let output = run(&[
            "event: response.output_text.delta\n\
             data: {\"type\":\"response.output_text.delta\",\n\
             data: \"delta\":\"Hi\"}\n\n"
                .to_string(),
            event("response.incomplete", ""),
        ]);

        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "Hi");
    }

    #[test]
    fn test_stream_without_terminal_event_fails() {
        let output = run(&[text_delta("Hi")]);

        assert!(!output.success);
        assert!(output.error.contains("terminal event"));
    }

    #[test]
    fn test_failed_event_fails_the_request() {
        let output = run(&[
            text_delta("Hi"),
            event(
                "response.failed",
                ",\"response\":{\"error\":{\"code\":\"server_error\"}}",
            ),
        ]);

        assert!(!output.success);
        assert!(output.error.contains("server_error"));
    }

    #[test]
    fn test_payload_uses_responses_fields() {
        let input = RequestFuncInput {
            prompt: Arc::from("hello"),
            model: "test-model".to_string(),
            output_len: 128,
            ignore_eos: true,
            extra_body: Some(serde_json::json!({"temperature": 0.5})),
            ..Default::default()
        };
        let payload = build_payload(&input);

        assert_eq!(payload["input"], "hello");
        assert_eq!(payload["max_output_tokens"], 128);
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["store"], false);
        assert_eq!(payload["ignore_eos"], true);
        assert_eq!(payload["temperature"], 0.5);
    }
}
