// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::error::Error as StdError;
use std::time::Instant;

use futures::StreamExt;

use super::streaming::{StreamedResponseHandler, trim_bytes};
use super::{
    CompletionChunk, RequestFuncInput, RequestFuncOutput, build_headers, update_server_metrics,
};
use crate::error::Result;

/// Backend for OpenAI-compatible Completions API (/v1/completions).
/// Used by "vllm" and "openai" backends.
#[derive(Clone)]
pub struct OpenAICompletionsBackend;

impl OpenAICompletionsBackend {
    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        let model = input.model_name.as_deref().unwrap_or(&input.model);

        // When prompt_token_ids are available, send them as the `prompt` value
        // (JSON array of integers). vLLM's completions API accepts both string
        // and token ID array as `prompt`, skipping server-side tokenization.
        let prompt_value = if let Some(ref token_ids) = input.prompt_token_ids {
            serde_json::json!(token_ids.as_ref())
        } else {
            serde_json::json!(input.prompt)
        };

        let mut payload = serde_json::json!({
            "model": model,
            "prompt": prompt_value,
            "max_tokens": input.output_len,
            "stream": true,
            "stream_options": {
                "include_usage": true,
            },
        });

        // Always include logprobs (null when not set) — matches Python which
        // sends logprobs=None explicitly rather than omitting the key.
        payload["logprobs"] = match input.logprobs {
            Some(n) => serde_json::json!(n),
            None => serde_json::Value::Null,
        };

        // Apply ignore_eos and extra_body
        if input.ignore_eos {
            payload["ignore_eos"] = serde_json::json!(true);
        }
        if let Some(serde_json::Value::Object(map)) = input.extra_body.as_ref() {
            for (k, v) in map {
                payload[k] = v.clone();
            }
        }

        let headers_map = build_headers(None, &input.extra_headers, &input.request_id);

        let mut output = RequestFuncOutput {
            prompt_len: input.prompt_len,
            itl: Vec::with_capacity(input.output_len.max(1)),
            ..Default::default()
        };

        let st = Instant::now();
        // start_time is overwritten by benchmark.rs with monotonic offset

        let mut most_recent_timestamp = st;
        let mut generated_text = String::new();
        let mut first_chunk_received = false;

        let mut request = client.post(&input.api_url).json(&payload);
        for (k, v) in &headers_map {
            request = request.header(k, v);
        }

        match request.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let mut handler = StreamedResponseHandler::new();
                    let mut stream = response.bytes_stream();

                    while let Some(chunk_result) = stream.next().await {
                        let chunk_bytes = match chunk_result {
                            Ok(b) => b,
                            Err(e) => {
                                output.success = false;
                                output.error = format!("Stream error: {e}");
                                return Ok(output);
                            }
                        };

                        let trimmed_bytes = trim_bytes(&chunk_bytes);
                        if trimmed_bytes.is_empty() {
                            continue;
                        }

                        let messages = handler.add_chunk(trimmed_bytes);
                        for message in messages {
                            // Skip SSE comments
                            if message.starts_with(':') {
                                continue;
                            }

                            // Handle multi-field SSE events (e.g., Dynamo sends
                            // "event: message\ndata: {...}"). Extract the data: line.
                            let raw = if message.contains('\n') {
                                match message.lines().find(|l| l.starts_with("data: ")) {
                                    Some(l) => l,
                                    None => continue,
                                }
                            } else {
                                message.as_str()
                            };

                            let chunk = raw.strip_prefix("data: ").unwrap_or(raw);

                            if chunk == "[DONE]" {
                                continue;
                            }

                            // Typed deserialization — avoids allocating a full
                            // serde_json::Value tree; only extracts needed fields.
                            let data: CompletionChunk = match serde_json::from_str(chunk) {
                                Ok(d) => d,
                                Err(_) => continue,
                            };

                            update_server_metrics(&mut output, data.metrics.as_ref());

                            if !data.choices.is_empty() {
                                let text = data.choices[0].text.as_deref().unwrap_or("");

                                let timestamp = Instant::now();

                                if !first_chunk_received {
                                    first_chunk_received = true;
                                    output.ttft = timestamp.duration_since(st).as_secs_f64();
                                } else {
                                    output.itl.push(
                                        timestamp
                                            .duration_since(most_recent_timestamp)
                                            .as_secs_f64(),
                                    );
                                }

                                most_recent_timestamp = timestamp;
                                generated_text.push_str(text);
                            }
                            // Separate `if` (not `else if`) — Dynamo may send
                            // both choices and usage in the same chunk.
                            if let Some(usage) = &data.usage {
                                usage.apply(&mut output);
                            }
                        }
                    }

                    if first_chunk_received {
                        output.success = true;
                    } else {
                        output.success = false;
                        output.error = "Never received a valid chunk to calculate TTFT. \
                                        This response will be marked as failed!"
                            .to_string();
                    }
                    output.generated_text = generated_text;
                    output.latency = most_recent_timestamp.duration_since(st).as_secs_f64();
                } else {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    output.error = if body.is_empty() {
                        format!("HTTP {status}")
                    } else {
                        format!("HTTP {status}: {body}")
                    };
                    output.success = false;
                }
            }
            Err(e) => {
                output.success = false;
                // Capture full error chain for debugging
                let mut error_msg = format!("{e}");
                let mut source = e.source();
                while let Some(cause) = source {
                    error_msg.push_str(&format!("\n  Caused by: {cause}"));
                    source = cause.source();
                }
                output.error = error_msg;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve `body` as one streaming completions response over a raw socket.
    /// Returns the endpoint URL.
    fn spawn_sse_server(body: &'static str) -> String {
        use std::io::{BufRead, BufReader, Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut content_length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap();
                }
                if line == "\r\n" {
                    break;
                }
            }
            reader.read_exact(&mut vec![0; content_length]).unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                         connection: close\r\n\r\n{body}"
                    )
                    .as_bytes(),
                )
                .unwrap();
        });
        format!("http://{addr}/v1/completions")
    }

    /// The server's `usage.prompt_tokens` replaces the client-side estimate,
    /// as in the chat backend.
    #[tokio::test]
    async fn test_usage_prompt_tokens_override_prompt_len() {
        let input = RequestFuncInput {
            api_url: spawn_sse_server(concat!(
                "data: {\"choices\":[{\"text\":\"a\"}]}\n\n",
                "data: {\"choices\":[{\"text\":\"b\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2,\"prompt_tokens\":128}}\n\n",
                "data: [DONE]\n\n",
            )),
            model: "test-model".to_string(),
            prompt_len: 100,
            output_len: 2,
            ..Default::default()
        };

        let output = OpenAICompletionsBackend
            .send_request(&input, &reqwest::Client::new())
            .await
            .unwrap();

        assert!(output.success, "{}", output.error);
        assert_eq!((output.prompt_len, output.output_tokens), (128, 2));
    }
}
