// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Instant;

use futures::StreamExt;

use super::streaming::{StreamedResponseHandler, trim_bytes};
use super::{ChatChunk, RequestFuncInput, RequestFuncOutput, build_headers};
use crate::error::Result;

/// Backend for OpenAI Chat Completions API (/v1/chat/completions).
#[derive(Clone)]
pub struct OpenAIChatBackend;

impl OpenAIChatBackend {
    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        // Content-Type is set below by `.json()` / `.header()`; keep it out of
        // headers_map to avoid a duplicate that strict gateways reject.
        let headers_map = build_headers(None, &input.extra_headers, &input.request_id);

        let mut output = RequestFuncOutput {
            prompt_len: input.prompt_len,
            itl: Vec::with_capacity(input.output_len.max(1)),
            ..Default::default()
        };

        let st = Instant::now();

        let mut most_recent_timestamp = st;
        let mut generated_text = String::new();
        let mut first_token_received = false;

        // Build request: use zero-copy raw JSON for multimodal, serde_json for text-only
        let mut request =
            if input.multi_modal_content.is_some() || input.chat_messages_json.is_some() {
                let payload_bytes = build_mm_payload(input);
                client
                    .post(&input.api_url)
                    .header("content-type", "application/json")
                    .body(payload_bytes)
            } else {
                let payload = build_text_payload(input);
                client.post(&input.api_url).json(&payload)
            };
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

                            // Python chat backend: timestamp is captured for ALL
                            // non-DONE messages, and most_recent_timestamp is updated
                            // unconditionally (outside `if choices:`). This differs from
                            // completions which only timestamps content chunks.
                            let timestamp = Instant::now();

                            let data: ChatChunk = match serde_json::from_str(chunk) {
                                Ok(d) => d,
                                Err(_) => continue,
                            };

                            if !data.choices.is_empty() {
                                let content = data.choices[0]
                                    .delta
                                    .as_ref()
                                    .and_then(|d| d.content.as_deref())
                                    .unwrap_or("");

                                if !first_token_received {
                                    first_token_received = true;
                                    output.ttft = timestamp.duration_since(st).as_secs_f64();
                                } else {
                                    output.itl.push(
                                        timestamp
                                            .duration_since(most_recent_timestamp)
                                            .as_secs_f64(),
                                    );
                                }

                                generated_text.push_str(content);
                            }
                            // Separate `if` (not `else if`) — Dynamo may send
                            // both choices and usage in the same chunk.
                            if let Some(ref usage) = data.usage
                                && let Some(ct) = usage.completion_tokens
                            {
                                output.output_tokens = ct as usize;
                            }

                            most_recent_timestamp = timestamp;
                        }
                    }

                    output.generated_text = generated_text;
                    output.success = true;
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
                output.error = format!("{e:#}");
            }
        }

        Ok(output)
    }
}

/// Build a JSON payload for text-only (non-multimodal) requests using serde_json.
fn build_text_payload(input: &RequestFuncInput) -> serde_json::Value {
    let model = input.model_name.as_deref().unwrap_or(&input.model);

    let messages = if let Some(ref msgs) = input.messages {
        msgs.clone()
    } else {
        let content = serde_json::json!([
            {"type": "text", "text": input.prompt}
        ]);
        serde_json::json!([{"role": "user", "content": content}])
    };

    let mut payload = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_completion_tokens": input.output_len,
        "stream": true,
        "stream_options": {
            "include_usage": true,
        },
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

/// Build the JSON payload as raw bytes for multimodal requests.
///
/// This is the zero-copy fast path: pre-serialized mm content fragments
/// (each ~200KB+ of base64 image data) are concatenated directly into the
/// output buffer without being parsed, cloned, or re-serialized.
///
/// Saves ~200KB of allocation + copy per image per request compared to
/// the serde_json::Value approach.
fn build_mm_payload(input: &RequestFuncInput) -> Vec<u8> {
    let model = input.model_name.as_deref().unwrap_or(&input.model);

    // Estimate total size: JSON overhead (~300 bytes) + prompt + mm fragments
    let mm_total: usize = input
        .multi_modal_content
        .as_ref()
        .map(|mm| mm.iter().map(|f| f.len() + 1).sum())
        .unwrap_or(0)
        + input.chat_messages_json.as_ref().map_or(0, |m| m.len());
    let estimated = 512 + input.prompt.len() * 2 + mm_total;
    let mut json = String::with_capacity(estimated);

    // {"model": <model>
    json.push_str(r#"{"model":"#);
    // serde_json::to_string on &str produces a JSON-escaped quoted string
    json.push_str(&serde_json::to_string(model).unwrap());

    json.push_str(r#","messages":"#);
    if let Some(ref msgs) = input.chat_messages_json {
        // --enable-multimodal-chat: the dataset pre-built the full messages
        // array (text + mm parts); splice it verbatim.
        json.push_str(msgs);
    } else {
        let mm = input.multi_modal_content.as_ref().unwrap();

        // [{"role":"user","content":[ <text part>
        json.push_str(r#"[{"role":"user","content":[{"type":"text","text":""#);
        // JSON-escape the prompt text (handles \n, \t, unicode, quotes)
        push_json_escaped_str(&mut json, &input.prompt);
        json.push_str(r#""}"#);

        // ,<mm fragment 1>,<mm fragment 2>,...
        for fragment in mm.iter() {
            json.push(',');
            json.push_str(fragment);
        }

        // Close content, message, messages
        json.push_str(r#"]}]"#);
    }

    // ,"max_completion_tokens": N, "stream": true, ...
    json.push_str(r##","max_completion_tokens":"##);
    json.push_str(&input.output_len.to_string());
    json.push_str(r##","stream":true,"stream_options":{"include_usage":true}"##);

    if input.ignore_eos {
        json.push_str(r#","ignore_eos":true"#);
    }

    // Merge extra_body key-value pairs, skipping keys already set above
    if let Some(serde_json::Value::Object(map)) = input.extra_body.as_ref() {
        for (k, v) in map {
            match k.as_str() {
                "model"
                | "messages"
                | "max_completion_tokens"
                | "stream"
                | "stream_options"
                | "ignore_eos" => continue,
                _ => {
                    json.push(',');
                    json.push_str(&serde_json::to_string(k).unwrap());
                    json.push(':');
                    json.push_str(&serde_json::to_string(v).unwrap());
                }
            }
        }
    }

    json.push('}');
    json.into_bytes()
}

/// Write a JSON-escaped string (without surrounding quotes) into the buffer.
///
/// Handles: `\n`, `\r`, `\t`, `\\`, `\"`, and control characters.
/// This avoids the allocation of `serde_json::to_string` which produces
/// a new String with surrounding quotes.
fn push_json_escaped_str(buf: &mut String, s: &str) {
    use std::fmt::Write;
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str(r#"\""#),
            '\\' => buf.push_str(r"\\"),
            '\n' => buf.push_str(r"\n"),
            '\r' => buf.push_str(r"\r"),
            '\t' => buf.push_str(r"\t"),
            c if c.is_control() => {
                // \uXXXX escape for control characters
                for unit in c.encode_utf16(&mut [0; 2]) {
                    write!(buf, "\\u{unit:04x}").unwrap();
                }
            }
            c => buf.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn mm_input() -> RequestFuncInput {
        let frag: Arc<str> =
            Arc::from(r#"{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AAAA"}}"#);
        RequestFuncInput {
            prompt: Arc::from("hello \"world\"\nline2"),
            model: "test-model".to_string(),
            output_len: 128,
            multi_modal_content: Some(Arc::from(vec![frag])),
            ..Default::default()
        }
    }

    /// Regression test: the assembled multimodal payload must be valid JSON
    /// (the text part once shipped without its opening quote — a raw-string
    /// delimiter eating the trailing `"` in `"text":"`).
    #[test]
    fn test_build_mm_payload_is_valid_json() {
        let payload = build_mm_payload(&mm_input());
        let v: serde_json::Value =
            serde_json::from_slice(&payload).expect("mm payload must be valid JSON");
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["messages"][0]["role"], "user");
        let content = v["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello \"world\"\nline2");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(v["max_completion_tokens"], 128);
        assert_eq!(v["stream"], true);
        assert_eq!(v["stream_options"]["include_usage"], true);
    }

    /// --enable-multimodal-chat (dataset pre-built messages) must produce a
    /// payload semantically identical to the fragment-assembly path.
    #[test]
    fn test_chat_messages_json_path_equivalent_to_fragment_path() {
        let base = mm_input();
        let fragment_payload = build_mm_payload(&base);

        let mut chat = base.clone();
        let mm = chat.multi_modal_content.take().unwrap();
        let msgs = crate::datasets::random_mm::build_chat_messages_json(&chat.prompt, Some(&mm));
        chat.chat_messages_json = Some(Arc::from(msgs.as_str()));
        let chat_payload = build_mm_payload(&chat);

        let a: serde_json::Value = serde_json::from_slice(&fragment_payload).unwrap();
        let b: serde_json::Value = serde_json::from_slice(&chat_payload).unwrap();
        assert_eq!(a, b);
    }

    /// ignore_eos and extra_body must survive the raw-splice path.
    #[test]
    fn test_mm_payload_tail_fields() {
        let mut input = mm_input();
        input.ignore_eos = true;
        input.extra_body = Some(serde_json::json!({"temperature": 0.5, "stream": false}));
        let v: serde_json::Value = serde_json::from_slice(&build_mm_payload(&input)).unwrap();
        assert_eq!(v["ignore_eos"], true);
        assert_eq!(v["temperature"], 0.5);
        // keys already set above must not be overridden by extra_body
        assert_eq!(v["stream"], true);
    }
}
