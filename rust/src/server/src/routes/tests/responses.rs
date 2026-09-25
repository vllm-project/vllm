// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HTTP-level coverage for the Rust OpenAI Responses API.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::json;
use serial_test::serial;
use tower::Service as _;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;

use super::*;

fn reasoning_answer_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (bytes_to_token_ids(b"<think>Need tool.</think>"), None),
        (bytes_to_token_ids(b"answer"), None),
        // '!' is the fake tokenizer's stop token; it is suppressed from the
        // visible text but carries the finish reason.
        (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
    ]
}

fn weather_function_tool() -> serde_json::Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}}
        }
    })
}

async fn responses_call(app: &axum::Router, body: serde_json::Value) -> axum::response::Response {
    app.clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_non_streaming_text_input_returns_response_object() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    let id = json["id"].as_str().expect("id");
    assert!(id.starts_with("resp_"), "{text}");
    assert_eq!(json["object"], "response");
    assert_eq!(json["status"], "completed");
    assert_eq!(json["model"], "Qwen/Qwen1.5-0.5B-Chat");

    let output = json["output"].as_array().expect("output items");
    assert_eq!(output.len(), 1, "{text}");
    let item = &output[0];
    assert_eq!(item["type"], "message");
    assert_eq!(item["role"], "assistant");
    assert_eq!(item["status"], "completed");
    assert!(
        item["id"].as_str().expect("item id").starts_with("msg_"),
        "{text}"
    );
    assert_eq!(item["content"][0]["type"], "output_text");
    // The fake tokenizer treats '!' as the stop token: it is counted in the
    // usage output tokens but suppressed from the visible text.
    assert_eq!(item["content"][0]["text"], "hi");

    assert_eq!(json["usage"]["input_tokens"], 22);
    assert_eq!(json["usage"]["output_tokens"], 3);
    assert_eq!(json["usage"]["total_tokens"], 25);
    assert_eq!(json["usage"]["input_tokens_details"]["cached_tokens"], 0);

    // Echoed request metadata with OpenAI defaults applied.
    assert_eq!(json["background"], false);
    assert_eq!(json["parallel_tool_calls"], true);
    assert_eq!(json["service_tier"], "auto");
    assert_eq!(json["temperature"], 1.0);
    assert_eq!(json["top_p"], 1.0);
    assert_eq!(json["tool_choice"], "none");
    assert_eq!(json["tools"], json!([]));
    assert_eq!(json["truncation"], "disabled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_accepts_auto_tool_choice_without_tools() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "tool_choice": "auto"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["status"], "completed");
    assert_eq!(json["tool_choice"], "none");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_accepts_output_presentation_controls() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "text": {"verbosity": "low"}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["status"], "completed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_header_request_id_takes_precedence() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("X-Request-Id", "header-req")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "body-req",
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["id"], "resp_header-req");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_body_request_id_derives_response_id() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "request_id": "body-req",
            "input": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["id"], "resp_body-req");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_non_streaming_length_finish_is_incomplete() {
    let (app, engine_task) = test_app_with_stream_output_specs(vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], Some(EngineCoreFinishReason::Length)),
    ])
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "max_output_tokens": 2
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    assert_eq!(json["status"], "incomplete", "{text}");
    assert_eq!(json["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(json["max_output_tokens"], 2);
    let message = json["output"]
        .as_array()
        .expect("output items")
        .iter()
        .find(|item| item["type"] == "message")
        .expect("message item");
    assert_eq!(message["status"], "completed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_streaming_error_includes_failed_response_error() {
    let (app, engine_task) =
        test_app_with_stream_output_specs(vec![(vec![], Some(EngineCoreFinishReason::Error))])
            .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "stream": true
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let payloads = sse_json_payloads(std::str::from_utf8(&body).expect("utf8 body"));

    let failed = payloads.last().expect("response.failed event");
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["status"], "failed");
    assert_eq!(failed["response"]["error"]["code"], "server_error");
    assert_eq!(
        failed["response"]["error"]["message"],
        "The model failed to generate a response."
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_non_streaming_includes_reasoning_item() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        reasoning_answer_output_specs(),
    )
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    let output = json["output"].as_array().expect("output items");
    let types: Vec<&str> = output.iter().map(|item| item["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["reasoning", "message"], "{text}");
    let reasoning = &output[0];
    assert!(
        reasoning["id"].as_str().expect("item id").starts_with("rs_"),
        "{text}"
    );
    assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
    assert_eq!(reasoning["content"][0]["text"], "Need tool.");
    assert_eq!(output[1]["content"][0]["text"], "answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_include_reasoning_false_excludes_reasoning_item() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        reasoning_answer_output_specs(),
    )
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "include_reasoning": false
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    let output = json["output"].as_array().expect("output items");
    assert_eq!(output.len(), 1, "{text}");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_streaming_hides_reasoning_when_requested() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        reasoning_answer_output_specs(),
    )
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "stream": true,
            "include_reasoning": false
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let payloads = sse_json_payloads(std::str::from_utf8(&body).expect("utf8 body"));
    assert!(
        payloads
            .iter()
            .all(|payload| !payload["type"].as_str().unwrap().contains("reasoning"))
    );

    let message_added = payloads
        .iter()
        .find(|payload| payload["type"] == "response.output_item.added")
        .expect("message added");
    assert_eq!(message_added["output_index"], 0);
    let completed = &payloads.last().expect("completed event")["response"];
    assert_eq!(completed["output"][0]["id"], message_added["item"]["id"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_function_call_roundtrip_across_turns() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        weather_tool_call_output_specs(),
    )
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "weather in Paris?",
            "tools": [weather_function_tool()]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    let output = json["output"].as_array().expect("output items");
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call item");
    assert_eq!(call["name"], "get_weather", "{text}");
    assert_eq!(call["arguments"], "{\"city\":\"Paris\"}", "{text}");
    assert!(
        call["id"].as_str().expect("item id").starts_with("fc_"),
        "{text}"
    );
    let call_id = call["call_id"].as_str().expect("call id");
    assert!(call_id.starts_with("call_"), "{text}");
    assert_eq!(json["tool_choice"], "auto");
    assert_eq!(json["tools"], json!([weather_function_tool()]));

    // Second turn: replay the history items plus the tool output.
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "max_output_tokens": 64,
            "input": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "sunny"
                }
            ]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");
    assert_eq!(json["status"], "completed", "{text}");
    assert_eq!(json["output"][0]["content"][0]["text"], "hi");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_streaming_emits_ordered_event_sequence() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "input": "hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    let payloads = sse_json_payloads(&text);
    let types: Vec<&str> =
        payloads.iter().map(|payload| payload["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ],
        "{text}"
    );

    // Every `data:` payload carries a globally increasing sequence number,
    // and the matching `event:` line names the same event type.
    let event_names: Vec<&str> =
        text.lines().filter_map(|line| line.strip_prefix("event: ")).collect();
    assert_eq!(event_names.len(), payloads.len(), "{text}");
    for (index, (event_name, payload)) in event_names.iter().zip(&payloads).enumerate() {
        assert_eq!(*event_name, payload["type"].as_str().unwrap(), "{text}");
        assert_eq!(payload["sequence_number"], index as u64, "{text}");
    }

    let created = &payloads[0]["response"];
    assert_eq!(created["status"], "in_progress");
    assert_eq!(created["output"], json!([]));
    assert!(created.get("usage").is_none(), "{text}");
    assert!(
        created["id"].as_str().expect("id").starts_with("resp_"),
        "{text}"
    );

    // Deltas join to the final message text ('!' is the fake tokenizer's
    // stop token and is suppressed).
    let deltas: String = payloads
        .iter()
        .filter(|payload| payload["type"] == "response.output_text.delta")
        .map(|payload| payload["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "hi");

    // The streamed item ID matches the terminal response payload.
    let added = payloads
        .iter()
        .find(|payload| payload["type"] == "response.output_item.added")
        .expect("added event");
    let item_id = added["item"]["id"].as_str().expect("item id");
    assert!(item_id.starts_with("msg_"), "{text}");
    assert_eq!(payloads[4]["item_id"], item_id);

    let completed = &payloads.last().expect("completed event")["response"];
    assert_eq!(completed["id"], created["id"]);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"][0]["id"], item_id);
    assert_eq!(completed["output"][0]["content"][0]["text"], "hi");
    assert_eq!(completed["usage"]["input_tokens"], 22);
    assert_eq!(completed["usage"]["output_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_streaming_emits_function_call_events() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        weather_tool_call_output_specs(),
    )
    .await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "input": "weather in Paris?",
            "tools": [weather_function_tool()]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_json_payloads(&text);

    let types: Vec<&str> =
        payloads.iter().map(|payload| payload["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        [
            "response.created",
            "response.in_progress",
            // Reasoning item ("Need tool.").
            "response.output_item.added",
            "response.reasoning_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.reasoning_part.done",
            "response.output_item.done",
            // Function call item.
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            // Trailing message item: the tool parser leaks the final
            // `}\n</tool_call` chunk as visible text (the same leak is
            // visible on /v1/chat/completions).
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed"
        ],
        "{text}"
    );

    let completed = &payloads.last().expect("completed event")["response"];
    let output = completed["output"].as_array().expect("output items");
    assert_eq!(output.len(), 3, "{text}");
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["text"], "Need tool.");
    let call = &output[1];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["arguments"], "{\"city\":\"Paris\"}");

    // The streamed function-call item ID matches the terminal payload.
    let call_added = payloads
        .iter()
        .filter(|payload| payload["type"] == "response.output_item.added")
        .nth(1)
        .expect("function call added event");
    assert_eq!(call_added["item"]["id"], call["id"]);
    assert_eq!(call_added["output_index"], 1);
    assert_eq!(call_added["item"]["call_id"], call["call_id"]);
    let arguments: String = payloads
        .iter()
        .filter(|payload| payload["type"] == "response.function_call_arguments.delta")
        .map(|payload| payload["delta"].as_str().unwrap())
        .collect();
    assert_eq!(arguments, "{\"city\":\"Paris\"}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_store_dependent_features_are_rejected() {
    let (app, engine_task) = test_app_with_engine_handle().await;

    // background=true requires a response store.
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "background": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["param"], "background");

    // previous_response_id cannot be resolved without a store; unknown IDs
    // fail like every other unknown response ID.
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "previous_response_id": "resp_missing"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["param"], "response_id");

    // Retrieval and cancellation of never-stored responses are 404.
    for (method, uri) in [
        ("GET", "/v1/responses/resp_missing"),
        ("POST", "/v1/responses/resp_missing/cancel"),
    ] {
        let response = app
            .clone()
            .call(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("call app");
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
    }

    // Built-in tool types are rejected: only function tools are supported.
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "tools": [{"type": "web_search_preview"}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["param"], "tools");

    // store=true executes normally (store-off Python parity).
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "store": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["status"], "completed");
}
