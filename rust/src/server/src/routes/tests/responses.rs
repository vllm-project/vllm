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
async fn responses_empty_model_uses_served_model() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(&app, json!({"model": "", "input": "hello"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["model"], "Qwen/Qwen1.5-0.5B-Chat");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_sampling_metadata_matches_engine_defaults_and_overrides() {
    for (overrides, temperature, top_p) in [
        (json!({}), 0.6, 0.95),
        (json!({"temperature": 0.2, "top_p": 0.8}), 0.2, 0.8),
    ] {
        let backend = FakeChatBackend {
            sampling_hints: vllm_text::SamplingHints {
                default_temperature: Some(0.6),
                default_top_p: Some(0.95),
                ..Default::default()
            },
            ..FakeChatBackend::new()
        };
        let (app, engine_task) =
            test_app_with_backend_and_engine_request_check(Arc::new(backend), move |request| {
                let params = request.sampling_params.as_ref().unwrap();
                assert_eq!(params.temperature, temperature);
                assert_eq!(params.top_p, top_p);
            })
            .await;
        let mut request = json!({"input": "hello", "stream": null});
        request.as_object_mut().unwrap().extend(overrides.as_object().unwrap().clone());
        let response = responses_call(&app, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        engine_task.await.expect("mock engine task");
        let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!((response["temperature"].as_f64().unwrap() - f64::from(temperature)).abs() < 1e-6);
        assert!((response["top_p"].as_f64().unwrap() - f64::from(top_p)).abs() < 1e-6);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_rejects_invalid_controls_before_inference() {
    let (app, _engine_task) = test_app_with_engine_handle().await;
    for (field, value) in [
        ("metadata", json!([])),
        ("metadata", json!({"nested": {"value": "no"}})),
        ("metadata", json!({"number": 1})),
        ("metadata", json!({"x".repeat(65): "value"})),
        ("metadata", json!({"key": "x".repeat(513)})),
        (
            "metadata",
            serde_json::to_value(
                (0..17)
                    .map(|i| (i.to_string(), "value"))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
            .unwrap(),
        ),
        ("service_tier", json!("bogus")),
        ("include", json!(["message.output_text.logprobs"])),
        ("include", json!(["unknown"])),
        ("top_logprobs", json!(1)),
        ("reasoning", json!({"summary": "unknown"})),
        ("text", json!({"verbosity": "unknown"})),
    ] {
        let response = responses_call(&app, json!({"input": "hello", field: value})).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_accepts_codex_controls_and_metadata_boundaries() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let metadata = (0..16)
        .map(|i| (format!("{i:064}"), "é".repeat(512)))
        .collect::<std::collections::HashMap<_, _>>();
    let response = responses_call(
        &app,
        json!({
            "input": "hello", "stream": null, "metadata": metadata,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"}, "text": {"verbosity": "low"},
            "service_tier": "default"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    engine_task.await.expect("mock engine task");
    let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["metadata"], json!(metadata));
    assert_eq!(response["service_tier"], "default");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_empty_history_is_client_error() {
    let (app, _engine_task) = test_app_with_engine_handle().await;
    for input in [
        json!([]),
        json!([
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "status": "incomplete", "content": [], "summary": []}
        ]),
    ] {
        let response = responses_call(&app, json!({"input": input})).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["param"], "input");
    }
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
async fn responses_invalid_tool_choice_reports_tool_choice_parameter() {
    let (app, _engine_task) = test_app_with_engine_handle().await;

    for request in [
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "tool_choice": "required",
        }),
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "tools": [weather_function_tool()],
            "tool_choice": {"type": "function", "name": "missing"},
        }),
    ] {
        let response = responses_call(&app, request).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
        assert_eq!(json["error"]["param"], "tool_choice");
    }
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
async fn responses_auto_truncation_reaches_engine() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(
        &app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
            "truncation": "auto"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["truncation"], "auto");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_watermarking_defaults_and_opt_out_reach_engine_core() {
    for watermarking in [None, Some(true), Some(false)] {
        let (app, engine_task) = test_app_with_backend_and_engine_request_check(
            Arc::new(FakeChatBackend::new()),
            move |request| {
                let params = request.sampling_params.as_ref().expect("sampling params");
                assert_eq!(params.watermarking, watermarking.unwrap_or(true));
            },
        )
        .await;

        let mut body = json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "hello",
        });
        if let Some(watermarking) = watermarking {
            body["watermarking"] = json!(watermarking);
        }
        let response = responses_call(&app, body).await;

        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), usize::MAX).await.expect("drain response");
        engine_task.await.expect("mock engine task");
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_streaming_is_rejected_until_supported() {
    let (app, _engine_task) = test_app_with_engine_handle().await;
    let response = responses_call(&app, json!({"input": "hello", "stream": true})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(error["error"]["param"], "stream");
}
