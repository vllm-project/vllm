// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires VLLM_USE_DYNAMO_RUST_PARSER=1"]
#[serial]
async fn dynamo_unified_http_stream_and_nonstream_cover_every_family() {
    assert_eq!(
        std::env::var("VLLM_USE_DYNAMO_RUST_PARSER").as_deref(),
        Ok("1")
    );
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../chat/tests/fixtures/dynamo_unified.json"
    ))
    .unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        for streaming in [false, true] {
            let input = case["input"].as_str().unwrap();
            let mut specs =
                input.chars().map(|ch| (token_ids(&ch.to_string()), None)).collect::<Vec<_>>();
            specs.push((vec![], Some(EngineCoreFinishReason::Stop)));
            let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
                Arc::new(FakeChatBackend::with_model_id(
                    case["model"].as_str().unwrap(),
                )),
                specs,
            )
            .await;
            let tools = case["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| json!({"type":"function", "function":tool}))
                .collect::<Vec<_>>();
            let response = app.clone().call(Request::builder()
                .method("POST").uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json!({
                    "model":"Qwen/Qwen1.5-0.5B-Chat", "stream":streaming,
                    "stream_options": if streaming {json!({"include_usage":true, "continuous_usage_stats":true})} else {json!(null)},
                    "messages":[{"role":"user","content":"Use the provided tools."}],
                    "tools":tools, "tool_choice":"auto"
                }).to_string())).unwrap()).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "{} streaming={streaming}",
                case["family"]
            );
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            engine_task.await.unwrap();
            let body = std::str::from_utf8(&body).unwrap();
            let mut content = String::new();
            let mut reasoning = String::new();
            let mut calls = std::collections::BTreeMap::<usize, (String, String)>::new();
            if streaming {
                let payloads = sse_json_payloads(body);
                assert!(!payloads.is_empty(), "{}", case["family"]);
                for payload in &payloads {
                    if let Some(delta) = payload["choices"][0].get("delta") {
                        if let Some(text) = delta["content"].as_str() {
                            content.push_str(text);
                        }
                        if let Some(text) = delta["reasoning"].as_str() {
                            reasoning.push_str(text);
                        }
                        if let Some(items) = delta["tool_calls"].as_array() {
                            for call in items {
                                let entry = calls
                                    .entry(call["index"].as_u64().unwrap() as usize)
                                    .or_default();
                                if let Some(name) = call["function"]["name"].as_str() {
                                    entry.0.push_str(name);
                                }
                                if let Some(args) = call["function"]["arguments"].as_str() {
                                    entry.1.push_str(args);
                                }
                            }
                        }
                    }
                }
                let usage = &payloads.last().unwrap()["usage"];
                assert!(
                    usage["completion_tokens_details"].is_null(),
                    "unknown reasoning usage: {usage}"
                );
            } else {
                let payload: serde_json::Value = serde_json::from_str(body).unwrap();
                let message = &payload["choices"][0]["message"];
                if let Some(text) = message["content"].as_str() {
                    content.push_str(text);
                }
                if let Some(text) = message["reasoning"].as_str() {
                    reasoning.push_str(text);
                }
                for (i, call) in message["tool_calls"].as_array().unwrap().iter().enumerate() {
                    calls.insert(
                        i,
                        (
                            call["function"]["name"].as_str().unwrap().to_string(),
                            call["function"]["arguments"].as_str().unwrap().to_string(),
                        ),
                    );
                }
                assert!(payload["usage"]["completion_tokens_details"].is_null());
            }
            let expected = case["expected"].as_array().unwrap();
            let expected_text = |kind| {
                expected
                    .iter()
                    .filter(|event| event["kind"] == kind)
                    .map(|event| event["text"].as_str().unwrap())
                    .collect::<String>()
            };
            assert_eq!(
                content,
                expected_text("text"),
                "{} streaming={streaming}",
                case["family"]
            );
            assert_eq!(
                reasoning,
                expected_text("reasoning"),
                "{} streaming={streaming}",
                case["family"]
            );
            let expected_calls = expected
                .iter()
                .filter(|event| event["kind"] == "tool_call")
                .map(|event| {
                    (
                        event["name"].as_str().unwrap().to_string(),
                        event["arguments"].clone(),
                    )
                })
                .collect::<Vec<_>>();
            let actual_calls = calls
                .into_values()
                .map(|(name, args)| {
                    (
                        name,
                        serde_json::from_str::<serde_json::Value>(&args).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual_calls, expected_calls,
                "{} streaming={streaming}",
                case["family"]
            );
        }
    }
}
