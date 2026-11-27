// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::*;
use axum::response::Response;
use futures::StreamExt as _;

fn messages_request() -> serde_json::Value {
    json!({
        "model": "Qwen/Qwen1.5-0.5B-Chat",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Hi"}],
    })
}

fn weather_tool() -> serde_json::Value {
    json!({"name": "get_weather", "input_schema": {
        "type": "object", "properties": {"city": {"type": "string"}}
    }})
}

async fn call_messages(app: &axum::Router, body: serde_json::Value) -> Response {
    app.clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn model_specific_parsers_produce_anthropic_thinking_and_tool_use() {
    let tokenizer = Arc::new(fake_chat_tokenizer().with_regular_token("｜DSML｜", 0xF007));
    for (model, tool_output, expected_id) in [
        (
            "deepseek-ai/DeepSeek-V4.1-Flash",
            concat!(
                "<｜DSML｜ calls>\n",
                "<｜DSML｜ invoke name=\"get_weather\">\n",
                "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n",
                "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
            ),
            None,
        ),
        (
            "zai-org/GLM-4.7-Flash",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            None,
        ),
        (
            "moonshotai/Kimi-K2.5",
            concat!(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0",
                "<|tool_call_argument_begin|>{\"city\":\"Paris\"}<|tool_call_end|>",
                "<|tool_calls_section_end|>"
            ),
            Some("functions.get_weather:0"),
        ),
    ] {
        for streaming in [false, true] {
            let backend = FakeChatBackend {
                model_id: model.to_owned(),
                tokenizer: Some(tokenizer.clone()),
                parsers: ResolvedParsers::new(
                    model,
                    tokenizer.clone(),
                    &ParserSelection::Auto,
                    &ParserSelection::Auto,
                )
                .unwrap(),
                ..FakeChatBackend::new()
            };
            let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
                Arc::new(backend),
                vec![
                    (token_ids("<think>Need weather.</think>"), None),
                    (tokenizer.encode(tool_output, false).unwrap(), None),
                    (token_ids("!"), Some(EngineCoreFinishReason::Stop)),
                ],
            )
            .await;
            let mut body = messages_request();
            body["stream"] = json!(streaming);
            body["tools"] = json!([weather_tool()]);
            let response = call_messages(&app, body).await;
            let status = response.status();
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let body = std::str::from_utf8(&bytes).unwrap();
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            engine_task.await.unwrap();
            if streaming {
                let events = sse_json_payloads(body);
                let tool = events
                    .iter()
                    .find(|event| event["content_block"]["type"] == "tool_use")
                    .unwrap_or_else(|| panic!("{model}: no streamed tool call in {body}"));
                let id = tool["content_block"]["id"].as_str().unwrap();
                if let Some(expected_id) = expected_id {
                    assert_eq!(id, expected_id, "{model}");
                }
                let thinking: String =
                    events.iter().filter_map(|event| event["delta"]["thinking"].as_str()).collect();
                assert_eq!(thinking, "Need weather.", "{model}: {body}");
                let args: String = events
                    .iter()
                    .filter_map(|event| event["delta"]["partial_json"].as_str())
                    .collect();
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&args).unwrap(),
                    json!({"city": "Paris"})
                );
                assert!(
                    events.iter().any(|event| event["delta"]["stop_reason"] == "tool_use"),
                    "{model}: {body}"
                );
            } else {
                let message: serde_json::Value = serde_json::from_str(body).unwrap();
                assert_eq!(
                    message["content"][0]["thinking"], "Need weather.",
                    "{model}: {body}"
                );
                let tool = message["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|block| block["type"] == "tool_use")
                    .unwrap_or_else(|| panic!("{model}: no tool call in {body}"));
                if let Some(expected_id) = expected_id {
                    assert_eq!(tool["id"], expected_id, "{model}");
                }
                assert_eq!(tool["input"], json!({"city": "Paris"}), "{model}: {body}");
                assert_eq!(message["stop_reason"], "tool_use", "{model}: {body}");
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streams_thinking_and_tools_in_order_with_named_events() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        weather_tool_call_output_specs(),
    )
    .await;
    let mut body = messages_request();
    body["stream"] = json!(true);
    body["thinking"] = json!({"type": "enabled"});
    body["metadata"] = json!({"user_id": "client-session"});
    body["safeguards"] = json!({});
    body["tools"] = json!([weather_tool()]);
    let response = call_messages(&app, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    engine_task.await.unwrap();
    let sse = std::str::from_utf8(&bytes).unwrap();
    let payloads = sse_json_payloads(sse);
    let event_names: Vec<_> = sse.lines().filter_map(|line| line.strip_prefix("event: ")).collect();
    assert_eq!(
        event_names,
        payloads
            .iter()
            .map(|payload| payload["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
    );
    assert!(!sse.contains("[DONE]"));
    assert_eq!(payloads[0]["message"]["stop_reason"], json!(null));
    assert_eq!(payloads[0]["message"]["content"], json!([]));
    let signature = payloads
        .iter()
        .position(|payload| payload["delta"]["type"] == "signature_delta")
        .unwrap();
    assert_eq!(payloads[signature]["index"], 0);
    assert!(
        uuid::Uuid::parse_str(payloads[signature]["delta"]["signature"].as_str().unwrap()).is_ok()
    );
    assert_eq!(
        payloads[signature + 1],
        json!({"type": "content_block_stop", "index": 0})
    );
    let tool = &payloads[signature + 2];
    assert_eq!(tool["index"], 1);
    assert_eq!(tool["content_block"]["type"], "tool_use");
    assert_eq!(tool["content_block"]["name"], "get_weather");
    assert!(!tool["content_block"]["id"].as_str().unwrap().is_empty());
    let args: String = payloads
        .iter()
        .filter(|payload| payload["index"] == 1 && payload["delta"]["type"] == "input_json_delta")
        .map(|payload| payload["delta"]["partial_json"].as_str().unwrap())
        .collect();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&args).unwrap(),
        json!({"city":"Paris"})
    );
    let terminal = &payloads[payloads.len() - 2];
    assert_eq!(terminal["delta"]["stop_reason"], "tool_use");
    let expected_tokens: usize =
        weather_tool_call_output_specs().iter().map(|(tokens, _)| tokens.len()).sum();
    assert_eq!(terminal["usage"]["output_tokens"], expected_tokens);
    assert_eq!(event_names.last(), Some(&"message_stop"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streamed_messages_report_cache_usage_in_final_delta() {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend_inner(
        b"engine-openai",
        |request_id| {
            RequestBatchOutputs {
                outputs: vec![
                    EngineCoreOutput {
                        prefill_stats: Some(Box::new(PrefillStats {
                            num_cached_tokens: 2,
                            num_cache_creation_tokens: 3,
                            ..Default::default()
                        })),
                        ..request_output(request_id, token_ids("a"), None)
                    },
                    request_output(
                        request_id,
                        token_ids("!"),
                        Some(EngineCoreFinishReason::Stop),
                    ),
                ],
                ..Default::default()
            }
            .into()
        },
        |_| {},
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    let app = build_router(Arc::new(
        AppState::new(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat).with_api_server_options(
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                ..Default::default()
            },
        ),
    ));
    let mut request = messages_request();
    request["stream"] = json!(true);
    let response = call_messages(&app, request).await;
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    engine_task.await.unwrap();
    let events = sse_json_payloads(std::str::from_utf8(&body).unwrap());
    let start = &events[0]["message"]["usage"];
    let finish = &events[events.len() - 2]["usage"];
    assert_eq!(start, &json!({"input_tokens": 0, "output_tokens": 0}));
    assert_eq!(finish["cache_read_input_tokens"], 2);
    assert_eq!(finish["cache_creation_input_tokens"], 3);
    assert_eq!(
        finish["input_tokens"],
        token_ids("user: Hi\nassistant:").len() - 5
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn count_tokens_accepts_tool_call_history() {
    let (mut app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        weather_tool_call_output_specs(),
    )
    .await;
    let mut body = messages_request();
    body["tools"] = json!([weather_tool()]);
    let (status, response) = post_json(&mut app, "/v1/messages", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    engine_task.await.unwrap();
    assert_eq!(response["stop_reason"], "tool_use");
    let tool = &response["content"][1];
    assert_eq!(tool["type"], "tool_use");
    assert_eq!(tool["input"], json!({"city": "Paris"}));
    body["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant", "content": response["content"]}),
        json!({"role":"user", "content":[{"type":"tool_result","tool_use_id": tool["id"]}]}),
    ]);
    body.as_object_mut().unwrap().remove("max_tokens");
    let (mut app, _engine_task) = test_app().await;
    let (status, response) = post_json(&mut app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert!(response["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_result_images_reach_engine_multimodal_features() {
    let (app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::with_multimodal_model_info(
            qwen_multimodal_model_info(),
        )),
        |request| {
            let features = request.mm_features.as_ref().unwrap();
            assert_eq!(features.len(), 1);
            assert_eq!(features[0].modality, MmModality::Image);
            assert!(features[0].data.as_ref().unwrap().contains_key("pixel_values"));
            let prompt = decode_prompt_token_ids(request).unwrap();
            assert!(prompt.contains(&151655));
        },
    )
    .await;
    let mut body = messages_request();
    body["messages"].as_array_mut().unwrap().extend([
        json!({"role": "assistant", "content": [
            {"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"Paris"}}
        ]}),
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "image", "source": {
                    "type":"base64", "media_type":"image/png",
                    "data": TINY_PNG_DATA_URI.strip_prefix("data:image/png;base64,").unwrap(),
                }}
            ]}
        ]}),
    ]);
    let response = call_messages(&app, body).await;
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_error_uses_anthropic_envelope_and_ends_the_stream() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::new()),
        vec![(vec![UNKNOWN_DECODE_TOKEN_ID], None)],
    )
    .await;
    let mut body = messages_request();
    body["stream"] = json!(true);
    let response = call_messages(&app, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    engine_task.await.unwrap();
    let sse = std::str::from_utf8(&bytes).unwrap();
    assert!(sse.contains("event: error\n"), "{sse}");
    assert!(!sse.contains("event: message_stop"), "{sse}");
    assert_eq!(
        sse_json_payloads(sse).last().unwrap()["error"]["type"],
        "api_error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn dropping_the_messages_stream_aborts_the_engine_request() {
    let (state, engine_task) =
        test_admin_state_with_ready_and_engine_script(default_ready_response(), |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request = decode_add_request(&add);
                let output = engine_outputs_for_request(
                    &request.request_id,
                    vec![(token_ids("hello"), None)],
                );
                send_outputs(push, output).await;
                let abort = recv_engine_message(dealer).await;
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(ids, [request.request_id]);
            })
        })
        .await;
    let app = build_router(state);
    let mut body = messages_request();
    body["stream"] = json!(true);
    let response = call_messages(&app, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let start = tokio::time::timeout(Duration::from_secs(3), body.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(std::str::from_utf8(&start).unwrap().contains("event: message_start"));
    drop(body);
    tokio::time::timeout(Duration::from_secs(3), engine_task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_text_generation_reports_content_and_stop() {
    for (output, finish, text, expected, stop_sequence) in [
        (
            "Hello",
            EngineCoreFinishReason::Length,
            "Hello",
            "max_tokens",
            None,
        ),
        ("", EngineCoreFinishReason::Stop, "", "end_turn", None),
        (
            "HelloSTOP",
            EngineCoreFinishReason::Length,
            "Hello",
            "stop_sequence",
            Some("STOP"),
        ),
    ] {
        let (mut app, engine_task) = test_app_with_stream_output_specs(vec![(
            bytes_to_token_ids(output.as_bytes()),
            Some(finish),
        )])
        .await;
        let (status, body) = post_json(
            &mut app,
            "/v1/messages",
            json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 10, "stop_sequences": ["STOP"],
                "messages": [{"role": "user", "content": "Hi"}]
            }),
        )
        .await;
        engine_task.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["content"], json!([{"type": "text", "text": text}]));
        assert_eq!(body["stop_reason"], expected);
        assert_eq!(body.get("stop_sequence"), Some(&json!(stop_sequence)));
        assert_eq!(body["usage"]["output_tokens"], output.len());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_rejects_invalid_sampling_before_generation() {
    let (mut app, _engine_task) = test_app().await;
    for (field, value) in [
        ("temperature", json!(-1)),
        ("temperature", json!(1.5)),
        ("top_p", json!(1.1)),
        ("stop_sequences", json!([""])),
    ] {
        let mut request = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}]});
        request[field] = value;
        let (actual, body) = post_json(&mut app, "/v1/messages", request).await;
        assert_eq!(actual, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["type"], "error");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_assistant_prefix_count_matches_generation_without_echo() {
    let (mut app, engine_task) = test_app_with_stream_output_specs(vec![(
        bytes_to_token_ids(b"continuation"),
        Some(EngineCoreFinishReason::Length),
    )])
    .await;
    let input = json!({"model": "Qwen/Qwen1.5-0.5B-Chat",
    "system": [
        {"type": "text", "text": "x-anthropic-billing-header: ignored"},
        {"type": "text", "text": "Be precise."}
    ], "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Com"}, {"type": "text", "text": "plete"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "pre"}, {"type": "text", "text": "fix"}
        ]}
    ]});
    let (status, response) = post_json(&mut app, "/v1/messages", input.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["type"], "invalid_request_error");
    let (status, count) = post_json(&mut app, "/v1/messages/count_tokens", input.clone()).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    // The fixture renderer appends `assistant:` only for a new turn.
    assert_eq!(
        count["input_tokens"],
        b"system: Be precise.\nuser: Complete\nassistant: prefix\n".len()
    );
    let mut generation = input;
    generation["max_tokens"] = json!(20);
    let (status, response) = post_json(&mut app, "/v1/messages", generation).await;
    engine_task.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["usage"]["input_tokens"], count["input_tokens"]);
    assert_eq!(
        response["content"],
        json!([{"type": "text", "text": "continuation"}])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_errors_cover_extraction_and_counting() {
    let (mut app, _engine_task) = test_app().await;
    for (uri, body, status, kind) in [
        (
            "/v1/messages",
            "{".to_owned(),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
        (
            "/v1/messages/count_tokens",
            "{".to_owned(),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
        (
            "/v1/messages/count_tokens",
            json!({"model": "missing", "messages": [{"role":"user","content":"Hi"}]}).to_string(),
            StatusCode::NOT_FOUND,
            "not_found_error",
        ),
        (
            "/v1/messages/count_tokens",
            json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "messages": [
                {"role": "user", "content": [{"type": "document", "source": {}}]}
            ]})
            .to_string(),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
    ] {
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], kind);
        assert!(value["error"].get("code").is_none());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_reasoning_output_preserves_blocks_and_usage() {
    let mut signatures = BTreeSet::new();
    // The decoder removes the terminal stop token ('!') from visible content.
    for (output, finish, expected_stop, expected_text) in [
        (
            "<think>reason</think>answer!",
            EngineCoreFinishReason::Stop,
            "end_turn",
            Some("answer"),
        ),
        (
            "<think>unfinished",
            EngineCoreFinishReason::Length,
            "max_tokens",
            None,
        ),
    ] {
        let output_ids = token_ids(output);
        let output_tokens = output_ids.len();
        let (mut app, engine_task) = test_app_with_backend_and_stream_output_specs(
            Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
            vec![(output_ids, Some(finish))],
        )
        .await;
        let (status, mut body) = post_json(
            &mut app,
            "/v1/messages",
            json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hi"}]
            }),
        )
        .await;
        engine_task.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        let signature = body["content"][0]["signature"].as_str().unwrap();
        let uuid = uuid::Uuid::parse_str(signature).unwrap();
        assert_eq!(uuid.get_version_num(), 4);
        assert_eq!(signature, uuid.simple().to_string());
        assert!(signatures.insert(signature.to_owned()));
        body["content"][0]["signature"] = json!("<signature>");
        let mut expected = vec![json!({"type": "thinking",
            "thinking": if expected_text.is_some() { "reason" } else { "unfinished" },
            "signature": "<signature>"})];
        if let Some(text) = expected_text {
            expected.push(json!({"type": "text", "text": text}));
        }
        assert_eq!(body["content"], json!(expected));
        assert_eq!(body["stop_reason"], expected_stop);
        assert_eq!(body["stop_sequence"], json!(null));
        assert_eq!(body["usage"]["output_tokens"], output_tokens);
    }
}

// A real HF renderer exposes replayed reasoning and effective effort in the prompt.
const ANTHROPIC_REASONING_TEMPLATE: &str = concat!(
    "{% if reasoning_effort == 'high' %}{{ raise_exception('Unsupported reasoning effort high') }}{% endif %}",
    "{{ reasoning_effort | default('unset') }}|",
    "{% if enable_thinking is defined %}{{ enable_thinking | tojson }}{% else %}unset{% endif %}|",
    "{% for message in messages %}{{ message.role }}:{{ message.reasoning or '' }}:{{ message.content }};{% endfor %}",
);

async fn anthropic_hf_backend(chat_template: &str) -> vllm_chat::DynChatBackend {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
    fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
    fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({
            "chat_template": chat_template
        })
        .to_string(),
    )
    .unwrap();
    let files = vllm_text::backend::hf::ResolvedModelFiles::new(dir.path().to_str().unwrap(), None)
        .await
        .unwrap();
    Arc::new(
        vllm_chat::HfChatBackend::from_resolved_model_files(
            files,
            "test-model".into(),
            vllm_chat::LoadModelBackendsOptions {
                language_model_only: true,
                ..Default::default()
            },
            Arc::new(fake_chat_tokenizer()),
        )
        .unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_replays_thinking_without_signatures_or_redacted_data() {
    let (mut app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![(
            token_ids("<think>reason</think>answer!"),
            Some(EngineCoreFinishReason::Stop),
        )],
    )
    .await;
    let (status, response) = post_json(
        &mut app,
        "/v1/messages",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}]
        }),
    )
    .await;
    engine_task.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{response}");
    let mut content = response["content"].as_array().unwrap().clone();
    content.push(json!({"type": "redacted_thinking", "data": "opaque-data"}));
    let expected = token_ids("unset|unset|user::Hi;assistant:reason:answer;user::Next;");
    let expected_count = expected.len();
    let backend = anthropic_hf_backend(ANTHROPIC_REASONING_TEMPLATE).await;
    let (mut app, engine_task) = test_app_with_chat_and_engine_request_check(
        move |llm| ChatLlm::new(TextLlm::new(llm, Arc::new(FakeChatBackend::new())), backend),
        move |request| assert_eq!(decode_prompt_token_ids(request).unwrap(), expected),
    )
    .await;
    let body = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 64,
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "messages": [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Next"}]});
    let mut count_body = body.clone();
    count_body["messages"][1]["content"][0]["signature"] = json!("different-signature");
    let (status, count) = post_json(&mut app, "/v1/messages/count_tokens", count_body).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    assert_eq!(count["input_tokens"], expected_count);
    let (status, response) = post_json(&mut app, "/v1/messages", body).await;
    engine_task.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{response}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_thinking_controls_reach_renderer_and_engine() {
    let backend = anthropic_hf_backend(ANTHROPIC_REASONING_TEMPLATE).await;
    for (thinking, effort, expected_controls, budget) in [
        (
            json!({"type": "enabled", "budget_tokens": 1024, "display": "omitted"}),
            json!("high"),
            "xhigh|true",
            Some(1024),
        ),
        (
            json!({"type": "adaptive", "display": "summarized"}),
            json!("xhigh"),
            "xhigh|true",
            None,
        ),
        (
            json!({"type": "adaptive", "display": "updates"}),
            json!(null),
            "unset|unset",
            None,
        ),
        (
            json!({"type": "disabled"}),
            json!("high"),
            "none|false",
            None,
        ),
        (
            json!({"type": "between_tools"}),
            json!("max"),
            "none|false",
            None,
        ),
    ] {
        let expected = token_ids(&format!("{expected_controls}|user::Hi;"));
        let expected_count = expected.len();
        let backend = backend.clone();
        let (mut app, engine_task) = test_app_with_chat_and_engine_request_check(
            move |llm| {
                ChatLlm::new(TextLlm::new(llm, Arc::new(FakeChatBackend::new())), backend)
                    .with_reasoning_effort_map([("high".to_owned(), "xhigh".to_owned())].into())
            },
            move |request| {
                assert_eq!(decode_prompt_token_ids(request).unwrap(), expected);
                assert_eq!(
                    request.sampling_params.as_ref().unwrap().thinking_token_budget,
                    budget
                );
            },
        )
        .await;
        let body = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": thinking, "output_config": {"effort": effort}});
        let (status, count) = post_json(&mut app, "/v1/messages/count_tokens", body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{count}");
        assert_eq!(count["input_tokens"], expected_count);
        let (status, response) = post_json(&mut app, "/v1/messages", body).await;
        engine_task.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{response}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_rejects_invalid_thinking_and_unsupported_content() {
    let (mut app, _engine_task) = test_app().await;
    for overrides in [
        json!({"thinking": {"type": "enabled", "budget_tokens": 1023}}),
        json!({"output_config": {"effort": "none"}}),
        json!({"context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}),
        json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
        json!({"messages": [{"role": "user", "content": [{"type": "thinking", "thinking": "reason"}]}]}),
    ] {
        let mut body = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}]});
        body.as_object_mut().unwrap().extend(overrides.as_object().unwrap().clone());
        for endpoint in ["/v1/messages", "/v1/messages/count_tokens"] {
            let (status, response) = post_json(&mut app, endpoint, body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {response}");
            assert_eq!(response["error"]["type"], "invalid_request_error");
        }
    }
    let mut body = messages_request();
    body["output_config"] = json!({"format": {"type": "json_schema", "schema": {}}});
    let (status, _) = post_json(&mut app, "/v1/messages", body.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_json(&mut app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_inline_system_uses_renderer_fallback() {
    let template = "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('System must be first') }}{% endif %}{{ m.role }}:{{ m.content }};{% endfor %}";
    let expected = "system:Base;user:Hi;assistant:Answer;user:Reminder\n\nNext;";
    let backend = anthropic_hf_backend(template).await;
    let expected = token_ids(expected);
    let expected_count = expected.len();
    let (mut app, engine_task) = test_app_with_chat_and_engine_request_check(
        move |llm| ChatLlm::new(TextLlm::new(llm, Arc::new(FakeChatBackend::new())), backend),
        move |request| assert_eq!(decode_prompt_token_ids(request).unwrap(), expected),
    )
    .await;
    let body = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 16,
        "system": "Base", "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Answer"},
            {"role": "system", "content": "Reminder"},
            {"role": "user", "content": "Next"}]});
    let (status, count) = post_json(&mut app, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    assert_eq!(count["input_tokens"], expected_count);
    let (status, response) = post_json(&mut app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn anthropic_inline_system_strips_billing_headers() {
    let backend =
        anthropic_hf_backend("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}")
            .await;
    let expected = token_ids(
        "system:Base;user:x-anthropic-billing-header: user text;assistant:Answer;system:Reminder;user:Next;",
    );
    let expected_count = expected.len();
    let (mut app, engine_task) = test_app_with_chat_and_engine_request_check(
        move |llm| ChatLlm::new(TextLlm::new(llm, Arc::new(FakeChatBackend::new())), backend),
        move |request| assert_eq!(decode_prompt_token_ids(request).unwrap(), expected),
    )
    .await;
    let mut messages = vec![
        json!({"role": "user", "content": "x-anthropic-billing-header: user text"}),
        json!({"role": "assistant", "content": "Answer"}),
    ];
    messages.extend([
        json!({"role": "system", "content": "x-anthropic-billing-header: hash-a"}),
        json!({"role": "system", "content": [
            {"type": "text", "text": "x-anthropic-billing-header: hash-b"}]}),
    ]);
    let content = json!([
        {"type": "text", "text": "x-anthropic-billing-header: hash-c"},
        {"type": "text", "text": "Reminder"}
    ]);
    messages.push(json!({"role": "system", "content": content}));
    messages.push(json!({"role": "user", "content": "Next"}));
    let body = json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "max_tokens": 16,
        "system": [{"type": "text", "text": "Base"}], "messages": messages});
    let (status, count) = post_json(&mut app, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{count}");
    assert_eq!(count["input_tokens"], expected_count);
    let (status, response) = post_json(&mut app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    engine_task.await.unwrap();
}
