// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /v1/decisions` against the mock engine.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::json;
use serial_test::serial;
use tower::Service as _;

use super::*;
use crate::routes::decisions::NextTokenReads;

/// A ChatML template like Qwen3's: with thinking off, the generation prompt
/// ends with an empty think block.
struct ChatMlRenderer;

impl ChatRenderer for ChatMlRenderer {
    fn render(&self, request: &ChatRequest) -> vllm_chat::Result<vllm_chat::RenderedPrompt> {
        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                message.role().as_str(),
                message.text_content()?
            ));
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("<|im_start|>assistant\n");
            if request.chat_options.template_kwargs.get("enable_thinking") == Some(&json!(false)) {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        }
        Ok(vllm_chat::RenderedPrompt {
            prompt: Prompt::Text(prompt),
            media_order: None,
            effective_template_kwargs: request.chat_options.template_kwargs.clone(),
        })
    }
}

/// The fake tokenizer with ChatML markers. `yes` and `no` are one token
/// each, as in real vocabularies.
fn chatml_tokenizer(with_yes_no: bool) -> TestTokenizer {
    let tokenizer = fake_chat_tokenizer()
        .with_special_token("<|im_start|>", 0xF100)
        .with_special_token("<|im_end|>", 0xF101);
    if with_yes_no {
        tokenizer.with_regular_token("yes", 0xF102).with_regular_token("no", 0xF103)
    } else {
        tokenizer
    }
}

fn chatml_backend(with_yes_no: bool) -> Arc<FakeChatBackend> {
    Arc::new(FakeChatBackend {
        tokenizer: Some(Arc::new(chatml_tokenizer(with_yes_no))),
        ..FakeChatBackend::with_chat_renderer(Arc::new(ChatMlRenderer))
    })
}

/// A state serving `/v1/decisions` with the probed reads.
fn with_probed_decisions(state: AppState) -> AppState {
    let reads = NextTokenReads::probe(&state.chat).expect("probe the chat template");
    state.with_decisions(Some(reads))
}

fn no_engine_requests<'a>(_: &'a mut DealerSocket, _: &'a mut PushSocket) -> TestFuture<'a> {
    boxed_test_future(async {})
}

async fn post_decision(
    app: &mut axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    post(app, "/v1/decisions", Some("req"), body).await
}

async fn post_systemone(
    app: &mut axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    post(app, "/v1/systemone", None, body).await
}

async fn post(
    app: &mut axum::Router,
    uri: &str,
    request_id: Option<&str>,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let mut request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(request_id) = request_id {
        request = request.header("X-Request-Id", request_id);
    }
    let response = app
        .call(request.body(Body::from(body.to_string())).expect("build request"))
        .await
        .expect("call app");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    (status, serde_json::from_slice(&bytes).expect("JSON body"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_read_each_question_from_its_label_logprobs() {
    let requests = Arc::default();
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        with_probed_decisions,
        generative_scoring_script(
            vec![vec![-0.1, -2.5], vec![-1.0, -0.5], vec![-0.2, -1.6]],
            Arc::clone(&requests),
        ),
    )
    .await;
    let mut app = build_router(state);

    let (status, json) = post_decision(
        &mut app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": [{"role": "user", "content": "The screen is broken."}],
            "questions": [
                {"type": "predicate", "instructions": "Is the screen broken?"},
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Who handles this?",
                    "choices": [{"value": "returns"}, {"value": false, "description": "nobody"}],
                },
                {
                    "type": "score",
                    "name": "damage",
                    "instructions": "Rate the damage.",
                    "levels": [{"label": "none"}, {"label": "total"}],
                },
            ],
            "cache_salt": "salt",
        }),
    )
    .await;
    engine_task.finish().await;
    assert_eq!(status, StatusCode::OK, "{json}");

    let tokenizer = chatml_tokenizer(true);
    let engine_requests = (requests.lock().unwrap().iter())
        .map(|request| {
            let params = request.sampling_params.as_ref().expect("sampling params");
            let prompt = decode_prompt_token_ids(request).expect("prompt token IDs");
            format!(
                "{}: temperature {}, max_tokens {}, logprobs {:?}, logprob_token_ids {:?}, \
                 cache_salt {:?}\n{}",
                request.request_id,
                params.temperature,
                params.max_tokens,
                params.logprobs,
                params.logprob_token_ids,
                request.cache_salt,
                tokenizer.decode(&prompt, false).unwrap(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n---\n");
    expect_test::expect![[r#"
        decision-req-0: temperature 0, max_tokens 1, logprobs None, logprob_token_ids Some([61698, 61699]), cache_salt Some("salt")
        <|im_start|>user
        The screen is broken.

        Question: Is the screen broken?
        yes
        no
        Answer with yes or no only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>


        ---
        decision-req-1: temperature 0, max_tokens 1, logprobs None, logprob_token_ids Some([65, 66]), cache_salt Some("salt")
        <|im_start|>user
        The screen is broken.

        Question: Who handles this?
        A: "returns"
        B: false - nobody
        Answer with the letter of one option only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>


        ---
        decision-req-2: temperature 0, max_tokens 1, logprobs None, logprob_token_ids Some([65, 66]), cache_salt Some("salt")
        <|im_start|>user
        The screen is broken.

        Question: Rate the damage.
        A: none
        B: total
        Answer with the letter of one option only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

    "#]]
    .assert_eq(&engine_requests);
    expect_test::expect![[r#"
        {
          "model": "Qwen/Qwen1.5-0.5B-Chat",
          "answers": [
            {
              "type": "predicate",
              "name": null,
              "probability": 0.9168273033924488
            },
            {
              "type": "choice",
              "name": "department",
              "choice": false,
              "probabilities": [
                {
                  "value": "returns",
                  "probability": 0.37754066879814546
                },
                {
                  "value": false,
                  "probability": 0.6224593312018546
                }
              ],
              "confidence": 0.6065306597126334
            },
            {
              "type": "score",
              "name": "damage",
              "score": 0.19781610813099337,
              "probabilities": [
                {
                  "value": 0,
                  "label": "none",
                  "probability": 0.8021838918690066
                },
                {
                  "value": 1,
                  "label": "total",
                  "probability": 0.19781610813099337
                }
              ],
              "confidence": 0.8187307506379741
            }
          ],
          "usage": {
            "input_tokens": 389,
            "input_tokens_details": {
              "cached_tokens": 0,
              "cache_write_tokens": 0
            },
            "output_tokens": 3,
            "output_tokens_details": {
              "reasoning_tokens": 0
            },
            "total_tokens": 392
          }
        }"#]]
    .assert_eq(&serde_json::to_string_pretty(&json).unwrap());
}

/// Like Python, a read sets `logprob_token_ids` without `logprobs`, so a
/// choice may have more options than `--max-logprobs` (20 by default).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_read_more_labels_than_max_logprobs() {
    let requests = Arc::default();
    let logprobs = (0..26).map(|i| -0.1 * i as f32).collect();
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        with_probed_decisions,
        generative_scoring_script(vec![logprobs], Arc::clone(&requests)),
    )
    .await;
    let mut app = build_router(state);
    let choices: Vec<_> =
        (b'a'..=b'z').map(|c| json!({"value": (c as char).to_string()})).collect();
    let (status, json) = post_decision(
        &mut app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "x",
            "questions": [{"type": "choice", "instructions": "Pick", "choices": choices}],
        }),
    )
    .await;
    engine_task.finish().await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["answers"][0]["choice"], "a");
    let requests = requests.lock().unwrap();
    let params = requests[0].sampling_params.as_ref().unwrap();
    assert_eq!(params.logprobs, None);
    assert_eq!(params.logprob_token_ids.as_ref().map(Vec::len), Some(26));
}

/// Without a read strategy Python answers 501 too, with these messages.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_without_a_read_strategy_answer_like_python() {
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        |state| state,
        no_engine_requests,
    )
    .await;
    let mut app = build_router(state);
    let mut messages = Vec::new();
    for uri in ["/v1/decisions", "/v1/systemone"] {
        let (status, json) = post(&mut app, uri, None, json!({})).await;
        messages.push(format!(
            "{uri}: {status} {}: {}",
            json["error"]["type"].as_str().unwrap(),
            json["error"]["message"].as_str().unwrap()
        ));
    }
    engine_task.finish().await;
    expect_test::expect![[r#"
        [
            "/v1/decisions: 501 Not Implemented NotImplementedError: The model does not support the Decisions API",
            "/v1/systemone: 501 Not Implemented NotImplementedError: The model does not support the structured decision API",
        ]
    "#]]
    .assert_debug_eq(&messages);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_need_every_label_to_be_one_token() {
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(false),
        |state| state,
        no_engine_requests,
    )
    .await;
    let error = NextTokenReads::probe(&state.chat).unwrap_err();
    engine_task.finish().await;
    expect_test::expect!["label 'yes' is not one distinct token after this model's chat prompt"]
        .assert_eq(&error);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decisions_refuse_what_python_refuses() {
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        with_probed_decisions,
        no_engine_requests,
    )
    .await;
    let mut app = build_router(state);
    let predicate = json!({"type": "predicate", "instructions": "Q?"});
    let mut messages = Vec::new();
    for body in [
        json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "input": "x", "questions": [predicate], "joint": true}),
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}],
            "questions": [predicate],
        }),
        json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "input": "x", "questions": vec![predicate.clone(); 65]}),
        json!({"model": "Qwen/Qwen1.5-0.5B-Chat", "input": "x", "questions": []}),
    ] {
        let (status, json) = post_decision(&mut app, body).await;
        messages.push(format!(
            "{status}: {}",
            json["error"]["message"].as_str().unwrap()
        ));
    }
    engine_task.finish().await;
    expect_test::expect![[r#"
        [
            "400 Bad Request: joint: this model reads each question on its own; joint reads need DiffusionGemma",
            "400 Bad Request: Failed to deserialize the JSON body into the target type: input: Decisions supports text input only; images are not supported at line 1 column 110",
            "400 Bad Request: questions: at most 64 for this model",
            "400 Bad Request: questions: must have between 1 and 200 items",
        ]
    "#]]
    .assert_debug_eq(&messages);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn systemone_reads_questions_by_id() {
    let requests = Arc::default();
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        with_probed_decisions,
        generative_scoring_script(
            vec![vec![-0.1, -2.5], vec![-1.0, -0.5, -3.0], vec![-0.2, -1.6]],
            Arc::clone(&requests),
        ),
    )
    .await;
    let mut app = build_router(state);

    let (status, mut json) = post_systemone(
        &mut app,
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "state": {"ticket": "Charged twice.", "amount": 2.5},
            "instructions": "You triage tickets.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is money moving?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "charges", "shipping": null, "security": null},
                },
                "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "angry"]},
            },
            "chat_template_kwargs": {"enable_thinking": null, "flavor": "x"},
            "cache_salt": "salt",
            "priority": 5,
            "request_id": "req",
        }),
    )
    .await;
    engine_task.finish().await;
    assert_eq!(status, StatusCode::OK, "{json}");
    json["created"] = json!("<created>");

    let tokenizer = chatml_tokenizer(true);
    let engine_requests = (requests.lock().unwrap().iter())
        .map(|request| {
            let params = request.sampling_params.as_ref().expect("sampling params");
            let prompt = decode_prompt_token_ids(request).expect("prompt token IDs");
            format!(
                "{}: logprob_token_ids {:?}, cache_salt {:?}, priority {}\n{}",
                request.request_id,
                params.logprob_token_ids,
                request.cache_salt,
                request.priority,
                tokenizer.decode(&prompt, false).unwrap(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n---\n");
    expect_test::expect![[r#"
        decision-req-0: logprob_token_ids Some([61698, 61699]), cache_salt Some("salt"), priority 5
        <|im_start|>system
        You triage tickets.<|im_end|>
        <|im_start|>user
        {"ticket": "Charged twice.", "amount": 2.5}

        Question: Is money moving?
        yes
        no
        Answer with yes or no only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>


        ---
        decision-req-1: logprob_token_ids Some([65, 66, 67]), cache_salt Some("salt"), priority 5
        <|im_start|>system
        You triage tickets.<|im_end|>
        <|im_start|>user
        {"ticket": "Charged twice.", "amount": 2.5}

        Question: Which team?
        A: billing - charges
        B: shipping
        C: security
        Answer with the letter of one option only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>


        ---
        decision-req-2: logprob_token_ids Some([48, 49]), cache_salt Some("salt"), priority 5
        <|im_start|>system
        You triage tickets.<|im_end|>
        <|im_start|>user
        {"ticket": "Charged twice.", "amount": 2.5}

        Question: How upset?
        0: calm
        1: angry
        Answer with the number of one level only.<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

    "#]]
    .assert_eq(&engine_requests);
    expect_test::expect![[r#"
        {
          "id": "decision-req",
          "object": "structured_decision",
          "created": "<created>",
          "model": "Qwen/Qwen1.5-0.5B-Chat",
          "answers": {
            "urgent": {
              "type": "noul",
              "noul": 0.9168273033924488,
              "probabilities": {
                "yes": 0.9168273033924488,
                "no": 0.08317269660755117
              },
              "confidence": 0.9048374166876468
            },
            "team": {
              "type": "choice",
              "choice": "shipping",
              "probabilities": {
                "billing": 0.3591881057825386,
                "shipping": 0.5922010701861591,
                "security": 0.04861082403130227
              },
              "confidence": 0.6065306597126333
            },
            "tone": {
              "type": "score",
              "score": 0.19781610813099337,
              "legend": {
                "0": "calm",
                "1": "angry"
              },
              "probabilities": {
                "0": 0.8021838918690066,
                "1": 0.19781610813099337
              },
              "confidence": 0.8187307506379741
            }
          },
          "usage": {
            "input_tokens": 540,
            "output_tokens": 3
          },
          "diagnostics": {
            "urgent": {
              "label_mass": 0.9869224153115456,
              "argmax_is_label": true
            },
            "team": {
              "label_mass": 1.0241971692519396,
              "argmax_is_label": true
            },
            "tone": {
              "label_mass": 1.0206272638190417,
              "argmax_is_label": true
            }
          },
          "joint": null
        }"#]]
    .assert_eq(&serde_json::to_string_pretty(&json).unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn systemone_refuses_what_python_refuses() {
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        chatml_backend(true),
        with_probed_decisions,
        no_engine_requests,
    )
    .await;
    let mut app = build_router(state);
    let choice = json!({"type": "choice", "criteria": {"a": null, "b": null}});
    let many: serde_json::Map<_, _> = (0..65).map(|i| (format!("q{i}"), choice.clone())).collect();
    let mut messages = Vec::new();
    for body in [
        json!({"state": "x", "questions": {"q": choice}, "images": ["https://example.com/a.png"]}),
        json!({"state": "x", "questions": {"q": choice}, "audio": ["a"], "videos": ["v"]}),
        json!({"state": "x", "questions": {"q": choice}, "chat_template_kwargs": {"enable_thinking": true}}),
        json!({"state": "x", "questions": {"q": choice}, "joint": true}),
        json!({"state": "x", "questions": many}),
        json!({"state": "x", "questions": {}}),
        json!({"state": "x", "questions": {"q": {"type": "noul", "criteria": ["yes", "no"]}}}),
    ] {
        let (status, json) = post_systemone(&mut app, body).await;
        messages.push(format!(
            "{status}: {}",
            json["error"]["message"].as_str().unwrap()
        ));
    }
    engine_task.finish().await;
    expect_test::expect![[r#"
        [
            "400 Bad Request: this model does not take image inputs",
            "400 Bad Request: this model does not take audio, video inputs",
            "400 Bad Request: a read is the reply's first token, so thinking must be off",
            "400 Bad Request: joint: this model reads each question on its own; joint reads need DiffusionGemma",
            "400 Bad Request: questions: at most 64 for this model",
            "400 Bad Request: questions: needs at least one question",
            "400 Bad Request: question 'q': noul criteria must be an object with true and false",
        ]
    "#]].assert_debug_eq(&messages);
}

/// A decision read whose output stream fails after submission is classified
/// like every other route's: a late rejection of the request is a 400, not a
/// 500.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn decision_read_stream_errors_are_classified() {
    let chat = ChatLlm::from_shared_backend(
        StreamErrorLlm::new(|| vllm_llm::Error::RemoteInvalidRequest {
            message: "the remote engine rejected the request".to_string(),
        }),
        chatml_backend(true),
    );
    let state = with_probed_decisions(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    ));
    let (status, json) = post_decision(
        &mut build_router(Arc::new(state)),
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "input": "x",
            "questions": [{"type": "predicate", "instructions": "Q?"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(
        json["error"]["message"],
        "the remote engine rejected the request"
    );
}
