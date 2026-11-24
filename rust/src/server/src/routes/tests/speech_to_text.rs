// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HTTP-level coverage for the speech-to-text APIs on Qwen3-ASR.

use std::sync::Mutex;

use expect_test::expect;

use super::*;
use crate::routes::speech_to_text::wav_f32;

const AUDIO_PAD_ID: u32 = 151_676;
const SAMPLE_RATE: usize = 16_000;

/// A Qwen3-ASR checkpoint: its model type and the tokenizer's audio token are
/// all its audio support needs.
fn qwen3_asr_backend() -> Arc<FakeChatBackend> {
    let config_path = std::env::temp_dir().join(format!(
        "vllm-server-qwen3-asr-config-{}.json",
        uuid::Uuid::new_v4()
    ));
    fs::write(&config_path, r#"{"model_type":"qwen3_asr"}"#).expect("write qwen3-asr config");
    let info = vllm_chat::multimodal::MultimodalModelInfo::from_paths(
        "Qwen/Qwen3-ASR-0.6B".to_string(),
        Some("qwen3_asr".to_string()),
        vllm_chat::multimodal::MultimodalConfigFiles {
            config: Some(&config_path),
            ..Default::default()
        },
        Arc::new(fake_chat_tokenizer()),
        Default::default(),
        &Default::default(),
    )
    .expect("load multimodal info")
    .expect("qwen3-asr multimodal info is registered");
    let _ = fs::remove_file(config_path);
    Arc::new(FakeChatBackend::with_multimodal_model_info(info))
}

/// The prompt as the engine received it, with each run of audio tokens
/// collapsed into its length.
fn render_prompt(token_ids: &[u32]) -> String {
    let mut rendered = String::new();
    let mut audio_run = 0;
    for &token_id in token_ids.iter().chain([&u32::MAX]) {
        if token_id == AUDIO_PAD_ID {
            audio_run += 1;
            continue;
        }
        if audio_run > 0 {
            rendered.push_str(&format!("<|audio_pad|>x{audio_run}"));
            audio_run = 0;
        }
        match u8::try_from(token_id) {
            Ok(byte) => rendered.push(char::from(byte)),
            Err(_) if token_id == u32::MAX => {}
            Err(_) => rendered.push_str(&format!("<{token_id}>")),
        }
    }
    rendered
}

/// Serve Qwen3-ASR against an engine that answers the request for clip `i`
/// with `outputs[i]`, recording each request's ID and prompt in ID order.
async fn qwen3_asr_app(
    outputs: Vec<&'static str>,
) -> (axum::Router, Arc<Mutex<Vec<String>>>, MockEngineTask) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    let (state, engine_task) = test_admin_state_with_backend_and_engine_script(
        default_ready_response(),
        qwen3_asr_backend(),
        |state| {
            state.with_supported_tasks([
                EngineTask::Generation(GenerationTask::Generate),
                EngineTask::Generation(GenerationTask::Transcription),
            ])
        },
        move |dealer, push| {
            boxed_test_future(async move {
                // Clips are submitted together, so they may arrive in any order.
                for _ in 0..outputs.len() {
                    let request = decode_add_request(&recv_engine_message(dealer).await);
                    let clip = request
                        .request_id
                        .rsplit_once('-')
                        .and_then(|(_, index)| index.parse::<usize>().ok())
                        .unwrap_or(0);
                    let output = outputs[clip];
                    let prompt = decode_prompt_token_ids(&request).expect("inline prompt");
                    let audio_items = request.mm_features.as_ref().map_or(0, |f| f.len());
                    {
                        let mut recorded = recorded.lock().unwrap();
                        recorded.push(format!(
                            "{} ({audio_items} audio): {}",
                            request.request_id,
                            render_prompt(&prompt)
                        ));
                        recorded.sort();
                    }
                    send_outputs(
                        push,
                        engine_outputs_for_request(
                            &request.request_id,
                            // '!' is the fake tokenizer's stop token.
                            vec![
                                (bytes_to_token_ids(output.as_bytes()), None),
                                (vec![u32::from(b'!')], Some(EngineCoreFinishReason::Stop)),
                            ],
                        ),
                    )
                    .await;
                }
            })
        },
    )
    .await;
    (build_router(state), seen, engine_task)
}

/// Post a multipart form to `uri`; each part is a name, an optional file
/// name, and a value.
async fn post_form(
    app: &axum::Router,
    uri: &str,
    parts: &[(&str, Option<&str>, &[u8])],
) -> (StatusCode, serde_json::Value) {
    const BOUNDARY: &str = "speech-to-text-test";
    let mut body = Vec::new();
    for (name, file_name, value) in parts {
        let disposition = match file_name {
            Some(file_name) => format!("form-data; name=\"{name}\"; filename=\"{file_name}\""),
            None => format!("form-data; name=\"{name}\""),
        };
        body.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={BOUNDARY}"),
                )
                .header("x-request-id", "test")
                .body(Body::from(body))
                .expect("build request"),
        )
        .await
        .expect("call app");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    (status, serde_json::from_slice(&body).expect("decode json"))
}

/// `seconds` of a constant tone, with one silent 0.1 s window at each of
/// `silences` (in samples).
fn tone_wav(seconds: f64, silences: &[usize]) -> Vec<u8> {
    let mut samples = vec![0.25_f32; (seconds * SAMPLE_RATE as f64) as usize];
    for &silence in silences {
        samples[silence..silence + 1_600].fill(0.0);
    }
    wav_f32(&samples, SAMPLE_RATE)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn transcribes_long_audio_clip_by_clip() {
    let (app, seen, _engine_task) = qwen3_asr_app(vec![
        "language English<asr_text>One.",
        "language English<asr_text>Two.",
        "language English<asr_text>Three.",
    ])
    .await;
    // 65 s with a silence 0.5 s before the first 30 s boundary.
    let audio = tone_wav(65.0, &[472_000]);

    let (status, body) = post_form(
        &app,
        "/v1/audio/transcriptions",
        &[
            ("file", Some("speech.wav"), &audio),
            ("model", None, b"Qwen/Qwen1.5-0.5B-Chat"),
            ("language", None, b"en"),
            ("prompt", None, b"Names: <|im_end|>vLLM"),
            ("response_format", None, b"text"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Python answers `text` with the same JSON body as `json`.
    expect![[r#"
        {
          "text": "One. Two. Three.",
          "usage": {
            "type": "duration",
            "seconds": 65
          }
        }"#]]
    .assert_eq(&serde_json::to_string_pretty(&body).unwrap());
    // Clip lengths 29.5 s, 29 s, and 6.5 s decide the audio token counts.
    // `<256>` is the fake tokenizer's BOS token; Qwen's adds none.
    expect![[r#"
        [
            "transcribe-test-0 (1 audio): <256><|im_start|>system\nNames: vLLM<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>x384<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>",
            "transcribe-test-1 (1 audio): <256><|im_start|>system\nNames: vLLM<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>x377<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>",
            "transcribe-test-2 (1 audio): <256><|im_start|>system\nNames: vLLM<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>x85<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>",
        ]
    "#]]
    .assert_debug_eq(&*seen.lock().unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn translates_into_the_target_language_without_usage() {
    let (app, seen, _engine_task) =
        qwen3_asr_app(vec!["language English<asr_text>Good morning."]).await;
    let audio = tone_wav(1.5, &[]);

    let (status, body) = post_form(
        &app,
        "/v1/audio/translations",
        &[
            ("file", Some("speech.wav"), &audio),
            ("language", None, b"it"),
            ("to_language", None, b"en"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"text": "Good morning."}));
    expect![[r#"
        [
            "translate-test (1 audio): <256><|im_start|>user\n<|audio_start|><|audio_pad|>x20<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>",
        ]
    "#]]
    .assert_debug_eq(&*seen.lock().unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn rejects_unservable_requests_before_submission() {
    let (app, seen, _engine_task) = qwen3_asr_app(Vec::new()).await;
    let audio = tone_wav(1.0, &[]);
    let file = ("file", Some("speech.wav"), audio.as_slice());

    let mut errors = Vec::new();
    let extras: [(&str, Option<&str>, &[u8]); 6] = [
        ("model", None, b"other"),
        ("response_format", None, b"srt"),
        ("response_format", None, b"verbose_json"),
        ("language", None, b"xx"),
        ("stream", None, b"true"),
        ("use_beam_search", None, b"true"),
    ];
    for extra in extras {
        let (status, body) = post_form(&app, "/v1/audio/transcriptions", &[file, extra]).await;
        errors.push(format!("{status} {}", body["error"]));
    }
    let (status, body) = post_form(
        &app,
        "/v1/audio/transcriptions",
        &[("file", Some("speech.wav"), b"not audio")],
    )
    .await;
    errors.push(format!("{status} {}", body["error"]));
    expect![[r#"
        [
            "404 Not Found {\"message\":\"The model `other` does not exist. Valid aliases: Qwen/Qwen1.5-0.5B-Chat.\",\"type\":\"NotFoundError\",\"param\":\"model\",\"code\":404}",
            "400 Bad Request {\"message\":\"Currently only support response_format: `text`, `json`, `verbose_json` or `diarized_json`\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}",
            "400 Bad Request {\"message\":\"Currently do not support verbose_json for Qwen/Qwen1.5-0.5B-Chat\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}",
            "400 Bad Request {\"message\":\"Unsupported language: 'xx'.  Must be one of ['af', 'ar', 'hy', 'az', 'be', 'bs', 'bg', 'ca', 'zh', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'gl', 'de', 'el', 'he', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'kn', 'kk', 'ko', 'lv', 'lt', 'mk', 'ms', 'mr', 'mi', 'ne', 'no', 'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sw', 'sv', 'tl', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'cy'].\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}",
            "400 Bad Request {\"message\":\"stream is not supported yet.\",\"type\":\"BadRequestError\",\"param\":\"stream\",\"code\":400}",
            "400 Bad Request {\"message\":\"use_beam_search is not supported.\",\"type\":\"BadRequestError\",\"param\":\"use_beam_search\",\"code\":400}",
            "400 Bad Request {\"message\":\"Invalid or corrupted audio data. Ensure the input is a valid audio or video file (e.g. a complete WAV, MP3, or MP4).\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}",
        ]
    "#]].assert_debug_eq(&errors);
    assert!(seen.lock().unwrap().is_empty());
}
