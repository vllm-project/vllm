// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI speech-to-text APIs: `/v1/audio/transcriptions` and
//! `/v1/audio/translations`.
//!
//! The uploaded audio is decoded, split into clips no longer than the model
//! takes at once, and each clip becomes one text request carrying one audio
//! item. The clip outputs are joined into one response.
//!
//! Original Python definition:
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/entrypoints/speech_to_text/base/serving.py>

mod chunk;
mod form;
mod language;
mod qwen3_asr;

use std::num::NonZeroU32;
use std::sync::Arc;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::future;
use llm_multimodal::audio::decode_audio_mono_f32;
use llm_multimodal::{MediaContentPart, MediaFault};
use serde::Serialize;
use thiserror_ext::AsReport as _;
use tracing::{debug, error};
use tracing_futures::Instrument as _;
use vllm_chat::ChatLlm;
use vllm_text::{
    FinishReason, Prompt, SamplingParams, TextDecodeOptions, TextOutputStreamExt as _, TextRequest,
    normalize_top_k,
};

use self::chunk::{ChunkingConfig, split_audio};
use self::form::{ResponseFormat, SpeechToTextForm, SpeechToTextTask};
use self::language::{ISO639_1_SUPPORTED_LANGS, validate_language};
use crate::error::{
    ApiError, bail_invalid_request, chat_submit_error, engine_error, server_error,
    text_submit_error,
};
use crate::state::AppState;
use crate::utils::resolve_request_context;

/// How the loaded model takes audio, fixed when the routes are built.
///
/// Only Qwen3-ASR is served so far.
#[derive(Debug, Clone, Copy)]
pub(super) struct SpeechToTextModel {
    /// Sample rate the feature extractor expects, in Hz.
    sample_rate: usize,
    /// Longest audio the model takes at once, in seconds: the feature
    /// extractor's `chunk_length`, like Python's `max_audio_clip_s`.
    max_clip_s: usize,
}

impl SpeechToTextModel {
    /// Resolve the loaded model's audio input, if the Rust frontend can serve
    /// speech-to-text for it.
    pub(super) fn resolve(chat: &ChatLlm) -> Option<Self> {
        let info = chat.multimodal_model_info()?;
        // TODO(#69115): Whisper and the other transcription families.
        if info.model_type() != Some("qwen3_asr") {
            return None;
        }
        let input = info.audio_input()?;
        // `Qwen3ASRFeatureExtractor` defaults.
        Some(Self {
            sample_rate: input.sample_rate.unwrap_or(16_000),
            max_clip_s: input.chunk_length_s.unwrap_or(30),
        })
    }

    /// Python's `SpeechToTextConfig` defaults: search the second before each
    /// clip boundary for the quietest 0.1 s, in the audio's own sample rate.
    fn chunking(self, sample_rate: usize) -> ChunkingConfig {
        ChunkingConfig {
            max_clip_samples: sample_rate * self.max_clip_s,
            search_samples: sample_rate,
            energy_window_samples: 1_600 * sample_rate / self.sample_rate,
        }
    }
}

/// Register the speech-to-text routes for `model`.
pub(super) fn register(
    router: Router<Arc<AppState>>,
    model: SpeechToTextModel,
) -> Router<Arc<AppState>> {
    router
        .route(
            "/v1/audio/transcriptions",
            post(
                move |State(state): State<Arc<AppState>>,
                      headers: HeaderMap,
                      multipart: Result<Multipart, MultipartRejection>| {
                    speech_to_text(
                        state,
                        headers,
                        multipart,
                        model,
                        SpeechToTextTask::Transcribe,
                    )
                },
            ),
        )
        .route(
            "/v1/audio/translations",
            post(
                move |State(state): State<Arc<AppState>>,
                      headers: HeaderMap,
                      multipart: Result<Multipart, MultipartRejection>| {
                    speech_to_text(
                        state,
                        headers,
                        multipart,
                        model,
                        SpeechToTextTask::Translate,
                    )
                },
            ),
        )
}

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
    usage: TranscriptionUsage,
}

/// Audio duration usage, rounded up to whole seconds as OpenAI specifies.
#[derive(Serialize)]
struct TranscriptionUsage {
    #[serde(rename = "type")]
    usage_type: &'static str,
    seconds: u64,
}

#[derive(Serialize)]
struct TranslationResponse {
    text: String,
}

async fn speech_to_text(
    state: Arc<AppState>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
    model: SpeechToTextModel,
    task: SpeechToTextTask,
) -> Result<Response, ApiError> {
    let form = SpeechToTextForm::parse(
        task,
        multipart,
        state.api_server_options.max_audio_clip_filesize_mb,
    )
    .await?;
    if form.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }
    // TODO(#69115): stream `transcription.chunk` events.
    if form.stream {
        bail_invalid_request!(param = "stream", "stream is not supported yet.");
    }

    let ctx = resolve_request_context(&headers, None)?;
    let lora_resolution = state.resolve_model_with_loras(form.model.as_deref()).await;
    lora_resolution.model_names.check(form.model.as_deref())?;
    let model_name = form
        .model
        .clone()
        .unwrap_or_else(|| lora_resolution.model_names.primary().to_string());

    match form.response_format {
        ResponseFormat::Json | ResponseFormat::Text => {}
        ResponseFormat::Srt | ResponseFormat::Vtt => bail_invalid_request!(
            "Currently only support response_format: `text`, `json`, `verbose_json` or \
             `diarized_json`"
        ),
        ResponseFormat::VerboseJson => {
            bail_invalid_request!("Currently do not support verbose_json for {model_name}")
        }
        ResponseFormat::DiarizedJson => {
            bail_invalid_request!("Currently do not support diarized_json for {model_name}")
        }
    }
    for language in [&form.language, &form.to_language].into_iter().flatten() {
        validate_language(language, ISO639_1_SUPPORTED_LANGS, "Qwen3-ASR")?;
    }

    let prompt_language = match task {
        SpeechToTextTask::Transcribe => form.language.as_deref(),
        SpeechToTextTask::Translate => form.to_language.as_deref(),
    };
    let prompt = qwen3_asr::generation_prompt(&form.prompt, prompt_language);
    let text = state.chat.text();
    let prompt_token_ids = text
        .tokenizer()
        .encode(&prompt, true)
        .map_err(|error| text_submit_error("failed to tokenize the prompt", error.into()))?;
    let sampling_params = sampling_params(&form)?;

    let audio = decode_audio_mono_f32(form.audio).await.map_err(|error| match error.fault() {
        // The decoder's report can name server paths, so it stays in the log.
        MediaFault::Client => {
            debug!(error = %error.as_report(), "failed to decode the audio file");
            ApiError::invalid_request(
                "Invalid or corrupted audio data. Ensure the input is a valid audio or video \
                 file (e.g. a complete WAV, MP3, or MP4).",
                None,
            )
        }
        _ => {
            error!(error = %error.as_report(), "failed to decode the audio file");
            server_error!("failed to decode the audio file")
        }
    })?;
    let duration_s = audio.samples.len() as f64 / audio.sample_rate as f64;
    let clips = split_audio(&audio.samples, model.chunking(audio.sample_rate));

    let request_id = format!("{}-{}", task.as_str(), ctx.request_id);
    let request_span = tracing::info_span!("speech_to_text", request_id = %request_id);
    let mut prepared = Vec::with_capacity(clips.len());
    for (index, clip) in clips.iter().enumerate() {
        let audio_item = MediaContentPart::AudioData {
            data: wav_f32(&audio.samples[clip.clone()], audio.sample_rate),
            mime_type: Some("audio/wav".to_string()),
            uuid: None,
        };
        let mut token_ids = prompt_token_ids.clone();
        let mm_features = state
            .chat
            .prepare_media(vec![audio_item], &mut token_ids)
            .instrument(vllm_chat::mm_request_span(&request_id))
            .await
            .map_err(|error| chat_submit_error("failed to prepare the audio", error))?;
        let text_request = TextRequest {
            request_id: if clips.len() == 1 {
                request_id.clone()
            } else {
                format!("{request_id}-{index}")
            },
            prompt: Prompt::TokenIds(token_ids),
            mm_features,
            sampling_params: sampling_params.clone(),
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: ctx.priority.unwrap_or_default(),
            cache_salt: None,
            prompt_truncation: None,
            add_special_tokens: false,
            data_parallel_rank: ctx.data_parallel_rank,
            session_id: ctx.session_id.clone(),
            kv_hints: None,
            reasoning_parser_kwargs: Default::default(),
            reasoning_ended: None,
            lora_request: lora_resolution.lora_request.clone(),
            arrival_time: None,
        };
        prepared.push(
            text.request_processor()
                .prepare(text_request)
                .map_err(|error| text_submit_error("failed to prepare the request", error))?,
        );
    }

    let outputs = future::try_join_all(prepared.into_iter().map(|prepared| async {
        let mut streams = text
            .generate_prepared_n(prepared, NonZeroU32::MIN)
            .await
            .map_err(|error| text_submit_error("failed to submit the request", error))?;
        let stream = streams.pop().expect("one output stream per sample");
        stream.collect_output().await.map_err(|error| {
            server_error!("failed to collect the output: {}", error.to_report_string())
        })
    }))
    .instrument(request_span)
    .await?;

    let mut clip_texts = Vec::with_capacity(outputs.len());
    for output in &outputs {
        // Like Python, an engine error fails the whole request.
        if let FinishReason::Error(error) = &output.finish_reason {
            return Err(engine_error(error.as_deref()));
        }
        clip_texts.push(qwen3_asr::post_process_output(&output.text));
    }
    // Languages written without spaces join their clips without one.
    let separator = match form.language.as_deref() {
        Some(language) if ["ja", "zh"].contains(&language.to_lowercase().as_str()) => "",
        _ => " ",
    };
    let text = clip_texts.join(separator);

    Ok(match task {
        SpeechToTextTask::Transcribe => Json(TranscriptionResponse {
            text,
            usage: TranscriptionUsage {
                usage_type: "duration",
                seconds: duration_s.ceil() as u64,
            },
        })
        .into_response(),
        // Python reports no usage for translations.
        SpeechToTextTask::Translate => Json(TranslationResponse { text }).into_response(),
    })
}

/// Python's `TranscriptionRequest.to_sampling_params`, whose `temperature`,
/// `frequency_penalty`, and `presence_penalty` default to 0 rather than to
/// the model's generation config.
fn sampling_params(form: &SpeechToTextForm) -> Result<SamplingParams, ApiError> {
    let top_k = form
        .top_k
        .map(|top_k| {
            normalize_top_k(top_k)
                .map(|top_k| top_k.unwrap_or(0))
                .map_err(|message| ApiError::invalid_request(message, Some("top_k")))
        })
        .transpose()?;
    let max_tokens = form
        .max_completion_tokens
        .map(|max_tokens| {
            u32::try_from(max_tokens).ok().filter(|&n| n >= 1).ok_or_else(|| {
                ApiError::invalid_request(
                    format!("max_tokens must be at least 1, got {max_tokens}."),
                    Some("max_completion_tokens"),
                )
            })
        })
        .transpose()?;
    Ok(SamplingParams {
        temperature: Some(form.temperature as f32),
        watermarking: form.watermarking,
        top_p: form.top_p.map(|top_p| top_p as f32),
        top_k,
        seed: form.seed,
        max_tokens,
        min_p: form.min_p.map(|min_p| min_p as f32),
        frequency_penalty: Some(form.frequency_penalty as f32),
        presence_penalty: Some(form.presence_penalty as f32),
        repetition_penalty: form.repetition_penalty.map(|penalty| penalty as f32),
        vllm_xargs: form.vllm_xargs.clone().map(|xargs| xargs.into_iter().collect()),
        ..SamplingParams::default()
    })
}

/// Encode mono samples as a 32-bit float WAV file, which the audio decoder
/// reads back without loss.
pub(super) fn wav_f32(samples: &[f32], sample_rate: usize) -> Vec<u8> {
    const HEADER_LEN: usize = 44;
    const FORMAT_IEEE_FLOAT: u16 = 3;
    let data_len = (samples.len() * 4) as u32;
    let sample_rate = sample_rate as u32;
    let mut wav = Vec::with_capacity(HEADER_LEN + samples.len() * 4);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes()); // channels
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    wav.extend_from_slice(&4_u16.to_le_bytes()); // block align
    wav.extend_from_slice(&32_u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

#[cfg(test)]
mod tests {
    use llm_multimodal::audio::DecodedAudio;

    use super::*;

    #[tokio::test]
    async fn wav_round_trips_through_the_audio_decoder() {
        let samples = [0.0, 0.25, -0.5, 1.0, -1.0, 1e-7];
        let DecodedAudio {
            samples: decoded,
            sample_rate,
        } = decode_audio_mono_f32(wav_f32(&samples, 22_050).into()).await.unwrap();
        assert_eq!(sample_rate, 22_050);
        assert_eq!(decoded, samples);
    }
}
