// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `multipart/form-data` parsing for the speech-to-text APIs.
//!
//! Python binds the form to `TranscriptionRequest` / `TranslationRequest`
//! through FastAPI's `Form()`: a repeated field keeps its last value, unknown
//! fields are ignored, and a value that fails to parse is a 400 naming the
//! field as `body.<name>`.
//!
//! Original Python definitions:
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/entrypoints/speech_to_text/transcription/protocol.py#L52-L323>
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/entrypoints/speech_to_text/translation/protocol.py#L52-L307>

use std::collections::HashMap;
use std::str::FromStr;

use axum::body::Bytes;
use axum::extract::Multipart;
use axum::extract::multipart::{Field, MultipartRejection};
use serde_json::{Map, Value};

use crate::error::{ApiError, bail_invalid_request};

/// The speech-to-text API a form was posted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeechToTextTask {
    /// `/v1/audio/transcriptions`
    Transcribe,
    /// `/v1/audio/translations`
    Translate,
}

impl SpeechToTextTask {
    /// Python's `task_type`, also the request ID prefix.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

/// `response_format` values the request model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
    /// Transcriptions only.
    DiarizedJson,
}

/// A parsed speech-to-text form. Fields keep Python's defaults.
#[derive(Debug)]
pub(crate) struct SpeechToTextForm {
    pub audio: Bytes,
    pub model: Option<String>,
    pub language: Option<String>,
    pub to_language: Option<String>,
    pub prompt: String,
    pub response_format: ResponseFormat,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub stream_continuous_usage_stats: bool,
    pub vllm_xargs: Option<Map<String, Value>>,
    pub use_beam_search: bool,
    pub temperature: f64,
    pub watermarking: Option<bool>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub seed: Option<i64>,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: Option<f64>,
    pub max_completion_tokens: Option<i64>,
}

impl SpeechToTextForm {
    /// Read and validate a form, rejecting an audio upload larger than
    /// `max_audio_mb` MiB without buffering the excess.
    pub(crate) async fn parse(
        task: SpeechToTextTask,
        multipart: Result<Multipart, MultipartRejection>,
        max_audio_mb: i64,
    ) -> Result<Self, ApiError> {
        let mut multipart =
            multipart.map_err(|rejection| ApiError::invalid_form(rejection.body_text(), None))?;
        let max_audio_bytes = max_audio_mb.saturating_mul(1 << 20);

        let mut audio = None;
        let mut fields = HashMap::<String, String>::new();
        while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
            let Some(name) = field.name().map(str::to_owned) else {
                continue;
            };
            if name == "file" {
                if field.file_name().is_none() {
                    bail_invalid_request!(
                        param = "file",
                        "Expected 'file' to be a file-like object, not 'str'."
                    );
                }
                // Like FastAPI, a repeated field keeps its last value.
                audio = Some(read_audio(field, max_audio_bytes).await?);
            } else if task == SpeechToTextTask::Transcribe && name == "timestamp_granularities[]" {
                // Accepted for OpenAI compatibility; Python never reads it.
                let value = field.text().await.map_err(multipart_error)?;
                if !matches!(value.as_str(), "word" | "segment") {
                    return Err(form_error(&name, "Input should be 'word' or 'segment'"));
                }
            } else {
                let value = field.text().await.map_err(multipart_error)?;
                fields.insert(name, value);
            }
        }

        let mut fields = FormFields(fields);
        let form = Self {
            audio: audio.ok_or_else(|| form_error("file", "Field required"))?,
            model: fields.take("model").filter(|model| !model.is_empty()),
            language: fields.take("language"),
            to_language: fields.take("to_language").filter(|lang| !lang.is_empty()),
            prompt: fields.take("prompt").unwrap_or_default(),
            response_format: fields
                .take("response_format")
                .map(|value| parse_response_format(task, &value))
                .transpose()?
                .unwrap_or(ResponseFormat::Json),
            stream: fields.parse_bool("stream")?.unwrap_or(false),
            stream_include_usage: fields.parse_bool("stream_include_usage")?.unwrap_or(false),
            stream_continuous_usage_stats: fields
                .parse_bool("stream_continuous_usage_stats")?
                .unwrap_or(false),
            vllm_xargs: fields
                .take("vllm_xargs")
                .map(|value| parse_vllm_xargs(&value))
                .transpose()?,
            use_beam_search: fields.parse_bool("use_beam_search")?.unwrap_or(false),
            temperature: fields.parse_number("temperature")?.unwrap_or(0.0),
            watermarking: fields.parse_bool("watermarking")?,
            top_p: fields.parse_number("top_p")?,
            top_k: fields.parse_integer("top_k")?,
            min_p: fields.parse_number("min_p")?,
            seed: fields.parse_integer("seed")?,
            frequency_penalty: fields.parse_number("frequency_penalty")?.unwrap_or(0.0),
            presence_penalty: fields.parse_number("presence_penalty")?.unwrap_or(0.0),
            repetition_penalty: fields.parse_number("repetition_penalty")?,
            max_completion_tokens: fields.parse_integer("max_completion_tokens")?,
        };
        // Python only type-checks these: `n` and `length_penalty` take effect
        // with beam search, and `include_stop_str_in_output` has no stop
        // strings to apply to.
        fields.parse_integer("n")?;
        fields.parse_number("length_penalty")?;
        fields.parse_bool("include_stop_str_in_output")?;

        let stream_options = [
            ("stream_include_usage", form.stream_include_usage),
            (
                "stream_continuous_usage_stats",
                form.stream_continuous_usage_stats,
            ),
        ];
        if let Some((option, _)) = stream_options.into_iter().find(|&(_, set)| set && !form.stream)
        {
            bail_invalid_request!(
                param = option,
                "Stream options can only be defined when `stream=True`."
            );
        }
        Ok(form)
    }
}

/// Read the uploaded audio, counting but not keeping bytes past the limit so
/// the error can report the full size like Python.
async fn read_audio(mut field: Field<'_>, max_bytes: i64) -> Result<Bytes, ApiError> {
    let mut audio = Vec::new();
    let mut size: i64 = 0;
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        size = size.saturating_add(chunk.len() as i64);
        if size <= max_bytes {
            audio.extend_from_slice(&chunk);
        }
    }
    if size > max_bytes {
        return Err(ApiError::invalid_request(
            format!(
                "Maximum file size exceeded. Set VLLM_MAX_AUDIO_CLIP_FILESIZE_MB to increase \
                 this limit. (parameter=audio_filesize_mb, value={:?})",
                size as f64 / f64::from(1 << 20)
            ),
            Some("audio_filesize_mb"),
        ));
    }
    Ok(audio.into())
}

/// Text form fields by name, consumed as they are parsed.
struct FormFields(HashMap<String, String>);

impl FormFields {
    fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name)
    }

    /// Parse a boolean the way pydantic's lax mode does.
    fn parse_bool(&mut self, name: &str) -> Result<Option<bool>, ApiError> {
        self.take(name)
            .map(|value| match value.to_ascii_lowercase().as_str() {
                "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
                "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
                _ => Err(form_error(
                    name,
                    "Input should be a valid boolean, unable to interpret input",
                )),
            })
            .transpose()
    }

    fn parse_integer(&mut self, name: &str) -> Result<Option<i64>, ApiError> {
        self.parse(
            name,
            "Input should be a valid integer, unable to parse string as an integer",
        )
    }

    fn parse_number(&mut self, name: &str) -> Result<Option<f64>, ApiError> {
        self.parse(
            name,
            "Input should be a valid number, unable to parse string as a number",
        )
    }

    fn parse<T: FromStr>(&mut self, name: &str, message: &str) -> Result<Option<T>, ApiError> {
        self.take(name)
            .map(|value| value.trim().parse().map_err(|_| form_error(name, message)))
            .transpose()
    }
}

fn parse_response_format(task: SpeechToTextTask, value: &str) -> Result<ResponseFormat, ApiError> {
    Ok(match value {
        "json" => ResponseFormat::Json,
        "text" => ResponseFormat::Text,
        "srt" => ResponseFormat::Srt,
        "verbose_json" => ResponseFormat::VerboseJson,
        "vtt" => ResponseFormat::Vtt,
        "diarized_json" if task == SpeechToTextTask::Transcribe => ResponseFormat::DiarizedJson,
        _ => {
            let message = match task {
                SpeechToTextTask::Transcribe => {
                    "Input should be 'json', 'text', 'srt', 'verbose_json', 'vtt' or \
                     'diarized_json'"
                }
                SpeechToTextTask::Translate => {
                    "Input should be 'json', 'text', 'srt', 'verbose_json' or 'vtt'"
                }
            };
            return Err(form_error("response_format", message));
        }
    })
}

/// Parse `vllm_xargs`, which forms carry as a JSON object string whose values
/// are strings, numbers, or lists of them.
fn parse_vllm_xargs(value: &str) -> Result<Map<String, Value>, ApiError> {
    let parsed: Value = serde_json::from_str(value).map_err(|error| {
        ApiError::invalid_request(
            format!("Failed to parse vllm_xargs. Must be valid JSON: {error}"),
            Some("vllm_xargs"),
        )
    })?;
    let Value::Object(map) = parsed else {
        return Err(form_error(
            "vllm_xargs",
            "Input should be a valid dictionary",
        ));
    };
    let is_scalar = |value: &Value| matches!(value, Value::String(_) | Value::Number(_));
    for (key, value) in &map {
        let valid = match value {
            Value::Array(items) => items.iter().all(is_scalar),
            value => is_scalar(value),
        };
        if !valid {
            return Err(ApiError::invalid_form(
                "Input should be a string, a number, or a list of strings and numbers",
                Some(format!("body.vllm_xargs.{key}")),
            ));
        }
    }
    Ok(map)
}

fn form_error(field: &str, message: &str) -> ApiError {
    ApiError::invalid_form(message, Some(format!("body.{field}")))
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::invalid_form(error.body_text(), None)
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::extract::{FromRequest as _, Multipart};
    use axum::http::{Request, header};
    use expect_test::expect;

    use super::*;

    const BOUNDARY: &str = "speech-to-text-test";

    /// A form part: name, file name (`Some` makes it a file part), and value.
    type Part<'a> = (&'a str, Option<&'a str>, &'a [u8]);

    async fn parse(
        task: SpeechToTextTask,
        parts: &[Part<'_>],
    ) -> Result<SpeechToTextForm, ApiError> {
        parse_with_limit(task, parts, 25).await
    }

    async fn parse_with_limit(
        task: SpeechToTextTask,
        parts: &[Part<'_>],
        max_audio_mb: i64,
    ) -> Result<SpeechToTextForm, ApiError> {
        let mut body = Vec::new();
        for (name, file_name, value) in parts {
            body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
            let disposition = match file_name {
                Some(file_name) => {
                    format!("form-data; name=\"{name}\"; filename=\"{file_name}\"")
                }
                None => format!("form-data; name=\"{name}\""),
            };
            body.extend_from_slice(
                format!("Content-Disposition: {disposition}\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .unwrap();
        SpeechToTextForm::parse(
            task,
            Multipart::from_request(request, &()).await,
            max_audio_mb,
        )
        .await
    }

    const AUDIO: Part<'static> = ("file", Some("audio.wav"), b"RIFF");

    #[tokio::test]
    async fn parses_python_form_semantics() {
        let form = parse(
            SpeechToTextTask::Transcribe,
            &[
                AUDIO,
                ("model", None, b""),
                ("language", None, b"en"),
                ("response_format", None, b"text"),
                ("temperature", None, b"0.2"),
                // A repeated field keeps its last value.
                ("temperature", None, b" 0.7 "),
                ("stream", None, b"True"),
                ("stream_include_usage", None, b"on"),
                ("seed", None, b"-3"),
                ("vllm_xargs", None, br#"{"a": 1, "b": ["x", 2.5]}"#),
                ("timestamp_granularities[]", None, b"word"),
                ("timestamp_granularities[]", None, b"segment"),
                // Unknown fields are ignored, as are fields Python only
                // type-checks.
                ("hotwords", None, b"vLLM"),
                ("n", None, b"2"),
                ("unknown", None, b"value"),
            ],
        )
        .await
        .unwrap();
        expect![[r#"
            SpeechToTextForm {
                audio: b"RIFF",
                model: None,
                language: Some(
                    "en",
                ),
                to_language: None,
                prompt: "",
                response_format: Text,
                stream: true,
                stream_include_usage: true,
                stream_continuous_usage_stats: false,
                vllm_xargs: Some(
                    {
                        "a": Number(1),
                        "b": Array [
                            String("x"),
                            Number(2.5),
                        ],
                    },
                ),
                use_beam_search: false,
                temperature: 0.7,
                watermarking: None,
                top_p: None,
                top_k: None,
                min_p: None,
                seed: Some(
                    -3,
                ),
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: None,
                max_completion_tokens: None,
            }
        "#]]
        .assert_debug_eq(&form);
    }

    #[tokio::test]
    async fn rejects_invalid_forms_with_python_params() {
        let cases: &[(SpeechToTextTask, &[Part<'_>])] = &[
            (SpeechToTextTask::Transcribe, &[("model", None, b"m")]),
            (
                SpeechToTextTask::Transcribe,
                &[("file", None, b"not a file")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("stream", None, b"maybe")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("temperature", None, b"abc")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("seed", None, b"1e30")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("response_format", None, b"xml")],
            ),
            (
                SpeechToTextTask::Translate,
                &[AUDIO, ("response_format", None, b"diarized_json")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("timestamp_granularities[]", None, b"char")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("vllm_xargs", None, b"{")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("vllm_xargs", None, b"[1]")],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("vllm_xargs", None, br#"{"a": {}}"#)],
            ),
            (
                SpeechToTextTask::Transcribe,
                &[AUDIO, ("stream_continuous_usage_stats", None, b"true")],
            ),
        ];
        let mut errors = Vec::new();
        for (task, parts) in cases {
            let error = parse(*task, parts).await.unwrap_err().to_error_response().error;
            errors.push(format!(
                "{:?} {:?}: {}",
                error.param, error.error_type, error.message
            ));
        }
        expect![[r#"
            [
                "Some(\"body.file\") \"Bad Request\": Field required",
                "Some(\"file\") \"BadRequestError\": Expected 'file' to be a file-like object, not 'str'.",
                "Some(\"body.stream\") \"Bad Request\": Input should be a valid boolean, unable to interpret input",
                "Some(\"body.temperature\") \"Bad Request\": Input should be a valid number, unable to parse string as a number",
                "Some(\"body.seed\") \"Bad Request\": Input should be a valid integer, unable to parse string as an integer",
                "Some(\"body.response_format\") \"Bad Request\": Input should be 'json', 'text', 'srt', 'verbose_json', 'vtt' or 'diarized_json'",
                "Some(\"body.response_format\") \"Bad Request\": Input should be 'json', 'text', 'srt', 'verbose_json' or 'vtt'",
                "Some(\"body.timestamp_granularities[]\") \"Bad Request\": Input should be 'word' or 'segment'",
                "Some(\"vllm_xargs\") \"BadRequestError\": Failed to parse vllm_xargs. Must be valid JSON: EOF while parsing an object at line 1 column 1",
                "Some(\"body.vllm_xargs\") \"Bad Request\": Input should be a valid dictionary",
                "Some(\"body.vllm_xargs.a\") \"Bad Request\": Input should be a string, a number, or a list of strings and numbers",
                "Some(\"stream_continuous_usage_stats\") \"BadRequestError\": Stream options can only be defined when `stream=True`.",
            ]
        "#]]
        .assert_debug_eq(&errors);
    }

    #[tokio::test]
    async fn rejects_oversized_audio_with_its_full_size() {
        let audio = vec![0; (1 << 20) + (1 << 19)];
        let error = parse_with_limit(
            SpeechToTextTask::Transcribe,
            &[("file", Some("audio.wav"), &audio)],
            1,
        )
        .await
        .unwrap_err()
        .to_error_response()
        .error;
        assert_eq!(error.param.as_deref(), Some("audio_filesize_mb"));
        assert_eq!(
            error.message,
            "Maximum file size exceeded. Set VLLM_MAX_AUDIO_CLIP_FILESIZE_MB to increase this \
             limit. (parameter=audio_filesize_mb, value=1.5)"
        );

        let form = parse_with_limit(
            SpeechToTextTask::Transcribe,
            &[("file", Some("audio.wav"), &audio[..1 << 20])],
            1,
        )
        .await
        .unwrap();
        assert_eq!(form.audio.len(), 1 << 20);
    }
}
