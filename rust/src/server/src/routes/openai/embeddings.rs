// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use futures::{StreamExt as _, TryStreamExt as _, stream};
use thiserror_ext::AsReport as _;
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::request::{EngineCorePoolingParams, EngineCoreRequest};
use vllm_llm::current_unix_timestamp_secs;
use vllm_text::tokenizer::Tokenizer;

use self::types::{
    EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData,
    EncodingFormat, TruncationSide,
};
use super::utils::types::Usage;
use super::utils::validated_json::ValidatedJson;
use crate::error::{ApiError, bail_invalid_request, server_error};
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

struct PreparedInput {
    token_ids: Vec<u32>,
    engine_request: EngineCoreRequest,
}

/// Serve one OpenAI-compatible embeddings request through engine-core pooling.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<EmbeddingRequest>,
) -> Response {
    match embeddings_inner(&state, &headers, body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn embeddings_inner(
    state: &AppState,
    headers: &HeaderMap,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse, ApiError> {
    validate_encoding_options(&request)?;
    let requested_model = request.model.as_deref().unwrap_or(state.primary_model_name());
    let lora_resolution = state.resolve_model_with_loras(Some(requested_model)).await;
    if !lora_resolution.model_names.iter().any(|name| name == requested_model) {
        return Err(ApiError::model_not_found(requested_model.to_string()));
    }

    let ctx = resolve_request_context(headers, request.request_id.as_deref());
    let response_id = format!("embd-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| state.primary_model_name().to_string());
    let tokenizer = state.chat.text().tokenizer();
    let inputs = prepare_inputs(
        request.input,
        tokenizer.as_ref(),
        request.add_special_tokens,
    )?;
    if inputs.is_empty() {
        bail_invalid_request!(param = "input", "input must not be empty.");
    }

    let max_model_len = state.engine_core_client().max_model_len() as usize;
    let prompt_vocab_size = tokenizer.vocab_size().max(state.chat.text().model_vocab_size());
    let mut prepared = Vec::with_capacity(inputs.len());
    for (index, token_ids) in inputs.into_iter().enumerate() {
        let token_ids = validate_and_truncate(
            token_ids,
            prompt_vocab_size,
            max_model_len,
            request.truncate_prompt_tokens,
            request.truncation_side.unwrap_or_default(),
        )?;
        let external_request_id = format!("{response_id}-{index}");
        let random_suffix = Uuid::new_v4().simple().to_string();
        let engine_request_id = format!("{external_request_id}-{}", &random_suffix[..8]);
        prepared.push(PreparedInput {
            token_ids: token_ids.clone(),
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(token_ids),
                mm_features: None,
                sampling_params: None,
                pooling_params: Some(EngineCorePoolingParams::embeddings(
                    request.dimensions,
                    request.use_activation,
                )),
                arrival_time: current_unix_timestamp_secs(),
                lora_request: lora_resolution.lora_request.clone(),
                cache_salt: request.cache_salt.clone(),
                data_parallel_rank: ctx.data_parallel_rank,
                prompt_embeds: None,
                prompt_is_token_ids: None,
                client_index: 0,
                current_wave: 0,
                priority: ctx.priority.or(request.priority).unwrap_or(0),
                trace_headers: None,
                resumable: false,
                external_req_id: Some(external_request_id),
                reasoning_ended: None,
                reasoning_parser_kwargs: None,
                abort_immediately: false,
                session_id: ctx.session_id.clone(),
            },
        });
    }

    let client = state.engine_core_client();
    let encoding_format = request.encoding_format;
    let results = stream::iter(prepared.into_iter().enumerate())
        .map(|(index, prepared)| async move {
            let prompt_tokens = prepared.token_ids.len();
            let mut outputs = client.call(prepared.engine_request).await.map_err(|error| {
                server_error!(
                    "failed to submit embedding request: {}",
                    error.to_report_string()
                )
            })?;
            let mut embedding = None;
            while let Some(output) = outputs.next().await {
                let output = output.map_err(|error| {
                    server_error!("embedding stream failed: {}", error.to_report_string())
                })?;
                if output.finish_reason == Some(EngineCoreFinishReason::Error) {
                    return Err(server_error!("embedding request failed in engine-core"));
                }
                if let Some(tensor) = &output.pooling_output {
                    if tensor.shape.len() != 1 {
                        return Err(server_error!(
                            "embedding output must be one-dimensional, got {:?}",
                            tensor.shape
                        ));
                    }
                    embedding = Some(tensor.to_f32_vec().map_err(|message| {
                        server_error!("failed to decode embedding output: {message}")
                    })?);
                }
            }
            let embedding = embedding
                .ok_or_else(|| server_error!("embedding stream completed without output"))?;
            Ok::<_, ApiError>((
                index,
                prompt_tokens,
                encode_embedding(embedding, encoding_format),
            ))
        })
        .buffer_unordered(32)
        .try_collect::<Vec<_>>()
        .await?;

    let prompt_tokens = results.iter().map(|(_, count, _)| count).sum();
    let mut data = results
        .into_iter()
        .map(|(index, _, embedding)| EmbeddingResponseData {
            object: "embedding",
            index,
            embedding,
        })
        .collect::<Vec<_>>();
    data.sort_unstable_by_key(|item| item.index);

    Ok(EmbeddingResponse {
        id: response_id,
        object: "list",
        created: unix_timestamp(),
        model: response_model,
        data,
        usage: Usage::from_counts(prompt_tokens, 0, None),
    })
}

fn validate_encoding_options(request: &EmbeddingRequest) -> Result<(), ApiError> {
    // TODO: Validate `dimensions` against the model's Matryoshka configuration
    // once the Rust text backend exposes pooler metadata.
    if request.dimensions == Some(0) {
        bail_invalid_request!(
            param = "dimensions",
            "dimensions must be greater than zero."
        );
    }
    if request.embed_dtype.as_deref().is_some_and(|value| value != "float32") {
        bail_invalid_request!(
            param = "embed_dtype",
            "The Rust frontend currently supports only embed_dtype=\"float32\"."
        );
    }
    if request.endianness.as_deref().is_some_and(|value| value != "native") {
        bail_invalid_request!(
            param = "endianness",
            "The Rust frontend currently supports only endianness=\"native\"."
        );
    }
    Ok(())
}

fn prepare_inputs(
    input: EmbeddingInput,
    tokenizer: &dyn Tokenizer,
    add_special_tokens: bool,
) -> Result<Vec<Vec<u32>>, ApiError> {
    match input {
        EmbeddingInput::TokenIds(token_ids) => Ok(vec![token_ids]),
        EmbeddingInput::TokenIdBatch(batch) => Ok(batch),
        EmbeddingInput::Text(text) => Ok(vec![tokenize(tokenizer, &text, add_special_tokens)?]),
        EmbeddingInput::TextBatch(texts) => texts
            .into_iter()
            .map(|text| tokenize(tokenizer, &text, add_special_tokens))
            .collect(),
    }
}

fn tokenize(
    tokenizer: &dyn Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>, ApiError> {
    tokenizer
        .encode(text, add_special_tokens)
        .map_err(|error| server_error!("failed to tokenize embedding input: {error}"))
}

fn validate_and_truncate(
    mut token_ids: Vec<u32>,
    vocab_size: usize,
    max_model_len: usize,
    truncate_prompt_tokens: Option<i64>,
    truncation_side: TruncationSide,
) -> Result<Vec<u32>, ApiError> {
    if token_ids.is_empty() {
        bail_invalid_request!(param = "input", "input token IDs must not be empty.");
    }
    if let Some(token_id) = token_ids.iter().find(|&&id| id as usize >= vocab_size) {
        bail_invalid_request!(
            param = "input",
            "Token id {token_id} is out of vocabulary; vocabulary size is {vocab_size}."
        );
    }
    let truncate_to = match truncate_prompt_tokens {
        None => None,
        Some(-1) => Some(max_model_len),
        Some(value) if value > 0 => Some(value as usize),
        Some(_) => {
            bail_invalid_request!(
                param = "truncate_prompt_tokens",
                "truncate_prompt_tokens must be -1 or a positive integer."
            );
        }
    };
    if let Some(limit) = truncate_to
        && token_ids.len() > limit
    {
        match truncation_side {
            TruncationSide::Right => token_ids.truncate(limit),
            TruncationSide::Left => {
                let start = token_ids.len() - limit;
                token_ids.drain(..start).for_each(drop);
            }
        }
    }
    if token_ids.len() > max_model_len {
        bail_invalid_request!(
            param = "input",
            "This model's maximum context length is {max_model_len} tokens, but the embedding input has {} tokens.",
            token_ids.len()
        );
    }
    Ok(token_ids)
}

fn encode_embedding(embedding: Vec<f32>, format: EncodingFormat) -> EmbeddingData {
    match format {
        EncodingFormat::Float => EmbeddingData::Float(embedding),
        EncodingFormat::Base64 => {
            let bytes = embedding.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>();
            EmbeddingData::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;

    #[test]
    fn truncation_preserves_the_requested_side() {
        assert_eq!(
            validate_and_truncate(vec![1, 2, 3, 4], 10, 10, Some(2), TruncationSide::Right)
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            validate_and_truncate(vec![1, 2, 3, 4], 10, 10, Some(2), TruncationSide::Left).unwrap(),
            vec![3, 4]
        );
    }

    #[test]
    fn base64_embedding_uses_native_float32_bytes() {
        let EmbeddingData::Base64(encoded) =
            encode_embedding(vec![1.0, -2.5], EncodingFormat::Base64)
        else {
            panic!("expected base64 embedding")
        };

        let decoded = base64::engine::general_purpose::STANDARD.decode(encoded).unwrap();
        let expected = [1.0_f32, -2.5].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }
}
