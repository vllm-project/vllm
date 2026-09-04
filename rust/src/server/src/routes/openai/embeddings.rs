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
use vllm_text::{EmbeddingParams, PromptTruncation, TruncationSide};

use self::types::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData, EncodingFormat,
    Endianness,
};
use super::utils::types::Usage;
use super::utils::validated_json::ValidatedJson;
use crate::error::{ApiError, bail_invalid_request, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, unix_timestamp};

const MAX_CONCURRENT_INPUTS: usize = 32;

struct PreparedRequest {
    response_id: String,
    response_model: String,
    encoding_format: EncodingFormat,
    endianness: Endianness,
    requests: Vec<vllm_text::EmbeddingRequest>,
}

/// Serve one OpenAI-compatible embeddings request through the shared text
/// facade.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<EmbeddingRequest>,
) -> Response {
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    let ctx = resolve_request_context(&headers, body.request_id.as_deref());
    let prepared = match prepare_request(body, &lora_resolution, ctx) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    match run_embeddings(state.chat.text(), prepared).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

fn prepare_request(
    request: EmbeddingRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    validate_request(&request, &lora_resolution.model_names)?;

    let prompt_truncation = request
        .truncate_prompt_tokens
        .map(|limit| {
            PromptTruncation::from_wire(
                limit,
                request.truncation_side.unwrap_or(TruncationSide::Right),
            )
        })
        .transpose()
        .map_err(|error| text_submit_error("invalid prompt truncation", error))?;
    let prompts = request.input.into_prompts();
    if prompts.is_empty() {
        bail_invalid_request!(param = "input", "input must not be empty.");
    }

    let response_id = format!("embd-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());
    let params = EmbeddingParams {
        dimensions: request.dimensions,
        use_activation: request.use_activation,
    };
    let priority = ctx.priority.or(request.priority).unwrap_or(0);
    let requests = prompts
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| vllm_text::EmbeddingRequest {
            request_id: format!("{response_id}-{index}"),
            prompt,
            params: params.clone(),
            prompt_truncation,
            add_special_tokens: request.add_special_tokens,
            priority,
            cache_salt: request.cache_salt.clone(),
            trace_headers: None,
            data_parallel_rank: ctx.data_parallel_rank,
            session_id: ctx.session_id.clone(),
            lora_request: lora_resolution.lora_request.clone(),
            arrival_time: None,
        })
        .collect();

    Ok(PreparedRequest {
        response_id,
        response_model,
        encoding_format: request.encoding_format,
        endianness: request.endianness,
        requests,
    })
}

fn validate_request(
    request: &EmbeddingRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref().filter(|model| !model.is_empty())
        && !served_model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }
    if request.embed_dtype.as_deref().is_some_and(|value| value != "float32") {
        bail_invalid_request!(
            param = "embed_dtype",
            "The Rust frontend currently supports only embed_dtype=\"float32\"."
        );
    }
    Ok(())
}

async fn run_embeddings(
    text: &vllm_text::TextLlm,
    prepared: PreparedRequest,
) -> Result<EmbeddingResponse, ApiError> {
    let encoding_format = prepared.encoding_format;
    let endianness = prepared.endianness;
    let results = stream::iter(prepared.requests.into_iter().enumerate())
        .map(|(index, request)| async move {
            let output = text
                .embed(request)
                .await
                .map_err(|error| text_submit_error("failed to submit embedding request", error))?;
            Ok::<_, ApiError>((
                index,
                output.prompt_token_ids.len(),
                encode_embedding(output.embedding, encoding_format, endianness),
            ))
        })
        .buffer_unordered(MAX_CONCURRENT_INPUTS)
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
        id: prepared.response_id,
        object: "list",
        created: unix_timestamp(),
        model: prepared.response_model,
        data,
        usage: Usage::from_counts(prompt_tokens, 0, None),
    })
}

fn encode_embedding(
    embedding: Vec<f32>,
    format: EncodingFormat,
    endianness: Endianness,
) -> EmbeddingData {
    match format {
        EncodingFormat::Float => EmbeddingData::Float(embedding),
        EncodingFormat::Base64 => {
            let native_bytes = || bytemuck::cast_slice::<f32, u8>(&embedding);
            let bytes: &[u8] = match endianness {
                Endianness::Native => native_bytes(),
                Endianness::Little if cfg!(target_endian = "little") => native_bytes(),
                Endianness::Big if cfg!(target_endian = "big") => native_bytes(),
                Endianness::Little => {
                    &embedding.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()
                }
                Endianness::Big => {
                    &embedding.iter().flat_map(|value| value.to_be_bytes()).collect::<Vec<_>>()
                }
            };
            EmbeddingData::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;

    #[test]
    fn base64_embedding_respects_endianness() {
        let values = [1.0_f32, -2.5];
        let cases = [
            (
                Endianness::Native,
                values.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>(),
            ),
            (
                Endianness::Little,
                values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
            ),
            (
                Endianness::Big,
                values.into_iter().flat_map(f32::to_be_bytes).collect::<Vec<_>>(),
            ),
        ];

        for (endianness, expected) in cases {
            let EmbeddingData::Base64(encoded) =
                encode_embedding(values.to_vec(), EncodingFormat::Base64, endianness)
            else {
                panic!("expected base64 embedding")
            };
            let decoded = base64::engine::general_purpose::STANDARD.decode(encoded).unwrap();
            assert_eq!(decoded, expected);
        }
    }
}
