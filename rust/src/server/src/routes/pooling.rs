// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Pooling HTTP protocols above the shared text encode facade.

mod types;

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use vllm_llm::{EncodeOutput, EngineTask, PoolingParams, PoolingTask};
use vllm_text::{PromptTruncation, TextEncodeRequest, TruncationSide};

use self::types::{EncodingFormat, PoolingRequest};
use crate::error::{
    ApiError, bail_invalid_request, invalid_request, server_error, text_submit_error,
};
use crate::routes::openai::utils::{types::Usage, validated_json::ValidatedJson};
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

#[derive(Clone, Copy)]
enum Endpoint {
    Pooling,
    Classify,
}

pub async fn pooling(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<PoolingRequest>,
) -> Response {
    handle_request(&state, &headers, body, Endpoint::Pooling).await
}

pub async fn classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<PoolingRequest>,
) -> Response {
    handle_request(&state, &headers, body, Endpoint::Classify).await
}

async fn handle_request(
    state: &AppState,
    headers: &HeaderMap,
    body: PoolingRequest,
    endpoint: Endpoint,
) -> Response {
    let prepared = match prepare_request(state, headers, body, endpoint).await {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    match run_pooling(state.chat.text(), prepared).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

struct PreparedRequest {
    response_id: String,
    response_model: String,
    encoding_format: EncodingFormat,
    endpoint: Endpoint,
    requests: Vec<TextEncodeRequest>,
}

async fn prepare_request(
    state: &AppState,
    headers: &HeaderMap,
    body: PoolingRequest,
    endpoint: Endpoint,
) -> Result<PreparedRequest, ApiError> {
    body.validate_options()?;
    let model = body.model.as_deref().filter(|model| !model.is_empty());
    let resolution = state.resolve_model_with_loras(model).await;
    if let Some(model) = model
        && !resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.to_owned()));
    }
    let fixed_task = match endpoint {
        Endpoint::Classify => Some(PoolingTask::Classify),
        Endpoint::Pooling => None,
    };
    if fixed_task.is_some() && body.task.is_some() && fixed_task != body.task {
        bail_invalid_request!(param = "task", "task does not match this endpoint");
    }
    if matches!(endpoint, Endpoint::Classify)
        && (body.dimensions.is_some() || body.encoding_format != EncodingFormat::Float)
    {
        bail_invalid_request!("classify does not accept dimensions or encoded outputs");
    }
    let add_special_tokens = body.add_special_tokens.unwrap_or(true);
    let ctx = resolve_request_context(headers, body.request_id.as_deref());
    let prompts = body.input.into_prompts();
    if prompts.is_empty() {
        bail_invalid_request!(param = "input", "input must contain at least one prompt");
    }
    let truncation = body
        .truncate_prompt_tokens
        .map(|limit| {
            PromptTruncation::from_wire(
                limit,
                body.truncation_side.unwrap_or(TruncationSide::Right),
            )
        })
        .transpose()
        .map_err(|error| text_submit_error("invalid truncation", error))?;
    let text = state.chat.text();
    let tasks = text
        .supported_tasks()
        .await
        .map_err(|error| text_submit_error("failed to discover pooling tasks", error))?;
    let task = fixed_task
        .or(body.task)
        .or_else(|| {
            [
                PoolingTask::Embed,
                PoolingTask::Classify,
                PoolingTask::TokenEmbed,
                PoolingTask::TokenClassify,
            ]
            .into_iter()
            .find(|task| tasks.contains(&EngineTask::Pooling(*task)))
        })
        .ok_or_else(|| {
            invalid_request!(
                param = "task",
                "model does not support a standard pooling task"
            )
        })?;
    if !tasks.contains(&EngineTask::Pooling(task)) {
        bail_invalid_request!(param = "task", "model does not support {task:?}");
    }
    // TODO: integrate IOProcessor plugins and composite pooling outputs.
    if matches!(
        task,
        PoolingTask::Plugin | PoolingTask::EmbedAndTokenClassify
    ) {
        bail_invalid_request!(
            param = "task",
            "plugin and composite pooling outputs are not supported yet"
        );
    }
    let prefix = match endpoint {
        Endpoint::Pooling => "pool",
        Endpoint::Classify => "classify",
    };
    let id = format!("{prefix}-{}", ctx.request_id);
    let model = resolution
        .lora_request
        .as_ref()
        .map(|lora| lora.lora_name.clone())
        .unwrap_or_else(|| resolution.model_names.first().cloned().unwrap_or_default());
    let requests = prompts
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| TextEncodeRequest {
            prompt,
            add_special_tokens,
            prompt_truncation: truncation,
            request_id: format!("{id}-{index}"),
            task,
            pooling_params: PoolingParams {
                dimensions: body.dimensions,
                use_activation: body.use_activation,
                ..Default::default()
            },
            arrival_time: Some(vllm_llm::current_unix_timestamp_secs()),
            cache_salt: body.cache_salt.clone(),
            trace_headers: None,
            priority: ctx.priority.unwrap_or(body.priority),
            data_parallel_rank: ctx.data_parallel_rank,
            session_id: ctx.session_id.clone(),
            lora_request: resolution.lora_request.clone(),
        })
        .collect();
    Ok(PreparedRequest {
        response_id: id,
        response_model: model,
        encoding_format: body.encoding_format,
        endpoint,
        requests,
    })
}

async fn run_pooling(
    text: &vllm_text::TextLlm,
    prepared: PreparedRequest,
) -> Result<Value, ApiError> {
    let encoding = prepared.encoding_format;
    let outputs = text
        .encode_batch(prepared.requests)
        .await
        .map_err(|error| text_submit_error("failed to encode pooling batch", error))?;
    let prompt_tokens = outputs.iter().map(|output| output.prompt_token_ids.len()).sum();
    let data = outputs.iter().enumerate().map(|(index, output)| {
        let values = &output.output.data;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(server_error!("pooling output contains non-finite values"));
        }
        match prepared.endpoint {
            Endpoint::Pooling => {
                let data = tensor_json(output)?;
                Ok(json!({"index": index, "object": "pooling", "data": if encoding == EncodingFormat::Base64 { encoded_values(values, encoding) } else { data }}))
            }
            Endpoint::Classify => {
                vector(output)?;
                let predicted = values.iter().enumerate().reduce(|best, item| if item.1 > best.1 { item } else { best })
                    .ok_or_else(|| server_error!("classifier returned no classes"))?.0;
                Ok(json!({"index": index, "label": text.classification_label(predicted), "probs": values, "num_classes": values.len()}))
            }
        }
    }).collect::<Result<Vec<_>, ApiError>>()?;
    Ok(
        json!({"id": prepared.response_id, "object": "list", "created": unix_timestamp(), "model": prepared.response_model,
        "data": data, "usage": Usage::from_counts(prompt_tokens, 0, None, 0)}),
    )
}

fn vector(output: &EncodeOutput) -> Result<&[f32], ApiError> {
    if output.output.shape != [output.output.data.len()] {
        return Err(server_error!(
            "expected a pooling vector, got shape {:?}",
            output.output.shape
        ));
    }
    if !output.output.data.iter().all(|value| value.is_finite()) {
        return Err(server_error!("pooling output contains non-finite values"));
    }
    Ok(&output.output.data)
}

fn encoded_values(values: &[f32], encoding: EncodingFormat) -> Value {
    match encoding {
        EncodingFormat::Float => json!(values),
        EncodingFormat::Base64 => json!(
            STANDARD
                .encode(values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>())
        ),
    }
}

fn tensor_json(output: &EncodeOutput) -> Result<Value, ApiError> {
    let tensor = &output.output;
    match tensor.shape.as_slice() {
        [n] if *n == tensor.data.len() => Ok(json!(tensor.data)),
        [rows, columns] if rows.checked_mul(*columns) == Some(tensor.data.len()) => {
            Ok(Value::Array(
                (0..*rows)
                    .map(|row| json!(&tensor.data[row * columns..(row + 1) * columns]))
                    .collect(),
            ))
        }
        _ => Err(server_error!(
            "unsupported pooling tensor shape {:?}",
            tensor.shape
        )),
    }
}
