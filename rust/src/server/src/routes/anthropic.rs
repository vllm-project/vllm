// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod convert;
mod response;
mod stream;
mod types;
mod validate;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use tracing_futures::Instrument as _;
use vllm_chat::{AssistantContentBlock, AssistantMessageExt as _, ChatTokenUsage};

use self::convert::{prepare_count_tokens_request, prepare_messages_request};
use self::response::{AnthropicError, convert_output_blocks, convert_usage, stop_reason};
use self::types::{
    ContextManagement, CountTokensRequest, CountTokensResponse, MessagesRequest, MessagesResponse,
    MessagesStopReason,
};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, text_submit_error};
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::routes::output_log::LoggedOutput;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response};

pub(super) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<CountTokensRequest>, JsonRejection>,
) -> Result<impl IntoResponse, AnthropicError> {
    let Json(body) = body.map_err(ApiError::from)?;
    let model_resolution = state.resolve_model_with_loras(Some(&body.input.model)).await;
    let ctx = resolve_request_context(&headers, None)?;
    let request = prepare_count_tokens_request(body, &model_resolution, ctx)?;
    let text_request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(request)
        .await
        .map_err(|error| chat_submit_error("count_tokens failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text_request)
        .map_err(|error| text_submit_error("count_tokens failed", error))?;
    Ok(Json(CountTokensResponse {
        input_tokens: tokens.len(),
        context_management: ContextManagement {
            original_input_tokens: tokens.len(),
        },
    }))
}

/// Anthropic request handling over the application's shared chat inference.
pub(super) async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, AnthropicError> {
    let Json(body) = body.map_err(ApiError::from)?;
    let model_resolution = state.resolve_model_with_loras(Some(&body.input.model)).await;
    let options = state.api_server_options;
    let ctx = resolve_request_context(&headers, None)?;
    let request = prepare_messages_request(body, &model_resolution, ctx)?;
    validate_max_output_tokens(
        "max_tokens",
        request.sampling_params.max_tokens,
        state.chat.text().llm().info().max_model_len,
    )?;
    let stream_requested = request.intermediate;
    let request_id = request.request_id.clone();
    let response_model = model_resolution.response_model();
    let span = tracing::info_span!("messages", request_id = %request_id, engine_request_id = tracing::field::Empty);
    let event_stream = state
        .chat
        .chat(request)
        .instrument(span.clone())
        .await
        .map_err(|error| chat_submit_error("failed to submit Messages request", error))?;
    if stream_requested {
        return Ok(sse_response(
            stream::messages_stream(event_stream, request_id, response_model, options)
                .instrument(span),
            options.sse_keep_alive_interval,
        ));
    }
    let response = async {
        let completion = event_stream
            .collect_message()
            .await
            .map_err(|error| chat_submit_error("failed to collect Messages response", error))?;
        let (reason, sequence) = stop_reason(&completion.finish_reason, &completion.message)?;
        log_completion(
            options,
            &response_model,
            &completion.message,
            completion.usage,
            reason,
            LoggedOutput::Response,
            Some(&completion.token_ids),
        );
        let content = convert_output_blocks(completion.message.content)?;
        Ok::<_, ApiError>(MessagesResponse {
            id: request_id,
            role: "assistant",
            content,
            model: response_model,
            stop_reason: reason,
            stop_sequence: sequence.map(str::to_owned),
            usage: convert_usage(
                completion.usage.engine,
                options.enable_prompt_tokens_details,
            ),
            kv_transfer_params: completion.kv_transfer_params,
            ec_transfer_params: completion.ec_transfer_params,
        })
    }
    .instrument(span)
    .await?;
    Ok(Json(response).into_response())
}

fn log_completion(
    options: ApiServerOptions,
    model: &str,
    message: &[AssistantContentBlock],
    usage: ChatTokenUsage,
    reason: Option<MessagesStopReason>,
    kind: LoggedOutput,
    token_ids: Option<&[u32]>,
) {
    let reason = reason.as_ref().map(MessagesStopReason::as_str);
    if options.enable_log_requests {
        tracing::info!(
            %model,
            prompt_tokens = usage.prompt_token_count,
            output_tokens = usage.output_token_count,
            finish_reason = reason,
            "Messages finished"
        );
    }
    if options.log_outputs.logs_complete() {
        let text = message.text();
        if !text.is_empty() {
            options.log_outputs.log(kind, &text, token_ids, reason);
        }
    }
}
