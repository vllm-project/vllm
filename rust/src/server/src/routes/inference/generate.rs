mod convert;
mod types;
mod validate;

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use openai_protocol::validated::ValidatedJson;
use thiserror_ext::AsReport as _;
use tracing::info;
use vllm_engine_core_client::protocol::{Logprobs, PositionLogprobs};
use vllm_llm::{CollectedGenerateOutput, GenerateOutputStreamExt as _};

use self::convert::prepare_generate_request;
use self::types::{GenerateLogprob, GenerateRequest, GenerateResponse, GenerateResponseChoice};
use crate::error::{ApiError, server_error};
use crate::routes::openai::utils::logprobs::clamp_logprob;
use crate::routes::openai::utils::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb};
use crate::state::AppState;

/// Validate one token-in/token-out request and proxy it into the shared `vllm-text` stack.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<GenerateRequest>,
) -> Response {
    let stream = body.stream;
    let prepared = match prepare_generate_request(body, &state.model_id) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    info!(request_id = %prepared.request_id, stream, "raw generate");
    let include_logprobs = prepared.include_logprobs;
    let include_prompt_logprobs = prepared.include_prompt_logprobs;

    let raw_stream = match state.chat.text().generate_raw(prepared.text_request).await {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit raw generate request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    let response = match collect_generate(
        raw_stream.collect_output().await,
        prepared.request_id,
        include_logprobs,
        include_prompt_logprobs,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

fn collect_generate(
    collected: vllm_llm::Result<CollectedGenerateOutput>,
    request_id: String,
    include_logprobs: bool,
    include_prompt_logprobs: bool,
) -> Result<GenerateResponse, ApiError> {
    let collected = collected.map_err(|error| {
        server_error!(
            "failed to collect raw generate response: {}",
            error.to_report_string()
        )
    })?;

    let logprobs = if include_logprobs {
        let logprobs = collected.logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested logprobs but generation returned none".to_string(),
            )
        })?;
        Some(raw_logprobs_to_openai_chat(logprobs)?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested prompt_logprobs but generation returned none"
                    .to_string(),
            )
        })?;
        Some(raw_prompt_logprobs_to_maps(prompt_logprobs))
    } else {
        None
    };

    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(collected.finish_reason.as_str().to_string()),
            token_ids: collected.token_ids,
        }],
        prompt_logprobs,
        kv_transfer_params: collected.kv_transfer_params,
    })
}

fn raw_logprobs_to_openai_chat(logprobs: &Logprobs) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(position_to_chat_logprobs_content)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

fn raw_prompt_logprobs_to_maps(
    prompt_logprobs: &Logprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    std::iter::once(None)
        .chain(
            prompt_logprobs
                .positions
                .iter()
                .map(|position| Some(position_to_logprob_map(position))),
        )
        .collect()
}

fn position_to_chat_logprobs_content(
    position: &PositionLogprobs,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        ApiError::server_error(
            "raw generate logprobs position unexpectedly had no token candidates".to_string(),
        )
    })?;
    let token = format_token_id(chosen.token_id);

    Ok(ChatLogProbsContent {
        token: token.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token.as_bytes().to_vec()),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token_id(entry.token_id);
                TopLogProb {
                    token: token.clone(),
                    logprob: clamp_logprob(entry.logprob),
                    bytes: Some(token.into_bytes()),
                }
            })
            .collect(),
    })
}

fn position_to_logprob_map(position: &PositionLogprobs) -> HashMap<u32, GenerateLogprob> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                entry.token_id,
                GenerateLogprob {
                    logprob: clamp_logprob(entry.logprob),
                    rank: Some(entry.rank),
                    decoded_token: Some(format_token_id(entry.token_id)),
                },
            )
        })
        .collect()
}

fn format_token_id(token_id: u32) -> String {
    format!("token_id:{token_id}")
}
