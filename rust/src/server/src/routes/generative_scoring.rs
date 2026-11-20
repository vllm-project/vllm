// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Generative scoring: the probability of given label tokens appearing as the
//! next token after a query+item prompt, for any generative model.
//!
//! Each item becomes one ordinary text request with `max_tokens=1` that asks
//! for the label tokens' logprobs, so sampling-parameter validation
//! (`max_logprobs`, vocabulary range, cache salt) stays in `vllm-text`.

mod types;

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures::future;
use thiserror_ext::AsReport as _;
use tracing_futures::Instrument as _;
use vllm_text::{
    CollectedTextOutput, FinishReason, Prompt, PromptTruncation, PromptTruncationLimit,
    SamplingParams, TextDecodeOptions, TextOutputStreamExt as _, TextRequest, TruncationSide,
};

use self::types::{
    GenerativeScoringItemResult, GenerativeScoringRequest, GenerativeScoringResponse,
};
use crate::error::{ApiError, bail_invalid_request, engine_error, server_error, text_submit_error};
use crate::routes::openai::utils::types::Usage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Score every item of one request, each as its own engine request.
pub async fn generative_scoring(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(request): ValidatedJson<GenerativeScoringRequest>,
) -> Result<Response, ApiError> {
    let ctx = resolve_request_context(&headers, request.request_id.as_deref())?;
    let lora_resolution = state.resolve_model_with_loras(request.model.as_deref()).await;
    lora_resolution.model_names.check(request.model.as_deref())?;

    if request.label_token_ids.is_empty() {
        bail_invalid_request!(
            param = "label_token_ids",
            "label_token_ids must contain at least one token ID."
        );
    }
    if request.items.is_empty() {
        bail_invalid_request!(param = "items", "items must contain at least one item.");
    }

    let request_id = format!("generative-scoring-{}", ctx.request_id);
    let request_span = tracing::info_span!("generative_scoring", request_id = %request_id);
    let text = state.chat.text();
    let GenerativeScoringRequest {
        model: _,
        query,
        items,
        label_token_ids,
        apply_softmax,
        item_first,
        add_special_tokens,
        cache_salt,
        priority,
        request_id: _,
    } = request;

    let tokenizer = text.tokenizer();
    let tokenize = |prompt: Prompt, add_special_tokens: bool| match prompt {
        Prompt::Text(prompt) => tokenizer
            .encode(&prompt, add_special_tokens)
            .map_err(|error| text_submit_error("failed to tokenize input", error.into())),
        Prompt::TokenIds(token_ids) => Ok(token_ids),
    };
    let query_token_ids = tokenize(query, add_special_tokens)?;
    let sampling_params = label_read_sampling_params(&label_token_ids);

    // Prepare every item before submitting any, so a request that fails
    // validation leaves nothing in flight.
    let prepared = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            // Don't add special tokens for items to avoid duplicate BOS/EOS.
            let item_token_ids = tokenize(item, false)?;
            let prompt_token_ids = if item_first {
                [&item_token_ids[..], &query_token_ids].concat()
            } else {
                [&query_token_ids[..], &item_token_ids].concat()
            };
            let text_request = TextRequest {
                request_id: format!("{request_id}-{index}"),
                prompt: Prompt::TokenIds(prompt_token_ids),
                mm_features: None,
                sampling_params: sampling_params.clone(),
                // The sampled token is never read, so it is not decoded either,
                // and token ID inputs need no tokenizer.
                decode_options: TextDecodeOptions {
                    detokenize: false,
                    ..TextDecodeOptions::default()
                },
                intermediate: false,
                priority: ctx.priority.unwrap_or(priority),
                cache_salt: cache_salt.clone(),
                // Keep the first `max_model_len - 1` tokens, leaving room for
                // the one output token.
                prompt_truncation: Some(PromptTruncation {
                    limit: PromptTruncationLimit::InputBudget,
                    side: TruncationSide::Right,
                }),
                add_special_tokens: false,
                data_parallel_rank: ctx.data_parallel_rank,
                session_id: ctx.session_id.clone(),
                kv_hints: None,
                reasoning_parser_kwargs: Default::default(),
                reasoning_ended: None,
                lora_request: lora_resolution.lora_request.clone(),
                arrival_time: None,
            };
            text.request_processor().prepare(text_request).map_err(|error| {
                text_submit_error("failed to prepare generative scoring request", error)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let created = unix_timestamp();
    let outputs = future::try_join_all(prepared.into_iter().map(|prepared| async {
        let mut streams =
            text.generate_prepared_n(prepared, NonZeroU32::MIN).await.map_err(|error| {
                text_submit_error("failed to submit generative scoring request", error)
            })?;
        let stream = streams.pop().expect("one output stream per sample");
        stream.collect_output().await.map_err(|error| {
            server_error!(
                "failed to collect generative scoring response: {}",
                error.to_report_string()
            )
        })
    }))
    .instrument(request_span)
    .await?;

    let mut data = Vec::with_capacity(outputs.len());
    let (mut prompt_tokens, mut completion_tokens) = (0, 0);
    for (index, output) in outputs.iter().enumerate() {
        // Like Python, an engine error fails the whole request.
        if let FinishReason::Error(error) = &output.finish_reason {
            return Err(engine_error(error.as_deref()));
        }
        let label_logprobs = label_logprobs(output, &label_token_ids, index)?;
        data.push(GenerativeScoringItemResult {
            index,
            object: "score",
            score: score(&label_token_ids, &label_logprobs, apply_softmax),
        });
        prompt_tokens += output.usage.prompt_token_count;
        completion_tokens += output.usage.output_token_count;
    }

    Ok(Json(GenerativeScoringResponse {
        id: request_id,
        object: "list",
        created,
        model: lora_resolution.response_model(),
        data,
        // Like Python, without prompt-token details.
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
    })
    .into_response())
}

/// One sampling step that returns the full-vocabulary logprobs of the label
/// tokens.
///
/// Like Python, every other parameter is vLLM's neutral default
/// ([`SamplingParams::neutral`]), not the model's generation config: under
/// `--logprobs-mode processed_logprobs` the logprobs are taken after
/// temperature and top-k/p, and the scored distribution must not depend on
/// the model's sampling defaults.
fn label_read_sampling_params(label_token_ids: &[u32]) -> SamplingParams {
    SamplingParams {
        watermarking: Some(false),
        max_tokens: Some(1),
        logprobs: Some(i32::try_from(label_token_ids.len()).unwrap_or(i32::MAX)),
        logprob_token_ids: Some(label_token_ids.to_vec()),
        ..SamplingParams::neutral()
    }
}

/// Read the logprob of each label token, in `label_token_ids` order, from the
/// item's single output position.
fn label_logprobs(
    output: &CollectedTextOutput,
    label_token_ids: &[u32],
    index: usize,
) -> Result<Vec<f64>, ApiError> {
    let Some(position) = output.logprobs.as_ref().and_then(|logprobs| logprobs.positions.first())
    else {
        return Err(server_error!("No logprobs available for item {index}."));
    };
    label_token_ids
        .iter()
        .map(|&token_id| {
            position
                .entries
                .iter()
                .find(|entry| entry.token_id == token_id)
                .map(|entry| f64::from(entry.logprob))
                .ok_or_else(|| {
                    server_error!("Token ID {token_id} not found in logprobs for item {index}.")
                })
        })
        .collect()
}

/// The probability of the first label token: normalized over the distinct
/// label tokens with `apply_softmax`, otherwise the model's probability over
/// the full vocabulary.
fn score(label_token_ids: &[u32], label_logprobs: &[f64], apply_softmax: bool) -> f64 {
    let first = label_logprobs[0];
    if !apply_softmax {
        return first.exp();
    }
    // A repeated label counts once, like Python's
    // `dict(zip(label_token_ids, logprobs))`.
    let mut seen = HashSet::new();
    let distinct: Vec<f64> = label_token_ids
        .iter()
        .zip(label_logprobs)
        .filter(|&(token_id, _)| seen.insert(token_id))
        .map(|(_, &logprob)| logprob)
        .collect();
    // Subtract the maximum for numerical stability.
    let max = distinct.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = distinct.iter().map(|logprob| (logprob - max).exp()).sum();
    (first - max).exp() / sum
}

#[cfg(test)]
mod tests {
    use super::score;

    #[test]
    fn score_is_first_label_probability() {
        let logprobs = [0.25_f64.ln(), 0.5_f64.ln(), 0.05_f64.ln()];
        // Full vocabulary: the model's own probability.
        assert!((score(&[1, 2, 3], &logprobs, false) - 0.25).abs() < 1e-12);
        // Over the labels only: 0.25 / (0.25 + 0.5 + 0.05).
        assert!((score(&[1, 2, 3], &logprobs, true) - 0.3125).abs() < 1e-12);
        // A repeated label counts once: 0.25 / (0.25 + 0.5).
        let repeated = [0.25_f64.ln(), 0.5_f64.ln(), 0.25_f64.ln()];
        assert!((score(&[1, 2, 1], &repeated, true) - 0.25 / 0.75).abs() < 1e-12);
    }
}
