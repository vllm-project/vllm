use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use tracing::info;
use tracing_futures::Instrument as _;
use validator::Validate;
use vllm_engine_core_client::protocol::logprobs::PositionLogprobs;
use vllm_llm::{CollectedGenerateOutput, FinishReason, GenerateOutputStreamExt as _, TokenUsage};
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use crate::error::{ApiError, bail_invalid_request, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::{Normalizable, Usage, default_true};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, unix_timestamp};

/// Request body for the vLLM generative scoring API.
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct GenerativeScoringRequest {
    pub model: Option<String>,
    pub query: GenerativeScoringInput,
    pub items: GenerativeScoringItems,
    pub label_token_ids: Vec<u32>,
    #[serde(default = "default_true")]
    pub apply_softmax: bool,
    #[serde(default)]
    pub item_first: bool,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    #[serde(default)]
    pub priority: i32,
    pub request_id: Option<String>,
}

impl Normalizable for GenerativeScoringRequest {}

/// Text or pre-tokenized IDs accepted by the generative scoring API.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum GenerativeScoringInput {
    Text(String),
    TokenIds(Vec<u32>),
}

/// Homogeneous text or token-ID item lists accepted by the API.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum GenerativeScoringItems {
    Text(Vec<String>),
    TokenIds(Vec<Vec<u32>>),
}

#[derive(Debug, Clone, Serialize)]
struct GenerativeScoringItemResult {
    index: usize,
    object: &'static str,
    score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GenerativeScoringResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    data: Vec<GenerativeScoringItemResult>,
    usage: Usage,
}

#[derive(Debug, Clone)]
struct PreparedGenerativeScoringRequest {
    request_id: String,
    response_model: String,
    label_token_ids: Vec<u32>,
    apply_softmax: bool,
    text_requests: Vec<TextRequest>,
}

/// Validate, build one prompt per item, score the configured label tokens, and
/// return probabilities in the Python `/generative_scoring` response shape.
pub async fn generative_scoring(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<GenerativeScoringRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;
    let prepared =
        match prepare_generative_scoring_request(body, &lora_resolution, request_context, &state) {
            Ok(prepared) => prepared,
            Err(error) => return error.into_response(),
        };

    let request_span = tracing::info_span!(
        "generative_scoring",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );
    let created = unix_timestamp();
    let enable_log_requests = state.api_server_options.enable_log_requests;

    let text = state.chat.text();
    let streams = match try_join_all(
        prepared
            .text_requests
            .into_iter()
            .map(|text_request| text.generate_raw(text_request).instrument(request_span.clone())),
    )
    .await
    {
        Ok(streams) => streams,
        Err(error) => {
            return text_submit_error("failed to submit generative scoring request", error)
                .into_response();
        }
    };

    let collected = match try_join_all(
        streams
            .into_iter()
            .map(|stream| stream.collect_output().instrument(request_span.clone())),
    )
    .await
    {
        Ok(outputs) => outputs,
        Err(error) => {
            return server_error!(
                "generative scoring stream failed: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    let response = match collect_generative_scoring(
        prepared.request_id,
        prepared.response_model,
        created,
        prepared.label_token_ids,
        prepared.apply_softmax,
        collected,
        enable_log_requests,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

fn prepare_generative_scoring_request(
    request: GenerativeScoringRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    state: &AppState,
) -> Result<PreparedGenerativeScoringRequest, ApiError> {
    validate_generative_scoring_request(&request, lora_resolution, state.chat.model_vocab_size())?;

    let request_id = format!("generative-scoring-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());
    let label_token_ids = request.label_token_ids;
    let apply_softmax = request.apply_softmax;
    let prompt_token_ids = build_prompt_token_ids(
        request.query,
        request.items,
        request.item_first,
        request.add_special_tokens,
        state,
    )?;

    let logprobs = i32::try_from(label_token_ids.len()).map_err(|_| {
        ApiError::invalid_request(
            "label_token_ids must fit within a signed 32-bit count.".to_string(),
            Some("label_token_ids"),
        )
    })?;

    let text_requests = prompt_token_ids
        .into_iter()
        .enumerate()
        .map(|(index, prompt_token_ids)| TextRequest {
            request_id: format!("{request_id}-{index}"),
            prompt: Prompt::TokenIds(prompt_token_ids),
            mm_features: None,
            sampling_params: SamplingParams {
                max_tokens: Some(1),
                logprobs: Some(logprobs),
                logprob_token_ids: Some(label_token_ids.clone()),
                ..Default::default()
            },
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: request.priority,
            cache_salt: None,
            add_special_tokens: false,
            data_parallel_rank: ctx.data_parallel_rank,
            lora_request: lora_resolution.lora_request.clone(),
        })
        .collect();

    Ok(PreparedGenerativeScoringRequest {
        request_id,
        response_model,
        label_token_ids,
        apply_softmax,
        text_requests,
    })
}

fn validate_generative_scoring_request(
    request: &GenerativeScoringRequest,
    lora_resolution: &LoraModelResolution,
    model_vocab_size: usize,
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref()
        && !lora_resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    if request.label_token_ids.is_empty() {
        bail_invalid_request!(
            param = "label_token_ids",
            "label_token_ids must contain at least one token ID."
        );
    }

    if request.items.is_empty() {
        bail_invalid_request!(param = "items", "items must contain at least one item.");
    }

    let invalid_token_ids: Vec<_> = request
        .label_token_ids
        .iter()
        .copied()
        .filter(|&token_id| token_id as usize >= model_vocab_size)
        .collect();
    if !invalid_token_ids.is_empty() {
        bail_invalid_request!(
            param = "label_token_ids",
            "label_token_id(s) {:?} are out of vocabulary range [0, {}).",
            invalid_token_ids,
            model_vocab_size
        );
    }

    Ok(())
}

impl GenerativeScoringItems {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(items) => items.is_empty(),
            Self::TokenIds(items) => items.is_empty(),
        }
    }
}

fn build_prompt_token_ids(
    query: GenerativeScoringInput,
    items: GenerativeScoringItems,
    item_first: bool,
    add_special_tokens: bool,
    state: &AppState,
) -> Result<Vec<Vec<u32>>, ApiError> {
    let tokenizer = state.chat.text().tokenizer();
    let query_token_ids = match query {
        GenerativeScoringInput::Text(query) => {
            tokenizer.encode(&query, add_special_tokens).map_err(|error| {
                server_error!(
                    "failed to tokenize generative scoring query: {}",
                    error.to_report_string()
                )
            })?
        }
        GenerativeScoringInput::TokenIds(token_ids) => token_ids,
    };
    let max_prompt_len = state.chat.engine_core_client().max_model_len().saturating_sub(1) as usize;

    match items {
        GenerativeScoringItems::Text(items) => items
            .into_iter()
            .map(|item| {
                let item_token_ids = tokenizer.encode(&item, false).map_err(|error| {
                    server_error!(
                        "failed to tokenize generative scoring item: {}",
                        error.to_report_string()
                    )
                })?;
                Ok(concat_and_truncate_prompt(
                    &query_token_ids,
                    &item_token_ids,
                    item_first,
                    max_prompt_len,
                ))
            })
            .collect(),
        GenerativeScoringItems::TokenIds(items) => Ok(items
            .into_iter()
            .map(|item_token_ids| {
                concat_and_truncate_prompt(
                    &query_token_ids,
                    &item_token_ids,
                    item_first,
                    max_prompt_len,
                )
            })
            .collect()),
    }
}

fn concat_and_truncate_prompt(
    query_token_ids: &[u32],
    item_token_ids: &[u32],
    item_first: bool,
    max_prompt_len: usize,
) -> Vec<u32> {
    let mut prompt = Vec::with_capacity(query_token_ids.len() + item_token_ids.len());
    if item_first {
        prompt.extend_from_slice(item_token_ids);
        prompt.extend_from_slice(query_token_ids);
    } else {
        prompt.extend_from_slice(query_token_ids);
        prompt.extend_from_slice(item_token_ids);
    }
    prompt.truncate(max_prompt_len);
    prompt
}

fn collect_generative_scoring(
    request_id: String,
    response_model: String,
    created: u64,
    label_token_ids: Vec<u32>,
    apply_softmax: bool,
    outputs: Vec<CollectedGenerateOutput>,
    enable_log_requests: bool,
) -> Result<GenerativeScoringResponse, ApiError> {
    let mut data = Vec::with_capacity(outputs.len());
    let mut usage = TokenUsage::default();

    for (index, output) in outputs.into_iter().enumerate() {
        if matches!(output.finish_reason, FinishReason::Error) {
            return Err(server_error!(
                "generation error for generative scoring item {index}"
            ));
        }

        let logprobs = output.logprobs.as_ref().ok_or_else(|| {
            server_error!(
                "no logprobs available for generative scoring item {index}; \
                 this might indicate an issue with logprobs configuration"
            )
        })?;
        let position = logprobs.positions.first().ok_or_else(|| {
            server_error!(
                "no logprobs available for generative scoring item {index}; \
                 this might indicate an issue with logprobs configuration"
            )
        })?;
        let score = compute_score(position, &label_token_ids, apply_softmax, index)?;

        data.push(GenerativeScoringItemResult {
            index,
            object: "score",
            score,
        });
        usage.prompt_token_count += output.usage.prompt_token_count;
        usage.output_token_count += output.usage.output_token_count;
        usage.cached_token_count += output.usage.cached_token_count;
    }

    if enable_log_requests {
        info!(
            prompt_tokens = usage.prompt_token_count,
            output_tokens = usage.output_token_count,
            "generative scoring finished"
        );
    }

    Ok(GenerativeScoringResponse {
        id: request_id,
        object: "list",
        created,
        model: response_model,
        data,
        usage: Usage::from_counts(usage.prompt_token_count, usage.output_token_count, None),
    })
}

fn compute_score(
    position: &PositionLogprobs,
    label_token_ids: &[u32],
    apply_softmax: bool,
    item_index: usize,
) -> Result<f64, ApiError> {
    let mut returned_logprobs = HashMap::new();
    for entry in &position.entries {
        returned_logprobs.insert(entry.token_id, f64::from(entry.logprob));
    }

    let mut label_logprobs = HashMap::new();
    let mut missing_token_ids = Vec::new();
    for token_id in label_token_ids {
        if let Some(logprob) = returned_logprobs.get(token_id) {
            label_logprobs.insert(*token_id, *logprob);
        } else {
            missing_token_ids.push(*token_id);
        }
    }

    if !missing_token_ids.is_empty() {
        return Err(server_error!(
            "Token IDs {:?} not found in logprobs for generative scoring item {}.",
            missing_token_ids,
            item_index
        ));
    }

    let first_label_id = label_token_ids[0];
    if apply_softmax {
        let max_logprob = label_logprobs.values().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 =
            label_logprobs.values().map(|logprob| (logprob - max_logprob).exp()).sum();
        let first_logprob = label_logprobs[&first_label_id];
        Ok((first_logprob - max_logprob).exp() / sum_exp)
    } else {
        Ok(label_logprobs[&first_label_id].exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core_client::protocol::logprobs::TokenLogprob;

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|name| (*name).to_string()).collect(),
            lora_request: None,
        }
    }

    fn base_request() -> GenerativeScoringRequest {
        serde_json::from_value(serde_json::json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "query": "question",
            "items": ["yes"],
            "label_token_ids": [10, 20]
        }))
        .expect("parse request")
    }

    #[test]
    fn generative_scoring_request_deserializes_text_and_token_inputs() {
        let text_request: GenerativeScoringRequest = serde_json::from_value(serde_json::json!({
            "query": "question",
            "items": ["yes", "no"],
            "label_token_ids": [10, 20]
        }))
        .expect("parse text request");
        assert_eq!(
            text_request.query,
            GenerativeScoringInput::Text("question".to_string())
        );
        assert_eq!(
            text_request.items,
            GenerativeScoringItems::Text(vec!["yes".to_string(), "no".to_string()])
        );
        assert!(text_request.apply_softmax);
        assert!(text_request.add_special_tokens);

        let token_request: GenerativeScoringRequest = serde_json::from_value(serde_json::json!({
            "query": [1, 2],
            "items": [[3, 4], [5, 6]],
            "label_token_ids": [10, 20],
            "apply_softmax": false,
            "item_first": true,
            "add_special_tokens": false
        }))
        .expect("parse token request");
        assert_eq!(
            token_request.query,
            GenerativeScoringInput::TokenIds(vec![1, 2])
        );
        assert_eq!(
            token_request.items,
            GenerativeScoringItems::TokenIds(vec![vec![3, 4], vec![5, 6]])
        );
        assert!(!token_request.apply_softmax);
        assert!(token_request.item_first);
        assert!(!token_request.add_special_tokens);
    }

    #[test]
    fn validate_generative_scoring_request_checks_model_items_and_labels() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);

        let wrong_model = GenerativeScoringRequest {
            model: Some("missing".to_string()),
            ..base_request()
        };
        assert!(validate_generative_scoring_request(&wrong_model, &served, 100).is_err());

        let empty_labels = GenerativeScoringRequest {
            label_token_ids: Vec::new(),
            ..base_request()
        };
        assert!(validate_generative_scoring_request(&empty_labels, &served, 100).is_err());

        let out_of_vocab = GenerativeScoringRequest {
            label_token_ids: vec![99, 100],
            ..base_request()
        };
        assert!(validate_generative_scoring_request(&out_of_vocab, &served, 100).is_err());
    }

    #[test]
    fn concat_and_truncate_prompt_respects_item_order() {
        assert_eq!(
            concat_and_truncate_prompt(&[100, 101], &[200, 201], false, 10),
            vec![100, 101, 200, 201]
        );
        assert_eq!(
            concat_and_truncate_prompt(&[100, 101], &[200, 201], true, 10),
            vec![200, 201, 100, 101]
        );
        assert_eq!(
            concat_and_truncate_prompt(&[100, 101], &[200, 201], false, 3),
            vec![100, 101, 200]
        );
    }

    #[test]
    fn compute_score_supports_softmax_and_true_probability_modes() {
        let position = PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: 10,
                    logprob: -0.5,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: 20,
                    logprob: -2.0,
                    rank: 2,
                },
            ],
        };

        let softmax_score = compute_score(&position, &[10, 20], true, 0).expect("softmax score");
        let expected = (-0.5_f64).exp() / ((-0.5_f64).exp() + (-2.0_f64).exp());
        assert!((softmax_score - expected).abs() < 1e-9);

        let true_probability = compute_score(&position, &[10, 20], false, 0).expect("true prob");
        assert!((true_probability - (-0.5_f64).exp()).abs() < 1e-9);
    }
}
