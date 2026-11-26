// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use vllm_text::Prompt;

use crate::routes::openai::utils::types::{Usage, default_true};

/// `POST /generative_scoring` request body.
///
/// Original Python definition: `GenerativeScoringRequest` in
/// `vllm/entrypoints/generate/generative_scoring/serving.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerativeScoringRequest {
    pub model: Option<String>,
    /// The query text or pre-tokenized query token IDs.
    pub query: Prompt,
    /// The item texts or pre-tokenized item token IDs, each scored after the
    /// query.
    pub items: Vec<Prompt>,
    /// Token IDs to compute next-token probabilities for. The score of an
    /// item is the probability of the first one.
    pub label_token_ids: Vec<u32>,
    /// If true, normalize probabilities using softmax over only the
    /// `label_token_ids`. If false, return the true model probabilities over
    /// the full vocab for those ids.
    #[serde(default = "default_true")]
    pub apply_softmax: bool,
    /// If true, prepend items to query. Otherwise append items to query.
    #[serde(default)]
    pub item_first: bool,
    /// Whether to add special tokens when tokenizing the query text. Item
    /// texts never get them, so a joined prompt has no duplicate BOS/EOS.
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    /// Salt for prefix cache isolation, applied to every item.
    pub cache_salt: Option<String>,
    /// The priority of the request (lower means earlier handling; default: 0).
    #[serde(default)]
    pub priority: i32,
    pub request_id: Option<String>,
}

/// `POST /generative_scoring` response body.
#[derive(Debug, Clone, Serialize)]
pub struct GenerativeScoringResponse {
    pub id: String,
    /// Always `list`.
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    /// One result per input item, in input order.
    pub data: Vec<GenerativeScoringItemResult>,
    /// Token usage summed over all items.
    pub usage: Usage,
}

/// Score of one input item.
#[derive(Debug, Clone, Serialize)]
pub struct GenerativeScoringItemResult {
    /// The index of this item in the input items list.
    pub index: usize,
    /// Always `score`.
    pub object: &'static str,
    /// The probability of the first label token.
    pub score: f64,
}
