// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod custom;
pub mod hf_dataset;
pub mod multi_turn;
pub mod prefix_repetition;
mod progress;
pub mod random;
pub mod random_mm;
pub mod random_rerank;
pub mod sharegpt;
pub mod sonnet;
pub mod speed_bench;

use std::sync::Arc;

/// Represents a single inference request for benchmarking.
/// Matches Python's SampleRequest dataclass from datasets.py:71-82.
///
/// `prompt` uses `Arc<str>` to avoid expensive String clones when distributing
/// requests across tokio tasks. At 100k prompts with 8k tokens each, this saves
/// ~3GB of peak memory vs cloning String per task.
#[derive(Debug, Clone)]
pub struct SampleRequest {
    pub prompt: Arc<str>,
    pub prompt_len: usize,
    pub expected_output_len: usize,
    pub request_id: Option<String>,
    /// Pre-computed token IDs for this prompt.
    /// When set, the completions backend sends these directly via `prompt_token_ids`
    /// instead of the text `prompt`, avoiding server-side re-tokenization.
    pub prompt_token_ids: Option<Arc<[u32]>>,
    /// Multimodal content items as pre-serialized JSON fragments.
    /// Each `Arc<str>` is a complete JSON object string, e.g.
    /// `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`
    ///
    /// Pre-serialized to avoid:
    /// 1. `serde_json::Value` tree overhead (3 Maps + keys per image)
    /// 2. Deep-cloning ~200KB+ base64 data when building request payloads
    ///
    /// Double-`Arc` for zero-cost sharing: outer Arc for the slice, inner Arc for each fragment.
    pub multi_modal_content: Option<Arc<[Arc<str>]>>,
    /// Pre-serialized OpenAI chat `messages` array as a complete JSON string,
    /// e.g. `[{"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url",...}]}]`.
    ///
    /// Set by datasets when `--enable-multimodal-chat` is on (mirrors Python's
    /// `apply_multimodal_chat_transformation`: the dataset builds the chat messages
    /// and the backend sends them verbatim). When set, `multi_modal_content` is None
    /// and the mm items are embedded here instead. `prompt` still holds the text part
    /// for token accounting and /tokenize verification.
    pub chat_messages_json: Option<Arc<str>>,
    /// Multiple text inputs for one request (pooling backends only).
    /// Embeddings send it as `"input": [t1, t2, ...]` (--random-batch-size);
    /// rerank sends `[0]` as the query and `[1..]` as documents (random-rerank).
    /// Mirrors Python's list-valued `SampleRequest.prompt`.
    pub prompt_list: Option<Arc<[Arc<str>]>>,
}

impl Default for SampleRequest {
    /// Empty request; struct-update base so dataset builders only spell out the
    /// fields they set (new optional fields then don't touch every call site).
    fn default() -> Self {
        Self {
            prompt: Arc::from(""),
            prompt_len: 0,
            expected_output_len: 0,
            request_id: None,
            prompt_token_ids: None,
            multi_modal_content: None,
            chat_messages_json: None,
            prompt_list: None,
        }
    }
}

/// Oversample `requests` up to `num_requests` by cloning random entries
/// (seeded by list length for determinism), renumbering their request ids.
/// No-op when enough samples exist, `no_oversample` is set, or the list is empty.
/// Mirrors Python `BenchmarkDataset.maybe_oversample_requests`.
pub fn oversample_requests(
    requests: &mut Vec<SampleRequest>,
    num_requests: usize,
    request_id_prefix: &str,
    no_oversample: bool,
) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    if requests.len() >= num_requests || requests.is_empty() {
        return;
    }
    if no_oversample {
        tracing::info!(
            samples = requests.len(),
            requested = num_requests,
            "skipping dataset oversampling"
        );
        return;
    }
    let original_len = requests.len();
    let mut rng = StdRng::seed_from_u64(original_len as u64);
    for i in 0..(num_requests - original_len) {
        let mut req = requests[rng.random_range(0..original_len)].clone();
        req.request_id = Some(format!("{request_id_prefix}{}", original_len + i));
        requests.push(req);
    }
    tracing::info!(
        original_samples = original_len,
        samples = requests.len(),
        "oversampled dataset"
    );
}

/// Group already-generated single-input requests into batched requests of
/// `batch_size` inputs each (embeddings/pooling only). Mirrors Python
/// `RandomDataset.sample` batching: prompt becomes a list, prompt_len is the
/// sum over the batch, request ids are renumbered per batch.
/// `batch_size <= 1` returns the input unchanged.
pub fn batch_requests(
    requests: Vec<SampleRequest>,
    batch_size: usize,
    request_id_prefix: &str,
) -> Vec<SampleRequest> {
    if batch_size <= 1 {
        return requests;
    }
    requests
        .chunks(batch_size)
        .enumerate()
        .map(|(batch_idx, batch)| SampleRequest {
            prompt_list: Some(batch.iter().map(|r| r.prompt.clone()).collect()),
            prompt_len: batch.iter().map(|r| r.prompt_len).sum(),
            expected_output_len: 0,
            request_id: Some(format!("{request_id_prefix}{batch_idx}")),
            ..Default::default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: &str, len: usize) -> SampleRequest {
        SampleRequest {
            prompt: Arc::from(prompt),
            prompt_len: len,
            expected_output_len: 128,
            ..Default::default()
        }
    }

    #[test]
    fn test_batch_requests_groups_and_sums() {
        let reqs = vec![
            req("a", 10),
            req("b", 20),
            req("c", 30),
            req("d", 40),
            req("e", 50),
        ];
        let batched = batch_requests(reqs, 2, "t-");
        assert_eq!(batched.len(), 3); // 2 + 2 + 1
        let first = batched[0].prompt_list.as_ref().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(&*first[0], "a");
        assert_eq!(batched[0].prompt_len, 30);
        assert_eq!(batched[0].expected_output_len, 0);
        assert_eq!(batched[0].request_id.as_deref(), Some("t-0"));
        assert_eq!(batched[2].prompt_list.as_ref().unwrap().len(), 1);
        assert_eq!(batched[2].prompt_len, 50);
    }

    #[test]
    fn test_batch_requests_size_one_is_identity() {
        let reqs = vec![req("a", 10), req("b", 20)];
        let out = batch_requests(reqs, 1, "t-");
        assert_eq!(out.len(), 2);
        assert!(out[0].prompt_list.is_none());
        assert_eq!(&*out[0].prompt, "a");
    }

    #[test]
    fn test_oversample_requests() {
        let mut reqs = vec![req("a", 10), req("b", 20)];
        oversample_requests(&mut reqs, 5, "t-", false);
        assert_eq!(reqs.len(), 5);
        assert_eq!(reqs[4].request_id.as_deref(), Some("t-4"));

        let mut reqs = vec![req("a", 10)];
        oversample_requests(&mut reqs, 5, "t-", true); // no_oversample
        assert_eq!(reqs.len(), 1);
    }
}

/// A single turn in a multi-turn conversation.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user_message: Arc<str>,
    pub user_message_len: usize,
    pub expected_output_len: usize,
}

/// A complete multi-turn conversation with all turns pre-generated.
#[derive(Debug, Clone)]
pub struct MultiTurnConversation {
    pub conversation_id: String,
    pub turns: Vec<ConversationTurn>,
}
