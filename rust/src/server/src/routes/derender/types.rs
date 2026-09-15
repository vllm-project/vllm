// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Wire types for the `/v1/chat/completions/derender` and
//! `/v1/completions/derender` endpoints.
//!
//! Mirrors the Python vLLM `Derender*Request` / `Derender*Stream*` classes in
//! `vllm/entrypoints/scale_out/token_in_token_out/protocol.py`.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use validator::{Validate, ValidationErrors};

use crate::error::{ApiError, bail_invalid_request};
use crate::routes::inference::generate::{GenerateResponse, PromptLogprobMaps};
use crate::routes::openai::CompletionRequest;
use crate::routes::openai::chat_completions::ChatCompletionRequest;
use crate::routes::openai::utils::types::{Normalizable, Usage};

/// Cap on the carried detok window in [`DerenderStreamState`].
///
/// INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET is small (5) and the trimmed
/// window is O(offset). A generous limit rejects unusually large or malformed
/// payloads without restricting legitimate multi-byte sequences.
const MAX_PREV_TOKENS: usize = 1024;

/// Per sequence state for stateless streaming derender.
///
/// The client carries this between successive per chunk HTTP calls to the
/// streaming derender endpoint. All fields are plain JSON serializable data.
/// No opaque tokenizer or parser internals are stored here.
///
/// The detokenization strategy carries the incremental decode offsets
/// directly rather than re-sending the whole token history each chunk.
/// `detokenize_incrementally` only ever reads the trailing token window
/// `prev_tokens[prefix_offset..]`, so we carry just that tail plus the two
/// offsets. Each chunk resumes exactly where the last one stopped, including
/// any partially processed multi-byte character (tracked by `read_offset`),
/// then trims and rebases the window so it never grows with generation
/// length.
///
/// Do not skip serializing `None` fields here: Python's `model_dump()` emits
/// them as explicit `null`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct DerenderStreamState {
    /// Trailing decode window. Token strings from `prefix_offset` onward.
    ///
    /// Bounded, trimmed and rebased each chunk to the tail
    /// `detokenize_incrementally` still reads, so it does not grow with the
    /// number of chunks.
    #[serde(default)]
    pub prev_tokens: Vec<String>,
    /// Token IDs parallel to `prev_tokens`, used to reconstruct the decode
    /// window without a lossy `id_to_token` → `token_to_id` round-trip
    /// (some backends store byte pieces as lossy UTF-8). Slots without a
    /// decodable ID carry `u32::MAX`.
    ///
    /// Rust-specific extension: states produced by the Python implementation
    /// lack this field (`null`/absent), in which case the pieces are mapped
    /// back through `token_to_id` as a best-effort fallback. Python ignores
    /// the unknown field when it parses our state.
    #[serde(default)]
    pub prev_token_ids: Option<Vec<u32>>,
    /// Prefix offset into `prev_tokens` for incremental detokenization.
    #[serde(default)]
    pub prefix_offset: usize,
    /// Read offset into `prev_tokens` for incremental detokenization.
    #[serde(default)]
    pub read_offset: usize,
    /// True once the initial `role: "assistant"` delta has been emitted.
    ///
    /// Prevents re-emitting the role on subsequent chunks even when the detok
    /// window is transiently empty (e.g. usage only final chunk).
    #[serde(default)]
    pub role_sent: bool,
    // TODO: Properties used in follow on PR for tool call parsing
    /// Last emitted cumulative assistant content text.
    pub last_content: Option<String>,
    /// Last emitted cumulative reasoning text.
    pub last_reasoning: Option<String>,
    /// Stable tool-call IDs, assigned once when each call first appears.
    ///
    /// Prevents ID regeneration across re-parsing.
    #[serde(default)]
    pub last_tool_call_ids: Vec<String>,
}

impl DerenderStreamState {
    /// Reject malformed caller supplied offsets/lengths.
    ///
    /// Python enforces the `prev_tokens` cap through a Pydantic field
    /// validator (surfaced as 400) and turns detok failures from out-of-range
    /// offsets into a 400 "invalid stream_state" error; both checks live here.
    // TODO: called by the phase-3 streaming endpoints.
    #[allow(dead_code)]
    pub(super) fn validate(&self) -> Result<(), ApiError> {
        if self.prev_tokens.len() > MAX_PREV_TOKENS {
            bail_invalid_request!(
                "prev_tokens length ({}) exceeds maximum ({})",
                self.prev_tokens.len(),
                MAX_PREV_TOKENS
            );
        }
        if self.prefix_offset > self.read_offset || self.read_offset > self.prev_tokens.len() {
            bail_invalid_request!(
                "invalid stream_state: detokenization failed (prefix_offset={}, \
                 read_offset={}, prev_tokens length={})",
                self.prefix_offset,
                self.read_offset,
                self.prev_tokens.len()
            );
        }
        if let Some(ids) = &self.prev_token_ids
            && ids.len() != self.prev_tokens.len()
        {
            bail_invalid_request!(
                "invalid stream_state: prev_token_ids length ({}) does not match \
                 prev_tokens length ({})",
                ids.len(),
                self.prev_tokens.len()
            );
        }
        Ok(())
    }
}

/// Reject `stream: true` on the non-streaming request body.
///
/// TODO: phase 3 re-adds the streaming request variant, which this error
/// steers the untagged union towards; until then a `stream: true` body fails
/// deserialization outright and the endpoint responds 400.
fn expect_stream_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = bool::deserialize(deserializer)?;
    if value {
        return Err(serde::de::Error::custom(
            "`stream` must be false or omitted for the non-streaming derender request",
        ));
    }
    Ok(false)
}

/// Request for the /v1/chat/completions/derender endpoint (non-streaming).
///
/// Wraps a complete GenerateResponse and caller supplied metadata needed to
/// produce a fully formed ChatCompletionResponse without a GPU.
#[derive(Debug, Deserialize)]
pub(crate) struct DerenderChatRequest {
    #[serde(default, deserialize_with = "expect_stream_false")]
    #[allow(dead_code)]
    stream: bool,
    /// Served model name. Defaults to the server's served model name.
    pub model: Option<String>,
    /// The complete token-in / token-out engine response to derender.
    pub generate_response: GenerateResponse,
    /// Prompt token count for usage; defaults to 0 if omitted.
    ///
    /// GenerateResponse carries only output tokens; the caller already has
    /// `len(GenerateRequest.token_ids)` from the render step.
    pub prompt_tokens: Option<usize>,
    /// The original (post-adjust_request) ChatCompletionRequest from /render.
    ///
    /// Phase 1 honours its `skip_special_tokens` option; phase 2 additionally
    /// feeds it to the reasoning/tool parsers so they receive the full
    /// request context they expect.
    pub chat_request: Option<ChatCompletionRequest>,
}

/// Request for the /v1/completions/derender endpoint (non-streaming).
///
/// Parallel to DerenderChatRequest but handles the multi-prompt completions
/// case: one GenerateResponse per prompt, mirroring the list[GenerateRequest]
/// returned by /v1/completions/render.
#[derive(Debug, Deserialize)]
pub(crate) struct DerenderCompletionRequest {
    #[serde(default, deserialize_with = "expect_stream_false")]
    #[allow(dead_code)]
    stream: bool,
    /// Served model name. Defaults to the server's served model name.
    pub model: Option<String>,
    /// One response per prompt, parallel to the list[GenerateRequest]
    /// returned by /v1/completions/render.
    pub generate_responses: Vec<GenerateResponse>,
    /// One prompt token count per response; each defaults to 0 if omitted.
    ///
    /// If provided, `len(prompt_tokens)` must equal `len(generate_responses)`.
    pub prompt_tokens: Option<Vec<usize>>,
    /// The original (post-adjust_request) CompletionRequest from /render.
    ///
    /// Mirrors chat_request on DerenderChatRequest.
    pub completion_request: Option<CompletionRequest>,
}

/// Validation and normalization of embedded requests (e.g. `chat_request`)
/// happen when the handler lowers them, not at extraction time.
macro_rules! impl_union_validation {
    ($union:ident) => {
        impl Validate for $union {
            fn validate(&self) -> Result<(), ValidationErrors> {
                Ok(())
            }
        }

        impl Normalizable for $union {}
    };
}

/// Request body for the chat derender endpoint.
///
/// TODO: phase 3 re-adds the `Streaming` variant, discriminated by the
/// `stream` field's literal value (a body without `stream` validates as the
/// non-streaming member, whose `stream` defaults to false).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DerenderChatRequestUnion {
    NonStreaming(DerenderChatRequest),
}

impl_union_validation!(DerenderChatRequestUnion);

/// See [`DerenderChatRequestUnion`].
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DerenderCompletionRequestUnion {
    NonStreaming(DerenderCompletionRequest),
}

impl_union_validation!(DerenderCompletionRequestUnion);

/// Mirrors the Python vLLM `ChatCompletionResponse` as produced by the
/// derender endpoint.
///
/// Unlike the normal chat path's response type, `prompt_logprobs` here
/// carries the engine-side `HashMap<u32, GenerateLogprob>` shape passed
/// through from `GenerateResponse`, and `usage` is always present.
///
/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DerenderChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<crate::routes::openai::chat_completions::ChatCompletionChoice>,
    pub usage: Usage,
    pub prompt_logprobs: Option<PromptLogprobMaps>,
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
}
