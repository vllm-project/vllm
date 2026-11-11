// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/request.py`: the engine-core request state the scheduler
//! reads and mutates.
//!
//! Opaque Python objects (the full `SamplingParams` / `PoolingParams`, the
//! LoRA request, prompt embeddings, multimodal data, trace headers, KV hints,
//! `kv_transfer_params` / `ec_transfer_params`, the structured output request,
//! and per-request spec-decode metrics) are not part of [`Request`]; the
//! binding layer keeps them keyed by request id. Only the scalars the
//! scheduler reads are mirrored here.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// The wire types of what the scheduler reports per request.
pub use vllm_engine_core_client::protocol::output::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason as FinishReason, StopReason,
};
pub use vllm_engine_core_client::protocol::stats::PrefillStats;

use super::multimodal::MultiModalFeatureSpec;
use super::stats::RequestSpecDecodeMetrics;
use crate::hash::BlockHash;
use crate::kv_cache::{EventExtraKey, EventExtraKeys, KvRequest};

/// Status of a request.
///
/// The discriminants match Python's `enum.auto()` values, and the derived
/// ordering matches `IntEnum` comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RequestStatus {
    Waiting = 1,
    WaitingForStructuredOutputGrammar = 2,
    WaitingForRemoteKvs = 3,
    WaitingForStreamingReq = 4,
    Running = 5,
    Preempted = 6,
    // Note: anything after PREEMPTED will be considered
    // as a finished status.
    FinishedStopped = 7,
    FinishedLengthCapped = 8,
    FinishedAborted = 9,
    FinishedIgnored = 10,
    FinishedError = 11,
    FinishedRepetition = 12,
}

impl RequestStatus {
    pub fn name(self) -> &'static str {
        match self {
            Self::Waiting => "WAITING",
            Self::WaitingForStructuredOutputGrammar => "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
            Self::WaitingForRemoteKvs => "WAITING_FOR_REMOTE_KVS",
            Self::WaitingForStreamingReq => "WAITING_FOR_STREAMING_REQ",
            Self::Running => "RUNNING",
            Self::Preempted => "PREEMPTED",
            Self::FinishedStopped => "FINISHED_STOPPED",
            Self::FinishedLengthCapped => "FINISHED_LENGTH_CAPPED",
            Self::FinishedAborted => "FINISHED_ABORTED",
            Self::FinishedIgnored => "FINISHED_IGNORED",
            Self::FinishedError => "FINISHED_ERROR",
            Self::FinishedRepetition => "FINISHED_REPETITION",
        }
    }

    pub fn is_finished(self) -> bool {
        self > Self::Preempted
    }

    /// Mapping of finished statuses to their finish reasons.
    ///
    /// NOTE: The ignored requests are the requests whose prompt lengths
    /// are longer than the model's length cap. Therefore, the stop
    /// reason should also be "length" as in OpenAI API.
    pub fn get_finished_reason(self) -> Option<FinishReason> {
        Some(match self {
            Self::FinishedStopped => FinishReason::Stop,
            Self::FinishedLengthCapped => FinishReason::Length,
            Self::FinishedAborted => FinishReason::Abort,
            Self::FinishedIgnored => FinishReason::Length,
            Self::FinishedError => FinishReason::Error,
            Self::WaitingForStreamingReq => FinishReason::Stop,
            Self::FinishedRepetition => FinishReason::Repetition,
            _ => return None,
        })
    }
}

impl fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What the scheduler records in a request's `PrefillStats`.
#[easy_ext::ext(PrefillStatsExt)]
pub impl PrefillStats {
    fn set(
        &mut self,
        num_prompt_tokens: usize,
        num_local_cached_tokens: usize,
        num_external_cached_tokens: usize,
    ) {
        let num_cached_tokens = num_local_cached_tokens + num_external_cached_tokens;
        assert!(num_cached_tokens <= num_prompt_tokens);

        self.num_prompt_tokens = num_prompt_tokens as u32;
        self.num_computed_tokens = (num_prompt_tokens - num_cached_tokens) as u32;
        self.num_cached_tokens = num_cached_tokens as u32;
        self.num_local_cached_tokens = num_local_cached_tokens as u32;
        self.num_external_cached_tokens = num_external_cached_tokens as u32;
    }

    fn finalize(&mut self, num_cached_tokens: usize) {
        self.num_cache_creation_tokens = (num_cached_tokens as u32)
            .min(self.num_prompt_tokens)
            .saturating_sub(self.num_cached_tokens);
    }
}

/// Parameters for detecting repetitive N-gram patterns in output tokens.
///
/// Python validates `max_pattern_size, min_pattern_size >= 0` and
/// `min_count >= 2` when enabled; a negative `min_count` only reaches
/// `check_sequence_repetition` with detection disabled, where it behaves as 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepetitionDetectionParams {
    /// Maximum size of N-gram pattern to detect for sequence repetition.
    /// Set to 0 to disable. Must be used together with min_count.
    pub max_pattern_size: usize,
    /// Minimum N-gram pattern size to check for sequence repetition.
    /// If set to 0, it defaults to 1.
    /// Must be <= max_pattern_size.
    pub min_pattern_size: usize,
    /// Minimum number of times an N-gram pattern must repeat to trigger
    /// detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    /// 3 times. Must be used together with max_pattern_size.
    pub min_count: usize,
}

/// `sampling_params.extra_args["kv_cache_report_mode"]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCacheReportMode {
    #[default]
    Incremental,
    /// Emit `BlockStored` events for every block of the request, not only
    /// newly stored ones. Python compares the string to `"full"`; any other
    /// value is [`KvCacheReportMode::Incremental`].
    Full,
}

/// The fields of `vllm.SamplingParams` (after `update_from_generation_config`)
/// that the engine core reads.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// `None` is rejected when constructing a [`Request`].
    pub max_tokens: Option<usize>,
    pub min_tokens: usize,
    /// `SamplingParams.eos_token_id`: the primary EOS token, `None` with
    /// `ignore_eos`.
    pub eos_token_id: Option<i32>,
    /// `SamplingParams.stop_token_ids` (`None` is empty). Includes the
    /// generation config's extra EOS ids unless `ignore_eos`.
    pub stop_token_ids: Vec<i32>,
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// `skip_reading_prefix_cache`, with Python's `None` as `false`.
    pub skip_reading_prefix_cache: bool,
    /// Whether `prompt_logprobs` is not `None`.
    pub has_prompt_logprobs: bool,
    /// Whether `prompt_logprob_token_ids` is not `None`.
    pub has_prompt_logprob_token_ids: bool,
    pub prompt_logprob_start: Option<usize>,
    /// `SamplingParams.num_logprobs`.
    pub num_logprobs: Option<usize>,
    /// Whether `structured_outputs` has a constraint, i.e.
    /// `StructuredOutputRequest.from_sampling_params` is not `None`.
    pub use_structured_output: bool,
    pub kv_cache_report_mode: KvCacheReportMode,
    pub watermarking: bool,
    pub temperature: f64,
    /// Whether `trace_decode_token_ids` is not `None`.
    pub has_trace_decode_token_ids: bool,
}

impl Default for SamplingParams {
    /// `SamplingParams()` with Python's defaults.
    fn default() -> Self {
        Self {
            max_tokens: Some(16),
            min_tokens: 0,
            eos_token_id: None,
            stop_token_ids: Vec::new(),
            repetition_detection: None,
            skip_reading_prefix_cache: false,
            has_prompt_logprobs: false,
            has_prompt_logprob_token_ids: false,
            prompt_logprob_start: None,
            num_logprobs: None,
            use_structured_output: false,
            kv_cache_report_mode: KvCacheReportMode::Incremental,
            watermarking: false,
            temperature: 1.0,
            has_trace_decode_token_ids: false,
        }
    }
}

/// The fields of `vllm.PoolingParams` that the engine core reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolingParams {
    /// `skip_reading_prefix_cache`, with Python's `None` as `false`.
    pub skip_reading_prefix_cache: bool,
}

/// Lightweight data for streaming session continuation.
///
/// Contains only the fields needed to update an existing streaming session
/// with new input data.
#[derive(Clone, Debug)]
pub struct StreamingUpdate {
    /// `None` and empty are equivalent to the scheduler.
    pub mm_features: Option<Vec<MultiModalFeatureSpec>>,
    pub prompt_token_ids: Option<Vec<i32>>,
    pub max_tokens: usize,
    pub arrival_time: f64,
    pub sampling_params: Option<SamplingParams>,
}

impl StreamingUpdate {
    pub fn from_request(request: &Request) -> Option<Self> {
        if !request.resumable {
            return None;
        }
        Some(Self {
            mm_features: Some(request.mm_features.clone()),
            prompt_token_ids: request.prompt_token_ids.clone(),
            max_tokens: request.max_tokens,
            arrival_time: request.arrival_time,
            sampling_params: request.sampling_params.clone(),
        })
    }
}

/// Computes the block hashes of a request's new full blocks
/// (`Callable[[Request], list[BlockHash]]` in Python, e.g. the closure from
/// `kv_cache_utils.get_request_block_hasher`).
// TODO: implement for `crate::hash::RequestBlockHasher` once the request
// carries its extra keys (LoRA name, cache salt, multimodal, prompt embeds).
pub trait BlockHasher: Send + Sync {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash>;
}

/// An invalid [`NewRequest`], raised as `ValueError` in Python.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidRequest {
    #[error("sampling_params and pooling_params can't both be unset")]
    MissingParams,
    #[error("Neither prompt_token_ids nor prompt_embeds were defined.")]
    MissingPrompt,
    #[error(
        "Prompt token ids and prompt embeds had different lengths \
         prompt_token_ids={prompt_token_ids} prompt_embeds={prompt_embeds}"
    )]
    PromptLengthMismatch {
        prompt_token_ids: usize,
        prompt_embeds: usize,
    },
}

/// The arguments of `Request.__init__`, without the opaque Python objects.
#[derive(Clone, Default)]
pub struct NewRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<i32>>,
    /// The length of `prompt_embeds`, if set. The tensor itself stays with
    /// the binding layer.
    pub num_prompt_embeds: Option<usize>,
    pub sampling_params: Option<SamplingParams>,
    pub pooling_params: Option<PoolingParams>,
    pub client_index: usize,
    /// Defaults to `time.time()`.
    pub arrival_time: Option<f64>,
    /// Per-position mask used in mixed-mode (chat completion with
    /// prompt_embeds).
    pub prompt_is_token_ids: Option<Vec<bool>>,
    pub mm_features: Vec<MultiModalFeatureSpec>,
    /// `lora_request.lora_int_id`, if a LoRA request is set.
    pub lora_int_id: Option<i64>,
    /// `lora_request.lora_name`, if a LoRA request is set.
    pub lora_name: Option<String>,
    pub cache_salt: Option<String>,
    pub priority: i64,
    pub block_hasher: Option<Arc<dyn BlockHasher>>,
    pub resumable: bool,
    pub session_id: Option<String>,
    pub abort_immediately: bool,
    pub block_hashes: Vec<BlockHash>,
    /// Whether `kv_transfer_params` (from `sampling_params.extra_args`) has a
    /// truthy `do_remote_prefill`.
    pub do_remote_prefill: bool,
}

/// Source of [`Request::seq`].
static NEXT_REQUEST_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct Request {
    pub request_id: String,
    pub client_index: usize,
    pub priority: i64,
    pub sampling_params: Option<SamplingParams>,
    pub pooling_params: Option<PoolingParams>,
    /// `lora_request.lora_int_id`, if a LoRA request is set.
    pub lora_int_id: Option<i64>,
    /// `lora_request.lora_name`, if a LoRA request is set.
    pub lora_name: Option<String>,
    /// Whether the request has a `StructuredOutputRequest`. Python derives
    /// it once from the initial sampling params.
    pub use_structured_output: bool,
    pub arrival_time: f64,

    pub status: RequestStatus,
    pub events: Vec<EngineCoreEvent>,
    pub stop_reason: Option<StopReason>,

    pub max_tokens: usize,
    /// `kv_cache_report_mode` from the initial sampling params; pooling
    /// requests read Python's `getattr` default.
    pub kv_cache_report_mode: KvCacheReportMode,

    pub prompt_token_ids: Option<Vec<i32>>,
    /// Whether `prompt_embeds` is set; the tensor stays with the binding layer.
    pub has_prompt_embeds: bool,
    /// Per-position mask used in mixed-mode (chat completion with
    /// prompt_embeds). `None` except when both `prompt_token_ids` and
    /// `prompt_embeds` are set and their positions are interleaved.
    pub prompt_is_token_ids: Option<Vec<bool>>,
    pub num_prompt_tokens: usize,
    output_token_ids: Vec<i32>,
    all_token_ids: Vec<i32>,

    /// Used in async scheduling.
    pub num_output_placeholders: usize,
    /// Tokens of output in flight when the request was preempted: delivered
    /// on return, but must not mutate the reset counters.
    pub num_stale_output_tokens: usize,
    /// Drop the stale output instead, for same-step preempt + resume
    /// (reset_prefix_cache).
    pub drop_stale_output: bool,

    /// Tokens of steps whose output is not yet processed (async scheduling
    /// and PP run ahead of the GPU); `num_computed_tokens` counts them
    /// optimistically.
    pub num_in_flight_tokens: usize,

    /// V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
    /// so the worker's broadcast slot ring stays consistent.
    pub next_decode_eligible_step: u64,

    /// Seq of the most recent step this request was scheduled in; fences
    /// deferred block freeing (see Scheduler._free_request_blocks).
    pub last_sched_seq: u64,

    pub spec_token_ids: Vec<i32>,
    pub num_computed_tokens: usize,
    pub cache_salt: Option<String>,

    // Multi-modal related
    pub mm_features: Vec<MultiModalFeatureSpec>,

    pub session_id: Option<String>,

    /// True if this request is scheduled as a non-final prefill chunk.
    pub is_prefill_chunk: bool,

    /// Block-aligned token position of a proven shared prefix worth pinning
    /// in the (sparse) prefix cache; 0 means none. Set at admission for
    /// hybrid/Mamba models when a shared prefix is detected (Marconi-style).
    pub shared_prefix_boundary: usize,
    /// DeepSeek-V4.1 only: SWA bounded replay. The request holds no
    /// sliding-window KV below this position; 0 when nothing replays.
    pub replay_start: usize,

    /// The number of NaNs in logits. A value greater than 0
    /// indicates that the output is corrupted
    pub num_nans_in_logits: usize,

    /// The number of times this request has been preempted by the scheduler.
    pub num_preemptions: usize,

    pub prefill_stats: Option<PrefillStats>,

    /// Prompt block hashes precomputed by the frontend; the hasher continues
    /// from the first block they don't cover.
    pub block_hashes: Vec<BlockHash>,
    block_hasher: Option<Arc<dyn BlockHasher>>,

    pub prefix_cache_hit_limit: Option<usize>,

    /// Used for streaming
    pub resumable: bool,
    /// `None` entry in the queue means finished.
    pub streaming_queue: Option<VecDeque<Option<StreamingUpdate>>>,

    /// If True, request should be aborted immediately after being added to
    /// the scheduler so the connector's request_finished hook runs.
    pub abort_immediately: bool,

    /// Whether `kv_transfer_params` has a truthy `do_remote_prefill`.
    pub do_remote_prefill: bool,

    /// Set by the scheduler unless per-request spec decode metrics are off.
    pub spec_decode_metrics: Option<RequestSpecDecodeMetrics>,

    /// Construction order, the final tiebreak of [`Request::lt`].
    ///
    /// Deviation: Python breaks full ties with `id(self) < id(other)`, an
    /// arbitrary memory address order; this is a process-wide monotonically
    /// increasing sequence number instead. Only requests with equal priority,
    /// arrival time, and request id reach it.
    seq: u64,
}

impl Request {
    pub fn new(args: NewRequest) -> Result<Self, InvalidRequest> {
        let NewRequest {
            request_id,
            prompt_token_ids,
            num_prompt_embeds,
            sampling_params,
            pooling_params,
            client_index,
            arrival_time,
            prompt_is_token_ids,
            mm_features,
            lora_int_id,
            lora_name,
            cache_salt,
            priority,
            block_hasher,
            resumable,
            session_id,
            abort_immediately,
            block_hashes,
            do_remote_prefill,
        } = args;

        let use_structured_output =
            sampling_params.as_ref().is_some_and(|sp| sp.use_structured_output);
        let arrival_time = arrival_time.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before the Unix epoch")
                .as_secs_f64()
        });

        let mut status = RequestStatus::Waiting;
        let max_tokens;
        let mut kv_cache_report_mode = KvCacheReportMode::Incremental;
        if pooling_params.is_some() {
            // Pooling models.
            max_tokens = 1;
        } else if let Some(sp) = &sampling_params {
            // Generative models.
            max_tokens = sp.max_tokens.expect("sampling_params.max_tokens is None");
            if use_structured_output {
                status = RequestStatus::WaitingForStructuredOutputGrammar;
            }
            kv_cache_report_mode = sp.kv_cache_report_mode;
        } else {
            return Err(InvalidRequest::MissingParams);
        }

        let num_prompt_tokens = match (&prompt_token_ids, num_prompt_embeds) {
            (None, None) => return Err(InvalidRequest::MissingPrompt),
            (None, Some(n)) => n,
            (Some(ids), Some(n)) if n != ids.len() => {
                return Err(InvalidRequest::PromptLengthMismatch {
                    prompt_token_ids: ids.len(),
                    prompt_embeds: n,
                });
            }
            (Some(ids), _) => ids.len(),
        };
        let all_token_ids = match (&prompt_token_ids, &prompt_is_token_ids) {
            (None, _) => vec![0; num_prompt_tokens],
            (Some(ids), None) => ids.clone(),
            // Mixed-mode prompt: positions covered by prompt_embeds hold a sentinel
            // special token id that may lie outside the embedding. Zero them, matching
            // the no-token-ids case above, so embedding gathers over these placeholder
            // ids stay in bounds; the actual inputs come from prompt_embeds.
            (Some(ids), Some(is_tok)) => {
                ids.iter().zip(is_tok).map(|(&t, &is_tok)| if is_tok { t } else { 0 }).collect()
            }
        };

        let mut request = Self {
            request_id,
            client_index,
            priority,
            sampling_params,
            pooling_params,
            lora_int_id,
            lora_name,
            use_structured_output,
            arrival_time,
            status,
            events: Vec::new(),
            stop_reason: None,
            max_tokens,
            kv_cache_report_mode,
            prompt_token_ids,
            has_prompt_embeds: num_prompt_embeds.is_some(),
            prompt_is_token_ids,
            num_prompt_tokens,
            output_token_ids: Vec::new(),
            all_token_ids,
            num_output_placeholders: 0,
            num_stale_output_tokens: 0,
            drop_stale_output: false,
            num_in_flight_tokens: 0,
            next_decode_eligible_step: 0,
            last_sched_seq: 0,
            spec_token_ids: Vec::new(),
            num_computed_tokens: 0,
            cache_salt,
            mm_features,
            session_id,
            is_prefill_chunk: false,
            shared_prefix_boundary: 0,
            replay_start: 0,
            num_nans_in_logits: 0,
            num_preemptions: 0,
            prefill_stats: Some(PrefillStats::default()),
            block_hashes,
            block_hasher,
            prefix_cache_hit_limit: None,
            resumable,
            streaming_queue: None,
            abort_immediately,
            do_remote_prefill,
            spec_decode_metrics: None,
            seq: NEXT_REQUEST_SEQ.fetch_add(1, Ordering::Relaxed),
        };
        request.update_block_hashes();
        request.prefix_cache_hit_limit = request.get_prefix_cache_hit_limit();
        Ok(request)
    }

    /// All token ids: the prompt (with mixed-mode embedding positions zeroed)
    /// followed by the output tokens.
    pub fn all_token_ids(&self) -> &[i32] {
        &self.all_token_ids
    }

    pub fn output_token_ids(&self) -> &[i32] {
        &self.output_token_ids
    }

    pub fn append_output_token_ids(&mut self, token_ids: &[i32]) {
        self.output_token_ids.extend_from_slice(token_ids);
        self.all_token_ids.extend_from_slice(token_ids);

        self.update_block_hashes();
    }

    /// Compute block hashes for any new full blocks and append them.
    pub fn update_block_hashes(&mut self) {
        if let Some(hasher) = self.block_hasher.clone() {
            let new_hashes = hasher.hash_new_blocks(self);
            self.block_hashes.extend(new_hashes);
        }
    }

    /// Discard hashes invalidated by truncating the request's tokens.
    // TODO: also clear the per-block prompt-embed hash cache once prompt
    // embeds extra keys are hashed in Rust.
    pub fn truncate_block_hashes(
        &mut self,
        hash_block_size: usize,
        num_hash_lookahead_tokens: usize,
    ) {
        let num_valid_hashes =
            self.num_tokens().saturating_sub(num_hash_lookahead_tokens) / hash_block_size;
        self.block_hashes.truncate(num_valid_hashes);
    }

    /// Truncate the token ids to the first `len` tokens, as
    /// `del request._all_token_ids[len:]`.
    pub fn truncate_all_token_ids(&mut self, len: usize) {
        self.all_token_ids.truncate(len);
    }

    /// `request._output_token_ids.clear()`.
    pub fn clear_output_token_ids(&mut self) {
        self.output_token_ids.clear();
    }

    /// `request._all_token_ids.extend(token_ids)`, without touching the
    /// output tokens or block hashes.
    pub fn extend_all_token_ids(&mut self, token_ids: &[i32]) {
        self.all_token_ids.extend_from_slice(token_ids);
    }

    pub fn use_structured_output(&self) -> bool {
        self.use_structured_output
    }

    /// All token ids, as the prompt_token_ids object itself while they hold
    /// the same tokens, so that pickling both sends them once.
    ///
    /// The contents always equal [`Request::all_token_ids`]; the binding
    /// layer can use the same identity trick when converting to Python.
    pub fn prefill_token_ids(&self) -> &[i32] {
        match &self.prompt_token_ids {
            Some(prompt)
                if self.prompt_is_token_ids.is_none()
                    && prompt.len() == self.all_token_ids.len() =>
            {
                prompt
            }
            _ => &self.all_token_ids,
        }
    }

    pub fn num_tokens(&self) -> usize {
        self.all_token_ids.len()
    }

    pub fn num_tokens_with_spec(&self) -> usize {
        self.all_token_ids.len() + self.spec_token_ids.len()
    }

    pub fn num_output_tokens(&self) -> usize {
        self.output_token_ids.len()
    }

    pub fn num_encoder_inputs(&self) -> usize {
        self.mm_features.len()
    }

    pub fn has_encoder_inputs(&self) -> bool {
        self.num_encoder_inputs() > 0
    }

    /// Longest prefix cache hit allowed, optionally tightened by bound.
    ///
    /// Python returns -1 for a request without tokens; requests always have
    /// a non-empty prompt, so this panics instead.
    pub fn get_max_cache_hit_length(&self, bound: Option<usize>) -> usize {
        // The last token is always recomputed to produce logits.
        let last = self.num_tokens().checked_sub(1).expect("request has no tokens");
        [Some(last), self.prefix_cache_hit_limit, bound]
            .into_iter()
            .flatten()
            .min()
            .unwrap()
    }

    pub fn skip_reading_external_kv(&self) -> bool {
        // KV connector hits are not bounded by prefix_cache_hit_limit, so a
        // bounded request reuses only the local prefix cache.
        self.prefix_cache_hit_limit.is_some()
    }

    /// Number of leading tokens that may be read from the prefix cache:
    /// 0 when the request skips cache reads, the first scored prompt row for
    /// prompt scoring, or None when unbounded.
    /// See docs/training/prompt_token_id_logprobs.md.
    pub fn get_prefix_cache_hit_limit(&self) -> Option<usize> {
        let skip = match (&self.sampling_params, &self.pooling_params) {
            (Some(sp), _) => sp.skip_reading_prefix_cache,
            (None, Some(pp)) => pp.skip_reading_prefix_cache,
            (None, None) => false,
        };
        if skip {
            return Some(0);
        }
        if let Some(sp) = &self.sampling_params
            && (sp.has_prompt_logprobs || sp.has_prompt_logprob_token_ids)
        {
            return Some(sp.prompt_logprob_start.unwrap_or(0));
        }
        None
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    pub fn get_finished_reason(&self) -> Option<FinishReason> {
        self.status.get_finished_reason()
    }

    pub fn get_num_encoder_embeds(&self, input_id: usize) -> usize {
        assert!(input_id < self.mm_features.len());
        self.mm_features[input_id].mm_position.get_num_embeds()
    }

    /// Python defaults `timestamp` to `time.monotonic()`.
    // TODO: default to the monotonic clock once the engine core owns one.
    pub fn record_event(&mut self, event_type: EngineCoreEventType, timestamp: f64) {
        self.events.push(EngineCoreEvent {
            r#type: event_type,
            timestamp,
        });
    }

    pub fn take_events(&mut self) -> Option<Vec<EngineCoreEvent>> {
        if self.events.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.events))
    }

    pub fn take_prefill_stats(&mut self) -> Option<PrefillStats> {
        self.prefill_stats.take()
    }

    /// Compare two requests based on priority, arrival time, and request ID.
    /// Used in priority scheduling.
    ///
    /// Python's `__lt__`; see [`Request::seq`] for the final tiebreak.
    pub fn lt(&self, other: &Self) -> bool {
        if self.priority != other.priority {
            return self.priority < other.priority;
        }
        if self.arrival_time != other.arrival_time {
            return self.arrival_time < other.arrival_time;
        }
        if self.request_id != other.request_id {
            // UTF-8 byte order is code point order, as Python compares `str`.
            return self.request_id < other.request_id;
        }
        self.seq < other.seq
    }

    /// Construction order, the final tiebreak of [`Request::lt`].
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

impl KvRequest for Request {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn block_hashes(&self) -> &[BlockHash] {
        &self.block_hashes
    }

    fn num_tokens(&self) -> usize {
        Request::num_tokens(self)
    }

    fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
    }

    fn num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }

    fn num_in_flight_tokens(&self) -> usize {
        self.num_in_flight_tokens
    }

    fn num_preemptions(&self) -> usize {
        self.num_preemptions
    }

    fn shared_prefix_boundary(&self) -> usize {
        self.shared_prefix_boundary
    }

    fn prefix_cache_hit_limit(&self) -> Option<usize> {
        self.prefix_cache_hit_limit
    }

    fn is_waiting_or_preempted(&self) -> bool {
        matches!(
            self.status,
            RequestStatus::Waiting | RequestStatus::Preempted
        )
    }

    fn kv_cache_report_mode_full(&self) -> bool {
        self.kv_cache_report_mode == KvCacheReportMode::Full
    }

    fn token_ids(&self, start: usize, end: usize) -> Vec<i32> {
        let end = end.min(self.all_token_ids.len());
        self.all_token_ids.get(start..end).unwrap_or_default().to_vec()
    }

    fn lora_id(&self) -> Option<i64> {
        self.lora_int_id
    }

    fn lora_name(&self) -> Option<String> {
        self.lora_name.clone()
    }

    fn session_id(&self) -> Option<String> {
        self.session_id.clone()
    }

    /// `generate_block_hash_extra_keys` converted by `to_event_extra_keys`:
    /// the LoRA name, the multi-modal `(identifier, offset)` pairs and the
    /// cache salt.
    // TODO: prompt-embeds digests; the binding layer rejects prompt-embeds
    // requests while KV cache events are enabled.
    fn event_extra_keys(
        &self,
        start: usize,
        end: usize,
        start_mm_idx: isize,
    ) -> (Option<EventExtraKeys>, isize) {
        let mut keys = EventExtraKeys::new();
        if let Some(lora_name) = &self.lora_name {
            keys.push(EventExtraKey::Str(lora_name.clone()));
        }
        let next_mm_idx = self.mm_event_extra_keys(start, end, start_mm_idx, &mut keys);
        if start == 0
            && let Some(salt) = self.cache_salt.as_ref().filter(|s| !s.is_empty())
        {
            keys.push(EventExtraKey::Str(salt.clone()));
        }
        ((!keys.is_empty()).then_some(keys), next_mm_idx)
    }

    fn num_mm_features(&self) -> usize {
        self.mm_features.len()
    }

    fn mm_feature_range(&self, idx: usize) -> (usize, usize) {
        let position = &self.mm_features[idx].mm_position;
        (position.offset, position.length)
    }
}

impl Request {
    /// `_gen_mm_extra_hash_keys` in event format: appends the
    /// `(identifier, offset)` of each multi-modal input in `[start, end)` and
    /// returns the next multi-modal index.
    fn mm_event_extra_keys(
        &self,
        start: usize,
        end: usize,
        start_mm_idx: isize,
        keys: &mut EventExtraKeys,
    ) -> isize {
        let Some(last) = self.mm_features.last() else {
            return start_mm_idx;
        };
        // Note that we assume mm_features are sorted by mm_position.offset.
        // We do not need to check all mm inputs if the start token index is out
        // of range. This usually happens in the late prefill phase and decoding
        // phase.
        if last.mm_position.offset + last.mm_position.length <= start {
            return start_mm_idx;
        }
        // Support start_mm_idx == -1 to indicate the last mm input.
        let mut curr_mm_idx = if start_mm_idx < 0 {
            assert!(start_mm_idx.unsigned_abs() <= self.mm_features.len());
            self.mm_features.len() as isize + start_mm_idx
        } else {
            start_mm_idx
        } as usize;
        while let Some(feature) = self.mm_features.get(curr_mm_idx) {
            let offset = feature.mm_position.offset;
            let length = feature.mm_position.length;
            if end <= offset {
                // This block has not reached the current mm input.
                break;
            }
            if start >= offset + length {
                // This block has passed the current mm input.
                curr_mm_idx += 1;
                continue;
            }
            keys.push(EventExtraKey::Mm {
                identifier: feature.identifier.clone(),
                offset: offset as i64 - start as i64,
            });
            if end >= offset + length {
                // If this block contains the end of the current mm input,
                // move to the next mm input as this block may also contain
                // the next mm input.
                curr_mm_idx += 1;
            } else {
                // Otherwise this block is done with mm inputs.
                break;
            }
        }
        curr_mm_idx as isize
    }
}

impl fmt::Debug for Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request")
            .field("request_id", &self.request_id)
            .field("status", &self.status)
            .field("priority", &self.priority)
            .field("arrival_time", &self.arrival_time)
            .field("num_prompt_tokens", &self.num_prompt_tokens)
            .field("num_tokens", &self.num_tokens())
            .field("num_computed_tokens", &self.num_computed_tokens)
            .finish_non_exhaustive()
    }
}

#[cfg(any(test, feature = "test-util"))]
impl NewRequest {
    /// A generative request with `prompt_token_ids` and default sampling
    /// params.
    pub fn for_test(request_id: &str, prompt_token_ids: Vec<i32>) -> Self {
        Self {
            request_id: request_id.to_owned(),
            prompt_token_ids: Some(prompt_token_ids),
            sampling_params: Some(SamplingParams::default()),
            arrival_time: Some(0.0),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(args: NewRequest) -> Request {
        Request::new(args).unwrap()
    }

    #[test]
    fn status_ordering_and_reasons() {
        use RequestStatus::*;
        let all = [
            Waiting,
            WaitingForStructuredOutputGrammar,
            WaitingForRemoteKvs,
            WaitingForStreamingReq,
            Running,
            Preempted,
            FinishedStopped,
            FinishedLengthCapped,
            FinishedAborted,
            FinishedIgnored,
            FinishedError,
            FinishedRepetition,
        ];
        let summary: Vec<_> = all
            .iter()
            .map(|s| {
                format!(
                    "{}={} finished={} reason={:?}",
                    s,
                    *s as u8,
                    s.is_finished(),
                    s.get_finished_reason()
                )
            })
            .collect();
        expect_test::expect![[r#"
            [
                "WAITING=1 finished=false reason=None",
                "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR=2 finished=false reason=None",
                "WAITING_FOR_REMOTE_KVS=3 finished=false reason=None",
                "WAITING_FOR_STREAMING_REQ=4 finished=false reason=Some(Stop)",
                "RUNNING=5 finished=false reason=None",
                "PREEMPTED=6 finished=false reason=None",
                "FINISHED_STOPPED=7 finished=true reason=Some(Stop)",
                "FINISHED_LENGTH_CAPPED=8 finished=true reason=Some(Length)",
                "FINISHED_ABORTED=9 finished=true reason=Some(Abort)",
                "FINISHED_IGNORED=10 finished=true reason=Some(Length)",
                "FINISHED_ERROR=11 finished=true reason=Some(Error)",
                "FINISHED_REPETITION=12 finished=true reason=Some(Repetition)",
            ]
        "#]]
        .assert_debug_eq(&summary);
    }

    #[test]
    fn init_validation_and_status() {
        let missing = NewRequest {
            sampling_params: None,
            ..NewRequest::for_test("r", vec![1])
        };
        assert_eq!(
            Request::new(missing).unwrap_err(),
            InvalidRequest::MissingParams
        );
        let mismatch = NewRequest {
            num_prompt_embeds: Some(2),
            ..NewRequest::for_test("r", vec![1])
        };
        assert!(matches!(
            Request::new(mismatch),
            Err(InvalidRequest::PromptLengthMismatch { .. })
        ));

        let structured = request(NewRequest {
            sampling_params: Some(SamplingParams {
                use_structured_output: true,
                ..Default::default()
            }),
            ..NewRequest::for_test("r", vec![1])
        });
        assert_eq!(
            structured.status,
            RequestStatus::WaitingForStructuredOutputGrammar
        );

        let pooling = request(NewRequest {
            sampling_params: None,
            pooling_params: Some(PoolingParams::default()),
            ..NewRequest::for_test("r", vec![1, 2])
        });
        assert_eq!(pooling.max_tokens, 1);
        assert_eq!(pooling.status, RequestStatus::Waiting);
    }

    #[test]
    fn token_storage() {
        let embeds_only = request(NewRequest {
            prompt_token_ids: None,
            num_prompt_embeds: Some(3),
            ..NewRequest::for_test("r", vec![])
        });
        assert_eq!(embeds_only.all_token_ids(), &[0, 0, 0]);
        assert_eq!(embeds_only.num_prompt_tokens, 3);

        let mixed = request(NewRequest {
            num_prompt_embeds: Some(3),
            prompt_is_token_ids: Some(vec![true, false, true]),
            ..NewRequest::for_test("r", vec![7, 99999, 9])
        });
        assert_eq!(mixed.all_token_ids(), &[7, 0, 9]);
        assert_eq!(mixed.prefill_token_ids(), &[7, 0, 9]);

        let mut r = request(NewRequest::for_test("r", vec![1, 2, 3]));
        assert_eq!(r.prefill_token_ids(), &[1, 2, 3]);
        r.append_output_token_ids(&[4]);
        r.append_output_token_ids(&[5, 6]);
        r.spec_token_ids = vec![7];
        assert_eq!(r.output_token_ids(), &[4, 5, 6]);
        assert_eq!(r.all_token_ids(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(r.prefill_token_ids(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            (
                r.num_tokens(),
                r.num_output_tokens(),
                r.num_tokens_with_spec()
            ),
            (6, 3, 7)
        );
    }

    struct CountingHasher {
        block_size: usize,
    }

    impl BlockHasher for CountingHasher {
        fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
            let start = request.block_hashes.len();
            let end = request.num_tokens() / self.block_size;
            (start..end).map(|i| BlockHash::from_slice(&[i as u8])).collect()
        }
    }

    #[test]
    fn block_hashes_follow_tokens() {
        let mut r = request(NewRequest {
            block_hasher: Some(Arc::new(CountingHasher { block_size: 2 })),
            block_hashes: vec![BlockHash::from_slice(&[0])],
            ..NewRequest::for_test("r", vec![1, 2, 3, 4, 5])
        });
        assert_eq!(r.block_hashes.len(), 2);
        r.append_output_token_ids(&[6]);
        assert_eq!(r.block_hashes.len(), 3);
        r.truncate_block_hashes(2, 3);
        assert_eq!(r.block_hashes.len(), 1);
        r.truncate_block_hashes(2, 100);
        assert_eq!(r.block_hashes.len(), 0);
    }

    #[test]
    fn prefix_cache_hit_limit() {
        let with = |sp: SamplingParams| {
            request(NewRequest {
                sampling_params: Some(sp),
                ..NewRequest::for_test("r", vec![1, 2, 3, 4, 5])
            })
        };

        let unbounded = with(SamplingParams::default());
        assert_eq!(unbounded.prefix_cache_hit_limit, None);
        assert!(!unbounded.skip_reading_external_kv());
        assert_eq!(unbounded.get_max_cache_hit_length(None), 4);
        assert_eq!(unbounded.get_max_cache_hit_length(Some(2)), 2);

        let skip = with(SamplingParams {
            skip_reading_prefix_cache: true,
            has_prompt_logprobs: true,
            prompt_logprob_start: Some(3),
            ..Default::default()
        });
        assert_eq!(skip.prefix_cache_hit_limit, Some(0));
        assert!(skip.skip_reading_external_kv());
        assert_eq!(skip.get_max_cache_hit_length(Some(2)), 0);

        let scored = with(SamplingParams {
            has_prompt_logprob_token_ids: true,
            prompt_logprob_start: Some(3),
            ..Default::default()
        });
        assert_eq!(scored.prefix_cache_hit_limit, Some(3));
        assert_eq!(scored.get_max_cache_hit_length(None), 3);

        let scored_from_start = with(SamplingParams {
            has_prompt_logprobs: true,
            ..Default::default()
        });
        assert_eq!(scored_from_start.prefix_cache_hit_limit, Some(0));

        let pooling = request(NewRequest {
            sampling_params: None,
            pooling_params: Some(PoolingParams {
                skip_reading_prefix_cache: true,
            }),
            ..NewRequest::for_test("r", vec![1])
        });
        assert_eq!(pooling.prefix_cache_hit_limit, Some(0));
    }

    #[test]
    fn lt_orders_by_priority_arrival_id_then_seq() {
        let mk = |id: &str, priority, arrival_time| {
            request(NewRequest {
                priority,
                arrival_time: Some(arrival_time),
                ..NewRequest::for_test(id, vec![1])
            })
        };
        assert!(mk("b", 0, 5.0).lt(&mk("a", 1, 0.0)));
        assert!(mk("b", 0, 1.0).lt(&mk("a", 0, 2.0)));
        assert!(mk("a", 0, 1.0).lt(&mk("b", 0, 1.0)));
        // Code point order: 'z' < 'é' < '中' < '😀'.
        assert!(mk("z", 0, 1.0).lt(&mk("é", 0, 1.0)));
        assert!(mk("é", 0, 1.0).lt(&mk("中", 0, 1.0)));
        assert!(mk("中", 0, 1.0).lt(&mk("😀", 0, 1.0)));
        let (first, second) = (mk("a", 0, 1.0), mk("a", 0, 1.0));
        assert!(first.lt(&second) && !second.lt(&first) && !first.lt(&first));
    }

    #[test]
    fn events_and_stats() {
        let mut r = request(NewRequest::for_test("r", vec![1]));
        assert_eq!(r.take_events(), None);
        r.record_event(EngineCoreEventType::Queued, 1.5);
        assert_eq!(r.take_events().unwrap().len(), 1);
        assert!(r.events.is_empty());

        let mut stats = r.take_prefill_stats().unwrap();
        assert_eq!(r.take_prefill_stats(), None);
        stats.set(10, 4, 2);
        stats.finalize(8);
        assert_eq!((stats.num_computed_tokens, stats.num_cached_tokens), (4, 6));
        assert_eq!(stats.num_cache_creation_tokens, 2);
        stats.finalize(3);
        assert_eq!(stats.num_cache_creation_tokens, 0);
    }

    #[test]
    fn streaming_update_from_request() {
        let r = request(NewRequest::for_test("r", vec![1]));
        assert!(StreamingUpdate::from_request(&r).is_none());
        let r = request(NewRequest {
            resumable: true,
            ..NewRequest::for_test("r", vec![1, 2])
        });
        let update = StreamingUpdate::from_request(&r).unwrap();
        assert_eq!(update.prompt_token_ids, Some(vec![1, 2]));
        assert_eq!(update.max_tokens, 16);
    }
}
